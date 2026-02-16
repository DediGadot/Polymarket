"""
Order Flow Imbalance (OFI) tracker. Accumulates aggressive buy/sell volume
from WebSocket delta events to produce a leading indicator for price moves.

OFI divergence between YES/NO tokens in the same event predicts imminent
rebalancing -- high-divergence arbs are more likely to fill before the book
adjusts.
"""

from __future__ import annotations

import logging
import threading
import time
from collections import defaultdict, deque
from dataclasses import dataclass

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class OFISnapshot:
    """Immutable snapshot of OFI state for a single token."""

    token_id: str
    ofi: float  # net aggressive volume (positive = buy pressure)
    normalized_ofi: float  # bounded -1 to +1
    total_volume: float
    event_count: int
    timestamp: float


class OFITracker:
    """
    Accumulates aggressive buy/sell volume from WS delta events.
    Per-token rolling window. Thread-safe via Lock.

    Usage:
        tracker = OFITracker(window_sec=30.0)
        tracker.record("token_a", "BUY", 100.0)
        tracker.record("token_a", "SELL", 50.0)
        ofi = tracker.get_ofi("token_a")          # 50.0
        norm = tracker.get_normalized_ofi("token_a")  # 0.333
        div = tracker.get_divergence("token_a", "token_b")
    """

    def __init__(self, window_sec: float = 30.0) -> None:
        self._window_sec = window_sec
        # token_id -> deque of (timestamp, signed_volume)
        self._events: dict[str, deque[tuple[float, float]]] = defaultdict(deque)
        # Rolling (signal, future_move) pairs for lightweight quality telemetry.
        self._quality_pairs: deque[tuple[float, float]] = deque(maxlen=2000)
        self._lock = threading.Lock()

    def record(
        self,
        token_id: str,
        side: str,
        size: float,
        timestamp: float | None = None,
    ) -> None:
        """
        Record an aggressive order event.

        Args:
            token_id: The token that was traded.
            side: "BUY" or "SELL".
            size: Volume of the trade.
            timestamp: Event timestamp (defaults to now).
        """
        ts = timestamp if timestamp is not None else time.time()
        signed = size if side.upper() == "BUY" else -size

        with self._lock:
            self._events[token_id].append((ts, signed))
            self._prune_token(token_id, ts)

    def record_aggressor(
        self,
        token_id: str,
        buy_volume: float,
        sell_volume: float,
        timestamp: float | None = None,
    ) -> None:
        """Record aggressor-flow approximation from book deltas."""
        ts = timestamp if timestamp is not None else time.time()
        if buy_volume > 0:
            self.record(token_id, "BUY", buy_volume, timestamp=ts)
        if sell_volume > 0:
            self.record(token_id, "SELL", sell_volume, timestamp=ts)

    def record_quality(self, ofi_signal: float, future_move: float) -> None:
        """
        Track OFI signal quality against realized short-horizon move.
        Positive correlation indicates predictive lift.
        """
        with self._lock:
            self._quality_pairs.append((ofi_signal, future_move))

    def get_ofi(self, token_id: str) -> float:
        """
        Raw OFI for a token: sum of signed volumes in the window.
        Positive = net buy pressure, negative = net sell pressure.
        """
        with self._lock:
            events = self._events.get(token_id)
            if not events:
                return 0.0
            return sum(v for _, v in events)

    def get_normalized_ofi(self, token_id: str) -> float:
        """
        Normalized OFI bounded to [-1, +1].
        normalized = ofi / total_volume.  Returns 0.0 if no volume.
        """
        with self._lock:
            events = self._events.get(token_id)
            if not events:
                return 0.0
            ofi = sum(v for _, v in events)
            total = sum(abs(v) for _, v in events)
            if total <= 0:
                return 0.0
            return max(-1.0, min(1.0, ofi / total))

    def get_divergence(self, token_a: str, token_b: str) -> float:
        """
        OFI divergence between two tokens.
        High absolute divergence = market about to correct.
        """
        ofi_a = self.get_ofi(token_a)
        ofi_b = self.get_ofi(token_b)
        return abs(ofi_a - ofi_b)

    def get_snapshot(self, token_id: str) -> OFISnapshot:
        """Get a frozen snapshot of OFI state for a token."""
        with self._lock:
            now = time.time()
            events = self._events.get(token_id)
            if not events:
                return OFISnapshot(
                    token_id=token_id,
                    ofi=0.0,
                    normalized_ofi=0.0,
                    total_volume=0.0,
                    event_count=0,
                    timestamp=now,
                )
            ofi = sum(v for _, v in events)
            total = sum(abs(v) for _, v in events)
            normalized = max(-1.0, min(1.0, ofi / total)) if total > 0 else 0.0
            return OFISnapshot(
                token_id=token_id,
                ofi=ofi,
                normalized_ofi=normalized,
                total_volume=total,
                event_count=len(events),
                timestamp=now,
            )

    def cleanup_stale(self, active_tokens: set[str]) -> int:
        """Remove tokens not in active set. Returns count removed."""
        with self._lock:
            stale = set(self._events.keys()) - active_tokens
            for tid in stale:
                del self._events[tid]
            return len(stale)

    def to_dict(self) -> dict:
        """Serialize for checkpoint persistence."""
        with self._lock:
            return {
                "window_sec": self._window_sec,
                "events": {k: list(v) for k, v in self._events.items()},
                "quality_pairs": list(self._quality_pairs),
            }

    @classmethod
    def from_dict(cls, data: dict) -> OFITracker:
        """Restore from checkpoint."""
        tracker = cls(window_sec=data.get("window_sec", 30.0))
        for k, v in data.get("events", {}).items():
            tracker._events[k] = deque(tuple(e) for e in v)
        for pair in data.get("quality_pairs", []):
            try:
                sig, move = pair
                tracker._quality_pairs.append((float(sig), float(move)))
            except Exception:
                continue
        return tracker

    @property
    def tracked_tokens(self) -> int:
        """Number of tokens currently being tracked."""
        return len(self._events)

    @property
    def quality_correlation(self) -> float:
        """Pearson correlation between OFI signal and realized move."""
        with self._lock:
            n = len(self._quality_pairs)
            if n < 5:
                return 0.0
            xs = [x for x, _ in self._quality_pairs]
            ys = [y for _, y in self._quality_pairs]
            mean_x = sum(xs) / n
            mean_y = sum(ys) / n
            cov = sum((x - mean_x) * (y - mean_y) for x, y in self._quality_pairs)
            var_x = sum((x - mean_x) ** 2 for x in xs)
            var_y = sum((y - mean_y) ** 2 for y in ys)
            if var_x <= 0 or var_y <= 0:
                return 0.0
            return cov / ((var_x * var_y) ** 0.5)

    def _prune_token(self, token_id: str, now: float) -> None:
        """Remove events older than window_sec. Must hold lock."""
        events = self._events.get(token_id)
        if not events:
            return
        cutoff = now - self._window_sec
        while events and events[0][0] < cutoff:
            events.popleft()
        if not events:
            del self._events[token_id]
