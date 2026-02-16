"""
Background market snapshot cache for Kalshi.

Daemon thread refreshes asynchronously so the cross-platform scanner
never blocks on Kalshi's paginated REST API (120s+ for full scan).
"""

from __future__ import annotations

import logging
import threading
import time
from dataclasses import dataclass, field

from client.kalshi import KalshiClient, KalshiMarket

logger = logging.getLogger(__name__)

_DEFAULT_REFRESH_SEC = 300.0
_DEFAULT_WARM_TIMEOUT_SEC = 180.0
_BACKOFF_START_SEC = 30.0
_BACKOFF_CAP_SEC = 300.0

@dataclass(frozen=True)
class KalshiMarketSnapshot:
    """Immutable point-in-time snapshot of all open Kalshi markets."""

    version: int
    timestamp: float
    markets: tuple[KalshiMarket, ...]
    by_event: dict[str, list[KalshiMarket]] = field(default_factory=dict)
    titles: dict[str, str] = field(default_factory=dict)

def _build_snapshot(
    markets: list[KalshiMarket], version: int,
) -> KalshiMarketSnapshot:
    """Build an immutable snapshot from a raw market list."""
    by_event: dict[str, list[KalshiMarket]] = {}
    titles: dict[str, str] = {}
    for m in markets:
        key = m.event_ticker
        if key not in by_event:
            by_event[key] = []
            titles[key] = m.title
        by_event[key].append(m)
    return KalshiMarketSnapshot(
        version=version,
        timestamp=time.time(),
        markets=tuple(markets),
        by_event=by_event,
        titles=titles,
    )

class KalshiMarketCache:
    """
    Background-refreshing cache for Kalshi markets.

    Thread safety: a single Lock protects the snapshot pointer swap.
    Readers get an immutable frozen dataclass — no copy needed.
    """

    def __init__(
        self,
        client: KalshiClient,
        refresh_sec: float = _DEFAULT_REFRESH_SEC,
        warm_timeout_sec: float = _DEFAULT_WARM_TIMEOUT_SEC,
    ) -> None:
        self._client = client
        self._refresh_sec = refresh_sec
        self._warm_timeout_sec = warm_timeout_sec

        self._lock = threading.Lock()
        self._snapshot: KalshiMarketSnapshot | None = None
        self._version = 0

        self._stop_event = threading.Event()
        self._thread: threading.Thread | None = None

    # -- Public API --

    def start(self) -> None:
        """Spawn background daemon that warms then refreshes periodically.

        Non-blocking: returns immediately.  The cross-platform scanner
        already handles ``snapshot() is None`` by skipping the Kalshi
        leg, so there is no need to block the main loop while the
        initial 123K-market fetch completes (~2 min on cold start).
        """
        self._thread = threading.Thread(
            target=self._warm_then_refresh, name="kalshi-cache", daemon=True,
        )
        self._thread.start()
        logger.info(
            "KalshiMarketCache started (warming in background, refresh every %.0fs)",
            self._refresh_sec,
        )

    def snapshot(self) -> KalshiMarketSnapshot | None:
        """Atomic read of current snapshot. None if not yet warmed."""
        with self._lock:
            return self._snapshot

    def stop(self) -> None:
        """Signal daemon to exit and wait for it."""
        self._stop_event.set()
        if self._thread is not None:
            self._thread.join(timeout=10.0)
            self._thread = None
        logger.info("KalshiMarketCache stopped")

    # -- Internals --

    def _warm_then_refresh(self) -> None:
        """Warm first, then enter the periodic refresh loop."""
        self._warm()
        self._refresh_loop()

    def _warm(self) -> None:
        """Blocking first fetch. On failure, log and leave snapshot as None."""
        try:
            markets = self._client.get_all_markets(status="open")
            self._version += 1
            snap = _build_snapshot(markets, self._version)
            with self._lock:
                self._snapshot = snap
            logger.info(
                "KalshiMarketCache warmed: v%d, %d markets, %d events",
                snap.version, len(snap.markets), len(snap.by_event),
            )
        except Exception:
            logger.exception("KalshiMarketCache warm fetch failed — snapshot stays None")

    def _refresh_loop(self) -> None:
        """Daemon loop: sleep, fetch, swap pointer. Stale-while-error."""
        consecutive_failures = 0

        while not self._stop_event.is_set():
            sleep_sec = self._refresh_sec if consecutive_failures == 0 else min(
                _BACKOFF_START_SEC * (2 ** (consecutive_failures - 1)),
                _BACKOFF_CAP_SEC,
            )
            if self._stop_event.wait(timeout=sleep_sec):
                break

            try:
                markets = self._client.get_all_markets(status="open")
                self._version += 1
                snap = _build_snapshot(markets, self._version)
                with self._lock:
                    self._snapshot = snap
                consecutive_failures = 0
                logger.info(
                    "KalshiMarketCache refreshed: v%d, %d markets, %d events",
                    snap.version, len(snap.markets), len(snap.by_event),
                )
            except Exception:
                consecutive_failures += 1
                logger.warning(
                    "KalshiMarketCache refresh failed (streak=%d, backoff=%.0fs)",
                    consecutive_failures,
                    min(_BACKOFF_START_SEC * (2 ** (consecutive_failures - 1)), _BACKOFF_CAP_SEC),
                    exc_info=True,
                )
