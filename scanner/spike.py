"""
Event-driven spike detection. Detects rapid price movements in one market
and checks sibling markets in the same event for lag -- where related markets
haven't yet adjusted proportionally.

During breaking news, one market reprices instantly while related markets
lag by 5-60 seconds. This is where edge is 10-50x wider than steady-state
rebalancing.
"""

from __future__ import annotations

import logging
import time
from collections import deque
from dataclasses import dataclass, field

from scanner.fees import MarketFeeModel
from scanner.book_cache import BookCache
from scanner.models import (
    Event,
    Market,
    OrderBook,
    Opportunity,
    OpportunityType,
    LegOrder,
    Side,
)

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class SpikeEvent:
    """A detected price spike in a specific token."""
    token_id: str
    event_id: str
    direction: float  # positive = price went up, negative = down
    magnitude_pct: float  # absolute % change
    velocity: float  # % change per second
    timestamp: float


@dataclass
class PriceHistory:
    """Rolling window of (timestamp, midpoint) for a single token."""
    max_window_sec: float = 300.0
    _points: deque[tuple[float, float]] = field(default_factory=deque)

    def record(self, price: float, timestamp: float) -> None:
        """Add a price observation and prune old entries."""
        self._points.append((timestamp, price))
        cutoff = timestamp - self.max_window_sec
        while self._points and self._points[0][0] < cutoff:
            self._points.popleft()

    def velocity(self, window_sec: float) -> float | None:
        """
        Price change per second over the last window_sec.
        Returns None if insufficient data.
        """
        if len(self._points) < 2:
            return None
        now = self._points[-1][0]
        cutoff = now - window_sec
        # Find oldest point within window
        oldest = None
        for ts, price in self._points:
            if ts >= cutoff:
                oldest = (ts, price)
                break
        if oldest is None or oldest[0] == self._points[-1][0]:
            return None
        dt = self._points[-1][0] - oldest[0]
        if dt <= 0:
            return None
        return (self._points[-1][1] - oldest[1]) / dt

    def pct_change(self, window_sec: float) -> float | None:
        """
        Percent change over the last window_sec.
        Returns None if insufficient data.
        """
        if len(self._points) < 2:
            return None
        now = self._points[-1][0]
        cutoff = now - window_sec
        # Find oldest point within window
        oldest_price = None
        for ts, price in self._points:
            if ts >= cutoff:
                oldest_price = price
                break
        if oldest_price is None or oldest_price <= 0:
            return None
        return ((self._points[-1][1] - oldest_price) / oldest_price) * 100.0

    @property
    def latest(self) -> tuple[float, float] | None:
        """Most recent (timestamp, price) or None."""
        return self._points[-1] if self._points else None

    def __len__(self) -> int:
        return len(self._points)


@dataclass
class SpikeDetector:
    """
    Detects price spikes across all tracked tokens and identifies
    lagging sibling markets for spike-lag arbitrage.
    """
    threshold_pct: float = 5.0
    window_sec: float = 30.0
    cooldown_sec: float = 60.0

    _histories: dict[str, PriceHistory] = field(default_factory=dict)
    # Track which tokens were recently spiked to avoid re-triggering
    _cooldowns: dict[str, float] = field(default_factory=dict)
    # Map token_id -> event_id for sibling lookup
    _token_events: dict[str, str] = field(default_factory=dict)

    def cleanup_stale(self, active_tokens: set[str]) -> None:
        """
        Remove tokens that are no longer active.

        Prevents memory leak in long-running sessions by pruning
        histories and token_event mappings for tokens no longer
        in the active market set.

        Args:
            active_tokens: Set of token_ids that are currently active
        """
        # Find tokens to remove (in our data but not active)
        stale_tokens = set(self._histories.keys()) - active_tokens

        for token_id in stale_tokens:
            self._histories.pop(token_id, None)
            self._token_events.pop(token_id, None)
            self._cooldowns.pop(token_id, None)

        if stale_tokens:
            logger.debug(
                "SpikeDetector: cleaned up %d stale tokens (%d remaining)",
                len(stale_tokens), len(self._histories),
            )

    def register_token(self, token_id: str, event_id: str) -> None:
        """Register a token's event membership for sibling lookup."""
        self._token_events[token_id] = event_id

    def update(self, token_id: str, price: float, timestamp: float | None = None) -> None:
        """Record a price update (from WS last_trade_price or book midpoint)."""
        ts = timestamp or time.time()
        if token_id not in self._histories:
            self._histories[token_id] = PriceHistory()
        self._histories[token_id].record(price, ts)

    def detect_spikes(self) -> list[SpikeEvent]:
        """
        Check all tracked tokens for spikes exceeding threshold.
        Returns SpikeEvents for tokens that spiked and are not in cooldown.
        """
        now = time.time()
        spikes: list[SpikeEvent] = []

        for token_id, history in self._histories.items():
            # Check cooldown
            if token_id in self._cooldowns and (now - self._cooldowns[token_id]) < self.cooldown_sec:
                continue

            pct = history.pct_change(self.window_sec)
            if pct is None:
                continue

            if abs(pct) >= self.threshold_pct:
                vel = history.velocity(self.window_sec)
                event_id = self._token_events.get(token_id, "")
                spikes.append(SpikeEvent(
                    token_id=token_id,
                    event_id=event_id,
                    direction=pct,
                    magnitude_pct=abs(pct),
                    velocity=vel or 0.0,
                    timestamp=now,
                ))
                self._cooldowns[token_id] = now

        return spikes

    def get_velocity(self, token_id: str, window_sec: float | None = None) -> float | None:
        """Get price velocity for a token."""
        history = self._histories.get(token_id)
        if not history:
            return None
        return history.velocity(window_sec or self.window_sec)


def scan_spike_opportunities(
    spike: SpikeEvent,
    event: Event,
    book_cache: BookCache,
    fee_model: MarketFeeModel,
    gas_cost_usd: float = 0.005,
    min_profit_usd: float = 0.50,
) -> list[Opportunity]:
    """
    Given a spike in one market of an event, check if sibling markets
    have lagged behind, creating an arb opportunity.

    For NegRisk events: if one outcome spikes down, others should spike up
    to maintain sum = 1. If they haven't adjusted, buy the lagging ones.
    """
    opps: list[Opportunity] = []

    if not event.neg_risk or len(event.markets) < 2:
        return opps

    # Get current books for all outcomes
    active_markets = [m for m in event.markets if m.active]
    books: dict[str, OrderBook | None] = {}
    for m in active_markets:
        books[m.yes_token_id] = book_cache.get_book(m.yes_token_id)

    # Check if sum of best asks < 1.0 (standard NegRisk arb, amplified by spike)
    total_ask = 0.0
    all_have_asks = True
    ask_data: list[tuple[Market, float, float]] = []

    for m in active_markets:
        book = books.get(m.yes_token_id)
        if not book or not book.best_ask:
            all_have_asks = False
            break
        ask_data.append((m, book.best_ask.price, book.best_ask.size))
        total_ask += book.best_ask.price

    if not all_have_asks or total_ask >= 1.0:
        return opps

    profit_per_set = 1.0 - total_ask
    max_sets = min(size for _, _, size in ask_data)
    if max_sets <= 0:
        return opps

    n_legs = len(ask_data)
    total_gas = gas_cost_usd * n_legs

    legs = tuple(
        LegOrder(
            token_id=m.yes_token_id,
            side=Side.BUY,
            price=price,
            size=max_sets,
        )
        for m, price, _ in ask_data
    )

    # Apply fees
    event_markets = [m for m, _, _ in ask_data]
    net_profit_per_set = fee_model.adjust_profit(profit_per_set, legs, markets=event_markets)
    net_profit = net_profit_per_set * max_sets - total_gas
    required_capital = total_ask * max_sets
    roi_pct = (net_profit / required_capital * 100) if required_capital > 0 else 0

    if net_profit < min_profit_usd:
        return opps

    logger.info(
        "SPIKE LAG ARB: %s | spike=%s %.1f%% | %d outcomes cost=%.4f net=$%.2f roi=%.2f%%",
        event.title[:40], spike.token_id[:10], spike.magnitude_pct,
        n_legs, total_ask, net_profit, roi_pct,
    )

    opps.append(Opportunity(
        type=OpportunityType.SPIKE_LAG,
        event_id=event.event_id,
        legs=legs,
        expected_profit_per_set=profit_per_set,
        net_profit_per_set=net_profit_per_set,
        max_sets=max_sets,
        gross_profit=profit_per_set * max_sets,
        estimated_gas_cost=total_gas,
        net_profit=net_profit,
        roi_pct=roi_pct,
        required_capital=required_capital,
    ))

    return opps
