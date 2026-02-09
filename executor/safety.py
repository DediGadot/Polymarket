"""
Pre-trade safety checks and circuit breakers. Fail-fast on violations.
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field

from client.gas import GasOracle
from scanner.depth import sweep_depth, sweep_cost
from scanner.models import Opportunity, OpportunityType, Side, OrderBook

from client.data import PositionTracker

logger = logging.getLogger(__name__)


class CircuitBreakerTripped(Exception):
    """Raised when a circuit breaker condition is met. Bot should halt."""
    pass


class SafetyCheckFailed(Exception):
    """Raised when a pre-trade safety check fails. Trade should be skipped."""
    pass


@dataclass
class CircuitBreaker:
    """
    Tracks losses and failures. Trips (raises) when limits are exceeded.
    """
    max_loss_per_hour: float
    max_loss_per_day: float
    max_consecutive_failures: int

    _hourly_losses: list[tuple[float, float]] = field(default_factory=list)  # (timestamp, loss)
    _daily_losses: list[tuple[float, float]] = field(default_factory=list)
    _consecutive_failures: int = 0

    def record_trade(self, pnl: float) -> None:
        """Record a trade result. Trips breaker if limits exceeded."""
        now = time.time()
        if pnl < 0:
            self._hourly_losses.append((now, abs(pnl)))
            self._daily_losses.append((now, abs(pnl)))
            self._consecutive_failures += 1
        else:
            self._consecutive_failures = 0

        self._check()

    def _check(self) -> None:
        """Check all breaker conditions. Raises CircuitBreakerTripped on violation."""
        now = time.time()

        # Prune old entries
        hour_ago = now - 3600
        day_ago = now - 86400
        self._hourly_losses = [(t, loss) for t, loss in self._hourly_losses if t > hour_ago]
        self._daily_losses = [(t, loss) for t, loss in self._daily_losses if t > day_ago]

        hourly_total = sum(loss for _, loss in self._hourly_losses)
        if hourly_total >= self.max_loss_per_hour:
            raise CircuitBreakerTripped(
                f"Hourly loss limit exceeded: ${hourly_total:.2f} >= ${self.max_loss_per_hour:.2f}"
            )

        daily_total = sum(loss for _, loss in self._daily_losses)
        if daily_total >= self.max_loss_per_day:
            raise CircuitBreakerTripped(
                f"Daily loss limit exceeded: ${daily_total:.2f} >= ${self.max_loss_per_day:.2f}"
            )

        if self._consecutive_failures >= self.max_consecutive_failures:
            raise CircuitBreakerTripped(
                f"Consecutive failures: {self._consecutive_failures} >= {self.max_consecutive_failures}"
            )


def verify_prices_fresh(
    opportunity: Opportunity,
    books: dict[str, OrderBook],
    max_slippage: float = 0.005,
) -> None:
    """
    Verify current prices haven't moved beyond slippage tolerance.
    Raises SafetyCheckFailed if quotes are stale.
    """
    for leg in opportunity.legs:
        book = books.get(leg.token_id)
        if not book:
            raise SafetyCheckFailed(f"No orderbook for {leg.token_id}")

        if leg.side == Side.BUY:
            if not book.best_ask:
                raise SafetyCheckFailed(f"No ask for {leg.token_id}")
            current_price = book.best_ask.price
            if current_price > leg.price + max_slippage:
                raise SafetyCheckFailed(
                    f"Ask moved: {leg.token_id} was {leg.price:.4f} now {current_price:.4f}"
                )
        else:
            if not book.best_bid:
                raise SafetyCheckFailed(f"No bid for {leg.token_id}")
            current_price = book.best_bid.price
            if current_price < leg.price - max_slippage:
                raise SafetyCheckFailed(
                    f"Bid moved: {leg.token_id} was {leg.price:.4f} now {current_price:.4f}"
                )


def verify_gas_reasonable(
    gas_oracle: GasOracle,
    opportunity: Opportunity,
    gas_per_order: int,
    max_gas_profit_ratio: float = 0.50,
    size: float | None = None,
) -> None:
    """
    Verify gas cost is not an unreasonable fraction of expected profit.
    Raises SafetyCheckFailed if gas > max_gas_profit_ratio * net_profit.
    """
    n_legs = len(opportunity.legs)
    gas_cost = gas_oracle.estimate_cost_usd(n_legs, gas_per_order)

    if size is None:
        expected_net_profit = opportunity.net_profit
        if expected_net_profit <= 0:
            raise SafetyCheckFailed(
                f"Opportunity has non-positive net profit: ${expected_net_profit:.2f}"
            )
    else:
        if size <= 0:
            raise SafetyCheckFailed(f"Invalid execution size for gas check: {size}")
        net_per_set = opportunity.net_profit_per_set
        expected_net_profit = net_per_set * size - gas_cost
        if expected_net_profit <= 0:
            raise SafetyCheckFailed(
                f"Opportunity has non-positive net profit after sizing: ${expected_net_profit:.4f}"
            )

    ratio = gas_cost / expected_net_profit
    if ratio > max_gas_profit_ratio:
        raise SafetyCheckFailed(
            f"Gas cost ${gas_cost:.4f} is {ratio:.0%} of net profit ${expected_net_profit:.2f} "
            f"(max {max_gas_profit_ratio:.0%})"
        )


def verify_max_legs(
    opportunity: Opportunity,
    max_legs: int,
) -> None:
    """
    Reject opportunities with too many legs for atomic execution.
    Polymarket batch endpoint processes max 15 orders per batch.
    More legs = more batches = higher partial-fill risk.
    Raises SafetyCheckFailed if leg count exceeds max_legs.
    """
    n_legs = len(opportunity.legs)
    if n_legs > max_legs:
        raise SafetyCheckFailed(
            f"Too many legs: {n_legs} > {max_legs} (multi-batch execution risk)"
        )


def verify_depth(
    opportunity: Opportunity,
    books: dict[str, OrderBook],
    max_slippage: float = 0.005,
) -> None:
    """
    Verify the orderbook has enough depth across all levels to fill our intended size.
    Uses sweep_depth() and sweep_cost() for multi-level analysis.
    Raises SafetyCheckFailed if depth is insufficient or fill cost exceeds slippage tolerance.
    """
    for leg in opportunity.legs:
        book = books.get(leg.token_id)
        if not book:
            raise SafetyCheckFailed(f"No orderbook for {leg.token_id}")

        if leg.side == Side.BUY:
            if not book.best_ask:
                raise SafetyCheckFailed(f"No ask for {leg.token_id}")
            # Check depth within slippage tolerance of opportunity price
            price_ceiling = leg.price * (1.0 + max_slippage)
            available = sweep_depth(book, Side.BUY, max_price=price_ceiling)
            if available < leg.size:
                raise SafetyCheckFailed(
                    f"Insufficient ask depth for {leg.token_id}: need {leg.size:.1f} have {available:.1f}"
                )
            # Verify actual fill cost doesn't exceed slippage tolerance
            try:
                actual_cost = sweep_cost(book, Side.BUY, leg.size)
                expected_cost = leg.price * leg.size
                if actual_cost > expected_cost * (1.0 + max_slippage):
                    raise SafetyCheckFailed(
                        f"Fill cost slippage for {leg.token_id}: expected ${expected_cost:.2f} actual ${actual_cost:.2f}"
                    )
            except ValueError as e:
                raise SafetyCheckFailed(str(e))
        else:
            if not book.best_bid:
                raise SafetyCheckFailed(f"No bid for {leg.token_id}")
            # Check depth within slippage tolerance
            price_floor = leg.price * (1.0 - max_slippage)
            available = sweep_depth(book, Side.SELL, max_price=price_floor)
            if available < leg.size:
                raise SafetyCheckFailed(
                    f"Insufficient bid depth for {leg.token_id}: need {leg.size:.1f} have {available:.1f}"
                )
            # Verify actual fill proceeds don't fall below slippage tolerance
            try:
                actual_proceeds = sweep_cost(book, Side.SELL, leg.size)
                expected_proceeds = leg.price * leg.size
                if actual_proceeds < expected_proceeds * (1.0 - max_slippage):
                    raise SafetyCheckFailed(
                        f"Fill proceeds slippage for {leg.token_id}: expected ${expected_proceeds:.2f} actual ${actual_proceeds:.2f}"
                    )
            except ValueError as e:
                raise SafetyCheckFailed(str(e))


# TTL thresholds per opportunity type (seconds)
_TTL_BY_TYPE = {
    OpportunityType.SPIKE_LAG: 0.5,
    OpportunityType.LATENCY_ARB: 0.5,
    OpportunityType.BINARY_REBALANCE: 2.0,
    OpportunityType.NEGRISK_REBALANCE: 2.0,
    OpportunityType.CROSS_PLATFORM_ARB: 2.0,
}


def verify_opportunity_ttl(
    opportunity: Opportunity,
    ttl_override_sec: float | None = None,
) -> None:
    """
    Reject stale opportunities. Time-sensitive arbs (spike, latency) get
    shorter TTLs than steady-state rebalancing.
    Raises SafetyCheckFailed if the opportunity is older than its TTL.
    """
    ttl = ttl_override_sec if ttl_override_sec is not None else _TTL_BY_TYPE.get(opportunity.type, 2.0)
    age = time.time() - opportunity.timestamp
    if age > ttl:
        raise SafetyCheckFailed(
            f"Opportunity stale: age={age:.3f}s > TTL={ttl:.1f}s for {opportunity.type.value}"
        )


def verify_edge_intact(
    opportunity: Opportunity,
    books: dict[str, OrderBook],
    min_edge_ratio: float = 0.50,
) -> None:
    """
    Recompute expected profit using fresh book data. If the edge has eroded
    below min_edge_ratio of the original estimate, abort.
    Raises SafetyCheckFailed if edge has deteriorated too much.
    """
    if opportunity.type in (OpportunityType.BINARY_REBALANCE, OpportunityType.NEGRISK_REBALANCE):
        # For buy-all arbs: recompute cost from fresh best asks
        fresh_cost = 0.0
        for leg in opportunity.legs:
            book = books.get(leg.token_id)
            if not book:
                raise SafetyCheckFailed(f"No fresh book for {leg.token_id}")
            if leg.side == Side.BUY:
                if not book.best_ask:
                    raise SafetyCheckFailed(f"No ask in fresh book for {leg.token_id}")
                fresh_cost += book.best_ask.price
            else:
                if not book.best_bid:
                    raise SafetyCheckFailed(f"No bid in fresh book for {leg.token_id}")
                fresh_cost += book.best_bid.price

        if opportunity.legs[0].side == Side.BUY:
            fresh_profit_per_set = 1.0 - fresh_cost
        else:
            fresh_profit_per_set = fresh_cost - 1.0

        if fresh_profit_per_set <= 0:
            raise SafetyCheckFailed(
                f"Edge gone: fresh profit/set={fresh_profit_per_set:.4f} (was {opportunity.expected_profit_per_set:.4f})"
            )

        ratio = fresh_profit_per_set / opportunity.expected_profit_per_set
        if ratio < min_edge_ratio:
            raise SafetyCheckFailed(
                f"Edge eroded: fresh={fresh_profit_per_set:.4f} vs original={opportunity.expected_profit_per_set:.4f} "
                f"(ratio={ratio:.2f} < {min_edge_ratio:.2f})"
            )


def verify_inventory(
    position_tracker: PositionTracker,
    opportunity: Opportunity,
    size: float,
) -> None:
    """
    Verify we hold enough tokens for every SELL leg in the opportunity.
    Buy-only opportunities (binary/negrisk rebalance) pass trivially.
    Raises SafetyCheckFailed if any SELL leg lacks sufficient inventory.
    """
    for leg in opportunity.legs:
        if leg.side != Side.SELL:
            continue
        held = position_tracker.get_position(leg.token_id)
        if held < size:
            raise SafetyCheckFailed(
                f"Insufficient inventory for SELL {leg.token_id}: need {size:.1f} have {held:.1f}"
            )


def verify_cross_platform_books(
    opportunity: Opportunity,
    pm_books: dict[str, OrderBook],
    kalshi_books: dict[str, OrderBook],
    min_depth: float = 1.0,
) -> None:
    """
    Verify both Polymarket and Kalshi orderbooks have sufficient depth for
    a cross-platform arbitrage opportunity.
    Raises SafetyCheckFailed if any leg's book is missing or has insufficient depth.
    """
    for leg in opportunity.legs:
        if leg.platform == "kalshi":
            book = kalshi_books.get(leg.token_id)
        else:
            book = pm_books.get(leg.token_id)

        if not book:
            raise SafetyCheckFailed(
                f"No orderbook for {leg.token_id} on {leg.platform or 'polymarket'}"
            )

        if leg.side == Side.BUY:
            if not book.best_ask:
                raise SafetyCheckFailed(
                    f"No ask for {leg.token_id} on {leg.platform or 'polymarket'}"
                )
            available = sweep_depth(book, Side.BUY, max_price=leg.price * 1.005)
        else:
            if not book.best_bid:
                raise SafetyCheckFailed(
                    f"No bid for {leg.token_id} on {leg.platform or 'polymarket'}"
                )
            available = sweep_depth(book, Side.SELL, max_price=leg.price * 0.995)

        if available < min_depth:
            raise SafetyCheckFailed(
                f"Insufficient depth for {leg.token_id} on {leg.platform or 'polymarket'}: "
                f"have {available:.1f} need {min_depth:.1f}"
            )
