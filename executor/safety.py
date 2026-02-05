"""
Pre-trade safety checks and circuit breakers. Fail-fast on violations.
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field

from py_clob_client.client import ClobClient

from client.clob import get_orderbooks
from scanner.models import Opportunity, Side

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
        self._hourly_losses = [(t, l) for t, l in self._hourly_losses if t > hour_ago]
        self._daily_losses = [(t, l) for t, l in self._daily_losses if t > day_ago]

        hourly_total = sum(l for _, l in self._hourly_losses)
        if hourly_total >= self.max_loss_per_hour:
            raise CircuitBreakerTripped(
                f"Hourly loss limit exceeded: ${hourly_total:.2f} >= ${self.max_loss_per_hour:.2f}"
            )

        daily_total = sum(l for _, l in self._daily_losses)
        if daily_total >= self.max_loss_per_day:
            raise CircuitBreakerTripped(
                f"Daily loss limit exceeded: ${daily_total:.2f} >= ${self.max_loss_per_day:.2f}"
            )

        if self._consecutive_failures >= self.max_consecutive_failures:
            raise CircuitBreakerTripped(
                f"Consecutive failures: {self._consecutive_failures} >= {self.max_consecutive_failures}"
            )


def verify_prices_fresh(
    client: ClobClient,
    opportunity: Opportunity,
    max_slippage: float = 0.005,
) -> None:
    """
    Re-fetch current prices and verify they haven't moved beyond slippage tolerance.
    Raises SafetyCheckFailed if quotes are stale.
    """
    token_ids = [leg.token_id for leg in opportunity.legs]
    books = get_orderbooks(client, token_ids)

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


def verify_depth(
    client: ClobClient,
    opportunity: Opportunity,
) -> None:
    """
    Verify the orderbook has enough depth to fill our intended size.
    Raises SafetyCheckFailed if depth is insufficient.
    """
    token_ids = [leg.token_id for leg in opportunity.legs]
    books = get_orderbooks(client, token_ids)

    for leg in opportunity.legs:
        book = books.get(leg.token_id)
        if not book:
            raise SafetyCheckFailed(f"No orderbook for {leg.token_id}")

        if leg.side == Side.BUY:
            if not book.best_ask or book.best_ask.size < leg.size:
                available = book.best_ask.size if book.best_ask else 0
                raise SafetyCheckFailed(
                    f"Insufficient ask depth for {leg.token_id}: need {leg.size:.1f} have {available:.1f}"
                )
        else:
            if not book.best_bid or book.best_bid.size < leg.size:
                available = book.best_bid.size if book.best_bid else 0
                raise SafetyCheckFailed(
                    f"Insufficient bid depth for {leg.token_id}: need {leg.size:.1f} have {available:.1f}"
                )
