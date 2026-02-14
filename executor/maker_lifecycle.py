"""
GTC maker order lifecycle manager. Tracks posted limit orders across scan cycles.

Manages:
- Posting new GTC limit orders
- Checking for fills on active orders
- Canceling stale orders (age-based)
- Canceling orders when price moves against position
- Tracking active exposure
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from typing import Callable

from scanner.models import Side, OrderBook


logger = logging.getLogger(__name__)


@dataclass
class MakerOrder:
    """A posted GTC limit order tracked by the lifecycle manager."""
    order_id: str
    token_id: str
    side: Side
    price: float
    size: float
    posted_at: float
    status: str  # "active", "filled", "cancelled"


@dataclass
class MakerConfig:
    """Configuration for maker order lifecycle."""
    max_age_sec: float = 30.0  # Cancel orders older than this
    max_orders: int = 20  # Maximum concurrent orders
    max_drift_ticks: int = 2  # Cancel if book moves this many ticks


class MakerLifecycle:
    """
    Track posted GTC maker orders across scan cycles.

    Thread-safe for single-writer (main loop) + single-reader (main loop).
    Not safe for concurrent writes from multiple threads.
    """

    def __init__(
        self,
        max_age_sec: float = 30.0,
        max_orders: int = 20,
        max_drift_ticks: int = 2,
    ) -> None:
        self._cfg = MakerConfig(
            max_age_sec=max_age_sec,
            max_orders=max_orders,
            max_drift_ticks=max_drift_ticks,
        )
        self._orders: dict[str, MakerOrder] = {}

    def post_order(
        self,
        order_id: str,
        token_id: str,
        side: Side,
        price: float,
        size: float,
    ) -> MakerOrder:
        """Record a newly posted GTC limit order."""
        now = time.time()
        order = MakerOrder(
            order_id=order_id,
            token_id=token_id,
            side=side,
            price=price,
            size=size,
            posted_at=now,
            status="active",
        )
        self._orders[order_id] = order
        logger.info(
            "Maker order posted: %s %s @ %.4f x %.0f",
            token_id, side.value, price, size,
        )
        return order

    def check_fills(
        self,
        client: Callable[[str], dict | None],
    ) -> list[MakerOrder]:
        """
        Check all active orders for fills.

        Args:
            client: Function(order_id) -> order status dict or None

        Returns:
            List of MakerOrder that newly filled (status transitioned active -> filled)
        """
        filled = []
        now = time.time()

        for order_id, order in list(self._orders.items()):
            if order.status != "active":
                continue

            status = client(order_id)
            if status is None:
                # API error - skip this order for now
                continue

            if status.get("filled", False):
                new_order = MakerOrder(
                    order_id=order.order_id,
                    token_id=order.token_id,
                    side=order.side,
                    price=order.price,
                    size=order.size,
                    posted_at=order.posted_at,
                    status="filled",
                )
                self._orders[order_id] = new_order
                filled.append(new_order)
                logger.info(
                    "Maker order filled: %s (age=%.1fs)",
                    order_id, now - order.posted_at,
                )
            elif status.get("cancelled", False):
                new_order = MakerOrder(
                    order_id=order.order_id,
                    token_id=order.token_id,
                    side=order.side,
                    price=order.price,
                    size=order.size,
                    posted_at=order.posted_at,
                    status="cancelled",
                )
                self._orders[order_id] = new_order
                logger.info("Maker order cancelled externally: %s", order_id)

        return filled

    def cancel_stale(
        self,
        client: Callable[[str], bool],
    ) -> list[str]:
        """
        Cancel orders older than max_age_sec.

        Args:
            client: Function(order_id) -> True if cancelled successfully

        Returns:
            List of cancelled order_ids
        """
        cancelled = []
        now = time.time()

        for order_id, order in list(self._orders.items()):
            if order.status != "active":
                continue

            age = now - order.posted_at
            if age > self._cfg.max_age_sec:
                if client(order_id):
                    new_order = MakerOrder(
                        order_id=order.order_id,
                        token_id=order.token_id,
                        side=order.side,
                        price=order.price,
                        size=order.size,
                        posted_at=order.posted_at,
                        status="cancelled",
                    )
                    self._orders[order_id] = new_order
                    cancelled.append(order_id)
                    logger.info(
                        "Cancelled stale order: %s (age=%.1fs > %.1fs)",
                        order_id, age, self._cfg.max_age_sec,
                    )

        return cancelled

    def cancel_all(
        self,
        client: Callable[[str], bool],
    ) -> list[str]:
        """
        Cancel all active orders. Used on shutdown.

        Args:
            client: Function(order_id) -> True if cancelled successfully

        Returns:
            List of cancelled order_ids
        """
        cancelled = []

        for order_id, order in list(self._orders.items()):
            if order.status != "active":
                continue

            if client(order_id):
                new_order = MakerOrder(
                    order_id=order.order_id,
                    token_id=order.token_id,
                    side=order.side,
                    price=order.price,
                    size=order.size,
                    posted_at=order.posted_at,
                    status="cancelled",
                )
                self._orders[order_id] = new_order
                cancelled.append(order_id)

        if cancelled:
            logger.info("Cancelled %d maker orders on shutdown", len(cancelled))

        return cancelled

    def cancel_if_price_moved(
        self,
        client: Callable[[str], bool],
        books: dict[str, OrderBook],
        max_drift_ticks: int | None = None,
    ) -> list[str]:
        """
        Cancel orders where the book has moved more than max_drift_ticks away.

        Args:
            client: Function(order_id) -> True if cancelled successfully
            books: Current orderbooks for price reference
            max_drift_ticks: Override config default

        Returns:
            List of cancelled order_ids
        """
        cancelled = []
        threshold_ticks = max_drift_ticks or self._cfg.max_drift_ticks

        for order_id, order in list(self._orders.items()):
            if order.status != "active":
                continue

            book = books.get(order.token_id)
            if not book:
                continue

            # Get current best price for our side
            if order.side == Side.BUY:
                best = book.best_bid
            else:
                best = book.best_ask

            if best is None:
                continue

            # Calculate tick drift (assuming 0.01 tick size for most markets)
            tick_size = 0.01
            if order.side == Side.BUY:
                # Our bid is too low (market moved up)
                drift_ticks = (best.price - order.price) / tick_size
            else:
                # Our ask is too high (market moved down)
                drift_ticks = (order.price - best.price) / tick_size

            if drift_ticks > threshold_ticks:
                if client(order_id):
                    new_order = MakerOrder(
                        order_id=order.order_id,
                        token_id=order.token_id,
                        side=order.side,
                        price=order.price,
                        size=order.size,
                        posted_at=order.posted_at,
                        status="cancelled",
                    )
                    self._orders[order_id] = new_order
                    cancelled.append(order_id)
                    logger.info(
                        "Cancelled order due to price move: %s (drift=%.1fticks > %d)",
                        order_id, drift_ticks, threshold_ticks,
                    )

        return cancelled

    @property
    def active_exposure(self) -> float:
        """Total $ tied up in active maker orders."""
        total = 0.0
        for order in self._orders.values():
            if order.status == "active":
                total += order.price * order.size
        return total

    @property
    def active_count(self) -> int:
        """Number of active orders."""
        return sum(
            1 for o in self._orders.values() if o.status == "active"
        )

    def prune_filled_and_cancelled(self) -> None:
        """Remove filled and cancelled orders from tracking (cleanup)."""
        to_remove = [
            order_id
            for order_id, order in self._orders.items()
            if order.status in ("filled", "cancelled")
        ]
        for order_id in to_remove:
            del self._orders[order_id]

        if to_remove:
            logger.debug("Pruned %d inactive orders from tracking", len(to_remove))
