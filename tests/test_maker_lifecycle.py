"""
Unit tests for executor/maker_lifecycle.py -- GTC maker order lifecycle manager.
"""

import time

from executor.maker_lifecycle import MakerLifecycle, MakerOrder, MakerConfig
from scanner.models import Side, OrderBook, PriceLevel


class TestPostOrder:
    def test_creates_entry_in_orders(self):
        lifecycle = MakerLifecycle()
        order = lifecycle.post_order(
            order_id="ord1",
            token_id="yes1",
            side=Side.BUY,
            price=0.45,
            size=100.0,
        )
        assert lifecycle.active_count == 1
        assert order.order_id == "ord1"
        assert order.status == "active"
        assert order.token_id == "yes1"

    def test_max_orders_limit(self):
        lifecycle = MakerLifecycle(max_orders=2)

        # Post 3 orders
        lifecycle.post_order("ord1", "y1", Side.BUY, 0.40, 100)
        lifecycle.post_order("ord2", "n1", Side.BUY, 0.40, 100)
        lifecycle.post_order("ord3", "y2", Side.BUY, 0.40, 100)

        # All tracked (lifecycle doesn't enforce limit on post, just tracks)
        assert lifecycle.active_count == 3


class TestCheckFills:
    def test_detects_filled_order(self):
        lifecycle = MakerLifecycle()
        lifecycle.post_order("ord1", "y1", Side.BUY, 0.45, 100)

        # Mock client that returns filled status
        def mock_client(order_id):
            if order_id == "ord1":
                return {"filled": True, "remainingSize": 0}
            return None

        filled = lifecycle.check_fills(mock_client)
        assert len(filled) == 1
        assert filled[0].order_id == "ord1"
        assert filled[0].status == "filled"
        assert lifecycle.active_count == 0

    def test_skips_orders_with_api_error(self):
        lifecycle = MakerLifecycle()
        lifecycle.post_order("ord1", "y1", Side.BUY, 0.45, 100)

        # Mock client that returns None (API error)
        def mock_client(order_id):
            return None

        filled = lifecycle.check_fills(mock_client)
        assert len(filled) == 0
        assert lifecycle.active_count == 1  # Still active

    def test_detects_multiple_fills(self):
        lifecycle = MakerLifecycle()
        lifecycle.post_order("ord1", "y1", Side.BUY, 0.45, 100)
        lifecycle.post_order("ord2", "n1", Side.SELL, 0.45, 100)

        def mock_client(order_id):
            if order_id == "ord1":
                return {"filled": True}
            return {"filled": False}

        filled = lifecycle.check_fills(mock_client)
        assert len(filled) == 1
        assert filled[0].order_id == "ord1"
        assert lifecycle.active_count == 1


class TestCancelStale:
    def test_removes_orders_older_than_max_age(self):
        lifecycle = MakerLifecycle(max_age_sec=10.0)
        lifecycle.post_order("ord1", "y1", Side.BUY, 0.45, 100)

        # Mock time - order is fresh
        def mock_cancel(order_id):
            return True

        cancelled = lifecycle.cancel_stale(mock_cancel)
        assert len(cancelled) == 0

        # Now age the order
        old_order = lifecycle._orders["ord1"]
        # Manually set posted_at to be old
        from dataclasses import replace
        lifecycle._orders["ord1"] = replace(old_order, posted_at=time.time() - 15)

        cancelled = lifecycle.cancel_stale(mock_cancel)
        assert len(cancelled) == 1
        assert cancelled[0] == "ord1"
        assert lifecycle.active_count == 0

    def test_only_cancels_active_orders(self):
        lifecycle = MakerLifecycle(max_age_sec=10.0)
        lifecycle.post_order("ord1", "y1", Side.BUY, 0.45, 100)

        # Mark as filled
        from dataclasses import replace
        old_order = lifecycle._orders["ord1"]
        lifecycle._orders["ord1"] = replace(old_order, status="filled")

        def mock_cancel(order_id):
            return True

        cancelled = lifecycle.cancel_stale(mock_cancel)
        assert len(cancelled) == 0  # Already filled, not cancelled


class TestCancelIfPriceMoved:
    def test_cancels_when_book_drifts_up(self):
        lifecycle = MakerLifecycle(max_drift_ticks=2)
        lifecycle.post_order("ord1", "y1", Side.BUY, 0.45, 100)

        # Book moved up: best bid is now 0.48 (3 ticks above our 0.45)
        book = OrderBook(
            token_id="y1",
            bids=(PriceLevel(0.48, 100),),
            asks=(PriceLevel(0.50, 100),),
        )

        def mock_cancel(order_id):
            return True

        cancelled = lifecycle.cancel_if_price_moved(mock_cancel, {"y1": book})
        assert len(cancelled) == 1
        assert cancelled[0] == "ord1"

    def test_cancels_when_book_drifts_down(self):
        lifecycle = MakerLifecycle(max_drift_ticks=2)
        lifecycle.post_order("ord1", "y1", Side.SELL, 0.55, 100)

        # Book moved down: best ask is now 0.52 (3 ticks below our 0.55)
        book = OrderBook(
            token_id="y1",
            bids=(PriceLevel(0.50, 100),),
            asks=(PriceLevel(0.52, 100),),
        )

        def mock_cancel(order_id):
            return True

        cancelled = lifecycle.cancel_if_price_moved(mock_cancel, {"y1": book})
        assert len(cancelled) == 1
        assert cancelled[0] == "ord1"

    def test_no_cancel_within_drift_threshold(self):
        lifecycle = MakerLifecycle(max_drift_ticks=2)
        lifecycle.post_order("ord1", "y1", Side.BUY, 0.45, 100)

        # Book moved only 1 tick (within threshold)
        book = OrderBook(
            token_id="y1",
            bids=(PriceLevel(0.46, 100),),
            asks=(PriceLevel(0.50, 100),),
        )

        def mock_cancel(order_id):
            return True

        cancelled = lifecycle.cancel_if_price_moved(mock_cancel, {"y1": book})
        assert len(cancelled) == 0

    def test_skips_orders_without_book(self):
        lifecycle = MakerLifecycle(max_drift_ticks=2)
        lifecycle.post_order("ord1", "y1", Side.BUY, 0.45, 100)

        # No book available for this token
        def mock_cancel(order_id):
            return True

        cancelled = lifecycle.cancel_if_price_moved(mock_cancel, {})
        assert len(cancelled) == 0


class TestActiveExposure:
    def test_sums_active_orders(self):
        lifecycle = MakerLifecycle()
        lifecycle.post_order("ord1", "y1", Side.BUY, 0.45, 100)
        lifecycle.post_order("ord2", "n1", Side.SELL, 0.45, 200)

        # Exposure = 0.45 * 100 + 0.45 * 200 = 135
        assert lifecycle.active_exposure == 135.0

    def test_excludes_filled_orders(self):
        lifecycle = MakerLifecycle()
        lifecycle.post_order("ord1", "y1", Side.BUY, 0.45, 100)

        from dataclasses import replace
        old_order = lifecycle._orders["ord1"]
        lifecycle._orders["ord1"] = replace(old_order, status="filled")

        assert lifecycle.active_exposure == 0.0


class TestActiveCount:
    def test_counts_only_active(self):
        lifecycle = MakerLifecycle()
        lifecycle.post_order("ord1", "y1", Side.BUY, 0.45, 100)
        lifecycle.post_order("ord2", "n1", Side.BUY, 0.40, 100)

        assert lifecycle.active_count == 2

        # Mark one as filled
        from dataclasses import replace
        old_order = lifecycle._orders["ord1"]
        lifecycle._orders["ord1"] = replace(old_order, status="filled")

        assert lifecycle.active_count == 1


class TestCancelAll:
    def test_cancels_all_active_orders(self):
        lifecycle = MakerLifecycle()
        lifecycle.post_order("ord1", "y1", Side.BUY, 0.45, 100)
        lifecycle.post_order("ord2", "n1", Side.BUY, 0.40, 100)

        # Mark one as already filled
        from dataclasses import replace
        old_order = lifecycle._orders["ord2"]
        lifecycle._orders["ord2"] = replace(old_order, status="filled")

        def mock_cancel(order_id):
            return True

        cancelled = lifecycle.cancel_all(mock_cancel)
        assert len(cancelled) == 1
        assert cancelled[0] == "ord1"


class TestPruneFilledAndCancelled:
    def test_removes_inactive_orders(self):
        lifecycle = MakerLifecycle()
        lifecycle.post_order("ord1", "y1", Side.BUY, 0.45, 100)
        lifecycle.post_order("ord2", "n1", Side.BUY, 0.40, 100)

        from dataclasses import replace
        old1 = lifecycle._orders["ord1"]
        old2 = lifecycle._orders["ord2"]
        lifecycle._orders["ord1"] = replace(old1, status="filled")
        lifecycle._orders["ord2"] = replace(old2, status="cancelled")

        lifecycle.prune_filled_and_cancelled()
        assert len(lifecycle._orders) == 0
