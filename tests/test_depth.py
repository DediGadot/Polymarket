"""
Unit tests for scanner/depth.py -- multi-level orderbook analysis.
"""

import pytest
from scanner.depth import (
    sweep_cost,
    sweep_depth,
    effective_price,
    worst_fill_price,
    depth_profile,
    find_deep_binary_arb,
    find_deep_negrisk_arb,
)
from scanner.models import OrderBook, PriceLevel, Side


def _make_book(token_id="tok1", bids=None, asks=None):
    return OrderBook(
        token_id=token_id,
        bids=tuple(bids or []),
        asks=tuple(asks or []),
    )


class TestSweepCost:
    def test_single_level_full_fill(self):
        book = _make_book(asks=[PriceLevel(0.50, 100)])
        cost = sweep_cost(book, Side.BUY, 100)
        assert abs(cost - 50.0) < 1e-9

    def test_multi_level_fill(self):
        book = _make_book(asks=[PriceLevel(0.50, 50), PriceLevel(0.52, 100)])
        cost = sweep_cost(book, Side.BUY, 80)
        # 50 @ 0.50 = 25, 30 @ 0.52 = 15.6
        assert abs(cost - 40.6) < 1e-9

    def test_sell_side(self):
        book = _make_book(bids=[PriceLevel(0.55, 100), PriceLevel(0.53, 50)])
        cost = sweep_cost(book, Side.SELL, 120)
        # 100 @ 0.55 = 55, 20 @ 0.53 = 10.6
        assert abs(cost - 65.6) < 1e-9

    def test_insufficient_depth_raises(self):
        book = _make_book(asks=[PriceLevel(0.50, 10)])
        with pytest.raises(ValueError, match="Insufficient depth"):
            sweep_cost(book, Side.BUY, 100)


class TestSweepDepth:
    def test_all_levels_under_ceiling(self):
        book = _make_book(asks=[PriceLevel(0.50, 100), PriceLevel(0.52, 50)])
        total = sweep_depth(book, Side.BUY, max_price=0.55)
        assert total == 150.0

    def test_partial_levels(self):
        book = _make_book(asks=[PriceLevel(0.50, 100), PriceLevel(0.52, 50), PriceLevel(0.60, 200)])
        total = sweep_depth(book, Side.BUY, max_price=0.55)
        assert total == 150.0  # only first two levels

    def test_sell_side_floor(self):
        book = _make_book(bids=[PriceLevel(0.55, 100), PriceLevel(0.53, 50), PriceLevel(0.40, 200)])
        total = sweep_depth(book, Side.SELL, max_price=0.50)
        assert total == 150.0  # first two bids above 0.50


class TestEffectivePrice:
    def test_single_level(self):
        book = _make_book(asks=[PriceLevel(0.50, 100)])
        price = effective_price(book, Side.BUY, 50)
        assert abs(price - 0.50) < 1e-9

    def test_multi_level_vwap(self):
        book = _make_book(asks=[PriceLevel(0.50, 50), PriceLevel(0.60, 50)])
        price = effective_price(book, Side.BUY, 100)
        # VWAP: (50*0.50 + 50*0.60) / 100 = 0.55
        assert abs(price - 0.55) < 1e-9

    def test_insufficient_depth_returns_none(self):
        book = _make_book(asks=[PriceLevel(0.50, 10)])
        assert effective_price(book, Side.BUY, 100) is None

    def test_empty_book_returns_none(self):
        book = _make_book(asks=[])
        assert effective_price(book, Side.BUY, 10) is None


class TestWorstFillPrice:
    def test_single_level(self):
        book = _make_book(asks=[PriceLevel(0.50, 100)])
        price = worst_fill_price(book, Side.BUY, 50)
        assert price == 0.50

    def test_multi_level_returns_last_needed(self):
        book = _make_book(asks=[PriceLevel(0.50, 50), PriceLevel(0.55, 50), PriceLevel(0.60, 50)])
        # Need 80: 50 from level1 + 30 from level2 → worst = 0.55
        price = worst_fill_price(book, Side.BUY, 80)
        assert price == 0.55

    def test_full_sweep_returns_deepest_level(self):
        book = _make_book(asks=[PriceLevel(0.50, 50), PriceLevel(0.55, 50), PriceLevel(0.60, 50)])
        price = worst_fill_price(book, Side.BUY, 150)
        assert price == 0.60

    def test_insufficient_depth_returns_none(self):
        book = _make_book(asks=[PriceLevel(0.50, 10)])
        assert worst_fill_price(book, Side.BUY, 100) is None

    def test_empty_book_returns_none(self):
        book = _make_book(asks=[])
        assert worst_fill_price(book, Side.BUY, 10) is None

    def test_sell_side(self):
        book = _make_book(bids=[PriceLevel(0.55, 50), PriceLevel(0.53, 50), PriceLevel(0.50, 50)])
        # Need 80: 50 from 0.55 + 30 from 0.53 → worst = 0.53
        price = worst_fill_price(book, Side.SELL, 80)
        assert price == 0.53

    def test_worst_is_higher_than_vwap_for_buy(self):
        """Worst fill should always be >= VWAP for buys."""
        book = _make_book(asks=[PriceLevel(0.50, 50), PriceLevel(0.60, 50)])
        worst = worst_fill_price(book, Side.BUY, 100)
        vwap = effective_price(book, Side.BUY, 100)
        assert worst >= vwap


class TestDepthProfile:
    def test_cumulative_sizes(self):
        book = _make_book(asks=[PriceLevel(0.50, 100), PriceLevel(0.52, 50), PriceLevel(0.55, 200)])
        profile = depth_profile(book, Side.BUY)
        assert len(profile) == 3
        assert profile[0] == (0.50, 100.0)
        assert profile[1] == (0.52, 150.0)
        assert profile[2] == (0.55, 350.0)


class TestFindDeepBinaryArb:
    def test_arb_at_depth(self):
        """Arb hidden at depth: top-of-book shows no arb, but VWAP shows arb."""
        yes_book = _make_book("yes", asks=[PriceLevel(0.52, 10), PriceLevel(0.48, 100)])
        no_book = _make_book("no", asks=[PriceLevel(0.49, 10), PriceLevel(0.45, 100)])
        result = find_deep_binary_arb(yes_book, no_book, target_size=50)
        assert result is not None
        yes_vwap, no_vwap, size = result
        assert yes_vwap + no_vwap < 1.0

    def test_no_arb_at_depth(self):
        yes_book = _make_book("yes", asks=[PriceLevel(0.55, 100)])
        no_book = _make_book("no", asks=[PriceLevel(0.50, 100)])
        result = find_deep_binary_arb(yes_book, no_book, target_size=50)
        assert result is None

    def test_insufficient_depth(self):
        yes_book = _make_book("yes", asks=[PriceLevel(0.40, 10)])
        no_book = _make_book("no", asks=[PriceLevel(0.40, 100)])
        result = find_deep_binary_arb(yes_book, no_book, target_size=50)
        assert result is None


class TestFindDeepNegriskArb:
    def test_arb_at_depth(self):
        books = {
            "y1": _make_book("y1", asks=[PriceLevel(0.30, 100)]),
            "y2": _make_book("y2", asks=[PriceLevel(0.30, 100)]),
            "y3": _make_book("y3", asks=[PriceLevel(0.30, 100)]),
        }
        result = find_deep_negrisk_arb(books, ["y1", "y2", "y3"], target_size=50)
        assert result is not None
        vwaps, size = result
        assert sum(vwaps) < 1.0

    def test_no_arb(self):
        books = {
            "y1": _make_book("y1", asks=[PriceLevel(0.40, 100)]),
            "y2": _make_book("y2", asks=[PriceLevel(0.40, 100)]),
            "y3": _make_book("y3", asks=[PriceLevel(0.40, 100)]),
        }
        result = find_deep_negrisk_arb(books, ["y1", "y2", "y3"], target_size=50)
        assert result is None

    def test_missing_book(self):
        books = {"y1": _make_book("y1", asks=[PriceLevel(0.30, 100)])}
        result = find_deep_negrisk_arb(books, ["y1", "y2"], target_size=50)
        assert result is None
