"""
Unit tests for scanner/negrisk.py -- NegRisk multi-outcome rebalancing detection.
"""

from unittest.mock import patch, MagicMock
from scanner.negrisk import scan_negrisk_events, _check_buy_all_arb, _check_sell_all_arb
from scanner.models import (
    Market,
    Event,
    OrderBook,
    PriceLevel,
    OpportunityType,
    Side,
)


def _make_market(cid, yes_id, no_id, eid="e1"):
    return Market(
        condition_id=cid,
        question=f"Outcome {cid}?",
        yes_token_id=yes_id,
        no_token_id=no_id,
        neg_risk=True,
        event_id=eid,
        min_tick_size="0.01",
        active=True,
    )


def _make_event(markets, eid="e1"):
    return Event(
        event_id=eid,
        title="Multi-outcome event",
        markets=tuple(markets),
        neg_risk=True,
    )


def _make_book(token_id, best_bid_price, best_bid_size, best_ask_price, best_ask_size):
    return OrderBook(
        token_id=token_id,
        bids=(PriceLevel(best_bid_price, best_bid_size),) if best_bid_price else (),
        asks=(PriceLevel(best_ask_price, best_ask_size),) if best_ask_price else (),
    )


class TestCheckBuyAllArb:
    def test_3_outcome_profitable(self):
        """3 outcomes at 0.30 each = 0.90 total -> 0.10 profit per set."""
        m1 = _make_market("c1", "y1", "n1")
        m2 = _make_market("c2", "y2", "n2")
        m3 = _make_market("c3", "y3", "n3")
        event = _make_event([m1, m2, m3])

        books = {
            "y1": _make_book("y1", 0.29, 100, 0.30, 100),
            "y2": _make_book("y2", 0.29, 100, 0.30, 100),
            "y3": _make_book("y3", 0.29, 100, 0.30, 100),
        }

        opp = _check_buy_all_arb(
            event, books,
            min_profit_usd=0.01, min_roi_pct=0.1,
            gas_per_order=150000, gas_price_gwei=30.0,
        )
        assert opp is not None
        assert opp.type == OpportunityType.NEGRISK_REBALANCE
        assert abs(opp.expected_profit_per_set - 0.10) < 1e-9
        assert len(opp.legs) == 3
        assert all(leg.side == Side.BUY for leg in opp.legs)

    def test_5_outcome_profitable(self):
        """5 outcomes: 0.15+0.15+0.15+0.15+0.15 = 0.75 -> 0.25 profit."""
        markets = [_make_market(f"c{i}", f"y{i}", f"n{i}") for i in range(5)]
        event = _make_event(markets)
        books = {f"y{i}": _make_book(f"y{i}", 0.14, 50, 0.15, 50) for i in range(5)}

        opp = _check_buy_all_arb(
            event, books,
            min_profit_usd=0.01, min_roi_pct=0.1,
            gas_per_order=150000, gas_price_gwei=30.0,
        )
        assert opp is not None
        assert abs(opp.expected_profit_per_set - 0.25) < 1e-9
        assert len(opp.legs) == 5

    def test_no_arb_when_sum_equals_1(self):
        """3 outcomes at 0.33, 0.33, 0.34 = 1.00 -> no arb."""
        m1 = _make_market("c1", "y1", "n1")
        m2 = _make_market("c2", "y2", "n2")
        m3 = _make_market("c3", "y3", "n3")
        event = _make_event([m1, m2, m3])

        books = {
            "y1": _make_book("y1", 0.32, 100, 0.33, 100),
            "y2": _make_book("y2", 0.32, 100, 0.33, 100),
            "y3": _make_book("y3", 0.33, 100, 0.34, 100),
        }

        opp = _check_buy_all_arb(
            event, books,
            min_profit_usd=0.01, min_roi_pct=0.1,
            gas_per_order=150000, gas_price_gwei=30.0,
        )
        assert opp is None

    def test_no_arb_when_sum_above_1(self):
        m1 = _make_market("c1", "y1", "n1")
        m2 = _make_market("c2", "y2", "n2")
        event = _make_event([m1, m2])

        books = {
            "y1": _make_book("y1", 0.55, 100, 0.60, 100),
            "y2": _make_book("y2", 0.55, 100, 0.60, 100),
        }

        opp = _check_buy_all_arb(
            event, books,
            min_profit_usd=0.01, min_roi_pct=0.1,
            gas_per_order=150000, gas_price_gwei=30.0,
        )
        assert opp is None

    def test_missing_book_returns_none(self):
        """If any outcome's book is missing, skip the entire event."""
        m1 = _make_market("c1", "y1", "n1")
        m2 = _make_market("c2", "y2", "n2")
        event = _make_event([m1, m2])

        books = {
            "y1": _make_book("y1", 0.30, 100, 0.30, 100),
            # y2 missing
        }

        opp = _check_buy_all_arb(
            event, books,
            min_profit_usd=0.01, min_roi_pct=0.1,
            gas_per_order=150000, gas_price_gwei=30.0,
        )
        assert opp is None

    def test_max_sets_limited_by_smallest_depth(self):
        m1 = _make_market("c1", "y1", "n1")
        m2 = _make_market("c2", "y2", "n2")
        m3 = _make_market("c3", "y3", "n3")
        event = _make_event([m1, m2, m3])

        books = {
            "y1": _make_book("y1", 0.25, 1000, 0.25, 1000),
            "y2": _make_book("y2", 0.25, 50, 0.25, 50),     # smallest
            "y3": _make_book("y3", 0.25, 500, 0.25, 500),
        }

        opp = _check_buy_all_arb(
            event, books,
            min_profit_usd=0.01, min_roi_pct=0.1,
            gas_per_order=150000, gas_price_gwei=30.0,
        )
        assert opp is not None
        assert opp.max_sets == 50.0


class TestCheckSellAllArb:
    def test_profitable_sell(self):
        """3 outcomes with bids summing > 1.0."""
        m1 = _make_market("c1", "y1", "n1")
        m2 = _make_market("c2", "y2", "n2")
        m3 = _make_market("c3", "y3", "n3")
        event = _make_event([m1, m2, m3])

        books = {
            "y1": _make_book("y1", 0.40, 100, 0.42, 100),
            "y2": _make_book("y2", 0.40, 100, 0.42, 100),
            "y3": _make_book("y3", 0.40, 100, 0.42, 100),
        }

        opp = _check_sell_all_arb(
            event, books,
            min_profit_usd=0.01, min_roi_pct=0.1,
            gas_per_order=150000, gas_price_gwei=30.0,
        )
        assert opp is not None
        assert abs(opp.expected_profit_per_set - 0.20) < 1e-9
        assert all(leg.side == Side.SELL for leg in opp.legs)

    def test_no_sell_arb_when_bids_below_1(self):
        m1 = _make_market("c1", "y1", "n1")
        m2 = _make_market("c2", "y2", "n2")
        event = _make_event([m1, m2])

        books = {
            "y1": _make_book("y1", 0.40, 100, 0.45, 100),
            "y2": _make_book("y2", 0.40, 100, 0.45, 100),
        }

        opp = _check_sell_all_arb(
            event, books,
            min_profit_usd=0.01, min_roi_pct=0.1,
            gas_per_order=150000, gas_price_gwei=30.0,
        )
        assert opp is None


class TestScanNegRiskEvents:
    @patch("scanner.negrisk.get_orderbooks")
    def test_filters_non_negrisk(self, mock_get_books):
        m = _make_market("c1", "y1", "n1")
        event = Event(event_id="e1", title="Binary", markets=(m,), neg_risk=False)
        result = scan_negrisk_events(
            MagicMock(), [event], 0.50, 2.0, 150000, 30.0,
        )
        assert result == []

    @patch("scanner.negrisk.get_orderbooks")
    def test_filters_single_market_events(self, mock_get_books):
        """Events with only 1 market can't have multi-outcome arb."""
        m = _make_market("c1", "y1", "n1")
        event = _make_event([m])
        result = scan_negrisk_events(
            MagicMock(), [event], 0.50, 2.0, 150000, 30.0,
        )
        assert result == []

    @patch("scanner.negrisk.get_orderbooks")
    def test_finds_arb(self, mock_get_books):
        m1 = _make_market("c1", "y1", "n1")
        m2 = _make_market("c2", "y2", "n2")
        m3 = _make_market("c3", "y3", "n3")
        event = _make_event([m1, m2, m3])

        mock_get_books.return_value = {
            "y1": _make_book("y1", 0.25, 100, 0.25, 100),
            "y2": _make_book("y2", 0.25, 100, 0.25, 100),
            "y3": _make_book("y3", 0.25, 100, 0.25, 100),
        }

        result = scan_negrisk_events(
            MagicMock(), [event], 0.01, 0.1, 150000, 30.0,
        )
        assert len(result) >= 1
        assert result[0].type == OpportunityType.NEGRISK_REBALANCE
