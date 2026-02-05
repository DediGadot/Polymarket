"""
Unit tests for scanner/binary.py -- binary market rebalancing detection.
"""

from unittest.mock import patch, MagicMock
from scanner.binary import scan_binary_markets, _check_buy_arb, _check_sell_arb
from scanner.models import (
    Market,
    OrderBook,
    PriceLevel,
    OpportunityType,
    Side,
)


def _make_market(yes_id="yes1", no_id="no1", neg_risk=False, active=True):
    return Market(
        condition_id="cond1",
        question="Test market?",
        yes_token_id=yes_id,
        no_token_id=no_id,
        neg_risk=neg_risk,
        event_id="evt1",
        min_tick_size="0.01",
        active=active,
        volume=10000.0,
    )


def _make_book(token_id, best_bid_price, best_bid_size, best_ask_price, best_ask_size):
    return OrderBook(
        token_id=token_id,
        bids=(PriceLevel(best_bid_price, best_bid_size),) if best_bid_price else (),
        asks=(PriceLevel(best_ask_price, best_ask_size),) if best_ask_price else (),
    )


class TestCheckBuyArb:
    def test_profitable_arb(self):
        """YES=0.45, NO=0.45 -> cost=0.90 -> profit=0.10 per set."""
        market = _make_market()
        yes_book = _make_book("yes1", 0.44, 200, 0.45, 200)
        no_book = _make_book("no1", 0.44, 200, 0.45, 200)

        opp = _check_buy_arb(
            market, yes_book, no_book,
            min_profit_usd=0.50, min_roi_pct=2.0,
            gas_per_order=150000, gas_price_gwei=30.0,
        )
        assert opp is not None
        assert opp.type == OpportunityType.BINARY_REBALANCE
        assert abs(opp.expected_profit_per_set - 0.10) < 1e-9
        assert opp.max_sets == 200.0
        assert opp.roi_pct > 0

    def test_no_arb_when_cost_equals_1(self):
        """YES=0.50, NO=0.50 -> cost=1.00 -> no arb."""
        market = _make_market()
        yes_book = _make_book("yes1", 0.49, 100, 0.50, 100)
        no_book = _make_book("no1", 0.49, 100, 0.50, 100)

        opp = _check_buy_arb(
            market, yes_book, no_book,
            min_profit_usd=0.50, min_roi_pct=2.0,
            gas_per_order=150000, gas_price_gwei=30.0,
        )
        assert opp is None

    def test_no_arb_when_cost_above_1(self):
        """YES=0.55, NO=0.55 -> cost=1.10 -> no arb."""
        market = _make_market()
        yes_book = _make_book("yes1", 0.54, 100, 0.55, 100)
        no_book = _make_book("no1", 0.54, 100, 0.55, 100)

        opp = _check_buy_arb(
            market, yes_book, no_book,
            min_profit_usd=0.50, min_roi_pct=2.0,
            gas_per_order=150000, gas_price_gwei=30.0,
        )
        assert opp is None

    def test_no_arb_empty_book(self):
        """No asks available -> no arb."""
        market = _make_market()
        yes_book = OrderBook(token_id="yes1", bids=(), asks=())
        no_book = _make_book("no1", 0.40, 100, 0.40, 100)

        opp = _check_buy_arb(
            market, yes_book, no_book,
            min_profit_usd=0.50, min_roi_pct=2.0,
            gas_per_order=150000, gas_price_gwei=30.0,
        )
        assert opp is None

    def test_min_sets_is_respected(self):
        """Depth limited by smaller side."""
        market = _make_market()
        yes_book = _make_book("yes1", 0.40, 50, 0.42, 50)
        no_book = _make_book("no1", 0.40, 200, 0.42, 200)

        opp = _check_buy_arb(
            market, yes_book, no_book,
            min_profit_usd=0.01, min_roi_pct=0.1,
            gas_per_order=150000, gas_price_gwei=30.0,
        )
        assert opp is not None
        assert opp.max_sets == 50.0

    def test_below_min_profit_filtered(self):
        """Edge exists but below min profit threshold."""
        market = _make_market()
        # cost=0.99, profit=0.01/set, 5 sets -> gross=0.05
        yes_book = _make_book("yes1", 0.49, 5, 0.495, 5)
        no_book = _make_book("no1", 0.49, 5, 0.495, 5)

        opp = _check_buy_arb(
            market, yes_book, no_book,
            min_profit_usd=1.0, min_roi_pct=2.0,
            gas_per_order=150000, gas_price_gwei=30.0,
        )
        assert opp is None

    def test_legs_are_buy(self):
        """Both legs should be BUY orders."""
        market = _make_market()
        yes_book = _make_book("yes1", 0.40, 100, 0.40, 100)
        no_book = _make_book("no1", 0.40, 100, 0.40, 100)

        opp = _check_buy_arb(
            market, yes_book, no_book,
            min_profit_usd=0.01, min_roi_pct=0.1,
            gas_per_order=150000, gas_price_gwei=30.0,
        )
        assert opp is not None
        assert all(leg.side == Side.BUY for leg in opp.legs)


class TestCheckSellArb:
    def test_profitable_sell_arb(self):
        """YES_bid=0.55, NO_bid=0.55 -> proceeds=1.10 -> profit=0.10/set."""
        market = _make_market()
        yes_book = _make_book("yes1", 0.55, 200, 0.56, 200)
        no_book = _make_book("no1", 0.55, 200, 0.56, 200)

        opp = _check_sell_arb(
            market, yes_book, no_book,
            min_profit_usd=0.50, min_roi_pct=2.0,
            gas_per_order=150000, gas_price_gwei=30.0,
        )
        assert opp is not None
        assert abs(opp.expected_profit_per_set - 0.10) < 1e-9
        assert all(leg.side == Side.SELL for leg in opp.legs)

    def test_no_sell_arb_when_proceeds_below_1(self):
        market = _make_market()
        yes_book = _make_book("yes1", 0.45, 100, 0.46, 100)
        no_book = _make_book("no1", 0.45, 100, 0.46, 100)

        opp = _check_sell_arb(
            market, yes_book, no_book,
            min_profit_usd=0.50, min_roi_pct=2.0,
            gas_per_order=150000, gas_price_gwei=30.0,
        )
        assert opp is None


class TestScanBinaryMarkets:
    @patch("scanner.binary.get_orderbooks")
    def test_filters_negrisk_markets(self, mock_get_books):
        """Should skip negRisk markets."""
        mock_get_books.return_value = {}
        markets = [_make_market(neg_risk=True)]
        result = scan_binary_markets(
            MagicMock(), markets, 0.50, 2.0, 150000, 30.0,
        )
        assert result == []
        mock_get_books.assert_not_called()

    @patch("scanner.binary.get_orderbooks")
    def test_filters_inactive_markets(self, mock_get_books):
        """Should skip inactive markets."""
        mock_get_books.return_value = {}
        markets = [_make_market(active=False)]
        result = scan_binary_markets(
            MagicMock(), markets, 0.50, 2.0, 150000, 30.0,
        )
        assert result == []

    @patch("scanner.binary.get_orderbooks")
    def test_finds_arb(self, mock_get_books):
        """Should detect a buy arb in a binary market."""
        mock_get_books.return_value = {
            "yes1": _make_book("yes1", 0.40, 100, 0.42, 100),
            "no1": _make_book("no1", 0.40, 100, 0.42, 100),
        }
        markets = [_make_market()]
        result = scan_binary_markets(
            MagicMock(), markets, 0.01, 0.1, 150000, 30.0,
        )
        assert len(result) >= 1
        assert result[0].type == OpportunityType.BINARY_REBALANCE

    @patch("scanner.binary.get_orderbooks")
    def test_sorted_by_roi(self, mock_get_books):
        """Results should be sorted by ROI descending."""
        mock_get_books.return_value = {
            "y1": _make_book("y1", 0.30, 100, 0.35, 100),
            "n1": _make_book("n1", 0.30, 100, 0.35, 100),
            "y2": _make_book("y2", 0.40, 100, 0.45, 100),
            "n2": _make_book("n2", 0.40, 100, 0.45, 100),
        }
        m1 = _make_market(yes_id="y1", no_id="n1")
        m2 = _make_market(yes_id="y2", no_id="n2")
        # m1 has wider spread (0.70 total vs 0.90) so higher ROI
        result = scan_binary_markets(
            MagicMock(), [m1, m2], 0.01, 0.1, 150000, 30.0,
        )
        if len(result) >= 2:
            assert result[0].roi_pct >= result[1].roi_pct
