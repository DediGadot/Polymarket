"""
Unit tests for scanner/models.py -- data models.
"""

import time
from scanner.models import (
    PriceLevel,
    OrderBook,
    Market,
    Event,
    LegOrder,
    Opportunity,
    OpportunityType,
    Side,
    TradeResult,
    is_market_stale,
)


class TestPriceLevel:
    def test_creation(self):
        pl = PriceLevel(price=0.55, size=100.0)
        assert pl.price == 0.55
        assert pl.size == 100.0

    def test_frozen(self):
        pl = PriceLevel(price=0.55, size=100.0)
        try:
            pl.price = 0.60
            assert False, "Should be frozen"
        except AttributeError:
            pass


class TestOrderBook:
    def _make_book(self):
        bids = (PriceLevel(0.50, 200.0), PriceLevel(0.49, 300.0))
        asks = (PriceLevel(0.52, 150.0), PriceLevel(0.53, 250.0))
        return OrderBook(token_id="tok1", bids=bids, asks=asks)

    def test_best_bid(self):
        book = self._make_book()
        assert book.best_bid.price == 0.50
        assert book.best_bid.size == 200.0

    def test_best_ask(self):
        book = self._make_book()
        assert book.best_ask.price == 0.52
        assert book.best_ask.size == 150.0

    def test_spread(self):
        book = self._make_book()
        assert abs(book.spread - 0.02) < 1e-9

    def test_midpoint(self):
        book = self._make_book()
        assert abs(book.midpoint - 0.51) < 1e-9

    def test_empty_book(self):
        book = OrderBook(token_id="empty", bids=(), asks=())
        assert book.best_bid is None
        assert book.best_ask is None
        assert book.spread is None
        assert book.midpoint is None

    def test_one_sided_book(self):
        book = OrderBook(
            token_id="one_side",
            bids=(PriceLevel(0.50, 100.0),),
            asks=(),
        )
        assert book.best_bid is not None
        assert book.best_ask is None
        assert book.spread is None


class TestMarket:
    def test_creation(self):
        m = Market(
            condition_id="cond1",
            question="Will it rain?",
            yes_token_id="yes1",
            no_token_id="no1",
            neg_risk=False,
            event_id="evt1",
            min_tick_size="0.01",
            active=True,
            volume=50000.0,
        )
        assert m.condition_id == "cond1"
        assert m.neg_risk is False
        assert m.active is True


class TestEvent:
    def test_creation(self):
        m1 = Market("c1", "Q1", "y1", "n1", True, "e1", "0.01", True)
        m2 = Market("c2", "Q2", "y2", "n2", True, "e1", "0.01", True)
        e = Event(event_id="e1", title="Test Event", markets=(m1, m2), neg_risk=True)
        assert len(e.markets) == 2
        assert e.neg_risk is True


class TestOpportunity:
    def _make_opp(self, net_profit=1.0, roi=5.0):
        return Opportunity(
            type=OpportunityType.BINARY_REBALANCE,
            event_id="e1",
            legs=(
                LegOrder("y1", Side.BUY, 0.48, 100.0),
                LegOrder("n1", Side.BUY, 0.48, 100.0),
            ),
            expected_profit_per_set=0.04,
            net_profit_per_set=0.04,
            max_sets=100.0,
            gross_profit=4.0,
            estimated_gas_cost=0.01,
            net_profit=net_profit,
            roi_pct=roi,
            required_capital=96.0,
        )

    def test_is_profitable_true(self):
        opp = self._make_opp(net_profit=1.0, roi=5.0)
        assert opp.is_profitable is True

    def test_is_profitable_false_negative_pnl(self):
        opp = self._make_opp(net_profit=-0.5, roi=-1.0)
        assert opp.is_profitable is False

    def test_is_profitable_false_zero_roi(self):
        opp = self._make_opp(net_profit=0.0, roi=0.0)
        assert opp.is_profitable is False

    def test_timestamp_auto(self):
        before = time.time()
        opp = self._make_opp()
        after = time.time()
        assert before <= opp.timestamp <= after


class TestIsSellArb:
    def _make_opp_with_legs(self, legs):
        return Opportunity(
            type=OpportunityType.BINARY_REBALANCE,
            event_id="e1",
            legs=tuple(legs),
            expected_profit_per_set=0.05,
            net_profit_per_set=0.05,
            max_sets=10.0,
            gross_profit=0.50,
            estimated_gas_cost=0.01,
            net_profit=0.49,
            roi_pct=5.0,
            required_capital=10.0,
        )

    def test_all_sell_legs(self):
        opp = self._make_opp_with_legs([
            LegOrder("t1", Side.SELL, 0.55, 10),
            LegOrder("t2", Side.SELL, 0.50, 10),
        ])
        assert opp.is_sell_arb is True
        assert opp.is_buy_arb is False

    def test_single_sell_leg(self):
        opp = self._make_opp_with_legs([
            LegOrder("t1", Side.SELL, 0.55, 10),
        ])
        assert opp.is_sell_arb is True
        assert opp.is_buy_arb is False

    def test_all_buy_legs(self):
        opp = self._make_opp_with_legs([
            LegOrder("t1", Side.BUY, 0.45, 10),
            LegOrder("t2", Side.BUY, 0.45, 10),
        ])
        assert opp.is_buy_arb is True
        assert opp.is_sell_arb is False

    def test_single_buy_leg(self):
        opp = self._make_opp_with_legs([
            LegOrder("t1", Side.BUY, 0.45, 10),
        ])
        assert opp.is_buy_arb is True
        assert opp.is_sell_arb is False

    def test_mixed_legs(self):
        opp = self._make_opp_with_legs([
            LegOrder("t1", Side.BUY, 0.45, 10),
            LegOrder("t2", Side.SELL, 0.55, 10),
        ])
        assert opp.is_sell_arb is False
        assert opp.is_buy_arb is False

    def test_empty_legs(self):
        opp = self._make_opp_with_legs([])
        assert opp.is_sell_arb is False
        assert opp.is_buy_arb is False


class TestIsMarketStale:
    def _make_market(self, end_date: str = "", closed: bool = False) -> Market:
        return Market(
            condition_id="cond1", question="Q?",
            yes_token_id="y1", no_token_id="n1",
            neg_risk=False, event_id="e1",
            min_tick_size="0.01", active=True,
            end_date=end_date, closed=closed,
        )

    def test_closed_is_stale(self):
        assert is_market_stale(self._make_market(closed=True)) is True

    def test_no_end_date_not_stale(self):
        assert is_market_stale(self._make_market(end_date="")) is False

    def test_future_date_not_stale(self):
        assert is_market_stale(self._make_market(end_date="2099-12-31T23:59:59Z")) is False

    def test_past_date_is_stale(self):
        assert is_market_stale(self._make_market(end_date="2020-01-01T00:00:00Z")) is True

    def test_cached_results_consistent(self):
        """Repeated calls with same end_date return same result (cache hit)."""
        m1 = self._make_market(end_date="2099-06-15T00:00:00Z")
        m2 = self._make_market(end_date="2099-06-15T00:00:00Z")
        result1 = is_market_stale(m1)
        result2 = is_market_stale(m2)
        assert result1 == result2
        assert result1 is False

    def test_different_dates_different_results(self):
        """Different end_dates can have different results."""
        future = self._make_market(end_date="2099-12-31T23:59:59Z")
        past = self._make_market(end_date="2020-01-01T00:00:00Z")
        assert is_market_stale(future) is False
        assert is_market_stale(past) is True

    def test_performance_cached(self):
        """Cached calls should be significantly faster than uncached."""
        markets = [
            self._make_market(end_date=f"2099-{m:02d}-15T00:00:00Z")
            for m in range(1, 13)
        ]
        # Warm the cache
        for m in markets:
            is_market_stale(m)

        # Benchmark cached calls (1000 reps)
        start = time.perf_counter()
        for _ in range(1000):
            for m in markets:
                is_market_stale(m)
        elapsed = time.perf_counter() - start
        per_call_us = elapsed / 12000 * 1e6
        # Cached calls should take < 1us each (vs ~2us uncached)
        assert per_call_us < 1.5, f"Cached is_market_stale too slow: {per_call_us:.2f}us"


class TestTradeResult:
    def test_creation(self):
        opp = Opportunity(
            type=OpportunityType.BINARY_REBALANCE,
            event_id="e1",
            legs=(LegOrder("y1", Side.BUY, 0.50, 10.0),),
            expected_profit_per_set=0.02,
            net_profit_per_set=0.02,
            max_sets=10.0,
            gross_profit=0.20,
            estimated_gas_cost=0.01,
            net_profit=0.19,
            roi_pct=3.8,
            required_capital=5.0,
        )
        result = TradeResult(
            opportunity=opp,
            order_ids=["o1"],
            fill_prices=[0.50],
            fill_sizes=[10.0],
            fees=0.0,
            gas_cost=0.01,
            net_pnl=0.19,
            execution_time_ms=50.0,
            fully_filled=True,
        )
        assert result.fully_filled is True
        assert result.net_pnl == 0.19
