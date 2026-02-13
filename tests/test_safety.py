"""
Unit tests for executor/safety.py -- circuit breakers and pre-trade checks.
"""

import time
import pytest

from executor.safety import (
    CircuitBreaker,
    CircuitBreakerTripped,
    SafetyCheckFailed,
    verify_prices_fresh,
    verify_depth,
    verify_gas_reasonable,
    verify_max_legs,
    verify_opportunity_ttl,
    verify_edge_intact,
    verify_inventory,
    verify_cross_platform_books,
)
from client.data import PositionTracker
from scanner.models import (
    Opportunity,
    OpportunityType,
    LegOrder,
    Side,
    OrderBook,
    PriceLevel,
)


def _make_opp(legs=None):
    if legs is None:
        legs = (
            LegOrder("y1", Side.BUY, 0.45, 100),
            LegOrder("n1", Side.BUY, 0.45, 100),
        )
    return Opportunity(
        type=OpportunityType.BINARY_REBALANCE,
        event_id="e1",
        legs=legs,
        expected_profit_per_set=0.10,
        net_profit_per_set=0.10,
        max_sets=100,
        gross_profit=10.0,
        estimated_gas_cost=0.01,
        net_profit=9.99,
        roi_pct=11.1,
        required_capital=90.0,
    )


def _make_book(token_id, bid_price, bid_size, ask_price, ask_size):
    return OrderBook(
        token_id=token_id,
        bids=(PriceLevel(bid_price, bid_size),),
        asks=(PriceLevel(ask_price, ask_size),),
    )


class TestCircuitBreaker:
    def test_no_trip_on_wins(self):
        cb = CircuitBreaker(
            max_loss_per_hour=50, max_loss_per_day=200, max_consecutive_failures=5,
        )
        for _ in range(20):
            cb.record_trade(1.0)  # all wins

    def test_trip_on_hourly_loss(self):
        cb = CircuitBreaker(
            max_loss_per_hour=10, max_loss_per_day=200, max_consecutive_failures=100,
        )
        with pytest.raises(CircuitBreakerTripped, match="Hourly loss"):
            for _ in range(20):
                cb.record_trade(-1.0)

    def test_trip_on_daily_loss(self):
        cb = CircuitBreaker(
            max_loss_per_hour=1000, max_loss_per_day=10, max_consecutive_failures=100,
        )
        with pytest.raises(CircuitBreakerTripped, match="Daily loss"):
            for _ in range(20):
                cb.record_trade(-1.0)

    def test_trip_on_consecutive_failures(self):
        cb = CircuitBreaker(
            max_loss_per_hour=1000, max_loss_per_day=1000, max_consecutive_failures=3,
        )
        with pytest.raises(CircuitBreakerTripped, match="Consecutive failures"):
            for _ in range(5):
                cb.record_trade(-0.01)

    def test_consecutive_reset_on_win(self):
        cb = CircuitBreaker(
            max_loss_per_hour=1000, max_loss_per_day=1000, max_consecutive_failures=3,
        )
        cb.record_trade(-0.01)
        cb.record_trade(-0.01)
        cb.record_trade(1.0)  # resets counter
        cb.record_trade(-0.01)
        cb.record_trade(-0.01)
        # Should not trip because we had a win in between

    def test_old_losses_pruned(self):
        cb = CircuitBreaker(
            max_loss_per_hour=10, max_loss_per_day=200, max_consecutive_failures=100,
        )
        # Inject old losses that should be pruned
        old_time = time.time() - 7200  # 2 hours ago
        cb._hourly_losses = [(old_time, 100.0)]
        # Should not trip because old losses are pruned
        cb.record_trade(-1.0)


class TestVerifyPricesFresh:
    def test_prices_within_tolerance(self):
        opp = _make_opp()
        books = {
            "y1": _make_book("y1", 0.44, 200, 0.45, 200),
            "n1": _make_book("n1", 0.44, 200, 0.45, 200),
        }
        # Should not raise
        verify_prices_fresh(opp, books)

    def test_ask_moved_up_raises(self):
        opp = _make_opp()
        books = {
            "y1": _make_book("y1", 0.50, 200, 0.52, 200),  # moved from 0.45 to 0.52
            "n1": _make_book("n1", 0.44, 200, 0.45, 200),
        }
        with pytest.raises(SafetyCheckFailed, match="Ask moved"):
            verify_prices_fresh(opp, books)

    def test_missing_book_raises(self):
        opp = _make_opp()
        books = {
            "y1": _make_book("y1", 0.44, 200, 0.45, 200),
            # n1 missing
        }
        with pytest.raises(SafetyCheckFailed, match="No orderbook"):
            verify_prices_fresh(opp, books)

    def test_sell_leg_bid_moved_down_raises(self):
        legs = (
            LegOrder("y1", Side.SELL, 0.55, 100),
            LegOrder("n1", Side.SELL, 0.55, 100),
        )
        opp = _make_opp(legs=legs)
        books = {
            "y1": _make_book("y1", 0.50, 200, 0.56, 200),  # bid moved from 0.55 to 0.50
            "n1": _make_book("n1", 0.55, 200, 0.56, 200),
        }
        with pytest.raises(SafetyCheckFailed, match="Bid moved"):
            verify_prices_fresh(opp, books)


class TestVerifyDepth:
    def test_sufficient_depth(self):
        opp = _make_opp()
        books = {
            "y1": _make_book("y1", 0.44, 200, 0.45, 200),
            "n1": _make_book("n1", 0.44, 200, 0.45, 200),
        }
        verify_depth(opp, books)  # should not raise

    def test_insufficient_depth_raises(self):
        opp = _make_opp()
        books = {
            "y1": _make_book("y1", 0.44, 200, 0.45, 50),  # only 50 available, need 100
            "n1": _make_book("n1", 0.44, 200, 0.45, 200),
        }
        with pytest.raises(SafetyCheckFailed, match="Insufficient ask depth"):
            verify_depth(opp, books)


class TestVerifyGasReasonable:
    def test_gas_within_ratio(self):
        from executor.safety import verify_gas_reasonable
        from client.gas import GasOracle
        oracle = GasOracle(default_gas_gwei=30.0, default_matic_usd=0.50)
        # Force use of defaults by setting cache_sec=0 and catching the error
        oracle._cached_gas_gwei = 30.0
        oracle._gas_ts = time.time()
        oracle._cached_matic_usd = 0.50
        oracle._matic_ts = time.time()

        opp = _make_opp()  # net_profit=9.99
        # Gas cost = 2 * 150000 * 30 * 1e9 / 1e18 * 0.50 = $0.0045 → ratio ~0.0005 → OK
        verify_gas_reasonable(oracle, opp, gas_per_order=150000, max_gas_profit_ratio=0.50)

    def test_gas_exceeds_ratio_raises(self):
        from executor.safety import verify_gas_reasonable
        from client.gas import GasOracle
        oracle = GasOracle()
        # Set very high gas to make ratio exceed threshold
        oracle._cached_gas_gwei = 100000.0
        oracle._gas_ts = time.time()
        oracle._cached_matic_usd = 100.0
        oracle._matic_ts = time.time()

        opp = _make_opp()  # net_profit=9.99
        with pytest.raises(SafetyCheckFailed, match="Gas cost"):
            verify_gas_reasonable(oracle, opp, gas_per_order=150000, max_gas_profit_ratio=0.50)

    def test_non_positive_profit_raises(self):
        from executor.safety import verify_gas_reasonable
        from client.gas import GasOracle
        oracle = GasOracle()
        oracle._cached_gas_gwei = 30.0
        oracle._gas_ts = time.time()
        oracle._cached_matic_usd = 0.50
        oracle._matic_ts = time.time()

        opp = Opportunity(
            type=OpportunityType.BINARY_REBALANCE,
            event_id="e1",
            legs=(LegOrder("y1", Side.BUY, 0.50, 100),),
            expected_profit_per_set=0.0,
            net_profit_per_set=0.0,
            max_sets=100,
            gross_profit=0.0,
            estimated_gas_cost=0.01,
            net_profit=0.0,
            roi_pct=0.0,
            required_capital=50.0,
        )
        with pytest.raises(SafetyCheckFailed, match="non-positive net profit"):
            verify_gas_reasonable(oracle, opp, gas_per_order=150000)

    def test_negative_profit_raises(self):
        from executor.safety import verify_gas_reasonable
        from client.gas import GasOracle
        oracle = GasOracle()
        oracle._cached_gas_gwei = 30.0
        oracle._gas_ts = time.time()
        oracle._cached_matic_usd = 0.50
        oracle._matic_ts = time.time()

        opp = Opportunity(
            type=OpportunityType.BINARY_REBALANCE,
            event_id="e1",
            legs=(LegOrder("y1", Side.BUY, 0.50, 100),),
            expected_profit_per_set=-0.05,
            net_profit_per_set=-0.05,
            max_sets=100,
            gross_profit=-5.0,
            estimated_gas_cost=0.01,
            net_profit=-5.01,
            roi_pct=-10.0,
            required_capital=50.0,
        )
        with pytest.raises(SafetyCheckFailed, match="non-positive net profit"):
            verify_gas_reasonable(oracle, opp, gas_per_order=150000)

    def test_sized_trade_can_fail_even_if_max_size_profit_passes(self):
        """Gas check must use execution size, not max-size net profit."""
        from client.gas import GasOracle

        oracle = GasOracle()
        # Gas = 2 * 250000 * 1000gwei * $1 / 1e9 = $0.50
        oracle._cached_gas_gwei = 1000.0
        oracle._gas_ts = time.time()
        oracle._cached_matic_usd = 1.0
        oracle._matic_ts = time.time()

        opp = Opportunity(
            type=OpportunityType.BINARY_REBALANCE,
            event_id="e1",
            legs=(
                LegOrder("y1", Side.BUY, 0.45, 1000),
                LegOrder("n1", Side.BUY, 0.45, 1000),
            ),
            expected_profit_per_set=0.05,
            net_profit_per_set=0.05,
            max_sets=1000,
            gross_profit=50.0,
            estimated_gas_cost=0.5,
            net_profit=49.5,
            roi_pct=5.5,
            required_capital=900.0,
        )

        with pytest.raises(SafetyCheckFailed):
            verify_gas_reasonable(
                oracle,
                opp,
                gas_per_order=250000,
                max_gas_profit_ratio=0.50,
                size=1.0,
            )


class TestVerifyDepthMultiLevel:
    def test_multi_level_sufficient_depth(self):
        """50 at level 1 + 100 at level 2 should satisfy need for 100 (within slippage)."""
        legs = (LegOrder("y1", Side.BUY, 0.45, 100),)
        opp = _make_opp(legs=legs)
        books = {
            "y1": OrderBook(
                token_id="y1",
                bids=(PriceLevel(0.44, 200),),
                asks=(PriceLevel(0.45, 50), PriceLevel(0.451, 100)),
            ),
        }
        verify_depth(opp, books)  # should not raise

    def test_multi_level_insufficient_depth(self):
        """50 at level 1 + 50 at level 2 = 100 but cost slips beyond tolerance for need 100."""
        legs = (LegOrder("y1", Side.BUY, 0.45, 100),)
        opp = _make_opp(legs=legs)
        books = {
            "y1": OrderBook(
                token_id="y1",
                bids=(PriceLevel(0.44, 200),),
                asks=(PriceLevel(0.45, 30), PriceLevel(0.60, 100)),  # level 2 way above
            ),
        }
        with pytest.raises(SafetyCheckFailed):
            verify_depth(opp, books)

    def test_no_book_raises(self):
        legs = (LegOrder("y1", Side.BUY, 0.45, 100),)
        opp = _make_opp(legs=legs)
        books = {}
        with pytest.raises(SafetyCheckFailed, match="No orderbook"):
            verify_depth(opp, books)


class TestVerifyDepthSellSide:
    def test_sell_depth_sufficient(self):
        legs = (
            LegOrder("y1", Side.SELL, 0.55, 50),
            LegOrder("n1", Side.SELL, 0.55, 50),
        )
        opp = _make_opp(legs=legs)
        books = {
            "y1": _make_book("y1", 0.55, 100, 0.56, 100),
            "n1": _make_book("n1", 0.55, 100, 0.56, 100),
        }
        verify_depth(opp, books)  # should not raise

    def test_sell_depth_insufficient(self):
        legs = (
            LegOrder("y1", Side.SELL, 0.55, 200),  # need 200
        )
        opp = _make_opp(legs=legs)
        books = {
            "y1": _make_book("y1", 0.55, 50, 0.56, 100),  # only 50 on bid side
        }
        with pytest.raises(SafetyCheckFailed, match="Insufficient bid depth"):
            verify_depth(opp, books)


class TestVerifyMaxLegs:
    def test_within_limit(self):
        """2-leg opportunity should pass max_legs=15."""
        opp = _make_opp()  # 2 legs
        verify_max_legs(opp, max_legs=15)  # should not raise

    def test_at_limit(self):
        """15-leg opportunity should pass max_legs=15."""
        legs = tuple(LegOrder(f"y{i}", Side.BUY, 0.05, 100) for i in range(15))
        opp = _make_opp(legs=legs)
        verify_max_legs(opp, max_legs=15)  # should not raise

    def test_exceeds_limit(self):
        """16-leg opportunity should fail max_legs=15."""
        legs = tuple(LegOrder(f"y{i}", Side.BUY, 0.05, 100) for i in range(16))
        opp = _make_opp(legs=legs)
        with pytest.raises(SafetyCheckFailed, match="Too many legs"):
            verify_max_legs(opp, max_legs=15)

    def test_large_event_rejected(self):
        """31-leg opportunity (like JD Vance 2028) should be rejected."""
        legs = tuple(LegOrder(f"y{i}", Side.BUY, 0.03, 50) for i in range(31))
        opp = _make_opp(legs=legs)
        with pytest.raises(SafetyCheckFailed, match="Too many legs"):
            verify_max_legs(opp, max_legs=15)


class TestVerifyOpportunityTTL:
    def test_fresh_opportunity_passes(self):
        """Opportunity created just now should pass TTL check."""
        opp = _make_opp()  # timestamp = time.time()
        verify_opportunity_ttl(opp)  # should not raise

    def test_stale_opportunity_rejected(self):
        """Opportunity created 10s ago should fail default 2s TTL."""
        opp = Opportunity(
            type=OpportunityType.BINARY_REBALANCE,
            event_id="e1",
            legs=(LegOrder("y1", Side.BUY, 0.45, 100),),
            expected_profit_per_set=0.10,
            net_profit_per_set=0.10,
            max_sets=100,
            gross_profit=10.0,
            estimated_gas_cost=0.01,
            net_profit=9.99,
            roi_pct=11.1,
            required_capital=90.0,
            timestamp=time.time() - 10.0,
        )
        with pytest.raises(SafetyCheckFailed, match="Opportunity stale"):
            verify_opportunity_ttl(opp)

    def test_spike_has_shorter_ttl(self):
        """Spike arb with 0.5s TTL should reject at 1s age."""
        opp = Opportunity(
            type=OpportunityType.SPIKE_LAG,
            event_id="e1",
            legs=(LegOrder("y1", Side.BUY, 0.45, 100),),
            expected_profit_per_set=0.10,
            net_profit_per_set=0.10,
            max_sets=100,
            gross_profit=10.0,
            estimated_gas_cost=0.01,
            net_profit=9.99,
            roi_pct=11.1,
            required_capital=90.0,
            timestamp=time.time() - 1.0,
        )
        with pytest.raises(SafetyCheckFailed, match="Opportunity stale"):
            verify_opportunity_ttl(opp)

    def test_ttl_override(self):
        """Custom TTL override should be respected."""
        opp = _make_opp()
        # Override with 0 TTL -- everything is stale
        with pytest.raises(SafetyCheckFailed, match="Opportunity stale"):
            verify_opportunity_ttl(opp, ttl_override_sec=0.0)


class TestVerifyEdgeIntact:
    def test_edge_intact_passes(self):
        """Fresh books with same prices should pass."""
        opp = _make_opp()
        books = {
            "y1": _make_book("y1", 0.44, 200, 0.45, 200),
            "n1": _make_book("n1", 0.44, 200, 0.45, 200),
        }
        verify_edge_intact(opp, books)  # should not raise

    def test_edge_eroded_raises(self):
        """If fresh asks sum to > $1 (no arb), edge is gone."""
        opp = _make_opp()
        books = {
            "y1": _make_book("y1", 0.44, 200, 0.55, 200),  # ask jumped
            "n1": _make_book("n1", 0.44, 200, 0.55, 200),  # ask jumped
        }
        with pytest.raises(SafetyCheckFailed, match="Edge gone"):
            verify_edge_intact(opp, books)

    def test_edge_partially_eroded_raises(self):
        """Edge dropped to 30% of original (below 50% threshold)."""
        opp = _make_opp()  # expected_profit_per_set=0.10, legs buy at 0.45+0.45=0.90
        books = {
            "y1": _make_book("y1", 0.44, 200, 0.48, 200),  # ask up from 0.45 to 0.48
            "n1": _make_book("n1", 0.44, 200, 0.49, 200),  # ask up from 0.45 to 0.49
        }
        # Fresh cost = 0.48 + 0.49 = 0.97, fresh profit = 0.03, original = 0.10
        # Ratio = 0.03/0.10 = 0.30 < 0.50
        with pytest.raises(SafetyCheckFailed, match="Edge eroded"):
            verify_edge_intact(opp, books)

    def test_missing_book_raises(self):
        opp = _make_opp()
        books = {"y1": _make_book("y1", 0.44, 200, 0.45, 200)}  # n1 missing
        with pytest.raises(SafetyCheckFailed, match="No fresh book"):
            verify_edge_intact(opp, books)

    def test_cross_platform_edge_intact_passes(self):
        opp = Opportunity(
            type=OpportunityType.CROSS_PLATFORM_ARB,
            event_id="e1",
            legs=(
                LegOrder("pm_yes", Side.BUY, 0.40, 10, platform="polymarket"),
                LegOrder("K-TEST", Side.SELL, 0.70, 10, platform="kalshi"),
            ),
            expected_profit_per_set=0.30,
            net_profit_per_set=0.30,
            max_sets=10,
            gross_profit=3.0,
            estimated_gas_cost=0.01,
            net_profit=2.99,
            roi_pct=30.0,
            required_capital=7.0,
        )
        books = {
            "pm_yes": _make_book("pm_yes", 0.39, 100, 0.40, 100),
            "K-TEST": _make_book("K-TEST", 0.70, 100, 0.71, 100),
        }
        verify_edge_intact(opp, books)

    def test_cross_platform_edge_gone_raises(self):
        opp = Opportunity(
            type=OpportunityType.CROSS_PLATFORM_ARB,
            event_id="e1",
            legs=(
                LegOrder("pm_yes", Side.BUY, 0.40, 10, platform="polymarket"),
                LegOrder("K-TEST", Side.SELL, 0.70, 10, platform="kalshi"),
            ),
            expected_profit_per_set=0.30,
            net_profit_per_set=0.30,
            max_sets=10,
            gross_profit=3.0,
            estimated_gas_cost=0.01,
            net_profit=2.99,
            roi_pct=30.0,
            required_capital=7.0,
        )
        books = {
            "pm_yes": _make_book("pm_yes", 0.39, 100, 0.54, 100),
            "K-TEST": _make_book("K-TEST", 0.52, 100, 0.53, 100),
        }
        with pytest.raises(SafetyCheckFailed, match="Edge gone"):
            verify_edge_intact(opp, books)


class TestVerifyInventory:
    def test_buy_only_passes_trivially(self):
        """Buy-only opportunities have no sell legs, so inventory check is a no-op."""
        opp = _make_opp()  # all BUY legs
        tracker = PositionTracker(profile_address="")  # no address = empty positions
        verify_inventory(tracker, opp, size=100)  # should not raise

    def test_sell_with_sufficient_inventory(self):
        """Sell legs should pass if we hold enough tokens."""
        legs = (
            LegOrder("y1", Side.SELL, 0.55, 100),
            LegOrder("n1", Side.SELL, 0.55, 100),
        )
        opp = _make_opp(legs=legs)
        tracker = PositionTracker(profile_address="0xfake")
        # Inject cached positions directly
        tracker._positions = {"y1": 200.0, "n1": 150.0}
        tracker._last_fetch = time.time()
        verify_inventory(tracker, opp, size=100)  # should not raise

    def test_sell_with_insufficient_inventory_raises(self):
        """Sell legs should fail if we don't hold enough tokens."""
        legs = (
            LegOrder("y1", Side.SELL, 0.55, 100),
            LegOrder("n1", Side.SELL, 0.55, 100),
        )
        opp = _make_opp(legs=legs)
        tracker = PositionTracker(profile_address="0xfake")
        tracker._positions = {"y1": 50.0}  # only 50, need 100; n1 not held at all
        tracker._last_fetch = time.time()
        with pytest.raises(SafetyCheckFailed, match="Insufficient inventory"):
            verify_inventory(tracker, opp, size=100)

    def test_mixed_legs_checks_only_sells(self):
        """Mixed buy+sell: only sell legs need inventory."""
        legs = (
            LegOrder("y1", Side.BUY, 0.45, 100),
            LegOrder("n1", Side.SELL, 0.55, 100),
        )
        opp = _make_opp(legs=legs)
        tracker = PositionTracker(profile_address="0xfake")
        tracker._positions = {"n1": 200.0}
        tracker._last_fetch = time.time()
        verify_inventory(tracker, opp, size=100)  # should not raise -- y1 is BUY, n1 has inventory

    def test_zero_position_for_sell_raises(self):
        """Selling a token we don't hold at all should fail."""
        legs = (LegOrder("y1", Side.SELL, 0.55, 100),)
        opp = _make_opp(legs=legs)
        tracker = PositionTracker(profile_address="0xfake")
        tracker._positions = {}
        tracker._last_fetch = time.time()
        with pytest.raises(SafetyCheckFailed, match="Insufficient inventory"):
            verify_inventory(tracker, opp, size=100)


class TestVerifyCrossPlatformBooks:
    def test_cross_platform_books_pass_with_sufficient_depth(self):
        opp = Opportunity(
            type=OpportunityType.CROSS_PLATFORM_ARB,
            event_id="e1",
            legs=(
                LegOrder("pm_yes", Side.BUY, 0.40, 10, platform="polymarket"),
                LegOrder("K-TEST", Side.SELL, 0.60, 10, platform="kalshi"),
            ),
            expected_profit_per_set=0.20,
            net_profit_per_set=0.20,
            max_sets=10,
            gross_profit=2.0,
            estimated_gas_cost=0.01,
            net_profit=1.99,
            roi_pct=24.0,
            required_capital=8.0,
        )
        pm_books = {"pm_yes": _make_book("pm_yes", 0.39, 50, 0.40, 50)}
        kalshi_books = {"K-TEST": _make_book("K-TEST", 0.60, 50, 0.61, 50)}
        verify_cross_platform_books(opp, pm_books, kalshi_books, min_depth=5.0)

    def test_cross_platform_missing_kalshi_book_raises(self):
        opp = Opportunity(
            type=OpportunityType.CROSS_PLATFORM_ARB,
            event_id="e1",
            legs=(LegOrder("K-TEST", Side.BUY, 0.40, 10, platform="kalshi"),),
            expected_profit_per_set=0.10,
            net_profit_per_set=0.10,
            max_sets=10,
            gross_profit=1.0,
            estimated_gas_cost=0.01,
            net_profit=0.99,
            roi_pct=10.0,
            required_capital=4.0,
        )
        with pytest.raises(SafetyCheckFailed, match="No orderbook"):
            verify_cross_platform_books(opp, pm_books={}, kalshi_books={}, min_depth=1.0)
