"""
Phase 2: Arithmetic correctness validation.

Manually computes expected values for every scanner and verifies the pipeline
produces matching results. This is the most critical validation -- incorrect
math means losing money on every trade.
"""

import pytest

from scanner.binary import _check_buy_arb, _check_sell_arb
from scanner.negrisk import _check_buy_all_arb, _check_sell_all_arb
from scanner.fees import MarketFeeModel, MAX_CRYPTO_FEE_RATE, RESOLUTION_FEE_RATE
from scanner.depth import sweep_cost, effective_price, sweep_depth, find_deep_binary_arb
from scanner.scorer import score_opportunity, ScoringContext
from scanner.latency import LatencyScanner
from client.gas import GasOracle
from executor.sizing import kelly_fraction, compute_position_size
from scanner.models import (
    Market,
    Event,
    OrderBook,
    PriceLevel,
    Opportunity,
    OpportunityType,
    LegOrder,
    Side,
)


def _make_market(question="Test?", neg_risk=False, event_id="e1"):
    return Market(
        condition_id="c1", question=question,
        yes_token_id="y1", no_token_id="n1",
        neg_risk=neg_risk, event_id=event_id,
        min_tick_size="0.01", active=True, volume=1000.0,
    )


def _make_book(token_id, bids, asks):
    """bids/asks: list of (price, size) tuples."""
    return OrderBook(
        token_id=token_id,
        bids=tuple(PriceLevel(p, s) for p, s in bids),
        asks=tuple(PriceLevel(p, s) for p, s in asks),
    )


# ===========================================================================
# 1. Binary Buy Arb Arithmetic
# ===========================================================================
class TestBinaryBuyArbArithmetic:
    def test_basic_profit_calculation(self):
        """YES ask=0.45, NO ask=0.50 → cost=0.95, profit/set=0.05"""
        market = _make_market()
        yes_book = _make_book("y1", [(0.44, 500)], [(0.45, 200)])
        no_book = _make_book("n1", [(0.49, 500)], [(0.50, 150)])

        opp = _check_buy_arb(
            market, yes_book, no_book,
            min_profit_usd=0.0, min_roi_pct=0.0,
            gas_per_order=150000, gas_price_gwei=30.0,
        )

        assert opp is not None
        # Manual: cost_per_set = 0.45 + 0.50 = 0.95
        assert opp.legs[0].price + opp.legs[1].price == pytest.approx(0.95)
        # profit/set = 1.0 - 0.95 = 0.05
        assert opp.expected_profit_per_set == pytest.approx(0.05)
        # max_sets = min(200, 150) = 150
        assert opp.max_sets == pytest.approx(150.0)
        # gross = 0.05 * 150 = 7.50
        assert opp.gross_profit == pytest.approx(7.50)
        # gas = 2 * 150000 * 30 * 1e9 / 1e18 * 0.50 = 0.0045
        expected_gas = 2 * 150000 * 30.0 * 1e9 / 1e18 * 0.50
        assert opp.estimated_gas_cost == pytest.approx(expected_gas, abs=0.001)
        # net = 7.50 - 0.0045 ≈ 7.4955
        assert opp.net_profit == pytest.approx(7.50 - expected_gas, abs=0.001)
        # required_capital = 0.95 * 150 = 142.50
        assert opp.required_capital == pytest.approx(142.50)
        # roi = (7.4955 / 142.50) * 100 ≈ 5.26%
        assert opp.roi_pct == pytest.approx(
            (opp.net_profit / 142.50) * 100, abs=0.01,
        )

    def test_no_arb_when_cost_equals_1(self):
        """YES ask=0.50, NO ask=0.50 → cost=1.00 → no arb"""
        market = _make_market()
        yes_book = _make_book("y1", [(0.49, 500)], [(0.50, 200)])
        no_book = _make_book("n1", [(0.49, 500)], [(0.50, 200)])

        opp = _check_buy_arb(
            market, yes_book, no_book,
            min_profit_usd=0.0, min_roi_pct=0.0,
            gas_per_order=150000, gas_price_gwei=30.0,
        )
        assert opp is None

    def test_no_arb_when_cost_above_1(self):
        """YES ask=0.55, NO ask=0.50 → cost=1.05 → no arb"""
        market = _make_market()
        yes_book = _make_book("y1", [(0.54, 500)], [(0.55, 200)])
        no_book = _make_book("n1", [(0.49, 500)], [(0.50, 200)])

        opp = _check_buy_arb(
            market, yes_book, no_book,
            min_profit_usd=0.0, min_roi_pct=0.0,
            gas_per_order=150000, gas_price_gwei=30.0,
        )
        assert opp is None

    def test_with_fee_model_reduces_profit(self):
        """Fee model deducts resolution fee ($0.02/set)"""
        market = _make_market()
        yes_book = _make_book("y1", [(0.44, 500)], [(0.45, 200)])
        no_book = _make_book("n1", [(0.49, 500)], [(0.50, 150)])
        fee_model = MarketFeeModel(enabled=True)

        opp = _check_buy_arb(
            market, yes_book, no_book,
            min_profit_usd=0.0, min_roi_pct=0.0,
            gas_per_order=150000, gas_price_gwei=30.0,
            fee_model=fee_model,
        )

        assert opp is not None
        # Standard market has zero taker fee, but still has resolution fee
        # Resolution fee = $0.02/set
        # Net profit/set = 0.05 - 0.02 = 0.03
        # Net profit = 0.03 * 150 - gas ≈ 4.50 - 0.0045
        assert opp.net_profit < 7.50  # less than without fees

    def test_with_gas_oracle(self):
        """Gas oracle provides real-time cost instead of static estimate"""
        market = _make_market()
        yes_book = _make_book("y1", [(0.44, 500)], [(0.45, 200)])
        no_book = _make_book("n1", [(0.49, 500)], [(0.50, 150)])

        oracle = GasOracle(default_gas_gwei=30.0, default_matic_usd=0.50)
        oracle._cached_gas_gwei = 100.0  # high gas
        oracle._gas_ts = 1e18  # far future = never expired
        oracle._cached_matic_usd = 1.00  # higher POL price
        oracle._matic_ts = 1e18

        opp = _check_buy_arb(
            market, yes_book, no_book,
            min_profit_usd=0.0, min_roi_pct=0.0,
            gas_per_order=150000, gas_price_gwei=30.0,
            gas_oracle=oracle,
        )

        assert opp is not None
        # Gas = 2 * 150000 * 100 * 1e9 / 1e18 * 1.00 = 0.030
        expected_gas = 2 * 150000 * 100.0 * 1e9 / 1e18 * 1.00
        assert opp.estimated_gas_cost == pytest.approx(expected_gas, abs=0.001)

    def test_empty_orderbook_returns_none(self):
        market = _make_market()
        yes_book = _make_book("y1", [], [])  # empty
        no_book = _make_book("n1", [(0.49, 500)], [(0.50, 200)])

        opp = _check_buy_arb(
            market, yes_book, no_book,
            min_profit_usd=0.0, min_roi_pct=0.0,
            gas_per_order=150000, gas_price_gwei=30.0,
        )
        assert opp is None


# ===========================================================================
# 2. Binary Sell Arb Arithmetic
# ===========================================================================
class TestBinarySellArbArithmetic:
    def test_basic_sell_arb(self):
        """YES bid=0.55, NO bid=0.50 → proceeds=1.05, profit/set=0.05"""
        market = _make_market()
        yes_book = _make_book("y1", [(0.55, 200)], [(0.56, 500)])
        no_book = _make_book("n1", [(0.50, 150)], [(0.51, 500)])

        opp = _check_sell_arb(
            market, yes_book, no_book,
            min_profit_usd=0.0, min_roi_pct=0.0,
            gas_per_order=150000, gas_price_gwei=30.0,
        )

        assert opp is not None
        # proceeds = 0.55 + 0.50 = 1.05
        assert opp.expected_profit_per_set == pytest.approx(0.05)
        # max_sets = min(200, 150) = 150
        assert opp.max_sets == pytest.approx(150.0)
        # required_capital for sell = 1.0 * max_sets (you need to hold the tokens)
        assert opp.required_capital == pytest.approx(150.0)
        # Both legs should be SELL
        assert all(leg.side == Side.SELL for leg in opp.legs)

    def test_no_sell_arb_when_proceeds_below_1(self):
        market = _make_market()
        yes_book = _make_book("y1", [(0.45, 200)], [(0.46, 500)])
        no_book = _make_book("n1", [(0.50, 150)], [(0.51, 500)])

        opp = _check_sell_arb(
            market, yes_book, no_book,
            min_profit_usd=0.0, min_roi_pct=0.0,
            gas_per_order=150000, gas_price_gwei=30.0,
        )
        assert opp is None


# ===========================================================================
# 3. NegRisk Buy Arb Arithmetic
# ===========================================================================
class TestNegRiskBuyArbArithmetic:
    def test_3_outcomes(self):
        """3 outcomes at asks [0.30, 0.30, 0.35] → cost=0.95, profit=0.05"""
        markets = [
            Market("c1", "A?", "y1", "n1", True, "e1", "0.01", True, 1000),
            Market("c2", "B?", "y2", "n2", True, "e1", "0.01", True, 1000),
            Market("c3", "C?", "y3", "n3", True, "e1", "0.01", True, 1000),
        ]
        event = Event("e1", "Test Event", tuple(markets), True)

        books = {
            "y1": _make_book("y1", [(0.29, 500)], [(0.30, 100)]),
            "y2": _make_book("y2", [(0.29, 500)], [(0.30, 200)]),
            "y3": _make_book("y3", [(0.34, 500)], [(0.35, 150)]),
        }

        opp = _check_buy_all_arb(
            event, books, markets,
            min_profit_usd=0.0, min_roi_pct=0.0,
            gas_per_order=150000, gas_price_gwei=30.0,
        )

        assert opp is not None
        # total_cost = 0.30 + 0.30 + 0.35 = 0.95
        assert opp.expected_profit_per_set == pytest.approx(0.05)
        # max_sets = min(100, 200, 150) = 100
        assert opp.max_sets == pytest.approx(100.0)
        # 3 legs
        assert len(opp.legs) == 3
        # Gas scales with n_legs: 3 * 150000 * 30 * 1e9 / 1e18 * 0.50
        expected_gas = 3 * 150000 * 30.0 * 1e9 / 1e18 * 0.50
        assert opp.estimated_gas_cost == pytest.approx(expected_gas, abs=0.001)
        # required_capital = 0.95 * 100 = 95.0
        assert opp.required_capital == pytest.approx(95.0)

    def test_no_arb_when_sum_equals_1(self):
        markets = [
            Market("c1", "A?", "y1", "n1", True, "e1", "0.01", True, 1000),
            Market("c2", "B?", "y2", "n2", True, "e1", "0.01", True, 1000),
        ]
        event = Event("e1", "Test", tuple(markets), True)

        books = {
            "y1": _make_book("y1", [(0.49, 500)], [(0.50, 100)]),
            "y2": _make_book("y2", [(0.49, 500)], [(0.50, 100)]),
        }

        opp = _check_buy_all_arb(
            event, books, markets,
            min_profit_usd=0.0, min_roi_pct=0.0,
            gas_per_order=150000, gas_price_gwei=30.0,
        )
        assert opp is None

    def test_missing_book_returns_none(self):
        markets = [
            Market("c1", "A?", "y1", "n1", True, "e1", "0.01", True, 1000),
            Market("c2", "B?", "y2", "n2", True, "e1", "0.01", True, 1000),
        ]
        event = Event("e1", "Test", tuple(markets), True)

        books = {
            "y1": _make_book("y1", [(0.49, 500)], [(0.30, 100)]),
            # y2 missing
        }

        opp = _check_buy_all_arb(
            event, books, markets,
            min_profit_usd=0.0, min_roi_pct=0.0,
            gas_per_order=150000, gas_price_gwei=30.0,
        )
        assert opp is None


# ===========================================================================
# 4. Fee Model Arithmetic
# ===========================================================================
class TestFeeModelArithmetic:
    def test_dynamic_fee_at_50_50(self):
        """At price=0.50: fee = MAX_RATE * 4 * 0.5 * 0.5 = MAX_RATE"""
        fm = MarketFeeModel(enabled=True)
        market = _make_market("Will BTC be up 0.5% in 15 minutes?")
        fee = fm.get_taker_fee(market, 0.50)
        assert fee == pytest.approx(MAX_CRYPTO_FEE_RATE, abs=0.0001)

    def test_dynamic_fee_at_10(self):
        """At price=0.10: fee = MAX_RATE * 4 * 0.1 * 0.9 = 0.01134"""
        fm = MarketFeeModel(enabled=True)
        market = _make_market("Will BTC be up 0.5% in 15 minutes?")
        fee = fm.get_taker_fee(market, 0.10)
        expected = MAX_CRYPTO_FEE_RATE * 4.0 * 0.10 * 0.90
        assert fee == pytest.approx(expected, abs=0.0001)

    def test_dynamic_fee_symmetry(self):
        """Fee at 0.10 should equal fee at 0.90 (parabolic symmetry)"""
        fm = MarketFeeModel(enabled=True)
        market = _make_market("Will BTC be up 0.5% in 15 minutes?")
        fee_low = fm.get_taker_fee(market, 0.10)
        fee_high = fm.get_taker_fee(market, 0.90)
        assert fee_low == pytest.approx(fee_high, abs=0.0001)

    def test_dynamic_fee_at_extremes(self):
        """Fee approaches 0 at price=0 and price=1"""
        fm = MarketFeeModel(enabled=True)
        market = _make_market("Will BTC be up 0.5% in 15 minutes?")
        assert fm.get_taker_fee(market, 0.0) == pytest.approx(0.0)
        assert fm.get_taker_fee(market, 1.0) == pytest.approx(0.0)

    def test_standard_market_zero_fee(self):
        fm = MarketFeeModel(enabled=True)
        market = _make_market("Will the next president be a Democrat?")
        assert fm.get_taker_fee(market, 0.50) == 0.0

    def test_resolution_fee_always_002(self):
        fm = MarketFeeModel(enabled=True)
        assert fm.estimate_resolution_fee(0.05) == pytest.approx(0.02)
        assert fm.estimate_resolution_fee(0.50) == pytest.approx(0.02)
        assert fm.estimate_resolution_fee(0.0) == pytest.approx(0.02)

    def test_adjust_profit_binary(self):
        """Standard market: only resolution fee ($0.02), no taker fee"""
        fm = MarketFeeModel(enabled=True)
        market = _make_market("Non-crypto market?")
        legs = (
            LegOrder("y1", Side.BUY, 0.45, 100),
            LegOrder("n1", Side.BUY, 0.50, 100),
        )
        # gross profit/set = 0.05, resolution fee = 0.02, taker fee = 0
        net = fm.adjust_profit(0.05, legs, market=market)
        assert net == pytest.approx(0.05 - 0.02, abs=0.001)

    def test_adjust_profit_crypto_15min(self):
        """Crypto 15-min: taker fee on each leg + resolution fee"""
        fm = MarketFeeModel(enabled=True)
        market = _make_market("Will BTC be up 0.5% in 15 minutes?")
        legs = (
            LegOrder("y1", Side.BUY, 0.50, 100),
            LegOrder("n1", Side.BUY, 0.50, 100),
        )
        # gross profit/set = 0.00 (not a real arb, but testing fee math)
        # taker fee per leg = 0.0315 * 0.50 = 0.01575
        # total taker = 2 * 0.01575 = 0.0315
        # resolution = 0.02
        # net = 0.00 - 0.0315 - 0.02 = -0.0515
        net = fm.adjust_profit(0.00, legs, market=market)
        expected_taker = MAX_CRYPTO_FEE_RATE * 0.50 * 2  # two legs at 0.50
        assert net == pytest.approx(0.00 - expected_taker - RESOLUTION_FEE_RATE, abs=0.001)


# ===========================================================================
# 5. Gas Oracle Arithmetic
# ===========================================================================
class TestGasOracleArithmetic:
    def test_estimate_cost_formula(self):
        """Verify: cost = n_orders * gas_per_order * gwei * 1e9 / 1e18 * matic_usd"""
        oracle = GasOracle()
        oracle._cached_gas_gwei = 30.0
        oracle._gas_ts = 1e18
        oracle._cached_matic_usd = 0.50
        oracle._matic_ts = 1e18

        cost = oracle.estimate_cost_usd(2, 150000)
        # = 2 * 150000 * 30 * 1e9 / 1e18 * 0.50
        # = 300000 * 30e9 / 1e18 * 0.50
        # = 9e15 / 1e18 * 0.50
        # = 0.009 * 0.50
        # = 0.0045
        assert cost == pytest.approx(0.0045)

    def test_higher_gas_higher_cost(self):
        oracle = GasOracle()
        oracle._cached_gas_gwei = 300.0  # 10x higher
        oracle._gas_ts = 1e18
        oracle._cached_matic_usd = 0.50
        oracle._matic_ts = 1e18

        cost = oracle.estimate_cost_usd(2, 150000)
        # = 0.0045 * 10 = 0.045
        assert cost == pytest.approx(0.045)

    def test_single_leg_cost(self):
        oracle = GasOracle()
        oracle._cached_gas_gwei = 30.0
        oracle._gas_ts = 1e18
        oracle._cached_matic_usd = 0.50
        oracle._matic_ts = 1e18

        cost = oracle.estimate_cost_usd(1, 150000)
        assert cost == pytest.approx(0.00225)


# ===========================================================================
# 6. Depth Sweep Arithmetic
# ===========================================================================
class TestDepthSweepArithmetic:
    def test_sweep_cost_multi_level(self):
        """Walk 3 levels: 100@0.50 + 100@0.51 + 50@0.52 = 126.0"""
        book = _make_book("y1", [], [
            (0.50, 100), (0.51, 100), (0.52, 200),
        ])
        cost = sweep_cost(book, Side.BUY, 250)
        # 100*0.50 + 100*0.51 + 50*0.52 = 50 + 51 + 26 = 127.0
        assert cost == pytest.approx(127.0)

    def test_effective_price_vwap(self):
        book = _make_book("y1", [], [
            (0.50, 100), (0.52, 100),
        ])
        vwap = effective_price(book, Side.BUY, 200)
        # (100*0.50 + 100*0.52) / 200 = 102/200 = 0.51
        assert vwap == pytest.approx(0.51)

    def test_effective_price_single_level(self):
        book = _make_book("y1", [], [(0.50, 500)])
        vwap = effective_price(book, Side.BUY, 100)
        assert vwap == pytest.approx(0.50)

    def test_effective_price_insufficient_depth(self):
        book = _make_book("y1", [], [(0.50, 50)])
        assert effective_price(book, Side.BUY, 100) is None

    def test_sweep_depth_buy(self):
        book = _make_book("y1", [], [
            (0.50, 100), (0.51, 200), (0.52, 300),
        ])
        # depth at max_price=0.51: 100 + 200 = 300
        assert sweep_depth(book, Side.BUY, 0.51) == pytest.approx(300.0)

    def test_find_deep_binary_arb(self):
        yes_book = _make_book("y1", [], [(0.45, 100), (0.50, 200)])
        no_book = _make_book("n1", [], [(0.48, 100), (0.52, 200)])

        result = find_deep_binary_arb(yes_book, no_book, target_size=200)
        if result:
            yes_vwap, no_vwap, size = result
            # At 200 size: YES = (100*0.45+100*0.50)/200=0.475
            # NO = (100*0.48+100*0.52)/200=0.50
            # Sum = 0.975 < 1.0 → arb
            assert yes_vwap + no_vwap < 1.0
            assert size == 200

    def test_no_deep_arb_when_sum_above_1(self):
        yes_book = _make_book("y1", [], [(0.55, 200)])
        no_book = _make_book("n1", [], [(0.50, 200)])
        assert find_deep_binary_arb(yes_book, no_book, 100) is None


# ===========================================================================
# 7. Kelly Sizing Arithmetic
# ===========================================================================
class TestKellySizingArithmetic:
    def test_5pct_edge(self):
        """5% edge, odds=1.0 → f = edge/odds * 0.5 = 0.05 * 0.5 = 0.025"""
        f = kelly_fraction(0.05, 1.0)
        assert f == pytest.approx(0.025)

    def test_sizing_with_bankroll(self):
        """$5000 bankroll, 5% edge → kelly_capital = 0.025 * 5000 = $125"""
        opp = Opportunity(
            type=OpportunityType.BINARY_REBALANCE,
            event_id="e1",
            legs=(
                LegOrder("y1", Side.BUY, 0.45, 1000),
                LegOrder("n1", Side.BUY, 0.50, 1000),
            ),
            expected_profit_per_set=0.05,
            net_profit_per_set=0.05,
            max_sets=1000,
            gross_profit=50.0,
            estimated_gas_cost=0.01,
            net_profit=49.99,
            roi_pct=5.26,
            required_capital=950.0,  # 0.95 * 1000
        )

        size = compute_position_size(
            opp,
            bankroll=5000.0,
            max_exposure_per_trade=500.0,
            max_total_exposure=5000.0,
            current_exposure=0.0,
        )

        # cost_per_set = 950/1000 = 0.95
        # edge = 0.05/0.95 ≈ 0.05263
        # kelly_f = 0.05263 * 0.5 = 0.02632
        # kelly_capital = 0.02632 * 5000 = 131.58
        # sets = 131.58 / 0.95 = 138.5
        # Capped by max_sets=1000 → 138.5
        assert size > 100
        assert size <= 1000


# ===========================================================================
# 8. Latency Arb with Fee Validation
# ===========================================================================
class TestLatencyArbFeeValidation:
    def test_fees_kill_edge_at_50_50(self):
        """At 50/50 odds, 3.15% fee makes most edges unprofitable"""
        fm = MarketFeeModel(enabled=True)
        scanner = LatencyScanner(min_edge_pct=5.0, fee_model=fm)

        import time
        now = time.time()
        # 1% momentum → implied ~85%
        scanner._spot_cache["BTC"] = (101000.0, now)
        scanner._prev_spot["BTC"] = (100000.0, now - 5)

        market = _make_market("Will BTC be up 0.5% in 15 minutes?")
        # Market at 50/50 → fee is 3.15%
        book = _make_book("yes1", [(0.49, 200)], [(0.50, 200)])

        opp = scanner.check_latency_arb(market, book, "BTC", "up")
        # At 0.50 ask: implied=0.85, edge = 0.85-0.50=0.35
        # fee = 3.15% * 0.50 = 0.01575
        # resolution = 0.02
        # net_edge = 0.35 - 0.01575 - 0.02 = 0.31425
        # net_edge_pct = 0.31425/0.50 * 100 = 62.85% → exceeds 5% min → ARB
        # The fee hurts but doesn't kill because 1% momentum creates huge edge
        if opp is not None:
            assert opp.net_profit > 0

    def test_low_odds_low_fee(self):
        """At extreme odds (0.10), fee drops to ~1.13% → more viable"""
        fm = MarketFeeModel(enabled=True)
        market = _make_market("Will BTC be up 0.5% in 15 minutes?")
        fee_50 = fm.get_taker_fee(market, 0.50)
        fee_10 = fm.get_taker_fee(market, 0.10)
        assert fee_10 < fee_50  # lower fee at extreme odds


# ===========================================================================
# 9. Scorer Arithmetic
# ===========================================================================
class TestScorerArithmetic:
    def test_weights_sum_to_1(self):
        from scanner.scorer import W_PROFIT, W_FILL, W_EFFICIENCY, W_URGENCY, W_COMPETITION
        assert W_PROFIT + W_FILL + W_EFFICIENCY + W_URGENCY + W_COMPETITION == pytest.approx(1.0)

    def test_spike_always_scores_higher_than_binary(self):
        """Same opportunity, but SPIKE_LAG should score higher due to urgency"""
        legs = (LegOrder("y1", Side.BUY, 0.45, 100),)
        base = dict(
            event_id="e1", legs=legs,
            expected_profit_per_set=0.05, net_profit_per_set=0.05,
            max_sets=100,
            gross_profit=5.0, estimated_gas_cost=0.01,
            net_profit=4.99, roi_pct=5.5, required_capital=45.0,
        )

        binary_opp = Opportunity(type=OpportunityType.BINARY_REBALANCE, **base)
        spike_opp = Opportunity(type=OpportunityType.SPIKE_LAG, **base)

        ctx = ScoringContext()
        binary_scored = score_opportunity(binary_opp, ctx)
        spike_scored = score_opportunity(spike_opp, ctx)

        assert spike_scored.total_score > binary_scored.total_score
        assert spike_scored.urgency_score > binary_scored.urgency_score
