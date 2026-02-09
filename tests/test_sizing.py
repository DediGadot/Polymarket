"""
Unit tests for executor/sizing.py -- Kelly criterion position sizing.
"""

from executor.sizing import kelly_fraction, compute_position_size
from scanner.models import (
    Opportunity,
    OpportunityType,
    LegOrder,
    Side,
)


def _make_opp(profit_per_set=0.10, max_sets=100.0, required_capital=90.0, gas=0.01):
    return Opportunity(
        type=OpportunityType.BINARY_REBALANCE,
        event_id="e1",
        legs=(
            LegOrder("y1", Side.BUY, 0.45, max_sets),
            LegOrder("n1", Side.BUY, 0.45, max_sets),
        ),
        expected_profit_per_set=profit_per_set,
        net_profit_per_set=profit_per_set,
        max_sets=max_sets,
        gross_profit=profit_per_set * max_sets,
        estimated_gas_cost=gas,
        net_profit=profit_per_set * max_sets - gas,
        roi_pct=(profit_per_set * max_sets - gas) / required_capital * 100 if required_capital > 0 else 0.0,
        required_capital=required_capital,
    )


class TestKellyFraction:
    def test_positive_edge(self):
        f = kelly_fraction(edge=0.10, odds=1.0)
        assert 0 < f <= 0.5  # half-Kelly capped

    def test_zero_edge(self):
        f = kelly_fraction(edge=0.0, odds=1.0)
        assert f == 0.0

    def test_negative_odds(self):
        f = kelly_fraction(edge=0.10, odds=-1.0)
        assert f == 0.0

    def test_zero_odds(self):
        f = kelly_fraction(edge=0.10, odds=0.0)
        assert f == 0.0

    def test_large_edge_capped(self):
        f = kelly_fraction(edge=5.0, odds=1.0)
        assert f <= 1.0

    def test_half_kelly_applied(self):
        """Kelly fraction should be halved for safety."""
        f = kelly_fraction(edge=0.10, odds=1.0)
        full_kelly = 0.10 / 1.0
        assert abs(f - full_kelly * 0.5) < 1e-9


class TestComputePositionSize:
    def test_basic_sizing(self):
        opp = _make_opp(profit_per_set=0.10, max_sets=100, required_capital=90.0)
        size = compute_position_size(
            opp, bankroll=5000, max_exposure_per_trade=500,
            max_total_exposure=5000, current_exposure=0,
        )
        assert size > 0
        assert size <= 100  # can't exceed max_sets

    def test_no_capital_available(self):
        opp = _make_opp()
        size = compute_position_size(
            opp, bankroll=5000, max_exposure_per_trade=500,
            max_total_exposure=5000, current_exposure=5000,  # fully deployed
        )
        assert size == 0.0

    def test_limited_by_max_exposure(self):
        opp = _make_opp(max_sets=10000, required_capital=9000.0)
        size = compute_position_size(
            opp, bankroll=100000, max_exposure_per_trade=100,
            max_total_exposure=100000, current_exposure=0,
        )
        # Should be limited by max_exposure_per_trade
        cost_per_set = 9000.0 / 10000  # 0.90
        max_from_exposure = 100 / cost_per_set  # ~111
        assert size <= max_from_exposure + 1

    def test_limited_by_max_sets(self):
        opp = _make_opp(max_sets=5, required_capital=4.5)
        size = compute_position_size(
            opp, bankroll=100000, max_exposure_per_trade=50000,
            max_total_exposure=100000, current_exposure=0,
        )
        assert size <= 5.0

    def test_zero_max_sets(self):
        opp = _make_opp(max_sets=0, required_capital=0)
        size = compute_position_size(
            opp, bankroll=5000, max_exposure_per_trade=500,
            max_total_exposure=5000, current_exposure=0,
        )
        assert size == 0.0

    def test_tiny_opportunity_skipped(self):
        """If Kelly says < 1 set, skip it."""
        opp = _make_opp(profit_per_set=0.001, max_sets=2, required_capital=1.998)
        size = compute_position_size(
            opp, bankroll=100, max_exposure_per_trade=50,
            max_total_exposure=100, current_exposure=99,
        )
        assert size == 0.0

    def test_skip_when_fixed_gas_makes_sized_trade_negative(self):
        """
        Opportunity can be net-positive at max size but net-negative at the
        smaller Kelly/exposure-constrained size once fixed gas is applied.
        """
        opp = Opportunity(
            type=OpportunityType.BINARY_REBALANCE,
            event_id="e1",
            legs=(
                LegOrder("y1", Side.BUY, 0.4975, 1000),
                LegOrder("n1", Side.BUY, 0.4975, 1000),
            ),
            expected_profit_per_set=0.005,
            net_profit_per_set=0.005,
            max_sets=1000,
            gross_profit=5.0,
            estimated_gas_cost=3.0,
            net_profit=2.0,
            roi_pct=0.2,
            required_capital=995.0,
        )
        size = compute_position_size(
            opp,
            bankroll=5000,
            max_exposure_per_trade=500,
            max_total_exposure=5000,
            current_exposure=0,
        )
        assert size == 0.0
