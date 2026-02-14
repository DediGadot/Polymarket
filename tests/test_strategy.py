"""
Unit tests for scanner/strategy.py -- adaptive strategy selection.
"""

from scanner.strategy import (
    StrategySelector,
    StrategyMode,
    MarketState,
)


class TestPickMode:
    def test_spike_hunt_on_active_spikes(self):
        sel = StrategySelector()
        state = MarketState(active_spike_count=2)
        params = sel.select(state)
        assert params.mode == StrategyMode.SPIKE_HUNT

    def test_latency_focus_on_crypto_momentum(self):
        sel = StrategySelector()
        state = MarketState(has_crypto_momentum=True)
        params = sel.select(state)
        assert params.mode == StrategyMode.LATENCY_FOCUS

    def test_aggressive_on_low_gas_wide_spread(self):
        sel = StrategySelector()
        state = MarketState(gas_price_gwei=15.0, avg_spread_pct=4.0)
        params = sel.select(state)
        assert params.mode == StrategyMode.AGGRESSIVE

    def test_conservative_on_high_gas(self):
        sel = StrategySelector()
        # Use gas above new HIGH_GAS_GWEI threshold (1000)
        state = MarketState(gas_price_gwei=1500.0, avg_spread_pct=2.0)
        params = sel.select(state)
        assert params.mode == StrategyMode.CONSERVATIVE

    def test_conservative_on_tight_spread(self):
        sel = StrategySelector()
        state = MarketState(gas_price_gwei=30.0, avg_spread_pct=0.5)
        params = sel.select(state)
        assert params.mode == StrategyMode.CONSERVATIVE

    def test_spike_overrides_latency(self):
        """Spikes take priority over crypto momentum."""
        sel = StrategySelector()
        state = MarketState(active_spike_count=1, has_crypto_momentum=True)
        params = sel.select(state)
        assert params.mode == StrategyMode.SPIKE_HUNT

    def test_default_aggressive_on_winning(self):
        sel = StrategySelector()
        state = MarketState(gas_price_gwei=30.0, avg_spread_pct=2.0, recent_win_rate=0.75)
        params = sel.select(state)
        assert params.mode == StrategyMode.AGGRESSIVE

    def test_default_conservative_on_losing(self):
        sel = StrategySelector()
        state = MarketState(gas_price_gwei=30.0, avg_spread_pct=2.0, recent_win_rate=0.30)
        params = sel.select(state)
        assert params.mode == StrategyMode.CONSERVATIVE


class TestScanParams:
    def test_spike_hunt_disables_binary(self):
        sel = StrategySelector()
        state = MarketState(active_spike_count=1)
        params = sel.select(state)
        assert params.binary_enabled is False
        assert params.negrisk_enabled is True
        assert params.spike_enabled is True

    def test_aggressive_lowers_thresholds(self):
        sel = StrategySelector(base_min_profit=1.0, base_min_roi=5.0)
        state = MarketState(gas_price_gwei=15.0, avg_spread_pct=4.0)
        params = sel.select(state)
        assert params.min_profit_usd < 1.0
        assert params.min_roi_pct < 5.0
        assert params.target_size_usd > 100.0

    def test_conservative_raises_thresholds(self):
        sel = StrategySelector(base_min_profit=1.0, base_min_roi=5.0)
        # Use gas above new HIGH_GAS_GWEI threshold (1000)
        state = MarketState(gas_price_gwei=1500.0)
        params = sel.select(state)
        assert params.min_profit_usd > 1.0
        assert params.min_roi_pct > 5.0
        assert params.target_size_usd < 100.0

    def test_latency_focus_all_scanners(self):
        sel = StrategySelector()
        state = MarketState(has_crypto_momentum=True)
        params = sel.select(state)
        assert params.binary_enabled is True
        assert params.latency_enabled is True
        assert params.spike_enabled is True


class TestGasCostUsd:
    """Tests for dollar-denominated gas cost mode selection."""

    def test_aggressive_on_cheap_gas_usd(self):
        """AGGRESSIVE selected when gas_cost_usd=$0.005 (cheap Polygon gas)."""
        sel = StrategySelector()
        state = MarketState(gas_cost_usd=0.005, avg_spread_pct=4.0)
        params = sel.select(state)
        assert params.mode == StrategyMode.AGGRESSIVE

    def test_conservative_on_expensive_gas_usd(self):
        """CONSERVATIVE selected when gas_cost_usd=$0.15 (expensive gas)."""
        sel = StrategySelector()
        state = MarketState(gas_cost_usd=0.15)
        params = sel.select(state)
        assert params.mode == StrategyMode.CONSERVATIVE

    def test_fallback_to_gwei_when_gas_cost_usd_none(self):
        """Fall back to gwei thresholds when gas_cost_usd is None."""
        sel = StrategySelector()
        # Low gas in gwei → AGGRESSIVE
        state = MarketState(gas_price_gwei=50.0, gas_cost_usd=None, avg_spread_pct=4.0)
        params = sel.select(state)
        assert params.mode == StrategyMode.AGGRESSIVE

        # High gas in gwei → CONSERVATIVE
        state2 = MarketState(gas_price_gwei=1500.0, gas_cost_usd=None)
        params2 = sel.select(state2)
        assert params2.mode == StrategyMode.CONSERVATIVE

    def test_polygon_100_gwei_to_aggressive(self):
        """
        Test that 100 gwei on Polygon (= ~$0.003) → AGGRESSIVE (not CONSERVATIVE).
        This was the bug: 100 gwei triggered CONSERVATIVE with old thresholds.
        """
        sel = StrategySelector()
        # 100 gwei is below LOW_GAS_GWEI=1000, should be AGGRESSIVE with wide spreads
        state = MarketState(gas_price_gwei=100.0, avg_spread_pct=4.0)
        params = sel.select(state)
        assert params.mode == StrategyMode.AGGRESSIVE

    def test_gas_cost_usd_preferred_over_gwei(self):
        """Dollar thresholds take precedence when both are available."""
        sel = StrategySelector()
        # gas_cost_usd says expensive, gwei says cheap → dollar wins
        state = MarketState(gas_price_gwei=50.0, gas_cost_usd=0.15)
        params = sel.select(state)
        assert params.mode == StrategyMode.CONSERVATIVE

    def test_gas_cost_usd_threshold_boundaries(self):
        """Test threshold boundary conditions."""
        sel = StrategySelector()

        # Exactly at HIGH_GAS_USD (0.10) → not >, so no CONSERVATIVE from gas
        # but win rate 0.50 defaults to AGGRESSIVE
        state1 = MarketState(gas_cost_usd=0.10, recent_win_rate=0.50)
        params1 = sel.select(state1)
        assert params1.mode == StrategyMode.AGGRESSIVE

        # Just above HIGH_GAS_USD → CONSERVATIVE
        state2 = MarketState(gas_cost_usd=0.11)
        params2 = sel.select(state2)
        assert params2.mode == StrategyMode.CONSERVATIVE

        # Exactly at LOW_GAS_USD (0.01) → not <, so no AGGRESSIVE from low gas
        # but tight spreads trigger CONSERVATIVE
        state3 = MarketState(gas_cost_usd=0.01, avg_spread_pct=0.5)
        params3 = sel.select(state3)
        assert params3.mode == StrategyMode.CONSERVATIVE

        # Just below LOW_GAS_USD + wide spreads → AGGRESSIVE
        state4 = MarketState(gas_cost_usd=0.009, avg_spread_pct=4.0)
        params4 = sel.select(state4)
        assert params4.mode == StrategyMode.AGGRESSIVE
