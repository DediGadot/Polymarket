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
        state = MarketState(gas_price_gwei=150.0, avg_spread_pct=2.0)
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
        state = MarketState(gas_price_gwei=150.0)
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
