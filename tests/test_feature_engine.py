"""Tests for scanner/feature_engine.py — ML feature extraction."""

from __future__ import annotations

import math

import numpy as np
import pytest

from scanner.feature_engine import (
    FEATURE_NAMES,
    N_FEATURES,
    FeatureEngine,
    RollingStats,
    _OPP_TYPES,
)
from scanner.models import LegOrder, Opportunity, OpportunityType, Side
from scanner.scorer import ScoringContext


# ---------------------------------------------------------------------------
# Factory
# ---------------------------------------------------------------------------

def _make_opp(**kwargs) -> Opportunity:
    defaults = dict(
        type=OpportunityType.BINARY_REBALANCE,
        event_id="evt1",
        legs=(LegOrder(token_id="tok1", side=Side.BUY, price=0.45, size=100),),
        expected_profit_per_set=0.05,
        net_profit_per_set=0.03,
        max_sets=100,
        gross_profit=5.0,
        estimated_gas_cost=0.01,
        net_profit=3.0,
        roi_pct=6.67,
        required_capital=45.0,
    )
    defaults.update(kwargs)
    return Opportunity(**defaults)


def _make_ctx(**kwargs) -> ScoringContext:
    return ScoringContext(**kwargs)


# ---------------------------------------------------------------------------
# Feature shape & consistency
# ---------------------------------------------------------------------------

class TestFeatureShape:
    def test_feature_names_length(self):
        assert len(FEATURE_NAMES) == N_FEATURES

    def test_extract_returns_correct_shape(self):
        engine = FeatureEngine(normalize=False)
        vec = engine.extract(_make_opp(), _make_ctx())
        assert isinstance(vec, np.ndarray)
        assert vec.shape == (N_FEATURES,)
        assert vec.dtype == np.float64

    def test_extract_normalized_returns_correct_shape(self):
        engine = FeatureEngine()
        vec = engine.extract_normalized(_make_opp(), _make_ctx())
        assert vec.shape == (N_FEATURES,)

    def test_n_features_property(self):
        engine = FeatureEngine()
        assert engine.n_features == N_FEATURES

    def test_feature_names_property(self):
        engine = FeatureEngine()
        assert engine.feature_names == list(FEATURE_NAMES)


# ---------------------------------------------------------------------------
# One-hot encoding
# ---------------------------------------------------------------------------

class TestOneHot:
    def test_binary_rebalance_one_hot(self):
        engine = FeatureEngine(normalize=False)
        opp = _make_opp(type=OpportunityType.BINARY_REBALANCE)
        vec = engine.extract(opp, _make_ctx())

        # Find one-hot region
        start = list(FEATURE_NAMES).index("type_binary_rebalance")
        end = start + len(_OPP_TYPES)
        one_hot = vec[start:end]

        assert one_hot[0] == 1.0  # BINARY_REBALANCE is first
        assert np.sum(one_hot) == 1.0  # exactly one bit set

    def test_maker_rebalance_one_hot(self):
        engine = FeatureEngine(normalize=False)
        opp = _make_opp(type=OpportunityType.MAKER_REBALANCE)
        vec = engine.extract(opp, _make_ctx())

        maker_idx = list(FEATURE_NAMES).index("type_maker_rebalance")
        assert vec[maker_idx] == 1.0

        # is_maker derived feature should also be 1.0
        is_maker_idx = list(FEATURE_NAMES).index("is_maker")
        assert vec[is_maker_idx] == 1.0

    def test_all_types_produce_valid_one_hot(self):
        engine = FeatureEngine(normalize=False)
        start = list(FEATURE_NAMES).index(f"type_{_OPP_TYPES[0].value}")
        end = start + len(_OPP_TYPES)

        for t in _OPP_TYPES:
            opp = _make_opp(type=t)
            vec = engine.extract(opp, _make_ctx())
            one_hot = vec[start:end]
            assert np.sum(one_hot) == 1.0, f"One-hot sum != 1 for {t}"

    def test_non_maker_type_is_maker_zero(self):
        engine = FeatureEngine(normalize=False)
        opp = _make_opp(type=OpportunityType.BINARY_REBALANCE)
        vec = engine.extract(opp, _make_ctx())
        is_maker_idx = list(FEATURE_NAMES).index("is_maker")
        assert vec[is_maker_idx] == 0.0


# ---------------------------------------------------------------------------
# Opportunity scalar features
# ---------------------------------------------------------------------------

class TestScalarFeatures:
    def test_opportunity_scalars(self):
        engine = FeatureEngine(normalize=False)
        opp = _make_opp(net_profit=3.0, roi_pct=6.67, required_capital=45.0, max_sets=100)
        vec = engine.extract(opp, _make_ctx())

        names = list(FEATURE_NAMES)
        assert vec[names.index("net_profit")] == 3.0
        assert vec[names.index("roi_pct")] == 6.67
        assert vec[names.index("required_capital")] == 45.0
        assert vec[names.index("max_sets")] == 100
        assert vec[names.index("n_legs")] == 1  # one leg in default factory

    def test_multi_leg_count(self):
        engine = FeatureEngine(normalize=False)
        legs = (
            LegOrder(token_id="t1", side=Side.BUY, price=0.45, size=100),
            LegOrder(token_id="t2", side=Side.BUY, price=0.50, size=100),
        )
        opp = _make_opp(legs=legs)
        vec = engine.extract(opp, _make_ctx())
        assert vec[list(FEATURE_NAMES).index("n_legs")] == 2


# ---------------------------------------------------------------------------
# Context features
# ---------------------------------------------------------------------------

class TestContextFeatures:
    def test_market_features(self):
        engine = FeatureEngine(normalize=False)
        ctx = _make_ctx(market_volume=50000.0, recent_trade_count=42,
                        time_to_resolution_hours=168.0)
        vec = engine.extract(_make_opp(), ctx)
        names = list(FEATURE_NAMES)
        assert vec[names.index("volume")] == 50000.0
        assert vec[names.index("trade_count")] == 42
        assert vec[names.index("time_to_resolution_hours")] == 168.0

    def test_book_features(self):
        engine = FeatureEngine(normalize=False)
        ctx = _make_ctx(book_depth_ratio=2.5)
        vec = engine.extract(_make_opp(), ctx)
        assert vec[list(FEATURE_NAMES).index("depth_ratio")] == 2.5

    def test_confidence_features(self):
        engine = FeatureEngine(normalize=False)
        ctx = _make_ctx(confidence=0.8, realized_ev_score=0.7)
        vec = engine.extract(_make_opp(), ctx)
        names = list(FEATURE_NAMES)
        assert vec[names.index("confidence")] == 0.8
        assert vec[names.index("realized_ev_score")] == 0.7


# ---------------------------------------------------------------------------
# Derived features
# ---------------------------------------------------------------------------

class TestDerivedFeatures:
    def test_log_profit(self):
        engine = FeatureEngine(normalize=False)
        opp = _make_opp(net_profit=10.0)
        vec = engine.extract(opp, _make_ctx())
        expected = math.log10(10.0)
        assert vec[list(FEATURE_NAMES).index("log_profit")] == pytest.approx(expected)

    def test_log_profit_clamped_at_zero(self):
        """net_profit=0 → clamped to log10(0.01)."""
        engine = FeatureEngine(normalize=False)
        opp = _make_opp(net_profit=0.0)
        vec = engine.extract(opp, _make_ctx())
        expected = math.log10(0.01)
        assert vec[list(FEATURE_NAMES).index("log_profit")] == pytest.approx(expected)

    def test_log_profit_negative(self):
        """Negative net_profit → clamped to log10(0.01)."""
        engine = FeatureEngine(normalize=False)
        opp = _make_opp(net_profit=-5.0)
        vec = engine.extract(opp, _make_ctx())
        expected = math.log10(0.01)
        assert vec[list(FEATURE_NAMES).index("log_profit")] == pytest.approx(expected)

    def test_capital_efficiency(self):
        engine = FeatureEngine(normalize=False)
        ctx = _make_ctx(time_to_resolution_hours=24.0)
        opp = _make_opp(roi_pct=10.0)
        vec = engine.extract(opp, ctx)
        expected = 10.0 * (8760.0 / 24.0)
        assert vec[list(FEATURE_NAMES).index("capital_efficiency")] == pytest.approx(expected)

    def test_capital_efficiency_short_resolution(self):
        """Very short resolution → clamp hours to 0.25."""
        engine = FeatureEngine(normalize=False)
        ctx = _make_ctx(time_to_resolution_hours=0.0)
        opp = _make_opp(roi_pct=10.0)
        vec = engine.extract(opp, ctx)
        expected = 10.0 * (8760.0 / 0.25)
        assert vec[list(FEATURE_NAMES).index("capital_efficiency")] == pytest.approx(expected)

    def test_is_spike_true(self):
        engine = FeatureEngine(normalize=False)
        ctx = _make_ctx(is_spike=True)
        vec = engine.extract(_make_opp(), ctx)
        assert vec[list(FEATURE_NAMES).index("is_spike")] == 1.0

    def test_is_spike_false(self):
        engine = FeatureEngine(normalize=False)
        ctx = _make_ctx(is_spike=False)
        vec = engine.extract(_make_opp(), ctx)
        assert vec[list(FEATURE_NAMES).index("is_spike")] == 0.0


# ---------------------------------------------------------------------------
# Normalization
# ---------------------------------------------------------------------------

class TestNormalization:
    def test_first_sample_returns_raw(self):
        """With only 1 sample, normalize returns raw features."""
        engine = FeatureEngine(normalize=True)
        opp = _make_opp()
        ctx = _make_ctx()
        raw = engine.extract(opp, ctx)
        normalized = engine.extract_normalized(opp, ctx)
        # First call: count=1 after update, normalize returns raw
        np.testing.assert_array_equal(normalized, raw)

    def test_normalized_reasonable_magnitude(self):
        """After many samples, normalized features should be near zero mean."""
        engine = FeatureEngine(normalize=True)
        rng = np.random.default_rng(42)

        for _ in range(100):
            opp = _make_opp(
                net_profit=rng.uniform(0.5, 20.0),
                roi_pct=rng.uniform(1.0, 30.0),
                required_capital=rng.uniform(10.0, 500.0),
                max_sets=rng.uniform(10, 500),
            )
            ctx = _make_ctx(
                market_volume=rng.uniform(1000, 100000),
                recent_trade_count=int(rng.uniform(0, 100)),
                time_to_resolution_hours=rng.uniform(1, 2000),
                confidence=rng.uniform(0, 1),
                realized_ev_score=rng.uniform(0, 1),
            )
            vec = engine.extract_normalized(opp, ctx)
            assert vec.shape == (N_FEATURES,)

        # After 100 samples, magnitudes should be reasonable (< 10 std devs)
        final = engine.extract_normalized(_make_opp(), _make_ctx())
        assert np.all(np.abs(final) < 50), f"Outlier normalized feature: {final}"


# ---------------------------------------------------------------------------
# Rolling stats
# ---------------------------------------------------------------------------

class TestRollingStats:
    def test_initial_state(self):
        stats = RollingStats()
        assert stats.count == 0
        np.testing.assert_array_equal(stats.mean, np.zeros(N_FEATURES))
        np.testing.assert_array_equal(stats.var, np.ones(N_FEATURES))

    def test_first_update_sets_mean(self):
        stats = RollingStats()
        vec = np.ones(N_FEATURES) * 5.0
        stats.update(vec)
        assert stats.count == 1
        np.testing.assert_array_equal(stats.mean, vec)

    def test_mean_convergence(self):
        """After many identical samples, mean should converge to that value."""
        stats = RollingStats(decay=0.95)
        vec = np.ones(N_FEATURES) * 3.0
        for _ in range(200):
            stats.update(vec)
        np.testing.assert_allclose(stats.mean, vec, atol=0.1)

    def test_normalize_returns_near_zero_for_constant(self):
        """Constant input → normalized ≈ 0."""
        stats = RollingStats(decay=0.95)
        vec = np.ones(N_FEATURES) * 7.0
        for _ in range(50):
            stats.update(vec)
        result = stats.normalize(vec)
        # variance is small for constant → near zero after normalization
        np.testing.assert_allclose(result, np.zeros(N_FEATURES), atol=1.0)

    def test_normalize_skips_with_low_count(self):
        stats = RollingStats()
        vec = np.ones(N_FEATURES) * 3.0
        stats.update(vec)
        result = stats.normalize(vec)
        # count < 2 → returns raw
        np.testing.assert_array_equal(result, vec)


# ---------------------------------------------------------------------------
# Batch extraction
# ---------------------------------------------------------------------------

class TestBatch:
    def test_batch_shape(self):
        engine = FeatureEngine(normalize=False)
        opps = [_make_opp(net_profit=float(i)) for i in range(5)]
        ctxs = [_make_ctx() for _ in range(5)]
        matrix = engine.extract_batch(opps, ctxs)
        assert matrix.shape == (5, N_FEATURES)

    def test_batch_matches_individual(self):
        engine = FeatureEngine(normalize=False)
        opps = [_make_opp(net_profit=float(i + 1)) for i in range(3)]
        ctxs = [_make_ctx(market_volume=float(i * 100)) for i in range(3)]
        matrix = engine.extract_batch(opps, ctxs)

        for i in range(3):
            single = engine.extract(opps[i], ctxs[i])
            np.testing.assert_array_equal(matrix[i], single)

    def test_batch_normalized_shape(self):
        engine = FeatureEngine(normalize=True)
        opps = [_make_opp(net_profit=float(i + 1)) for i in range(5)]
        ctxs = [_make_ctx() for _ in range(5)]
        matrix = engine.extract_batch_normalized(opps, ctxs)
        assert matrix.shape == (5, N_FEATURES)

    def test_batch_mismatched_ctx_uses_defaults(self):
        """If fewer contexts than opps, defaults fill in."""
        engine = FeatureEngine(normalize=False)
        opps = [_make_opp() for _ in range(3)]
        ctxs = [_make_ctx()]  # only 1 context for 3 opps
        matrix = engine.extract_batch(opps, ctxs)
        assert matrix.shape == (3, N_FEATURES)

    def test_empty_batch(self):
        engine = FeatureEngine(normalize=False)
        matrix = engine.extract_batch([], [])
        assert matrix.shape == (0, N_FEATURES)


# ---------------------------------------------------------------------------
# Serialization
# ---------------------------------------------------------------------------

class TestSerialization:
    def test_round_trip(self):
        engine = FeatureEngine(normalize=True)
        # Feed some data to build stats
        for i in range(10):
            engine.extract_normalized(
                _make_opp(net_profit=float(i + 1)),
                _make_ctx(market_volume=float(i * 1000)),
            )

        data = engine.to_dict()
        restored = FeatureEngine.from_dict(data)

        assert restored._normalize == engine._normalize
        assert restored._stats.count == engine._stats.count
        np.testing.assert_array_almost_equal(restored._stats.mean, engine._stats.mean)
        np.testing.assert_array_almost_equal(restored._stats.var, engine._stats.var)

        # Verify identical extraction
        opp = _make_opp(net_profit=5.0)
        ctx = _make_ctx(confidence=0.9)
        vec_orig = engine.extract(opp, ctx)
        vec_restored = restored.extract(opp, ctx)
        np.testing.assert_array_equal(vec_orig, vec_restored)

    def test_rolling_stats_round_trip(self):
        stats = RollingStats(decay=0.95)
        vec = np.arange(N_FEATURES, dtype=np.float64)
        for _ in range(5):
            stats.update(vec)

        data = stats.to_dict()
        restored = RollingStats.from_dict(data)

        assert restored.count == stats.count
        assert restored.decay == stats.decay
        np.testing.assert_array_almost_equal(restored.mean, stats.mean)
        np.testing.assert_array_almost_equal(restored.var, stats.var)


# ---------------------------------------------------------------------------
# Stats property
# ---------------------------------------------------------------------------

class TestStatsProperty:
    def test_stats_dict(self):
        engine = FeatureEngine(normalize=True)
        s = engine.stats
        assert s["n_features"] == N_FEATURES
        assert s["samples_seen"] == 0
        assert s["normalize"] is True

    def test_stats_after_extraction(self):
        engine = FeatureEngine()
        engine.extract_normalized(_make_opp(), _make_ctx())
        assert engine.stats["samples_seen"] == 1
