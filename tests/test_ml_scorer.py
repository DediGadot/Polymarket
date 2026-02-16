"""Tests for scanner/ml_scorer.py — ML opportunity scorer."""

from __future__ import annotations

import tempfile
from pathlib import Path

import numpy as np
import pytest

from scanner.feature_engine import FeatureEngine, N_FEATURES
from scanner.ml_scorer import MLScorer, MLScorerConfig, TrainingSample
from scanner.models import LegOrder, Opportunity, OpportunityType, Side
from scanner.scorer import ScoringContext


# ---------------------------------------------------------------------------
# Factories
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


def _build_scorer_with_data(n_positive: int = 60, n_negative: int = 60) -> MLScorer:
    """Build a scorer pre-loaded with synthetic labeled data."""
    config = MLScorerConfig(min_samples=50)
    scorer = MLScorer(config=config)
    rng = np.random.default_rng(42)

    for _ in range(n_positive):
        opp = _make_opp(
            net_profit=rng.uniform(1.0, 20.0),
            roi_pct=rng.uniform(3.0, 30.0),
            required_capital=rng.uniform(10, 200),
        )
        ctx = _make_ctx(
            market_volume=rng.uniform(5000, 100000),
            confidence=rng.uniform(0.6, 1.0),
            book_depth_ratio=rng.uniform(1.0, 5.0),
        )
        scorer.add_sample(opp, ctx, profitable=True)

    for _ in range(n_negative):
        opp = _make_opp(
            net_profit=rng.uniform(-2.0, 0.5),
            roi_pct=rng.uniform(-5.0, 1.0),
            required_capital=rng.uniform(10, 200),
        )
        ctx = _make_ctx(
            market_volume=rng.uniform(100, 5000),
            confidence=rng.uniform(0.0, 0.4),
            book_depth_ratio=rng.uniform(0.1, 0.8),
        )
        scorer.add_sample(opp, ctx, profitable=False)

    return scorer


# ---------------------------------------------------------------------------
# Fallback behavior
# ---------------------------------------------------------------------------

class TestFallback:
    def test_untrained_returns_default(self):
        scorer = MLScorer()
        assert scorer.predict(_make_opp(), _make_ctx()) == 0.5

    def test_untrained_batch_returns_defaults(self):
        scorer = MLScorer()
        opps = [_make_opp() for _ in range(3)]
        ctxs = [_make_ctx() for _ in range(3)]
        result = scorer.predict_batch(opps, ctxs)
        np.testing.assert_array_equal(result, [0.5, 0.5, 0.5])

    def test_insufficient_samples_returns_false(self):
        config = MLScorerConfig(min_samples=100)
        scorer = MLScorer(config=config)
        for i in range(50):
            scorer.add_sample(_make_opp(), _make_ctx(), profitable=True)
        assert scorer.train() is False
        assert not scorer.is_trained

    def test_single_class_returns_false(self):
        """Need both positive and negative labels."""
        config = MLScorerConfig(min_samples=10)
        scorer = MLScorer(config=config)
        for i in range(20):
            scorer.add_sample(_make_opp(), _make_ctx(), profitable=True)
        assert scorer.train() is False


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------

class TestTraining:
    def test_train_succeeds_with_enough_data(self):
        scorer = _build_scorer_with_data()
        assert scorer.train() is True
        assert scorer.is_trained

    def test_predictions_after_training(self):
        scorer = _build_scorer_with_data()
        scorer.train()

        # Good opportunity should score higher
        good_opp = _make_opp(net_profit=10.0, roi_pct=15.0)
        good_ctx = _make_ctx(confidence=0.9, book_depth_ratio=3.0, market_volume=50000)
        good_score = scorer.predict(good_opp, good_ctx)

        # Bad opportunity should score lower
        bad_opp = _make_opp(net_profit=-1.0, roi_pct=-2.0)
        bad_ctx = _make_ctx(confidence=0.1, book_depth_ratio=0.2, market_volume=500)
        bad_score = scorer.predict(bad_opp, bad_ctx)

        assert good_score > bad_score

    def test_predict_returns_probability(self):
        scorer = _build_scorer_with_data()
        scorer.train()
        score = scorer.predict(_make_opp(), _make_ctx())
        assert 0.0 <= score <= 1.0

    def test_batch_predictions(self):
        scorer = _build_scorer_with_data()
        scorer.train()
        opps = [_make_opp(net_profit=float(i)) for i in range(5)]
        ctxs = [_make_ctx() for _ in range(5)]
        result = scorer.predict_batch(opps, ctxs)
        assert result.shape == (5,)
        assert np.all((result >= 0.0) & (result <= 1.0))


# ---------------------------------------------------------------------------
# Retraining
# ---------------------------------------------------------------------------

class TestRetraining:
    def test_maybe_retrain_triggers_on_cycles(self):
        scorer = _build_scorer_with_data()
        config = MLScorerConfig(min_samples=50, retrain_every_cycles=5)
        scorer._config = config
        for _ in range(4):
            assert scorer.maybe_retrain() is False
        # 5th cycle triggers retrain
        assert scorer.maybe_retrain() is True
        assert scorer.is_trained

    def test_maybe_retrain_triggers_on_trades(self):
        scorer = _build_scorer_with_data(n_positive=60, n_negative=60)
        config = MLScorerConfig(min_samples=50, retrain_every_trades=10, retrain_every_cycles=9999)
        scorer._config = config
        # Add 10 more trades to trigger
        for i in range(10):
            scorer.add_sample(_make_opp(), _make_ctx(), profitable=i % 2 == 0)
        assert scorer.maybe_retrain() is True

    def test_background_retrain_runs(self):
        scorer = _build_scorer_with_data()
        scorer.retrain_background()
        # Wait for thread to finish
        if scorer._bg_thread is not None:
            scorer._bg_thread.join(timeout=10)
        assert scorer.is_trained


# ---------------------------------------------------------------------------
# Sample management
# ---------------------------------------------------------------------------

class TestSamples:
    def test_add_sample_increments_count(self):
        scorer = MLScorer()
        assert scorer.sample_count == 0
        scorer.add_sample(_make_opp(), _make_ctx(), profitable=True)
        assert scorer.sample_count == 1

    def test_add_sample_raw(self):
        scorer = MLScorer()
        features = np.ones(N_FEATURES)
        scorer.add_sample_raw(features, label=1)
        assert scorer.sample_count == 1
        # Verify it's a copy
        features[0] = 999.0
        assert scorer._samples[0].features[0] == 1.0


# ---------------------------------------------------------------------------
# Persistence
# ---------------------------------------------------------------------------

class TestPersistence:
    def test_save_and_load(self, tmp_path: Path):
        scorer = _build_scorer_with_data()
        scorer.train()
        model_path = tmp_path / "ml_model.joblib"
        scorer.save(model_path)
        assert model_path.exists()

        loaded = MLScorer.load(model_path)
        assert loaded.is_trained
        assert loaded.sample_count == scorer.sample_count

        # Predictions should match
        opp = _make_opp(net_profit=5.0)
        ctx = _make_ctx(confidence=0.7)
        orig_score = scorer.predict(opp, ctx)
        loaded_score = loaded.predict(opp, ctx)
        assert orig_score == pytest.approx(loaded_score, abs=1e-6)

    def test_save_untrained(self, tmp_path: Path):
        scorer = MLScorer()
        scorer.add_sample(_make_opp(), _make_ctx(), profitable=True)
        model_path = tmp_path / "untrained.joblib"
        scorer.save(model_path)

        loaded = MLScorer.load(model_path)
        assert not loaded.is_trained
        assert loaded.sample_count == 1
        assert loaded.predict(_make_opp(), _make_ctx()) == 0.5

    def test_save_creates_parent_dirs(self, tmp_path: Path):
        scorer = MLScorer()
        model_path = tmp_path / "sub" / "dir" / "model.joblib"
        scorer.save(model_path)
        assert model_path.exists()


# ---------------------------------------------------------------------------
# Stats
# ---------------------------------------------------------------------------

class TestStats:
    def test_stats_untrained(self):
        scorer = MLScorer()
        s = scorer.stats
        assert s["is_trained"] is False
        assert s["sample_count"] == 0

    def test_stats_after_training(self):
        scorer = _build_scorer_with_data()
        scorer.train()
        s = scorer.stats
        assert s["is_trained"] is True
        assert s["sample_count"] == 120
