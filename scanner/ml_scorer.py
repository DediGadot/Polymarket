"""
ML-based opportunity scorer. Wraps a GradientBoostingClassifier that predicts
whether an opportunity will result in a profitable trade.

Falls back to the hand-tuned scorer when training data is insufficient.
Supports background retraining and model persistence via joblib.
"""

from __future__ import annotations

import logging
import threading
from dataclasses import dataclass, field
from pathlib import Path

import joblib
import numpy as np
from sklearn.ensemble import GradientBoostingClassifier

from scanner.feature_engine import FeatureEngine, N_FEATURES
from scanner.models import Opportunity
from scanner.scorer import ScoringContext

logger = logging.getLogger(__name__)

MIN_TRAINING_SAMPLES = 100
DEFAULT_RETRAIN_INTERVAL_CYCLES = 50
DEFAULT_RETRAIN_INTERVAL_TRADES = 200


@dataclass
class TrainingSample:
    """A labeled feature vector for training."""

    features: np.ndarray  # shape (N_FEATURES,)
    label: int  # 1 = profitable, 0 = not profitable


@dataclass
class MLScorerConfig:
    """Configuration for the ML scorer."""

    min_samples: int = MIN_TRAINING_SAMPLES
    retrain_every_cycles: int = DEFAULT_RETRAIN_INTERVAL_CYCLES
    retrain_every_trades: int = DEFAULT_RETRAIN_INTERVAL_TRADES
    n_estimators: int = 100
    max_depth: int = 4
    learning_rate: float = 0.1
    model_path: str = ""


class MLScorer:
    """
    ML-augmented opportunity scorer.

    Trains a GradientBoostingClassifier on labeled trade outcomes.
    Falls back to a default score (0.5) when insufficient training data.
    """

    def __init__(
        self,
        feature_engine: FeatureEngine | None = None,
        config: MLScorerConfig | None = None,
    ) -> None:
        self._engine = feature_engine or FeatureEngine(normalize=False)
        self._config = config or MLScorerConfig()
        self._model: GradientBoostingClassifier | None = None
        self._samples: list[TrainingSample] = []
        self._cycles_since_train: int = 0
        self._trades_since_train: int = 0
        self._is_trained: bool = False
        self._training_lock = threading.Lock()
        self._bg_thread: threading.Thread | None = None

    @property
    def is_trained(self) -> bool:
        return self._is_trained

    @property
    def sample_count(self) -> int:
        return len(self._samples)

    def add_sample(self, opp: Opportunity, ctx: ScoringContext, profitable: bool) -> None:
        """Record a labeled training sample."""
        features = self._engine.extract(opp, ctx)
        sample = TrainingSample(features=features, label=int(profitable))
        self._samples.append(sample)
        self._trades_since_train += 1

    def add_sample_raw(self, features: np.ndarray, label: int) -> None:
        """Record a pre-extracted feature vector with label."""
        self._samples.append(TrainingSample(features=features.copy(), label=label))
        self._trades_since_train += 1

    def predict(self, opp: Opportunity, ctx: ScoringContext) -> float:
        """
        Predict probability that this opportunity results in a profitable trade.
        Returns 0.5 (fallback) if model is not trained.
        """
        if not self._is_trained or self._model is None:
            return 0.5
        features = self._engine.extract(opp, ctx).reshape(1, -1)
        try:
            proba = self._model.predict_proba(features)[0]
            # Index 1 = probability of class 1 (profitable)
            pos_idx = list(self._model.classes_).index(1)
            return float(proba[pos_idx])
        except Exception:
            logger.warning("ML predict failed, returning fallback", exc_info=True)
            return 0.5

    def predict_batch(self, opps: list[Opportunity], ctxs: list[ScoringContext]) -> np.ndarray:
        """Predict probabilities for a batch of opportunities."""
        if not self._is_trained or self._model is None:
            return np.full(len(opps), 0.5)
        matrix = self._engine.extract_batch(opps, ctxs)
        try:
            proba = self._model.predict_proba(matrix)
            pos_idx = list(self._model.classes_).index(1)
            return proba[:, pos_idx]
        except Exception:
            logger.warning("ML batch predict failed, returning fallback", exc_info=True)
            return np.full(len(opps), 0.5)

    def train(self) -> bool:
        """
        Train the model on accumulated samples.
        Returns True if training succeeded, False if insufficient data.
        """
        if len(self._samples) < self._config.min_samples:
            logger.info(
                "Insufficient samples for training: %d / %d",
                len(self._samples),
                self._config.min_samples,
            )
            return False

        X = np.array([s.features for s in self._samples])
        y = np.array([s.label for s in self._samples])

        # Need both classes present
        unique = np.unique(y)
        if len(unique) < 2:
            logger.info("Need both positive and negative labels to train (have %s)", unique)
            return False

        model = GradientBoostingClassifier(
            n_estimators=self._config.n_estimators,
            max_depth=self._config.max_depth,
            learning_rate=self._config.learning_rate,
            random_state=42,
        )
        model.fit(X, y)

        with self._training_lock:
            self._model = model
            self._is_trained = True
            self._cycles_since_train = 0
            self._trades_since_train = 0

        logger.info("ML scorer trained on %d samples", len(self._samples))
        return True

    def maybe_retrain(self) -> bool:
        """
        Check if retraining is due and trigger if so.
        Returns True if retraining was triggered.
        """
        self._cycles_since_train += 1
        should_retrain = (
            self._cycles_since_train >= self._config.retrain_every_cycles
            or self._trades_since_train >= self._config.retrain_every_trades
        )
        if not should_retrain:
            return False
        return self.train()

    def retrain_background(self) -> None:
        """Trigger retraining in a background thread (non-blocking)."""
        if self._bg_thread is not None and self._bg_thread.is_alive():
            return  # already training

        def _train() -> None:
            self.train()

        self._bg_thread = threading.Thread(target=_train, daemon=True, name="ml-retrain")
        self._bg_thread.start()

    def save(self, path: str | Path) -> None:
        """Save model and training data to disk."""
        path = Path(path)
        data = {
            "model": self._model,
            "samples_features": np.array([s.features for s in self._samples]) if self._samples else np.empty((0, N_FEATURES)),
            "samples_labels": np.array([s.label for s in self._samples]) if self._samples else np.empty(0, dtype=int),
            "is_trained": self._is_trained,
            "config": {
                "min_samples": self._config.min_samples,
                "retrain_every_cycles": self._config.retrain_every_cycles,
                "retrain_every_trades": self._config.retrain_every_trades,
                "n_estimators": self._config.n_estimators,
                "max_depth": self._config.max_depth,
                "learning_rate": self._config.learning_rate,
            },
            "engine": self._engine.to_dict(),
        }
        path.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(data, path)
        logger.info("ML scorer saved to %s (%d samples)", path, len(self._samples))

    @classmethod
    def load(cls, path: str | Path) -> MLScorer:
        """Load model and training data from disk."""
        path = Path(path)
        data = joblib.load(path)
        config = MLScorerConfig(**data.get("config", {}))
        engine = FeatureEngine.from_dict(data.get("engine", {}))
        scorer = cls(feature_engine=engine, config=config)
        scorer._model = data.get("model")
        scorer._is_trained = data.get("is_trained", False)

        features = data.get("samples_features", np.empty((0, N_FEATURES)))
        labels = data.get("samples_labels", np.empty(0, dtype=int))
        for i in range(len(labels)):
            scorer._samples.append(TrainingSample(features=features[i], label=int(labels[i])))

        logger.info("ML scorer loaded from %s (%d samples)", path, len(scorer._samples))
        return scorer

    @property
    def stats(self) -> dict:
        return {
            "is_trained": self._is_trained,
            "sample_count": len(self._samples),
            "cycles_since_train": self._cycles_since_train,
            "trades_since_train": self._trades_since_train,
            "min_samples": self._config.min_samples,
        }
