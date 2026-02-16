"""
Feature extraction engine for ML scoring. Converts Opportunity + ScoringContext
into fixed-width numpy arrays for classifier training and inference.

Features are normalized via rolling z-score statistics updated per cycle.
"""

from __future__ import annotations

import logging
import math
from dataclasses import dataclass, field

import numpy as np

from scanner.models import Opportunity, OpportunityType
from scanner.scorer import ScoringContext

logger = logging.getLogger(__name__)

# OpportunityType one-hot encoding order
_OPP_TYPES = [
    OpportunityType.BINARY_REBALANCE,
    OpportunityType.NEGRISK_REBALANCE,
    OpportunityType.NEGRISK_VALUE,
    OpportunityType.LATENCY_ARB,
    OpportunityType.SPIKE_LAG,
    OpportunityType.CROSS_PLATFORM_ARB,
    OpportunityType.RESOLUTION_SNIPE,
    OpportunityType.STALE_QUOTE_ARB,
    OpportunityType.MAKER_REBALANCE,
    OpportunityType.CORRELATION_ARB,
]

# Feature names for interpretability
FEATURE_NAMES: tuple[str, ...] = (
    # Opportunity features (5 scalar + len(_OPP_TYPES) one-hot)
    "net_profit",
    "roi_pct",
    "required_capital",
    "n_legs",
    "max_sets",
    *(f"type_{t.value}" for t in _OPP_TYPES),
    # Market features
    "volume",
    "trade_count",
    "time_to_resolution_hours",
    "spread_width",
    # Book features
    "depth_ratio",
    "bid_ask_imbalance",
    # Confidence features
    "confidence",
    "realized_ev_score",
    # Derived features
    "log_profit",
    "capital_efficiency",
    "is_spike",
    "is_maker",
)

N_FEATURES: int = len(FEATURE_NAMES)


@dataclass
class RollingStats:
    """Rolling mean/variance for z-score normalization (exponential moving average)."""

    mean: np.ndarray = field(default_factory=lambda: np.zeros(N_FEATURES))
    var: np.ndarray = field(default_factory=lambda: np.ones(N_FEATURES))
    count: int = 0
    decay: float = 0.99

    def update(self, features: np.ndarray) -> None:
        """Update running statistics with a new feature vector."""
        self.count += 1
        if self.count == 1:
            self.mean = features.copy()
            self.var = np.ones(N_FEATURES)
        else:
            delta = features - self.mean
            self.mean = self.decay * self.mean + (1 - self.decay) * features
            self.var = self.decay * self.var + (1 - self.decay) * (delta**2)

    def normalize(self, features: np.ndarray) -> np.ndarray:
        """Z-score normalize using rolling statistics."""
        if self.count < 2:
            return features
        std = np.sqrt(np.maximum(self.var, 1e-8))
        return (features - self.mean) / std

    def to_dict(self) -> dict:
        return {
            "mean": self.mean.tolist(),
            "var": self.var.tolist(),
            "count": self.count,
            "decay": self.decay,
        }

    @classmethod
    def from_dict(cls, data: dict) -> RollingStats:
        stats = cls(decay=data.get("decay", 0.99))
        stats.mean = np.array(data.get("mean", np.zeros(N_FEATURES)))
        stats.var = np.array(data.get("var", np.ones(N_FEATURES)))
        stats.count = data.get("count", 0)
        return stats


class FeatureEngine:
    """
    Extracts fixed-width feature vectors from opportunities.

    Usage:
        engine = FeatureEngine()
        features = engine.extract(opportunity, context)  # raw features
        normalized = engine.extract_normalized(opportunity, context)  # z-scored
    """

    def __init__(self, normalize: bool = True) -> None:
        self._normalize = normalize
        self._stats = RollingStats()

    def extract(self, opp: Opportunity, ctx: ScoringContext) -> np.ndarray:
        """Extract raw feature vector from opportunity + context."""
        features = np.zeros(N_FEATURES, dtype=np.float64)
        idx = 0

        # --- Opportunity features ---
        features[idx] = opp.net_profit
        idx += 1
        features[idx] = opp.roi_pct
        idx += 1
        features[idx] = opp.required_capital
        idx += 1
        features[idx] = len(opp.legs)
        idx += 1
        features[idx] = opp.max_sets
        idx += 1

        # One-hot for opportunity type
        for t in _OPP_TYPES:
            features[idx] = 1.0 if opp.type == t else 0.0
            idx += 1

        # --- Market features ---
        features[idx] = ctx.market_volume
        idx += 1
        features[idx] = ctx.recent_trade_count
        idx += 1
        features[idx] = ctx.time_to_resolution_hours
        idx += 1
        features[idx] = 0.0  # spread_width (placeholder, populated by caller if available)
        idx += 1

        # --- Book features ---
        features[idx] = ctx.book_depth_ratio
        idx += 1
        features[idx] = 0.0  # bid_ask_imbalance (placeholder)
        idx += 1

        # --- Confidence features ---
        features[idx] = ctx.confidence
        idx += 1
        features[idx] = ctx.realized_ev_score
        idx += 1

        # --- Derived features ---
        features[idx] = math.log10(max(opp.net_profit, 0.01))  # log_profit
        idx += 1
        hours = max(ctx.time_to_resolution_hours, 0.25)
        features[idx] = opp.roi_pct * (8760.0 / hours)  # capital_efficiency (annualized)
        idx += 1
        features[idx] = 1.0 if ctx.is_spike else 0.0  # is_spike
        idx += 1
        features[idx] = 1.0 if opp.type == OpportunityType.MAKER_REBALANCE else 0.0
        idx += 1

        return features

    def extract_normalized(self, opp: Opportunity, ctx: ScoringContext) -> np.ndarray:
        """Extract and z-score normalize feature vector."""
        raw = self.extract(opp, ctx)
        self._stats.update(raw)
        if self._normalize:
            return self._stats.normalize(raw)
        return raw

    def extract_batch(
        self, opps: list[Opportunity], ctxs: list[ScoringContext]
    ) -> np.ndarray:
        """Extract feature matrix for a batch of opportunities."""
        n = len(opps)
        matrix = np.zeros((n, N_FEATURES), dtype=np.float64)
        for i in range(n):
            ctx = ctxs[i] if i < len(ctxs) else ScoringContext()
            matrix[i] = self.extract(opps[i], ctx)
        return matrix

    def extract_batch_normalized(
        self, opps: list[Opportunity], ctxs: list[ScoringContext]
    ) -> np.ndarray:
        """Extract and normalize feature matrix for a batch."""
        raw = self.extract_batch(opps, ctxs)
        for i in range(len(opps)):
            self._stats.update(raw[i])
        if self._normalize and self._stats.count >= 2:
            std = np.sqrt(np.maximum(self._stats.var, 1e-8))
            raw = (raw - self._stats.mean) / std
        return raw

    @property
    def feature_names(self) -> list[str]:
        return list(FEATURE_NAMES)

    @property
    def n_features(self) -> int:
        return N_FEATURES

    @property
    def stats(self) -> dict:
        return {
            "n_features": N_FEATURES,
            "samples_seen": self._stats.count,
            "normalize": self._normalize,
        }

    def to_dict(self) -> dict:
        return {
            "normalize": self._normalize,
            "stats": self._stats.to_dict(),
        }

    @classmethod
    def from_dict(cls, data: dict) -> FeatureEngine:
        engine = cls(normalize=data.get("normalize", True))
        engine._stats = RollingStats.from_dict(data.get("stats", {}))
        return engine
