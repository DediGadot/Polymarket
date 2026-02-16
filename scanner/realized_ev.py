"""
Realized-EV tracker for opportunity ranking.

Learns from observed candidate frequency and execution outcomes to estimate
whether headline quoted edge is likely to convert into realized profit.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

from scanner.models import Opportunity, OpportunityType


def _opp_key(opp: Opportunity) -> str:
    token_ids = ",".join(sorted(leg.token_id for leg in opp.legs))
    return f"{opp.type.value}:{opp.event_id}:{token_ids}"


@dataclass
class EVStats:
    observations: int = 0
    full_fills: int = 0
    orphan_hedges: int = 0
    realized_pnl: float = 0.0

    def to_dict(self) -> dict:
        """Serialize to a JSON-safe dict."""
        return {
            "observations": self.observations,
            "full_fills": self.full_fills,
            "orphan_hedges": self.orphan_hedges,
            "realized_pnl": self.realized_pnl,
        }

    @classmethod
    def from_dict(cls, data: dict) -> EVStats:
        """Restore from a serialized dict."""
        return cls(
            observations=data.get("observations", 0),
            full_fills=data.get("full_fills", 0),
            orphan_hedges=data.get("orphan_hedges", 0),
            realized_pnl=data.get("realized_pnl", 0.0),
        )


@dataclass
class RealizedEVTracker:
    """
    Bayesian realized-edge estimator.

    Prior assumptions are intentionally conservative for maker arbs:
    - paired fill probability starts modest
    - orphan-leg risk is non-zero
    """

    alpha_full: float = 2.0
    beta_full: float = 6.0
    alpha_orphan: float = 1.0
    beta_orphan: float = 19.0
    orphan_loss_ratio: float = 0.12
    _stats: dict[str, EVStats] = field(default_factory=dict)

    def observe_candidates(self, opportunities: list[Opportunity]) -> None:
        """Record that these opportunities were seen in a scan cycle."""
        seen_keys: set[str] = set()
        for opp in opportunities:
            if opp.type != OpportunityType.MAKER_REBALANCE:
                continue
            key = _opp_key(opp)
            if key in seen_keys:
                continue
            seen_keys.add(key)
            stats = self._stats.setdefault(key, EVStats())
            stats.observations += 1

    def record_full_fill(self, opportunity: Opportunity, net_pnl: float) -> None:
        """Record a successful paired fill."""
        key = _opp_key(opportunity)
        stats = self._stats.setdefault(key, EVStats())
        stats.full_fills += 1
        stats.realized_pnl += net_pnl

    def record_orphan_hedge(self, opportunity: Opportunity, net_pnl: float) -> None:
        """Record a one-leg fill that required hedging."""
        key = _opp_key(opportunity)
        stats = self._stats.setdefault(key, EVStats())
        stats.orphan_hedges += 1
        stats.realized_pnl += net_pnl

    def estimate_realized_ev(self, opportunity: Opportunity) -> float:
        """Estimated realized EV in USD for this opportunity."""
        if opportunity.type != OpportunityType.MAKER_REBALANCE:
            return opportunity.net_profit

        stats = self._stats.get(_opp_key(opportunity), EVStats())
        obs = max(stats.observations, 1)

        p_full = (stats.full_fills + self.alpha_full) / (obs + self.alpha_full + self.beta_full)
        p_orphan = (stats.orphan_hedges + self.alpha_orphan) / (
            obs + self.alpha_orphan + self.beta_orphan
        )
        orphan_loss = max(0.50, opportunity.required_capital * self.orphan_loss_ratio)

        return p_full * max(0.0, opportunity.net_profit) - p_orphan * orphan_loss

    def to_dict(self) -> dict:
        """Serialize tracker state to a JSON-safe dict."""
        return {
            "alpha_full": self.alpha_full,
            "beta_full": self.beta_full,
            "alpha_orphan": self.alpha_orphan,
            "beta_orphan": self.beta_orphan,
            "orphan_loss_ratio": self.orphan_loss_ratio,
            "stats": {k: v.to_dict() for k, v in self._stats.items()},
        }

    @classmethod
    def from_dict(cls, data: dict) -> RealizedEVTracker:
        """Restore tracker from a serialized dict."""
        tracker = cls(
            alpha_full=data.get("alpha_full", 2.0),
            beta_full=data.get("beta_full", 6.0),
            alpha_orphan=data.get("alpha_orphan", 1.0),
            beta_orphan=data.get("beta_orphan", 19.0),
            orphan_loss_ratio=data.get("orphan_loss_ratio", 0.12),
        )
        tracker._stats = {k: EVStats.from_dict(v) for k, v in data.get("stats", {}).items()}
        return tracker

    def score(self, opportunity: Opportunity) -> float:
        """
        Convert estimated realized EV to a bounded 0-1 signal for rank scoring.
        0.5 is neutral, >0.5 is favorable realized EV.
        """
        ev = self.estimate_realized_ev(opportunity)
        scale = max(1.0, abs(opportunity.net_profit))
        # Logistic mapping keeps large outliers bounded and comparable.
        return 1.0 / (1.0 + math.exp(-ev / scale))

