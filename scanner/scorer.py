"""
Composite opportunity scoring. Replaces simple ROI sort with a weighted
multi-factor score that considers fill probability, capital efficiency,
urgency, and competition.

Not all arbs are created equal:
- 50% ROI on $10 depth < 5% ROI on $10,000 depth
- Spike arbs need instant execution (time-sensitive)
- Latency arbs have short windows (~200ms) vs steady-state (minutes)
- Markets with heavy trade activity = more bots competing
"""

from __future__ import annotations

import logging
import math
from dataclasses import dataclass

from scanner.models import Opportunity, OpportunityType

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class ScoringContext:
    """Extra context for scoring an opportunity beyond what's in Opportunity itself."""
    market_volume: float = 0.0
    recent_trade_count: int = 0
    time_to_resolution_hours: float = 720.0  # default 30 days
    is_spike: bool = False
    book_depth_ratio: float = 1.0  # available_depth / requested_size


@dataclass(frozen=True)
class ScoredOpportunity:
    """An opportunity with its composite score and breakdown."""
    opportunity: Opportunity
    total_score: float
    profit_score: float
    fill_score: float
    efficiency_score: float
    urgency_score: float
    competition_score: float


# Scoring weights
W_PROFIT = 0.25
W_FILL = 0.25
W_EFFICIENCY = 0.20
W_URGENCY = 0.20
W_COMPETITION = 0.10


def score_opportunity(
    opp: Opportunity,
    ctx: ScoringContext,
) -> ScoredOpportunity:
    """
    Score an opportunity using a weighted composite of 5 factors.
    Returns ScoredOpportunity with total_score and per-factor breakdown.
    Higher score = better opportunity.
    """
    profit_score = _score_profit(opp)
    fill_score = _score_fill(opp, ctx)
    efficiency_score = _score_efficiency(opp, ctx)
    urgency_score = _score_urgency(opp, ctx)
    competition_score = _score_competition(ctx)

    total = (
        W_PROFIT * profit_score
        + W_FILL * fill_score
        + W_EFFICIENCY * efficiency_score
        + W_URGENCY * urgency_score
        + W_COMPETITION * competition_score
    )

    return ScoredOpportunity(
        opportunity=opp,
        total_score=total,
        profit_score=profit_score,
        fill_score=fill_score,
        efficiency_score=efficiency_score,
        urgency_score=urgency_score,
        competition_score=competition_score,
    )


def _score_profit(opp: Opportunity) -> float:
    """
    Expected dollar profit, log-scaled to 0-1.
    log($0.50) ≈ 0.30, log($5) ≈ 0.70, log($50) ≈ 1.0
    """
    if opp.net_profit <= 0:
        return 0.0
    # Log scale: $0.10 → 0.0, $100 → 1.0
    raw = math.log10(max(opp.net_profit, 0.10)) + 1.0  # shift so log(0.1)=0
    return max(0.0, min(1.0, raw / 3.0))  # normalize: log(1000)=3


def _score_fill(opp: Opportunity, ctx: ScoringContext) -> float:
    """
    Fill probability based on depth ratio.
    depth_ratio >= 2.0 → score 1.0
    depth_ratio = 1.0 → score 0.70
    depth_ratio = 0.5 → score 0.35
    """
    ratio = ctx.book_depth_ratio
    if ratio <= 0:
        return 0.0
    return min(1.0, ratio * 0.50)


def _score_efficiency(opp: Opportunity, ctx: ScoringContext) -> float:
    """
    Capital efficiency: ROI adjusted for time-to-resolution.
    High ROI + short resolution = excellent.
    """
    if opp.roi_pct <= 0:
        return 0.0

    # Annualize the ROI
    hours = max(ctx.time_to_resolution_hours, 0.25)  # minimum 15 minutes
    annual_factor = 8760.0 / hours  # hours in a year
    annualized_roi = opp.roi_pct * annual_factor

    # Cap at 1.0 for 10000% annualized ROI
    return min(1.0, annualized_roi / 10000.0)


def _score_urgency(opp: Opportunity, ctx: ScoringContext) -> float:
    """
    Time-sensitivity of the opportunity.
    Spike arbs: 1.0 (must execute immediately)
    Latency arbs: 0.85 (short window)
    Steady-state (binary/negrisk): 0.50
    """
    if ctx.is_spike or opp.type == OpportunityType.SPIKE_LAG:
        return 1.0
    if opp.type == OpportunityType.LATENCY_ARB:
        return 0.85
    return 0.50


def _score_competition(ctx: ScoringContext) -> float:
    """
    Inverse competition: fewer recent trades = less competition = higher score.
    0 trades → 1.0
    10 trades → 0.50
    50+ trades → 0.10
    """
    if ctx.recent_trade_count <= 0:
        return 1.0
    # Exponential decay
    return max(0.10, math.exp(-ctx.recent_trade_count / 20.0))


def rank_opportunities(
    opps: list[Opportunity],
    contexts: list[ScoringContext] | None = None,
) -> list[ScoredOpportunity]:
    """
    Score and rank a list of opportunities. Returns sorted by total_score descending.
    If contexts is None, uses default ScoringContext for all.
    """
    if not opps:
        return []

    if contexts is None:
        contexts = [ScoringContext() for _ in opps]

    scored = [score_opportunity(opp, ctx) for opp, ctx in zip(opps, contexts)]
    scored.sort(key=lambda s: s.total_score, reverse=True)
    return scored
