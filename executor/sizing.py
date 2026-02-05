"""
Position sizing using Kelly Criterion with hard caps.
"""

from __future__ import annotations

import logging

from scanner.models import Opportunity

logger = logging.getLogger(__name__)


def kelly_fraction(edge: float, odds: float) -> float:
    """
    Kelly criterion: f* = (b*p - q) / b
    where b = odds (profit/risk), p = probability of success, q = 1-p.

    For arbitrage, p is high (probability of fill) and edge is the guaranteed profit.
    We use a fractional Kelly (half-Kelly) to be conservative.
    """
    if odds <= 0:
        return 0.0
    # For arb: edge = profit/cost, p = estimated fill probability
    # Simplified: f = edge / odds (since p ~= 1 for confirmed arb)
    f = edge / odds
    # Half-Kelly for safety
    return max(0.0, min(f * 0.5, 1.0))


def compute_position_size(
    opportunity: Opportunity,
    bankroll: float,
    max_exposure_per_trade: float,
    max_total_exposure: float,
    current_exposure: float,
) -> float:
    """
    Compute the number of sets to trade for a given opportunity.
    Returns 0 if the opportunity should be skipped.
    """
    available_capital = min(
        bankroll,
        max_exposure_per_trade,
        max_total_exposure - current_exposure,
    )
    if available_capital <= 0:
        logger.warning("No available capital: bankroll=%.2f exposure=%.2f", bankroll, current_exposure)
        return 0.0

    # Kelly sizing
    cost_per_set = opportunity.required_capital / opportunity.max_sets if opportunity.max_sets > 0 else 0
    if cost_per_set <= 0:
        return 0.0

    edge = opportunity.expected_profit_per_set / cost_per_set
    odds = 1.0  # risk = cost_per_set, potential payout = cost_per_set + profit_per_set
    kelly_f = kelly_fraction(edge, odds)

    kelly_capital = kelly_f * bankroll
    capital_to_deploy = min(kelly_capital, available_capital)

    # Convert capital to number of sets
    sets = capital_to_deploy / cost_per_set

    # Never exceed what the orderbook can support
    sets = min(sets, opportunity.max_sets)

    # Must be at least 1 set
    if sets < 1.0:
        return 0.0

    logger.info(
        "Sizing: edge=%.4f kelly_f=%.4f capital=$%.2f sets=%.1f",
        edge, kelly_f, capital_to_deploy, sets,
    )
    return sets
