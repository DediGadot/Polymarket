"""
Position sizing using Kelly Criterion with hard caps.
"""

from __future__ import annotations

import logging

from scanner.models import Opportunity, OpportunityType

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
    f = edge / odds
    # Half-Kelly for safety
    return max(0.0, min(f * 0.5, 1.0))


def compute_position_size(
    opportunity: Opportunity,
    bankroll: float,
    max_exposure_per_trade: float,
    max_total_exposure: float,
    current_exposure: float,
    kelly_odds_confirmed: float = 0.65,
    kelly_odds_cross_platform: float = 0.40,
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

    net_profit_per_set = opportunity.net_profit / opportunity.max_sets if opportunity.max_sets > 0 else 0
    if net_profit_per_set <= 0:
        return 0.0

    edge = net_profit_per_set / cost_per_set

    # Select odds based on arb type
    if opportunity.type == OpportunityType.CROSS_PLATFORM_ARB:
        odds = kelly_odds_cross_platform  # Higher execution risk
    else:
        odds = kelly_odds_confirmed  # Confirmed arb (10:1 implied)

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

    # Fixed gas can dominate small Kelly sizes; skip if sized trade is net negative.
    realized_net = net_profit_per_set * sets - opportunity.estimated_gas_cost
    if realized_net <= 0:
        logger.info(
            "Skipping size %.2f: fixed gas overwhelms edge (net=$%.4f)",
            sets, realized_net,
        )
        return 0.0

    logger.info(
        "Sizing: edge=%.4f kelly_f=%.4f capital=$%.2f sets=%.1f",
        edge, kelly_f, capital_to_deploy, sets,
    )
    return sets
