"""
Helpers for resolving user-facing labels for opportunities.

For single-market opportunities (e.g. binary/maker), use the exact market
question from leg token IDs. For multi-market baskets (e.g. negRisk), fall
back to the event title when available.
"""

from __future__ import annotations

from scanner.models import Opportunity


def resolve_opportunity_label(
    opp: Opportunity,
    event_questions: dict[str, str] | None = None,
    market_questions: dict[str, str] | None = None,
) -> str:
    """
    Return the best human-readable label for an opportunity.

    Priority:
    1) If all leg tokens map to the same market question, return that question.
    2) Otherwise return event title by event_id if available.
    3) Otherwise return the first known leg question.
    4) Fallback to truncated event_id.
    """
    event_questions = event_questions or {}
    market_questions = market_questions or {}

    leg_questions = [
        market_questions.get(leg.token_id, "")
        for leg in opp.legs
        if market_questions.get(leg.token_id)
    ]

    if leg_questions:
        unique = {q for q in leg_questions}
        if len(unique) == 1:
            return leg_questions[0]

    event_label = event_questions.get(opp.event_id)
    if event_label:
        return event_label

    if leg_questions:
        return leg_questions[0]

    return opp.event_id[:14]
