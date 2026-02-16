from __future__ import annotations

from scanner.labels import resolve_opportunity_label
from scanner.models import Opportunity, OpportunityType, LegOrder, Side


def _make_opp(event_id: str, token_ids: tuple[str, ...]) -> Opportunity:
    legs = tuple(
        LegOrder(token_id=t, side=Side.BUY, price=0.5, size=10.0)
        for t in token_ids
    )
    return Opportunity(
        type=OpportunityType.MAKER_REBALANCE,
        event_id=event_id,
        legs=legs,
        expected_profit_per_set=0.01,
        net_profit_per_set=0.01,
        max_sets=10.0,
        gross_profit=0.1,
        estimated_gas_cost=0.01,
        net_profit=0.09,
        roi_pct=1.0,
        required_capital=9.0,
    )


def test_uses_market_question_when_all_legs_same_market() -> None:
    opp = _make_opp("evt_1", ("tok_yes", "tok_no"))
    label = resolve_opportunity_label(
        opp,
        event_questions={"evt_1": "Wrong event title"},
        market_questions={"tok_yes": "Actual market?", "tok_no": "Actual market?"},
    )
    assert label == "Actual market?"


def test_falls_back_to_event_title_for_multi_market_basket() -> None:
    opp = _make_opp("evt_2", ("tok_a", "tok_b"))
    label = resolve_opportunity_label(
        opp,
        event_questions={"evt_2": "Event title"},
        market_questions={"tok_a": "Market A?", "tok_b": "Market B?"},
    )
    assert label == "Event title"
