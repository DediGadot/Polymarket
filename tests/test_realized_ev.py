"""Tests for scanner.realized_ev.RealizedEVTracker."""

from __future__ import annotations

from scanner.models import LegOrder, Opportunity, OpportunityType, Side
from scanner.realized_ev import RealizedEVTracker


def _maker_opp(event_id: str = "evt1", net_profit: float = 5.0) -> Opportunity:
    return Opportunity(
        type=OpportunityType.MAKER_REBALANCE,
        event_id=event_id,
        legs=(
            LegOrder("y1", Side.BUY, 0.45, 50.0),
            LegOrder("n1", Side.BUY, 0.45, 50.0),
        ),
        expected_profit_per_set=0.10,
        net_profit_per_set=0.08,
        max_sets=50.0,
        gross_profit=5.0,
        estimated_gas_cost=0.01,
        net_profit=net_profit,
        roi_pct=10.0,
        required_capital=45.0,
    )


def test_observe_candidates_updates_history():
    tracker = RealizedEVTracker()
    opp = _maker_opp()
    tracker.observe_candidates([opp, opp])  # deduped by key per call
    assert tracker.estimate_realized_ev(opp) != 0.0


def test_full_fill_improves_score():
    tracker = RealizedEVTracker()
    opp = _maker_opp()
    tracker.observe_candidates([opp])
    base = tracker.score(opp)
    tracker.record_full_fill(opp, net_pnl=3.0)
    boosted = tracker.score(opp)
    assert boosted > base


def test_orphan_hedge_reduces_score():
    tracker = RealizedEVTracker()
    opp = _maker_opp()
    tracker.observe_candidates([opp])
    base = tracker.score(opp)
    tracker.record_orphan_hedge(opp, net_pnl=-2.0)
    reduced = tracker.score(opp)
    assert reduced < base

