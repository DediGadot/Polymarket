"""Tests for monitor.scan_tracker.ScanTracker."""

from __future__ import annotations

import time

from scanner.models import (
    Opportunity,
    OpportunityType,
    LegOrder,
    Side,
)
from monitor.scan_tracker import ScanTracker


def _make_opp(
    opp_type: OpportunityType = OpportunityType.BINARY_REBALANCE,
    event_id: str = "evt_aaa",
    net_profit: float = 5.0,
    roi_pct: float = 2.5,
) -> Opportunity:
    leg = LegOrder(token_id="tok_1", side=Side.BUY, price=0.48, size=100.0)
    return Opportunity(
        type=opp_type,
        event_id=event_id,
        legs=(leg,),
        expected_profit_per_set=0.04,
        max_sets=100.0,
        gross_profit=net_profit + 0.5,
        estimated_gas_cost=0.5,
        net_profit=net_profit,
        roi_pct=roi_pct,
        required_capital=net_profit / (roi_pct / 100.0),
    )


class TestEmptyTracker:
    def test_summary_all_zeros(self):
        t = ScanTracker()
        s = t.summary()
        assert s["total_cycles"] == 0
        assert s["markets_scanned"] == 0
        assert s["opportunities_found"] == 0
        assert s["unique_events"] == 0
        assert s["by_type"] == {}
        assert s["best_roi_pct"] == 0.0
        assert s["best_profit_usd"] == 0.0
        assert s["total_theoretical_profit_usd"] == 0.0
        assert s["avg_roi_pct"] == 0.0
        assert s["avg_profit_usd"] == 0.0

    def test_duration_increases(self):
        t = ScanTracker(_session_start=time.time() - 10.0)
        s = t.summary()
        assert s["duration_sec"] >= 10.0


class TestAccumulation:
    def test_single_cycle(self):
        t = ScanTracker()
        opps = [_make_opp(net_profit=5.0, roi_pct=2.5)]
        t.record_cycle(1, 100, opps)

        s = t.summary()
        assert s["total_cycles"] == 1
        assert s["markets_scanned"] == 100
        assert s["opportunities_found"] == 1
        assert s["unique_events"] == 1

    def test_multiple_cycles_accumulate(self):
        t = ScanTracker()
        t.record_cycle(1, 100, [_make_opp(event_id="evt_a")])
        t.record_cycle(2, 150, [_make_opp(event_id="evt_b"), _make_opp(event_id="evt_c")])
        t.record_cycle(3, 200, [])

        s = t.summary()
        assert s["total_cycles"] == 3
        assert s["markets_scanned"] == 450
        assert s["opportunities_found"] == 3
        assert s["unique_events"] == 3


class TestByType:
    def test_type_breakdown(self):
        t = ScanTracker()
        opps = [
            _make_opp(opp_type=OpportunityType.BINARY_REBALANCE),
            _make_opp(opp_type=OpportunityType.BINARY_REBALANCE),
            _make_opp(opp_type=OpportunityType.NEGRISK_REBALANCE),
        ]
        t.record_cycle(1, 50, opps)

        s = t.summary()
        assert s["by_type"] == {
            "binary_rebalance": 2,
            "negrisk_rebalance": 1,
        }


class TestBestAvgCalculations:
    def test_best_values(self):
        t = ScanTracker()
        opps = [
            _make_opp(net_profit=3.0, roi_pct=1.5),
            _make_opp(net_profit=12.5, roi_pct=4.82),
            _make_opp(net_profit=7.0, roi_pct=3.0),
        ]
        t.record_cycle(1, 200, opps)

        s = t.summary()
        assert s["best_roi_pct"] == 4.82
        assert s["best_profit_usd"] == 12.5

    def test_avg_values(self):
        t = ScanTracker()
        opps = [
            _make_opp(net_profit=3.0, roi_pct=1.5),
            _make_opp(net_profit=12.0, roi_pct=4.5),
        ]
        t.record_cycle(1, 100, opps)

        s = t.summary()
        assert s["avg_roi_pct"] == 3.0
        assert s["avg_profit_usd"] == 7.5
        assert s["total_theoretical_profit_usd"] == 15.0


class TestUniqueEventDedup:
    def test_same_event_across_cycles(self):
        t = ScanTracker()
        t.record_cycle(1, 100, [_make_opp(event_id="evt_dup")])
        t.record_cycle(2, 100, [_make_opp(event_id="evt_dup")])
        t.record_cycle(3, 100, [_make_opp(event_id="evt_new")])

        s = t.summary()
        assert s["opportunities_found"] == 3
        assert s["unique_events"] == 2

    def test_same_event_within_cycle(self):
        t = ScanTracker()
        opps = [
            _make_opp(event_id="evt_x"),
            _make_opp(event_id="evt_x"),
            _make_opp(event_id="evt_y"),
        ]
        t.record_cycle(1, 50, opps)

        s = t.summary()
        assert s["opportunities_found"] == 3
        assert s["unique_events"] == 2
