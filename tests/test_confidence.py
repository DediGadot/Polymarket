"""
Unit tests for scanner/confidence.py -- ArbTracker confidence model.
Tracks opportunity persistence across scan cycles to assign confidence scores.
"""

from scanner.confidence import ArbTracker
from scanner.scorer import (
    ScoringContext,
    score_opportunity,
    W_PROFIT,
    W_FILL,
    W_EFFICIENCY,
    W_URGENCY,
    W_COMPETITION,
    W_PERSISTENCE,
)
from scanner.models import (
    Opportunity,
    OpportunityType,
    LegOrder,
    Side,
)


def _make_opp(
    event_id: str = "e1",
    opp_type=OpportunityType.BINARY_REBALANCE,
    net_profit: float = 5.0,
    roi_pct: float = 10.0,
    max_sets: float = 100.0,
    side: Side = Side.BUY,
) -> Opportunity:
    return Opportunity(
        type=opp_type,
        event_id=event_id,
        legs=(LegOrder("y1", side, 0.45, max_sets),),
        expected_profit_per_set=net_profit / max_sets if max_sets > 0 else 0,
        net_profit_per_set=net_profit / max_sets if max_sets > 0 else 0,
        max_sets=max_sets,
        gross_profit=net_profit + 0.01,
        estimated_gas_cost=0.01,
        net_profit=net_profit,
        roi_pct=roi_pct,
        required_capital=net_profit / (roi_pct / 100) if roi_pct > 0 else 1.0,
    )


class TestArbTrackerNew:
    def test_new_tracker_has_no_history(self):
        tracker = ArbTracker()
        assert tracker._history == {}

    def test_new_tracker_default_max_history(self):
        tracker = ArbTracker()
        assert tracker._max_history == 10


class TestArbTrackerRecord:
    def test_record_single_cycle(self):
        tracker = ArbTracker()
        opp = _make_opp(event_id="e1")
        tracker.record(1, [opp])
        assert "e1" in tracker._history
        assert tracker._history["e1"] == [1]

    def test_record_multiple_cycles(self):
        tracker = ArbTracker()
        opp = _make_opp(event_id="e1")
        tracker.record(1, [opp])
        tracker.record(2, [opp])
        tracker.record(3, [opp])
        assert tracker._history["e1"] == [1, 2, 3]

    def test_record_multiple_events(self):
        tracker = ArbTracker()
        opp_a = _make_opp(event_id="a")
        opp_b = _make_opp(event_id="b")
        tracker.record(1, [opp_a, opp_b])
        assert "a" in tracker._history
        assert "b" in tracker._history

    def test_record_does_not_duplicate_cycle(self):
        tracker = ArbTracker()
        opp = _make_opp(event_id="e1")
        tracker.record(5, [opp])
        tracker.record(5, [opp])
        assert tracker._history["e1"] == [5]


class TestArbTrackerConfidence:
    def test_consecutive_cycles_full_confidence(self):
        """Event seen 2+ consecutive cycles = 1.0."""
        tracker = ArbTracker()
        opp = _make_opp(event_id="e1")
        tracker.record(1, [opp])
        tracker.record(2, [opp])
        assert tracker.confidence("e1") == 1.0

    def test_three_consecutive_cycles(self):
        tracker = ArbTracker()
        opp = _make_opp(event_id="e1")
        tracker.record(5, [opp])
        tracker.record(6, [opp])
        tracker.record(7, [opp])
        assert tracker.confidence("e1") == 1.0

    def test_first_seen_deep_book(self):
        """First-seen event with depth_ratio >= 2.0 = 0.7."""
        tracker = ArbTracker()
        opp = _make_opp(event_id="e1")
        tracker.record(1, [opp])
        assert tracker.confidence("e1", depth_ratio=2.0) == 0.7
        assert tracker.confidence("e1", depth_ratio=3.0) == 0.7

    def test_first_seen_thin_book(self):
        """First-seen event with depth_ratio < 2.0 = 0.3."""
        tracker = ArbTracker()
        opp = _make_opp(event_id="e1")
        tracker.record(1, [opp])
        assert tracker.confidence("e1", depth_ratio=1.0) == 0.3
        assert tracker.confidence("e1", depth_ratio=0.5) == 0.3

    def test_sell_side_without_inventory(self):
        """Sell-side without inventory = 0.1."""
        tracker = ArbTracker()
        opp = _make_opp(event_id="e1")
        tracker.record(1, [opp])
        assert tracker.confidence("e1", has_inventory=False) == 0.1

    def test_sell_no_inventory_overrides_depth(self):
        """No-inventory penalty takes precedence over depth."""
        tracker = ArbTracker()
        opp = _make_opp(event_id="e1")
        tracker.record(1, [opp])
        tracker.record(2, [opp])
        assert tracker.confidence("e1", has_inventory=False) == 0.1

    def test_unknown_event(self):
        """Event never recorded = 0.0."""
        tracker = ArbTracker()
        assert tracker.confidence("unknown") == 0.0

    def test_non_consecutive_cycles_no_persistence(self):
        """Non-consecutive cycles don't count as persistent."""
        tracker = ArbTracker()
        opp = _make_opp(event_id="e1")
        tracker.record(1, [opp])
        tracker.record(5, [opp])  # gap of 4
        # Only last sighting matters; not consecutive
        assert tracker.confidence("e1", depth_ratio=1.0) == 0.3


class TestArbTrackerPurge:
    def test_stale_entries_purged(self):
        """Entries >10 cycles old are purged."""
        tracker = ArbTracker()
        opp = _make_opp(event_id="old")
        tracker.record(1, [opp])
        # Record something at cycle 12 to trigger purge
        opp2 = _make_opp(event_id="new")
        tracker.record(12, [opp2])
        assert "old" not in tracker._history
        assert "new" in tracker._history

    def test_recent_entries_not_purged(self):
        tracker = ArbTracker()
        opp_a = _make_opp(event_id="a")
        opp_b = _make_opp(event_id="b")
        tracker.record(5, [opp_a])
        tracker.record(8, [opp_b])
        # Cycle 8: a's last cycle is 5, which is 3 ago (within 10)
        assert "a" in tracker._history
        assert "b" in tracker._history

    def test_purge_removes_only_stale(self):
        tracker = ArbTracker()
        opp_old = _make_opp(event_id="old")
        opp_recent = _make_opp(event_id="recent")
        tracker.record(1, [opp_old])
        tracker.record(10, [opp_recent])
        # Cycle 20: old's last=1 (19 ago), recent's last=10 (10 ago, boundary)
        opp_new = _make_opp(event_id="new")
        tracker.record(20, [opp_new])
        assert "old" not in tracker._history
        assert "new" in tracker._history


class TestScorerIntegration:
    def test_confidence_in_scoring_context(self):
        """ScoringContext accepts confidence field."""
        ctx = ScoringContext(confidence=0.8)
        assert ctx.confidence == 0.8

    def test_default_confidence(self):
        ctx = ScoringContext()
        assert ctx.confidence == 0.5

    def test_weights_sum_to_one(self):
        total = W_PROFIT + W_FILL + W_EFFICIENCY + W_URGENCY + W_COMPETITION + W_PERSISTENCE
        assert abs(total - 1.0) < 1e-9

    def test_persistence_affects_score(self):
        """Higher confidence should produce a higher total score."""
        opp = _make_opp()
        ctx_high = ScoringContext(confidence=1.0)
        ctx_low = ScoringContext(confidence=0.0)
        scored_high = score_opportunity(opp, ctx_high)
        scored_low = score_opportunity(opp, ctx_low)
        assert scored_high.total_score > scored_low.total_score

    def test_persistence_score_in_breakdown(self):
        """ScoredOpportunity should include persistence_score."""
        opp = _make_opp()
        ctx = ScoringContext(confidence=0.75)
        scored = score_opportunity(opp, ctx)
        assert scored.persistence_score == 0.75
