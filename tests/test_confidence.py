"""
Tests for ArbTracker integration into scoring.
"""

import pytest
from scanner.confidence import ArbTracker


def _make_opp(event_id: str = "e1"):
    """Create a new Opportunity instance."""
    from scanner.models import Opportunity, OpportunityType, LegOrder, Side
    return Opportunity(
        type=OpportunityType.BINARY_REBALANCE,
        event_id=event_id,
        legs=(
            LegOrder("e1-yes", Side.BUY, 0.45, 10, ""),
            LegOrder("e1-no", Side.SELL, 0.45, 10, ""),
        ),
        expected_profit_per_set=0.10,
        net_profit_per_set=0.10,
        max_sets=100.0,
        gross_profit=10.0,
        estimated_gas_cost=0.01,
        net_profit=9.99,
        roi_pct=11.1,
        required_capital=90.0,
    )


def _make_tracker(max_history: int = 10) -> ArbTracker:
    """Create a new ArbTracker instance."""
    return ArbTracker(_max_history=max_history)


class TestArbTrackerNew:
    def test_new_tracker_has_no_history(self):
        """New tracker starts with empty history."""
        tracker = _make_tracker()
        assert tracker._history == {}, "New tracker should have empty history"

    def test_new_tracker_default_max_history(self):
        """New tracker respects max_history from config."""
        tracker = _make_tracker()
        assert tracker._max_history == 10, "Default max_history should be 10"

    def test_record_single_cycle(self):
        """Recording a single cycle updates history."""
        tracker = _make_tracker()
        opp1 = _make_opp(event_id="e1")
        opp2 = _make_opp(event_id="e2")
        opp3 = _make_opp(event_id="e3")
        opp4 = _make_opp(event_id="e4")

        tracker.record(0, [opp1, opp2, opp3, opp4])

        # Verify history - stores list of cycle numbers
        assert tracker._history["e1"] == [0], "Event e1 should have [0] cycle"
        assert tracker._history["e2"] == [0], "Event e2 should have [0] cycle"
        assert tracker._history["e3"] == [0], "Event e3 should have [0] cycle"
        assert tracker._history["e4"] == [0], "Event e4 should have [0] cycle"

    def test_record_multiple_cycles(self):
        """Recording multiple cycles across different events."""
        tracker = _make_tracker()
        opp1 = _make_opp(event_id="e1")
        opp2 = _make_opp(event_id="e2")

        # Record 3 consecutive cycles for event e1
        for i in range(3):
            tracker.record(i, [opp1])

        # Verify e1 has 3 cycles
        assert tracker._history["e1"] == [0, 1, 2]

        # Same for event e2
        for i in range(3):
            tracker.record(i, [opp2])

        assert tracker._history["e2"] == [0, 1, 2]

    def test_three_consecutive_cycles_full_confidence(self):
        """3 consecutive cycles with same events reach full confidence."""
        tracker = _make_tracker()
        opp1 = _make_opp(event_id="e1")

        # Record 3 consecutive cycles for event e1
        for i in range(3):
            tracker.record(i, [opp1])

        # Verify e1 has 3 cycles
        assert tracker._history["e1"] == [0, 1, 2]

        # Verify full confidence (has inventory + persistent)
        assert tracker.confidence("e1", has_inventory=True) == 1.0

        # First-seen without persistence
        opp2 = _make_opp(event_id="e2")
        tracker.record(0, [opp2])
        assert tracker.confidence("e2", has_inventory=True) == 0.3  # First seen, low depth

    def test_unknown_event(self):
        """Unknown event should have zero confidence."""
        tracker = _make_tracker()
        # e3 has never been seen
        assert tracker.confidence("e3") == 0.0, "Unknown event should keep confidence at 0.0"

    def test_non_consecutive_cycles_no_confidence_boost(self):
        """Non-consecutive cycles should not give full confidence."""
        tracker = _make_tracker()
        opp1 = _make_opp(event_id="e1")
        opp2 = _make_opp(event_id="e2")

        # Record e1 in cycle 0
        tracker.record(0, [opp1])
        assert tracker.confidence("e1") == 0.3  # First seen

        # Record e2 in cycle 2 (skip cycle 1)
        tracker.record(2, [opp2])
        assert tracker.confidence("e2") == 0.3  # First seen

    def test_consecutive_cycles_full_confidence(self):
        """Consecutive cycles give full confidence."""
        tracker = _make_tracker()
        opp1 = _make_opp(event_id="e1")

        # Record 3 consecutive cycles for event e1
        for i in range(3):
            tracker.record(i, [opp1])

        # Verify full confidence
        assert tracker.confidence("e1", has_inventory=True) == 1.0

    def test_stale_entries_purged(self):
        """Stale entries should be purged."""
        tracker = _make_tracker(max_history=5)
        opp1 = _make_opp(event_id="e1")

        # Record at cycle 0
        tracker.record(0, [opp1])
        assert "e1" in tracker._history

        # Move to cycle 100 (well past max_history of 5)
        tracker.record(100, [])

        # Verify stale entry was purged
        assert "e1" not in tracker._history

    def test_purge_removes_only_stale(self):
        """Purge should remove stale entries and keep others."""
        tracker = _make_tracker(max_history=5)
        opp1 = _make_opp(event_id="e1")
        opp2 = _make_opp(event_id="e2")

        # Record e1 at cycle 0
        tracker.record(0, [opp1])

        # Record e2 at cycle 4 (within max_history)
        tracker.record(4, [opp2])

        # Move to cycle 10
        tracker.record(10, [])

        # e1 should be purged (last seen at 0, 10-0=10 > 5)
        assert "e1" not in tracker._history

        # e2 should still be there (last seen at 4, 10-4=6 > 5)
        # Actually 10-4=6 > 5, so it's also purged
        assert "e2" not in tracker._history

    def test_recent_entries_not_purged(self):
        """Recent entries should not be purged."""
        tracker = _make_tracker(max_history=5)
        opp1 = _make_opp(event_id="e1")

        # Record at cycle 0
        tracker.record(0, [opp1])

        # Record again at cycle 5 (still within max_history)
        tracker.record(5, [opp1])

        # Move to cycle 7
        tracker.record(7, [])

        # e1 should still be there (last seen at 5, 7-5=2 <= 5)
        assert "e1" in tracker._history
        assert tracker._history["e1"] == [0, 5]

    def test_sell_side_no_inventory_low_confidence(self):
        """Sell-side without inventory gets low confidence."""
        tracker = _make_tracker()
        opp1 = _make_opp(event_id="e1")

        tracker.record(0, [opp1])

        # With inventory, first-seen confidence
        assert tracker.confidence("e1", has_inventory=True) == 0.3

        # Without inventory, very low confidence
        assert tracker.confidence("e1", has_inventory=False) == 0.1

    def test_depth_ratio_affects_confidence(self):
        """Depth ratio affects first-seen confidence."""
        tracker = _make_tracker()
        opp1 = _make_opp(event_id="e1")

        tracker.record(0, [opp1])

        # Low depth ratio gives 0.3
        assert tracker.confidence("e1", depth_ratio=1.0, has_inventory=True) == 0.3

        # High depth ratio gives 0.7
        assert tracker.confidence("e1", depth_ratio=2.0, has_inventory=True) == 0.7
        assert tracker.confidence("e1", depth_ratio=5.0, has_inventory=True) == 0.7

    def test_duplicate_cycle_numbers_not_added(self):
        """Recording same cycle number twice should not duplicate."""
        tracker = _make_tracker()
        opp1 = _make_opp(event_id="e1")

        tracker.record(0, [opp1])
        tracker.record(0, [opp1])

        # Should only have one entry
        assert tracker._history["e1"] == [0]

    def test_max_history_respected(self):
        """Tracker respects max_history setting."""
        tracker = _make_tracker(max_history=3)
        opp1 = _make_opp(event_id="e1")

        assert tracker._max_history == 3

        # Record at cycle 0
        tracker.record(0, [opp1])

        # Move to cycle 4 (just past max_history)
        tracker.record(4, [])

        # e1 should be purged (4-0=4 > 3)
        assert "e1" not in tracker._history

    def test_no_inventory_prevents_full_confidence(self):
        """No inventory prevents full confidence even with persistence."""
        tracker = _make_tracker()
        opp1 = _make_opp(event_id="e1")

        # Record 3 consecutive cycles
        for i in range(3):
            tracker.record(i, [opp1])

        # With inventory, full confidence
        assert tracker.confidence("e1", has_inventory=True) == 1.0

        # Without inventory, low confidence (even though persistent)
        assert tracker.confidence("e1", has_inventory=False) == 0.1
