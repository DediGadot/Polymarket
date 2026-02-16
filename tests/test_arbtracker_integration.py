"""
Integration tests for ArbTracker connected to the scoring pipeline.
Tests that ArbTracker confidence flows through the full pipeline.
"""

import pytest
from scanner.confidence import ArbTracker
from scanner.models import Opportunity, OpportunityType, LegOrder, Side
from scanner.scorer import ScoringContext, score_opportunity, rank_opportunities


def _make_opp(event_id: str = "evt-123", net_profit: float = 5.0) -> Opportunity:
    """Create a new Opportunity instance."""
    return Opportunity(
        type=OpportunityType.BINARY_REBALANCE,
        event_id=event_id,
        legs=(
            LegOrder(f"{event_id}-yes", Side.BUY, 0.45, 10, ""),
            LegOrder(f"{event_id}-no", Side.SELL, 0.45, 10, ""),
        ),
        expected_profit_per_set=net_profit / 10.0,
        net_profit_per_set=net_profit / 10.0,
        max_sets=10.0,
        gross_profit=net_profit + 0.01,
        estimated_gas_cost=0.01,
        net_profit=net_profit,
        roi_pct=11.1,
        required_capital=90.0,
    )


class TestArbTrackerIntegration:
    def test_first_seen_opportunity_gets_low_confidence(self):
        """First-seen opportunity should have 0.1 confidence (thin book, has inventory)."""
        tracker = ArbTracker()
        opp1 = _make_opp(event_id="evt-123")

        # Record first sighting
        tracker.record(cycle_num=1, opportunities=[opp1])

        # Confidence should be 0.1 for first-seen with thin depth
        conf = tracker.confidence("evt-123", depth_ratio=1.0, has_inventory=True)
        assert conf == 0.1, f"First-seen should have 0.1 confidence, got {conf}"

    def test_persistent_opportunity_gets_full_confidence(self):
        """Opportunity seen in consecutive cycles should have 1.0 confidence."""
        tracker = ArbTracker()
        opp1 = _make_opp(event_id="evt-123")

        # Record consecutive cycles
        tracker.record(cycle_num=1, opportunities=[opp1])
        tracker.record(cycle_num=2, opportunities=[opp1])

        # Confidence should be 1.0 for persistent
        conf = tracker.confidence("evt-123", depth_ratio=1.0, has_inventory=True)
        assert conf == 1.0, f"Persistent should have 1.0 confidence, got {conf}"

    def test_deep_book_first_seen_gets_moderate_confidence(self):
        """First-seen with deep book (ratio >= 2.0) should have 0.3 confidence."""
        tracker = ArbTracker()
        opp1 = _make_opp(event_id="evt-123")

        # Record first sighting
        tracker.record(cycle_num=1, opportunities=[opp1])

        # Confidence should be 0.3 for first-seen with deep book
        conf = tracker.confidence("evt-123", depth_ratio=2.0, has_inventory=True)
        assert conf == 0.3, f"First-seen with deep book should have 0.3 confidence, got {conf}"

    def test_unknown_event_zero_confidence(self):
        """Unknown event should have 0.0 confidence."""
        tracker = ArbTracker()
        opp1 = _make_opp(event_id="evt-123")

        # Record sighting for different event
        tracker.record(cycle_num=1, opportunities=[opp1])

        # Unknown event should have 0.0 confidence
        conf = tracker.confidence("evt-999", depth_ratio=2.0, has_inventory=True)
        assert conf == 0.0, f"Unknown event should have 0.0 confidence, got {conf}"

    def test_no_inventory_low_confidence(self):
        """Sell-side without inventory should have 0.1 confidence."""
        tracker = ArbTracker()
        opp1 = _make_opp(event_id="evt-123")

        # Record first sighting
        tracker.record(cycle_num=1, opportunities=[opp1])

        # No inventory should have 0.1 confidence
        conf = tracker.confidence("evt-123", depth_ratio=2.0, has_inventory=False)
        assert conf == 0.1, f"No inventory should have 0.1 confidence, got {conf}"

    def test_confidence_affects_scoring_rank(self):
        """Higher confidence should improve the score and rank."""
        tracker = ArbTracker()

        # Two opportunities with same profit but different event IDs
        opp1 = _make_opp(event_id="evt-persistent", net_profit=5.0)
        opp2 = _make_opp(event_id="evt-new", net_profit=5.0)

        # Make opp1 persistent (seen in 3 consecutive cycles)
        for i in range(1, 4):
            tracker.record(cycle_num=i, opportunities=[opp1])

        # opp2 is new (first seen)
        tracker.record(cycle_num=3, opportunities=[opp2])

        # Build ScoringContexts with confidence from ArbTracker
        conf1 = tracker.confidence("evt-persistent", depth_ratio=1.0, has_inventory=True)
        conf2 = tracker.confidence("evt-new", depth_ratio=1.0, has_inventory=True)

        ctx1 = ScoringContext(book_depth_ratio=1.0, confidence=conf1)
        ctx2 = ScoringContext(book_depth_ratio=1.0, confidence=conf2)

        # Score both opportunities
        scored1 = score_opportunity(opp1, ctx1)
        scored2 = score_opportunity(opp2, ctx2)

        # Persistent should have higher score due to higher confidence
        assert scored1.persistence_score > scored2.persistence_score
        assert scored1.total_score > scored2.total_score

    def test_stale_entries_cleaned_up(self):
        """Stale entries should be removed from history."""
        tracker = ArbTracker(_max_history=5)
        opp1 = _make_opp(event_id="evt-123")

        # Record at cycle 1
        tracker.record(cycle_num=1, opportunities=[opp1])
        assert "evt-123" in tracker._history

        # Move far past max_history
        tracker.record(cycle_num=100, opportunities=[])

        # Stale entry should be purged
        assert "evt-123" not in tracker._history

        # Confidence should return 0.0 for purged event
        conf = tracker.confidence("evt-123", depth_ratio=2.0, has_inventory=True)
        assert conf == 0.0

    def test_duplicate_cycle_not_recorded_twice(self):
        """Recording same cycle twice should not duplicate entries."""
        tracker = ArbTracker()
        opp1 = _make_opp(event_id="evt-123")

        # Record same cycle twice
        tracker.record(cycle_num=1, opportunities=[opp1])
        tracker.record(cycle_num=1, opportunities=[opp1])

        # Should only have one entry
        assert tracker._history["evt-123"] == [1]

        # Should not be considered persistent (needs 2+ consecutive cycles)
        conf = tracker.confidence("evt-123", depth_ratio=1.0, has_inventory=True)
        assert conf == 0.1  # First-seen, not persistent


class TestArbTrackerScoringPipeline:
    """Tests for end-to-end integration with the scoring pipeline."""

    def test_rank_opportunities_uses_confidence(self):
        """rank_opportunities should use confidence from ScoringContext."""
        tracker = ArbTracker()

        # Create opportunities with different persistence levels
        opp_persistent = _make_opp(event_id="evt-persistent", net_profit=3.0)
        opp_new = _make_opp(event_id="evt-new", net_profit=3.0)

        # Make persistent appear in 3 consecutive cycles
        for i in range(1, 4):
            tracker.record(cycle_num=i, opportunities=[opp_persistent])

        # New opportunity appears in cycle 3 only
        tracker.record(cycle_num=3, opportunities=[opp_new])

        # Build contexts with ArbTracker confidence
        conf_persistent = tracker.confidence("evt-persistent", depth_ratio=1.0, has_inventory=True)
        conf_new = tracker.confidence("evt-new", depth_ratio=1.0, has_inventory=True)

        contexts = [
            ScoringContext(book_depth_ratio=1.0, confidence=conf_persistent),
            ScoringContext(book_depth_ratio=1.0, confidence=conf_new),
        ]

        # Rank opportunities
        ranked = rank_opportunities([opp_persistent, opp_new], contexts=contexts)

        # Persistent should rank higher due to confidence
        assert ranked[0].opportunity.event_id == "evt-persistent"
        assert ranked[1].opportunity.event_id == "evt-new"
        assert ranked[0].total_score > ranked[1].total_score

    def test_confidence_zero_for_unknown_event(self):
        """Unknown events should have 0.0 confidence in scoring."""
        tracker = ArbTracker()
        opp = _make_opp(event_id="evt-unknown")

        # Never recorded
        conf = tracker.confidence("evt-unknown", depth_ratio=2.0, has_inventory=True)
        assert conf == 0.0

        # Score with zero confidence
        ctx = ScoringContext(book_depth_ratio=2.0, confidence=conf)
        scored = score_opportunity(opp, ctx)

        # Persistence score should be 0.0
        assert scored.persistence_score == 0.0
