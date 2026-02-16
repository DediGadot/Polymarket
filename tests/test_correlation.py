"""Tests for the correlation scanner."""

from __future__ import annotations

import pytest

from scanner.correlation import (
    CorrelationScanner,
    CorrelationViolation,
    MarketRelation,
    RelationType,
    extract_entities,
    extract_temporal,
)
from scanner.models import (
    Event,
    LegOrder,
    Market,
    OpportunityType,
    OrderBook,
    PriceLevel,
    Side,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_market(
    condition_id: str = "cond1",
    question: str = "Will X happen?",
    yes_token_id: str = "yes1",
    no_token_id: str = "no1",
    event_id: str = "evt1",
    active: bool = True,
    volume: float = 1000.0,
) -> Market:
    return Market(
        condition_id=condition_id,
        question=question,
        yes_token_id=yes_token_id,
        no_token_id=no_token_id,
        neg_risk=False,
        event_id=event_id,
        min_tick_size="0.01",
        active=active,
        volume=volume,
    )


def _make_event(
    event_id: str,
    title: str,
    markets: tuple[Market, ...] | None = None,
) -> Event:
    if markets is None:
        markets = (
            _make_market(
                condition_id=f"cond_{event_id}",
                yes_token_id=f"yes_{event_id}",
                no_token_id=f"no_{event_id}",
                event_id=event_id,
            ),
        )
    return Event(
        event_id=event_id,
        title=title,
        markets=markets,
        neg_risk=False,
    )


def _make_book(
    token_id: str,
    best_ask: float = 0.50,
    best_bid: float = 0.45,
    ask_size: float = 100.0,
    bid_size: float = 100.0,
) -> OrderBook:
    asks = (PriceLevel(price=best_ask, size=ask_size),) if best_ask else ()
    bids = (PriceLevel(price=best_bid, size=bid_size),) if best_bid else ()
    return OrderBook(token_id=token_id, bids=bids, asks=asks)


# ---------------------------------------------------------------------------
# Test entity extraction
# ---------------------------------------------------------------------------

class TestExtractEntities:
    def test_basic_names(self):
        result = extract_entities("Will Biden win Ohio?")
        assert "biden" in result
        assert "ohio" in result

    def test_multi_word_names(self):
        result = extract_entities("Will Donald Trump win the election?")
        assert "donald trump" in result

    def test_quoted_terms(self):
        result = extract_entities('Will "Bitcoin ETF" be approved?')
        assert "bitcoin etf" in result

    def test_stop_words_excluded(self):
        result = extract_entities("Will the market crash?")
        # "the" and "will" are stop words
        assert "the" not in result
        assert "will" not in result

    def test_short_words_excluded(self):
        # Single uppercase letter followed by lowercase (like "Is") has len<=2
        result = extract_entities("Is AI the future?")
        # "Is" is length 2, excluded
        assert "is" not in result

    def test_empty_string(self):
        assert extract_entities("") == frozenset()


# ---------------------------------------------------------------------------
# Test temporal extraction
# ---------------------------------------------------------------------------

class TestExtractTemporal:
    def test_basic_month_year(self):
        result = extract_temporal("Bitcoin to $100K by March 2026")
        assert result is not None
        assert result["month"] == 3
        assert result["year"] == 2026

    def test_month_without_year(self):
        result = extract_temporal("Will it happen by June?")
        assert result is not None
        assert result["month"] == 6
        assert result["year"] == 2026  # default

    def test_no_temporal(self):
        result = extract_temporal("Will Biden win?")
        assert result is None

    def test_december(self):
        result = extract_temporal("GDP growth by December 2025")
        assert result is not None
        assert result["month"] == 12
        assert result["year"] == 2025


# ---------------------------------------------------------------------------
# Test parent-child detection
# ---------------------------------------------------------------------------

class TestParentChildDetection:
    def test_parent_child_found(self):
        # "Trump" is parent entity; "Trump" + "Ohio" is child
        parent = _make_event("e1", "Will Trump win the presidency?")
        child = _make_event("e2", "Will Trump win Ohio?")
        scanner = CorrelationScanner()
        rels = scanner.build_relationship_graph([parent, child])
        pc = [r for r in rels if r.relation_type == RelationType.PARENT_CHILD]
        assert len(pc) == 1
        assert pc[0].source_event_id == "e1"  # parent (fewer entities)
        assert pc[0].target_event_id == "e2"  # child (more entities)

    def test_parent_child_violation(self):
        """Parent prob (0.40) < child prob (0.60) → violation with edge 0.20."""
        parent = _make_event("e1", "Will Trump win the presidency?")
        child = _make_event("e2", "Will Trump win Ohio?")

        books = {
            "yes_e1": _make_book("yes_e1", best_ask=0.40, best_bid=0.35),
            "yes_e2": _make_book("yes_e2", best_ask=0.60, best_bid=0.55),
        }

        scanner = CorrelationScanner(min_edge_pct=1.0)
        scanner.build_relationship_graph([parent, child])
        opps = scanner.scan([parent, child], books, gas_cost_usd=0.001)

        assert len(opps) == 1
        opp = opps[0]
        assert opp.type == OpportunityType.CORRELATION_ARB
        # Edge = 0.60 - 0.40 = 0.20
        assert abs(opp.expected_profit_per_set - 0.20) < 1e-6
        # Legs: BUY parent ask, SELL child bid
        assert opp.legs[0].side == Side.BUY
        assert opp.legs[1].side == Side.SELL

    def test_no_violation_when_consistent(self):
        """Parent prob >= child prob → no violation."""
        parent = _make_event("e1", "Will Trump win the presidency?")
        child = _make_event("e2", "Will Trump win Ohio?")

        books = {
            "yes_e1": _make_book("yes_e1", best_ask=0.70, best_bid=0.65),
            "yes_e2": _make_book("yes_e2", best_ask=0.40, best_bid=0.35),
        }

        scanner = CorrelationScanner(min_edge_pct=1.0)
        scanner.build_relationship_graph([parent, child])
        opps = scanner.scan([parent, child], books, gas_cost_usd=0.001)
        assert len(opps) == 0

    def test_parent_child_prefers_long_only_when_child_no_book_available(self):
        parent = _make_event("e1", "Will Trump win the presidency?")
        child = _make_event("e2", "Will Trump win Ohio?")
        books = {
            "yes_e1": _make_book("yes_e1", best_ask=0.40, best_bid=0.35, ask_size=120.0, bid_size=120.0),
            "yes_e2": _make_book("yes_e2", best_ask=0.70, best_bid=0.65, ask_size=120.0, bid_size=120.0),
            "no_e2": _make_book("no_e2", best_ask=0.30, best_bid=0.25, ask_size=120.0, bid_size=120.0),
        }

        scanner = CorrelationScanner(min_edge_pct=1.0)
        scanner.build_relationship_graph([parent, child])
        opps = scanner.scan([parent, child], books, gas_cost_usd=0.001)

        assert len(opps) == 1
        opp = opps[0]
        assert opp.reason_code.startswith("corr_parent_child_buy")
        assert all(leg.side == Side.BUY for leg in opp.legs)
        assert {leg.token_id for leg in opp.legs} == {"yes_e1", "no_e2"}

    def test_parent_child_rejects_weak_semantic_entity_subset(self):
        """
        Shared entity token alone ("gta") should not imply parent/child.
        """
        parent = _make_event("e1", "Will GTA 6 cost $100+?")
        child = _make_event("e2", "Will Jesus Christ return before GTA VI?")
        scanner = CorrelationScanner()
        rels = scanner.build_relationship_graph([parent, child])
        pc = [r for r in rels if r.relation_type == RelationType.PARENT_CHILD]
        assert pc == []


# ---------------------------------------------------------------------------
# Test temporal detection
# ---------------------------------------------------------------------------

class TestTemporalDetection:
    def test_temporal_relation_found(self):
        earlier = _make_event("e1", "Bitcoin to $100K by March 2026")
        later = _make_event("e2", "Bitcoin to $100K by June 2026")
        scanner = CorrelationScanner()
        rels = scanner.build_relationship_graph([earlier, later])
        temp = [r for r in rels if r.relation_type == RelationType.TEMPORAL]
        assert len(temp) == 1
        assert temp[0].source_event_id == "e1"  # earlier
        assert temp[0].target_event_id == "e2"  # later

    def test_temporal_violation(self):
        """Earlier deadline has higher prob than later → violation."""
        earlier = _make_event("e1", "Bitcoin to $100K by March 2026")
        later = _make_event("e2", "Bitcoin to $100K by June 2026")

        books = {
            "yes_e1": _make_book("yes_e1", best_ask=0.50, best_bid=0.45),
            "yes_e2": _make_book("yes_e2", best_ask=0.30, best_bid=0.25),
        }

        scanner = CorrelationScanner(min_edge_pct=1.0)
        scanner.build_relationship_graph([earlier, later])
        opps = scanner.scan([earlier, later], books, gas_cost_usd=0.001)

        assert len(opps) == 1
        opp = opps[0]
        assert opp.type == OpportunityType.CORRELATION_ARB
        # Edge = 0.50 - 0.30 = 0.20
        assert abs(opp.expected_profit_per_set - 0.20) < 1e-6
        # Sell earlier (overpriced), buy later (underpriced)
        assert opp.legs[0].side == Side.SELL
        assert opp.legs[1].side == Side.BUY

    def test_no_temporal_violation_when_consistent(self):
        """Earlier prob <= later prob → no violation."""
        earlier = _make_event("e1", "Bitcoin to $100K by March 2026")
        later = _make_event("e2", "Bitcoin to $100K by June 2026")

        books = {
            "yes_e1": _make_book("yes_e1", best_ask=0.20, best_bid=0.15),
            "yes_e2": _make_book("yes_e2", best_ask=0.50, best_bid=0.45),
        }

        scanner = CorrelationScanner(min_edge_pct=1.0)
        scanner.build_relationship_graph([earlier, later])
        opps = scanner.scan([earlier, later], books, gas_cost_usd=0.001)
        assert len(opps) == 0

    def test_temporal_prefers_long_only_when_earlier_no_book_available(self):
        earlier = _make_event("e1", "Bitcoin to $100K by March 2026")
        later = _make_event("e2", "Bitcoin to $100K by June 2026")
        books = {
            "yes_e1": _make_book("yes_e1", best_ask=0.70, best_bid=0.65, ask_size=100.0, bid_size=100.0),
            "yes_e2": _make_book("yes_e2", best_ask=0.40, best_bid=0.35, ask_size=100.0, bid_size=100.0),
            "no_e1": _make_book("no_e1", best_ask=0.20, best_bid=0.15, ask_size=100.0, bid_size=100.0),
        }

        scanner = CorrelationScanner(min_edge_pct=1.0)
        scanner.build_relationship_graph([earlier, later])
        opps = scanner.scan([earlier, later], books, gas_cost_usd=0.001)

        assert len(opps) == 1
        opp = opps[0]
        assert opp.reason_code.startswith("corr_temporal_buy")
        assert all(leg.side == Side.BUY for leg in opp.legs)
        assert {leg.token_id for leg in opp.legs} == {"no_e1", "yes_e2"}

    def test_temporal_rejects_different_predicates_same_entity(self):
        """
        Same entity + different deadlines is insufficient when predicate differs.
        """
        e1 = _make_event("e1", "Will Trump resign by March 2026?")
        e2 = _make_event("e2", "Will Trump win Ohio by June 2026?")
        scanner = CorrelationScanner()
        rels = scanner.build_relationship_graph([e1, e2])
        temp = [r for r in rels if r.relation_type == RelationType.TEMPORAL]
        assert temp == []


# ---------------------------------------------------------------------------
# Test complement detection
# ---------------------------------------------------------------------------

class TestComplementDetection:
    def test_complement_found(self):
        # Titles with 2+ separate shared entities: "Super Bowl" and "MVP"
        e1 = _make_event("e1", "Will Mahomes win Super Bowl MVP?")
        e2 = _make_event("e2", "Will Allen win Super Bowl MVP?")
        scanner = CorrelationScanner()
        rels = scanner.build_relationship_graph([e1, e2])
        comp = [r for r in rels if r.relation_type == RelationType.COMPLEMENT]
        assert len(comp) == 1

    def test_complement_violation(self):
        """P(A) + P(B) > 1.0 → complement violation."""
        e1 = _make_event("e1", "Will Mahomes win Super Bowl MVP?")
        e2 = _make_event("e2", "Will Allen win Super Bowl MVP?")

        books = {
            "yes_e1": _make_book("yes_e1", best_ask=0.60, best_bid=0.55),
            "yes_e2": _make_book("yes_e2", best_ask=0.55, best_bid=0.50),
        }

        scanner = CorrelationScanner(min_edge_pct=1.0)
        scanner.build_relationship_graph([e1, e2])
        # Complement violations → no opp (Phase 5b), but violation is detected
        violations = []
        for rel in scanner._relations:
            from scanner.correlation import _check_violation
            v = _check_violation(rel, e1, e2, books)
            if v:
                violations.append(v)
        # 0.60 + 0.55 = 1.15 → edge = 0.15
        assert len(violations) == 1
        assert abs(violations[0].violation_magnitude - 0.15) < 1e-6


# ---------------------------------------------------------------------------
# Test no false positives
# ---------------------------------------------------------------------------

class TestNoFalsePositives:
    def test_unrelated_events(self):
        """Completely unrelated events produce no relations."""
        e1 = _make_event("e1", "Will Bitcoin hit $100K?")
        e2 = _make_event("e2", "Will Lakers win the NBA Finals?")
        scanner = CorrelationScanner()
        rels = scanner.build_relationship_graph([e1, e2])
        assert len(rels) == 0

    def test_single_shared_entity_insufficient(self):
        """Only 1 shared entity is not enough for complement."""
        e1 = _make_event("e1", "Will Trump win?")
        e2 = _make_event("e2", "Will Trump resign?")
        scanner = CorrelationScanner()
        rels = scanner.build_relationship_graph([e1, e2])
        comp = [r for r in rels if r.relation_type == RelationType.COMPLEMENT]
        assert len(comp) == 0


class TestGraphInvalidation:
    def test_scan_rebuilds_graph_when_event_set_changes(self):
        parent = _make_event("e1", "Will Trump win the presidency?")
        child = _make_event("e2", "Will Trump win Ohio?")
        scanner = CorrelationScanner()
        scanner.build_relationship_graph([parent, child])
        assert len(scanner._relations) == 1

        unrelated_a = _make_event("u1", "Will Bitcoin hit $100K?")
        unrelated_b = _make_event("u2", "Will Lakers win the NBA Finals?")
        scanner.scan([unrelated_a, unrelated_b], books={})

        assert len(scanner._relations) == 0


# ---------------------------------------------------------------------------
# Test edge filter
# ---------------------------------------------------------------------------

class TestEdgeFilter:
    def test_below_min_edge_filtered(self):
        """Violation below min_edge_pct → no opportunity emitted."""
        parent = _make_event("e1", "Will Trump win the presidency?")
        child = _make_event("e2", "Will Trump win Ohio?")

        # Edge = 0.02 (2%), min_edge_pct=3.0 → filtered
        books = {
            "yes_e1": _make_book("yes_e1", best_ask=0.48, best_bid=0.43),
            "yes_e2": _make_book("yes_e2", best_ask=0.50, best_bid=0.45),
        }

        scanner = CorrelationScanner(min_edge_pct=3.0)
        scanner.build_relationship_graph([parent, child])
        opps = scanner.scan([parent, child], books, gas_cost_usd=0.001)
        assert len(opps) == 0


# ---------------------------------------------------------------------------
# Test opportunity generation
# ---------------------------------------------------------------------------

class TestOpportunityGeneration:
    def test_opportunity_fields(self):
        """Verify opportunity has correct legs, profit, and type."""
        parent = _make_event("e1", "Will Trump win the presidency?")
        child = _make_event("e2", "Will Trump win Ohio?")

        books = {
            "yes_e1": _make_book("yes_e1", best_ask=0.40, best_bid=0.35, ask_size=50.0),
            "yes_e2": _make_book("yes_e2", best_ask=0.60, best_bid=0.55, bid_size=80.0),
        }

        scanner = CorrelationScanner(min_edge_pct=1.0)
        scanner.build_relationship_graph([parent, child])
        opps = scanner.scan([parent, child], books, gas_cost_usd=0.001)

        assert len(opps) == 1
        opp = opps[0]
        assert opp.type == OpportunityType.CORRELATION_ARB
        assert opp.event_id == "e1"
        assert len(opp.legs) == 2
        # Size = min(ask_size=50, bid_size=80) = 50
        assert opp.max_sets == 50.0
        # Profit per set = 0.20, gas = 0.001 * 2 = 0.002
        assert abs(opp.expected_profit_per_set - 0.20) < 1e-6
        assert abs(opp.estimated_gas_cost - 0.002) < 1e-6
        assert opp.net_profit > 0
        assert opp.roi_pct > 0


class TestPrecisionHardening:
    def test_aggregates_event_level_prob_not_first_market(self):
        """
        Low-volume outlier market should not dominate event implied probability.
        """
        parent = _make_event(
            "e1",
            "Will Trump win the presidency?",
            markets=(
                _make_market(
                    condition_id="c1a",
                    yes_token_id="yes_e1_outlier",
                    no_token_id="no_e1_outlier",
                    event_id="e1",
                    volume=5.0,  # should be filtered out
                ),
                _make_market(
                    condition_id="c1b",
                    yes_token_id="yes_e1_liq",
                    no_token_id="no_e1_liq",
                    event_id="e1",
                    volume=5000.0,
                ),
            ),
        )
        child = _make_event("e2", "Will Trump win Ohio?")
        books = {
            "yes_e1_outlier": _make_book("yes_e1_outlier", best_ask=0.10, best_bid=0.05),
            "yes_e1_liq": _make_book("yes_e1_liq", best_ask=0.80, best_bid=0.75),
            "yes_e2": _make_book("yes_e2", best_ask=0.60, best_bid=0.55),
        }

        scanner = CorrelationScanner(min_edge_pct=1.0, min_market_volume=100.0)
        scanner.build_relationship_graph([parent, child])
        opps = scanner.scan([parent, child], books, gas_cost_usd=0.001)
        assert opps == []

    def test_depth_filter_suppresses_thin_book(self):
        parent = _make_event("e1", "Will Trump win the presidency?")
        child = _make_event("e2", "Will Trump win Ohio?")
        books = {
            "yes_e1": _make_book("yes_e1", best_ask=0.40, best_bid=0.35, ask_size=1.0, bid_size=1.0),
            "yes_e2": _make_book("yes_e2", best_ask=0.70, best_bid=0.65, ask_size=1.0, bid_size=1.0),
        }
        scanner = CorrelationScanner(min_edge_pct=1.0, min_book_depth=50.0)
        scanner.build_relationship_graph([parent, child])
        opps = scanner.scan([parent, child], books, gas_cost_usd=0.001)
        assert opps == []

    def test_roi_cap_filters_unrealistic_research_signals(self):
        parent = _make_event("e1", "Will Trump win the presidency?")
        child = _make_event("e2", "Will Trump win Ohio?")
        books = {
            "yes_e1": _make_book("yes_e1", best_ask=0.01, best_bid=0.01, ask_size=1000.0, bid_size=1000.0),
            "yes_e2": _make_book("yes_e2", best_ask=0.99, best_bid=0.98, ask_size=1000.0, bid_size=1000.0),
        }
        scanner = CorrelationScanner(min_edge_pct=1.0, max_theoretical_roi_pct=50.0)
        scanner.build_relationship_graph([parent, child])
        opps = scanner.scan([parent, child], books, gas_cost_usd=0.0001)
        assert opps == []

    def test_unprofitable_after_gas(self):
        """Tiny edge eaten by gas → no opportunity."""
        parent = _make_event("e1", "Will Trump win the presidency?")
        child = _make_event("e2", "Will Trump win Ohio?")

        books = {
            "yes_e1": _make_book("yes_e1", best_ask=0.49, best_bid=0.44, ask_size=1.0),
            "yes_e2": _make_book("yes_e2", best_ask=0.53, best_bid=0.48, bid_size=1.0),
        }

        scanner = CorrelationScanner(min_edge_pct=1.0)
        scanner.build_relationship_graph([parent, child])
        # Gas cost per leg = 5.0 → total = 10.0, edge = 0.04 * 1 = 0.04
        opps = scanner.scan([parent, child], books, gas_cost_usd=5.0)
        assert len(opps) == 0

    def test_persistence_gate_requires_consecutive_cycles(self):
        parent = _make_event("e1", "Will Trump win the presidency?")
        child = _make_event("e2", "Will Trump win Ohio?")
        books = {
            "yes_e1": _make_book("yes_e1", best_ask=0.35, best_bid=0.30),
            "yes_e2": _make_book("yes_e2", best_ask=0.65, best_bid=0.60),
        }
        scanner = CorrelationScanner(min_edge_pct=1.0, min_persistence_cycles=2)
        scanner.build_relationship_graph([parent, child])

        first = scanner.scan([parent, child], books, gas_cost_usd=0.001)
        second = scanner.scan([parent, child], books, gas_cost_usd=0.001)

        assert first == []
        assert len(second) == 1

    def test_caps_size_by_max_capital_per_opportunity(self):
        parent = _make_event("e1", "Will Trump win the presidency?")
        child = _make_event("e2", "Will Trump win Ohio?")
        books = {
            "yes_e1": _make_book("yes_e1", best_ask=0.40, best_bid=0.35, ask_size=1000.0, bid_size=1000.0),
            "yes_e2": _make_book("yes_e2", best_ask=0.70, best_bid=0.65, ask_size=1000.0, bid_size=1000.0),
        }
        scanner = CorrelationScanner(
            min_edge_pct=1.0,
            min_persistence_cycles=1,
            max_capital_per_opportunity=40.0,
        )
        scanner.build_relationship_graph([parent, child])
        opps = scanner.scan([parent, child], books, gas_cost_usd=0.001)

        assert len(opps) == 1
        assert opps[0].max_sets == pytest.approx(100.0)
        assert opps[0].required_capital <= 40.0 + 1e-6


# ---------------------------------------------------------------------------
# Test empty inputs
# ---------------------------------------------------------------------------

class TestEmptyInputs:
    def test_empty_events(self):
        scanner = CorrelationScanner()
        rels = scanner.build_relationship_graph([])
        assert rels == []

    def test_scan_empty_events(self):
        scanner = CorrelationScanner()
        opps = scanner.scan([], {})
        assert opps == []

    def test_scan_empty_books(self):
        parent = _make_event("e1", "Will Trump win the presidency?")
        child = _make_event("e2", "Will Trump win Ohio?")
        scanner = CorrelationScanner(min_edge_pct=1.0)
        scanner.build_relationship_graph([parent, child])
        opps = scanner.scan([parent, child], {})
        assert opps == []


# ---------------------------------------------------------------------------
# Test confidence filter
# ---------------------------------------------------------------------------

class TestConfidenceFilter:
    def test_low_confidence_filtered(self):
        """Relations below min_confidence are skipped during scan."""
        parent = _make_event("e1", "Will Trump win the presidency?")
        child = _make_event("e2", "Will Trump win Ohio?")

        books = {
            "yes_e1": _make_book("yes_e1", best_ask=0.30, best_bid=0.25),
            "yes_e2": _make_book("yes_e2", best_ask=0.70, best_bid=0.65),
        }

        # Parent-child confidence defaults to 0.7, set min_confidence high
        scanner = CorrelationScanner(min_edge_pct=1.0, min_confidence=0.9)
        scanner.build_relationship_graph([parent, child])
        opps = scanner.scan([parent, child], books, gas_cost_usd=0.001)
        assert len(opps) == 0


# ---------------------------------------------------------------------------
# Test missing books
# ---------------------------------------------------------------------------

class TestMissingBooks:
    def test_missing_source_book(self):
        parent = _make_event("e1", "Will Trump win the presidency?")
        child = _make_event("e2", "Will Trump win Ohio?")

        # Only child book present
        books = {
            "yes_e2": _make_book("yes_e2", best_ask=0.60, best_bid=0.55),
        }

        scanner = CorrelationScanner(min_edge_pct=1.0)
        scanner.build_relationship_graph([parent, child])
        opps = scanner.scan([parent, child], books, gas_cost_usd=0.001)
        assert len(opps) == 0

    def test_inactive_market_skipped(self):
        inactive_market = _make_market(
            condition_id="cond_e1",
            yes_token_id="yes_e1",
            no_token_id="no_e1",
            event_id="e1",
            active=False,
        )
        parent = _make_event("e1", "Will Trump win the presidency?", markets=(inactive_market,))
        child = _make_event("e2", "Will Trump win Ohio?")

        books = {
            "yes_e1": _make_book("yes_e1", best_ask=0.30, best_bid=0.25),
            "yes_e2": _make_book("yes_e2", best_ask=0.70, best_bid=0.65),
        }

        scanner = CorrelationScanner(min_edge_pct=1.0)
        scanner.build_relationship_graph([parent, child])
        opps = scanner.scan([parent, child], books, gas_cost_usd=0.001)
        assert len(opps) == 0
