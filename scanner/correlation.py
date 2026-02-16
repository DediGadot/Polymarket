"""
Correlation scanner: detects logical inconsistencies across related markets.
Rule-based approach using Gamma API metadata (categories, titles, tags).

Phase 1: flag-only (emit opportunities but mark for alerting, not auto-execution).
Multi-event execution is deferred to Phase 5b.
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass
from enum import Enum
from statistics import median

from scanner.models import (
    Event,
    LegOrder,
    Market,
    Opportunity,
    OpportunityType,
    OrderBook,
    Side,
)

logger = logging.getLogger(__name__)


class RelationType(Enum):
    PARENT_CHILD = "parent_child"
    COMPLEMENT = "complement"
    TEMPORAL = "temporal"


@dataclass(frozen=True)
class MarketRelation:
    """A detected relationship between two markets/events."""

    source_event_id: str
    target_event_id: str
    relation_type: RelationType
    confidence: float  # 0.0-1.0, higher = more certain
    description: str = ""


@dataclass(frozen=True)
class CorrelationViolation:
    """A detected probability constraint violation."""

    relation: MarketRelation
    expected_constraint: str  # human-readable constraint description
    actual_values: dict  # {market_id: implied_probability}
    violation_magnitude: float  # how much the constraint is violated (edge)


@dataclass(frozen=True)
class EventQuoteView:
    """Aggregated event-level quote used for robust correlation checks."""

    event_id: str
    implied_prob: float
    representative_book: OrderBook
    representative_market_volume: float
    sample_count: int
    aggregation: str


_STOP_WORDS = frozenset(
    {
        "the",
        "will",
        "win",
        "wins",
        "to",
        "in",
        "by",
        "at",
        "for",
        "and",
        "or",
        "not",
        "be",
        "is",
        "are",
        "was",
        "were",
        "has",
        "have",
        "had",
        "do",
        "does",
        "did",
        "can",
        "could",
        "would",
        "should",
        "may",
        "might",
        "shall",
        "must",
        "yes",
        "no",
        "who",
        "what",
        "how",
        "when",
        "where",
        "which",
        "that",
        "this",
        "than",
        "more",
        "most",
        "next",
        "before",
        "after",
    }
)

_MONTHS = {
    "january": 1,
    "february": 2,
    "march": 3,
    "april": 4,
    "may": 5,
    "june": 6,
    "july": 7,
    "august": 8,
    "september": 9,
    "october": 10,
    "november": 11,
    "december": 12,
}

_WIN_PATTERN = re.compile(r"(\w+)\s+(?:win|wins|to win)")
_ENTITY_PATTERN = re.compile(r"\b([A-Z][a-z]+(?:\s+[A-Z][a-z]+)*)\b")
_ACRONYM_PATTERN = re.compile(r"\b([A-Z]{2,})\b")
_QUOTED_PATTERN = re.compile(r'"([^"]+)"')
_YEAR_PATTERN = re.compile(r"20\d{2}")
_TOKEN_PATTERN = re.compile(r"[a-z0-9]+")
_ROMAN_NUMERAL_MAP = {
    "i": "1",
    "ii": "2",
    "iii": "3",
    "iv": "4",
    "v": "5",
    "vi": "6",
    "vii": "7",
    "viii": "8",
    "ix": "9",
    "x": "10",
}
_LEMMA_MAP = {
    "wins": "win",
    "won": "win",
    "winning": "win",
    "costs": "cost",
    "costing": "cost",
    "returns": "return",
    "returned": "return",
    "resigns": "resign",
    "resigned": "resign",
    "deports": "deport",
    "deported": "deport",
    "launches": "launch",
    "launched": "launch",
    "captures": "capture",
    "captured": "capture",
    "tests": "test",
    "tested": "test",
    "confirms": "confirm",
    "confirmed": "confirm",
    "approves": "approve",
    "approved": "approve",
    "raises": "raise",
    "raised": "raise",
    "meets": "meet",
    "met": "meet",
    "creates": "create",
    "created": "create",
    "invades": "invade",
    "invaded": "invade",
    "signs": "sign",
    "signed": "sign",
    "postponed": "postpone",
    "postpones": "postpone",
}
_SEMANTIC_STOP_WORDS = _STOP_WORDS - {"win", "wins"}
_GENERIC_CORE_TOKENS = frozenset(
    {
        "market",
        "markets",
        "price",
        "prices",
        "day",
        "days",
        "year",
        "years",
        "month",
        "months",
        "week",
        "weeks",
        "event",
        "events",
        "time",
        "times",
    }
)


class CorrelationScanner:
    """
    Builds a relationship graph from event metadata and detects
    probability constraint violations.
    """

    def __init__(
        self,
        min_edge_pct: float = 3.0,
        min_confidence: float = 0.7,
        *,
        aggregation: str = "liquidity_weighted",
        max_markets_per_event: int = 5,
        min_market_volume: float = 500.0,
        min_book_depth: float = 50.0,
        max_theoretical_roi_pct: float = 250.0,
        min_buy_total_prob: float = 0.15,
        min_persistence_cycles: int = 1,
        max_capital_per_opportunity: float = 1000.0,
    ):
        self._min_edge = min_edge_pct / 100.0
        self._min_confidence = min_confidence
        self._aggregation = aggregation
        self._max_markets_per_event = max(1, max_markets_per_event)
        self._min_market_volume = max(0.0, min_market_volume)
        self._min_book_depth = max(0.0, min_book_depth)
        self._max_theoretical_roi_pct = max(0.0, max_theoretical_roi_pct)
        self._min_buy_total_prob = max(0.0, min(1.0, min_buy_total_prob))
        self._min_persistence_cycles = max(1, int(min_persistence_cycles))
        self._max_capital_per_opportunity = max(1.0, max_capital_per_opportunity)
        self._relations: list[MarketRelation] = []
        self._graph_signature: tuple[tuple[str, str, int, int], ...] = ()
        self._violation_streak: dict[tuple[str, str, str, str], int] = {}

    def build_relationship_graph(self, events: list[Event]) -> list[MarketRelation]:
        """
        Analyze events to find logically related markets.
        Uses title analysis, category matching, and entity extraction.
        Returns a new list of relations (also stored internally for scan()).
        """
        relations: list[MarketRelation] = []

        entity_map: dict[str, frozenset[str]] = {}
        temporal_map: dict[str, dict] = {}

        for event in events:
            entities = extract_entities(event.title)
            entity_map[event.event_id] = entities

            temporal = extract_temporal(event.title)
            if temporal:
                temporal_map[event.event_id] = temporal

        relations.extend(_find_complement_relations(events, entity_map))
        relations.extend(_find_temporal_relations(events, entity_map, temporal_map))
        relations.extend(_find_parent_child_relations(events, entity_map))

        logger.debug(
            "CorrelationScanner: built graph with %d relations from %d events",
            len(relations),
            len(events),
        )
        self._relations = relations
        self._graph_signature = self._compute_graph_signature(events)
        return list(relations)

    @staticmethod
    def _compute_graph_signature(events: list[Event]) -> tuple[tuple[str, str, int, int], ...]:
        """
        Build an order-independent event signature for graph invalidation.
        Refreshes relations when event identity/title or market counts change.
        """
        return tuple(
            sorted(
                (
                    e.event_id,
                    e.title,
                    len(e.markets),
                    1 if e.neg_risk else 0,
                )
                for e in events
            )
        )

    def scan(
        self,
        events: list[Event],
        books: dict[str, OrderBook],
        gas_cost_usd: float = 0.005,
    ) -> list[Opportunity]:
        """
        Scan for probability constraint violations across related events.
        Returns opportunities where constraints are violated beyond min_edge.
        """
        current_signature = self._compute_graph_signature(events)
        if current_signature != self._graph_signature:
            self.build_relationship_graph(events)

        event_map = {e.event_id: e for e in events}
        event_views: dict[str, EventQuoteView] = {}
        for event in events:
            view = _build_event_quote_view(
                event=event,
                books=books,
                aggregation=self._aggregation,
                max_markets=self._max_markets_per_event,
                min_market_volume=self._min_market_volume,
                min_book_depth=self._min_book_depth,
            )
            if view is not None:
                event_views[event.event_id] = view

        violations: list[CorrelationViolation] = []

        current_keys: set[tuple[str, str, str, str]] = set()
        for rel in self._relations:
            if rel.confidence < self._min_confidence:
                continue

            source_view = event_views.get(rel.source_event_id)
            target_view = event_views.get(rel.target_event_id)
            if source_view is None or target_view is None:
                continue

            if rel.relation_type == RelationType.COMPLEMENT:
                total = source_view.implied_prob + target_view.implied_prob
                if total < self._min_buy_total_prob:
                    continue

            violation = _check_violation(rel, source_view, target_view)
            if violation and violation.violation_magnitude >= self._min_edge:
                direction = _violation_direction(rel, source_view, target_view)
                key = (rel.source_event_id, rel.target_event_id, rel.relation_type.value, direction)
                current_keys.add(key)
                streak = self._violation_streak.get(key, 0) + 1
                self._violation_streak[key] = streak
                if streak < self._min_persistence_cycles:
                    continue
                violations.append(violation)

        for key in list(self._violation_streak):
            if key not in current_keys:
                self._violation_streak.pop(key, None)

        opps: list[Opportunity] = []
        for v in violations:
            opp = _violation_to_opportunity(
                v,
                event_map,
                event_views,
                books,
                gas_cost_usd,
                max_theoretical_roi_pct=self._max_theoretical_roi_pct,
                max_capital_per_opportunity=self._max_capital_per_opportunity,
            )
            if opp:
                opps.append(opp)

        if opps:
            logger.info(
                "CorrelationScanner: found %d violations from %d relations",
                len(opps),
                len(self._relations),
            )

        return opps


# ---------------------------------------------------------------------------
# Entity / temporal extraction (pure functions)
# ---------------------------------------------------------------------------


def extract_entities(title: str) -> frozenset[str]:
    """Extract named entities from event title using regex patterns."""
    entities: set[str] = set()

    for match in _ENTITY_PATTERN.finditer(title):
        raw = match.group(1)
        # Strip leading/trailing stop words from multi-word captures
        words = raw.split()
        words = _strip_stop_words(words)
        if not words:
            continue
        entity = " ".join(w.lower() for w in words)
        if len(entity) > 2:
            entities.add(entity)

    # Capture acronyms (e.g., MVP, NFL, GDP)
    for match in _ACRONYM_PATTERN.finditer(title):
        entities.add(match.group(1).lower())

    for match in _QUOTED_PATTERN.finditer(title):
        entities.add(match.group(1).lower())

    return frozenset(entities)


def _strip_stop_words(words: list[str]) -> list[str]:
    """Strip leading and trailing stop words from a word list."""
    # Strip from front
    while words and words[0].lower() in _STOP_WORDS:
        words = words[1:]
    # Strip from back
    while words and words[-1].lower() in _STOP_WORDS:
        words = words[:-1]
    return words


def extract_temporal(title: str) -> dict | None:
    """Extract temporal markers from title (e.g., 'by March 2026')."""
    title_lower = title.lower()
    for month_name, month_num in _MONTHS.items():
        if month_name in title_lower:
            year_match = _YEAR_PATTERN.search(title)
            year = int(year_match.group()) if year_match else 2026
            return {"month": month_num, "year": year, "month_name": month_name}
    return None


# ---------------------------------------------------------------------------
# Relation finders (pure functions)
# ---------------------------------------------------------------------------


def _find_complement_relations(
    events: list[Event],
    entity_map: dict[str, frozenset[str]],
) -> list[MarketRelation]:
    """Find events that represent mutually exclusive outcomes of the same question."""
    relations: list[MarketRelation] = []
    for i, e1 in enumerate(events):
        for e2 in events[i + 1 :]:
            shared = entity_map.get(e1.event_id, frozenset()) & entity_map.get(
                e2.event_id, frozenset()
            )
            if len(shared) >= 2 and _looks_complementary(e1.title, e2.title):
                relations.append(
                    MarketRelation(
                        source_event_id=e1.event_id,
                        target_event_id=e2.event_id,
                        relation_type=RelationType.COMPLEMENT,
                        confidence=0.6 + 0.1 * min(len(shared), 4),
                        description=f"Complement: shared entities {shared}",
                    )
                )
    return relations


def _find_temporal_relations(
    events: list[Event],
    entity_map: dict[str, frozenset[str]],
    temporal_map: dict[str, dict],
) -> list[MarketRelation]:
    """Find events with same entity but different temporal deadlines."""
    relations: list[MarketRelation] = []
    temporal_events = list(temporal_map.items())

    for i, (eid1, t1) in enumerate(temporal_events):
        for eid2, t2 in temporal_events[i + 1 :]:
            shared = entity_map.get(eid1, frozenset()) & entity_map.get(
                eid2, frozenset()
            )
            if not shared:
                continue
            ents1 = entity_map.get(eid1, frozenset())
            ents2 = entity_map.get(eid2, frozenset())
            e1 = next((event for event in events if event.event_id == eid1), None)
            e2 = next((event for event in events if event.event_id == eid2), None)
            if e1 is None or e2 is None:
                continue
            if not _is_semantically_compatible_temporal(e1.title, e2.title, ents1, ents2):
                continue

            # Same month+year → not a temporal pair
            if t1["year"] == t2["year"] and t1["month"] == t2["month"]:
                continue

            if t1["year"] < t2["year"] or (
                t1["year"] == t2["year"] and t1["month"] < t2["month"]
            ):
                earlier, later = eid1, eid2
            else:
                earlier, later = eid2, eid1

            relations.append(
                MarketRelation(
                    source_event_id=earlier,
                    target_event_id=later,
                    relation_type=RelationType.TEMPORAL,
                    confidence=0.8,
                    description=f"Temporal: {shared} (earlier->later)",
                )
            )
    return relations


def _find_parent_child_relations(
    events: list[Event],
    entity_map: dict[str, frozenset[str]],
) -> list[MarketRelation]:
    """Find general/specific event pairs (e.g., 'wins presidency' vs 'wins Ohio')."""
    relations: list[MarketRelation] = []
    for i, e1 in enumerate(events):
        for e2 in events[i + 1 :]:
            ents1 = entity_map.get(e1.event_id, frozenset())
            ents2 = entity_map.get(e2.event_id, frozenset())

            if not ents1 or not ents2:
                continue

            if ents1 < ents2:
                if not _is_semantically_compatible_parent_child(e1.title, e2.title, ents1, ents2):
                    continue
                # e1 is more general (parent), e2 is more specific (child)
                relations.append(
                    MarketRelation(
                        source_event_id=e1.event_id,
                        target_event_id=e2.event_id,
                        relation_type=RelationType.PARENT_CHILD,
                        confidence=0.7,
                        description=f"Parent-child: {e1.title[:30]} >= {e2.title[:30]}",
                    )
                )
            elif ents2 < ents1:
                if not _is_semantically_compatible_parent_child(e2.title, e1.title, ents2, ents1):
                    continue
                relations.append(
                    MarketRelation(
                        source_event_id=e2.event_id,
                        target_event_id=e1.event_id,
                        relation_type=RelationType.PARENT_CHILD,
                        confidence=0.7,
                        description=f"Parent-child: {e2.title[:30]} >= {e1.title[:30]}",
                    )
                )
    return relations


def _looks_complementary(title1: str, title2: str) -> bool:
    """Heuristic: do two titles look like they describe mutually exclusive outcomes?"""
    t1 = title1.lower()
    t2 = title2.lower()

    m1 = _WIN_PATTERN.search(t1)
    m2 = _WIN_PATTERN.search(t2)
    if m1 and m2 and m1.group(1) != m2.group(1):
        return True

    return False


def _semantic_core_tokens(title: str, entities: frozenset[str]) -> frozenset[str]:
    """
    Extract normalized predicate-ish tokens from a title.

    Strategy:
    - lowercase + tokenize
    - normalize roman numerals and common verb inflections
    - drop stop words, years, month names, and entity words
    - keep only compact non-trivial signal tokens
    """
    month_tokens = set(_MONTHS.keys())
    entity_words: set[str] = set()
    for ent in entities:
        for tok in _TOKEN_PATTERN.findall(ent.lower()):
            tok = _ROMAN_NUMERAL_MAP.get(tok, tok)
            if tok:
                entity_words.add(tok)

    out: set[str] = set()
    for raw in _TOKEN_PATTERN.findall(title.lower()):
        tok = _ROMAN_NUMERAL_MAP.get(raw, raw)
        tok = _LEMMA_MAP.get(tok, tok)
        if tok in month_tokens:
            continue
        if _YEAR_PATTERN.fullmatch(tok):
            continue
        if tok in _SEMANTIC_STOP_WORDS:
            continue
        if tok in entity_words:
            continue
        if len(tok) <= 1:
            continue
        out.add(tok)
    return frozenset(out)


def _has_meaningful_shared_tokens(a: frozenset[str], b: frozenset[str]) -> bool:
    shared = a & b
    if not shared:
        return False
    meaningful = {
        t
        for t in shared
        if t not in _GENERIC_CORE_TOKENS and not t.isdigit() and len(t) >= 2
    }
    return bool(meaningful)


def _is_semantically_compatible_parent_child(
    parent_title: str,
    child_title: str,
    parent_entities: frozenset[str],
    child_entities: frozenset[str],
) -> bool:
    """
    Parent/child relations must share proposition-level semantics, not just
    entity overlap. This blocks unrelated pairs like:
      "Will GTA 6 cost $100+?" vs "Will Jesus Christ return before GTA VI?"
    """
    parent_core = _semantic_core_tokens(parent_title, parent_entities)
    child_core = _semantic_core_tokens(child_title, child_entities)
    if not parent_core or not child_core:
        return False
    if not _has_meaningful_shared_tokens(parent_core, child_core):
        return False
    overlap = len(parent_core & child_core) / max(1, min(len(parent_core), len(child_core)))
    return overlap >= 0.5


def _is_semantically_compatible_temporal(
    earlier_title: str,
    later_title: str,
    earlier_entities: frozenset[str],
    later_entities: frozenset[str],
) -> bool:
    """
    Temporal relations should represent the *same proposition* across two
    deadlines. Require stronger core overlap than parent-child.
    """
    earlier_core = _semantic_core_tokens(earlier_title, earlier_entities)
    later_core = _semantic_core_tokens(later_title, later_entities)
    if not earlier_core or not later_core:
        return False
    if not _has_meaningful_shared_tokens(earlier_core, later_core):
        return False
    overlap = len(earlier_core & later_core) / max(1, min(len(earlier_core), len(later_core)))
    return overlap >= 0.7


# ---------------------------------------------------------------------------
# Violation checking (pure functions)
# ---------------------------------------------------------------------------


def _check_violation(
    rel: MarketRelation,
    source: EventQuoteView | Event,
    target: EventQuoteView | Event,
    books: dict[str, OrderBook] | None = None,
) -> CorrelationViolation | None:
    """Check if a relationship's probability constraint is violated."""
    if isinstance(source, Event):
        if books is None:
            return None
        source_view = _build_event_quote_view(source, books)
        target_view = _build_event_quote_view(target, books) if isinstance(target, Event) else target
    else:
        source_view = source
        target_view = target if isinstance(target, EventQuoteView) else (
            _build_event_quote_view(target, books) if books is not None else None
        )

    if source_view is None or target_view is None:
        return None

    source_prob = source_view.implied_prob
    target_prob = target_view.implied_prob

    if rel.relation_type == RelationType.PARENT_CHILD:
        # Parent should have >= probability than child
        if source_prob < target_prob:
            edge = target_prob - source_prob
            return CorrelationViolation(
                relation=rel,
                expected_constraint=(
                    f"P(parent={source_view.event_id[:10]}) >= P(child={target_view.event_id[:10]})"
                ),
                actual_values={
                    source_view.event_id: source_prob,
                    target_view.event_id: target_prob,
                },
                violation_magnitude=edge,
            )

    elif rel.relation_type == RelationType.TEMPORAL:
        # Earlier deadline should have <= probability than later
        if source_prob > target_prob:
            edge = source_prob - target_prob
            return CorrelationViolation(
                relation=rel,
                expected_constraint=(
                    f"P(earlier={source_view.event_id[:10]}) <= P(later={target_view.event_id[:10]})"
                ),
                actual_values={
                    source_view.event_id: source_prob,
                    target_view.event_id: target_prob,
                },
                violation_magnitude=edge,
            )

    elif rel.relation_type == RelationType.COMPLEMENT:
        total = source_prob + target_prob
        if total > 1.0:
            edge = total - 1.0
            return CorrelationViolation(
                relation=rel,
                expected_constraint=(
                    f"P({source_view.event_id[:10]}) + P({target_view.event_id[:10]}) <= 1.0"
                ),
                actual_values={
                    source_view.event_id: source_prob,
                    target_view.event_id: target_prob,
                },
                violation_magnitude=edge,
            )
        elif total < 1.0:
            # BUY arb: P(A) + P(B) < 1.0 means we can buy both and profit if one wins.
            # However, COMPLEMENT means they are mutually exclusive BUT they might not
            # cover the entire probability space. If they DO (full complement), then
            # P(A) + P(B) SHOULD be 1.0. If P(A) + P(B) < 1.0, it's a BUY arb.
            edge = 1.0 - total
            return CorrelationViolation(
                relation=rel,
                expected_constraint=(
                    f"P({source_view.event_id[:10]}) + P({target_view.event_id[:10]}) >= 1.0"
                ),
                actual_values={
                    source_view.event_id: source_prob,
                    target_view.event_id: target_prob,
                },
                violation_magnitude=edge,
            )

    return None


def _build_event_quote_view(
    event: Event,
    books: dict[str, OrderBook],
    aggregation: str = "liquidity_weighted",
    max_markets: int = 5,
    min_market_volume: float = 0.0,
    min_book_depth: float = 0.0,
) -> EventQuoteView | None:
    """
    Build a robust event-level implied probability view by aggregating across
    liquid markets instead of using a single first market.
    """
    candidates: list[tuple[float, float, OrderBook]] = []
    for m in event.markets:
        if not m.active:
            continue
        if m.volume < min_market_volume:
            continue
        book = books.get(m.yes_token_id)
        if not book or not book.best_ask:
            continue
        if book.best_ask.size < min_book_depth:
            continue
        # liquidity score: prefer deeper + higher-volume markets
        bid_size = book.best_bid.size if book.best_bid else 0.0
        ask_size = book.best_ask.size
        depth = max(0.0, bid_size + ask_size)
        liquidity = max(1.0, depth + m.volume * 0.01)
        candidates.append((book.best_ask.price, liquidity, book))

    if not candidates:
        return None

    candidates.sort(key=lambda x: x[1], reverse=True)
    selected = candidates[: max(1, max_markets)]
    prices = [price for price, _, _ in selected]
    liquidities = [liq for _, liq, _ in selected]

    if aggregation == "median":
        implied = float(median(prices))
    elif aggregation == "top_liquidity":
        implied = float(selected[0][0])
    else:
        # Default: liquidity-weighted average
        total_w = sum(liquidities)
        implied = (
            sum(price * liq for price, liq, _ in selected) / total_w
            if total_w > 0
            else float(sum(prices) / len(prices))
        )

    representative = selected[0][2]
    return EventQuoteView(
        event_id=event.event_id,
        implied_prob=max(0.0, min(1.0, implied)),
        representative_book=representative,
        representative_market_volume=max(liquidities),
        sample_count=len(selected),
        aggregation=aggregation,
    )


# ---------------------------------------------------------------------------
# Violation → Opportunity conversion
# ---------------------------------------------------------------------------


def _violation_direction(
    rel: MarketRelation,
    source_view: EventQuoteView,
    target_view: EventQuoteView,
) -> str:
    """Stable direction key for persistence gating across cycles."""
    if rel.relation_type == RelationType.COMPLEMENT:
        total = source_view.implied_prob + target_view.implied_prob
        return "buy" if total < 1.0 else "sell"
    if rel.relation_type == RelationType.PARENT_CHILD:
        return "parent_lt_child"
    if rel.relation_type == RelationType.TEMPORAL:
        return "earlier_gt_later"
    return "unknown"


def _cap_size_by_capital(
    size: float,
    required_capital_per_set: float,
    max_capital_per_opportunity: float,
) -> float:
    """Limit set size so required capital stays within configured cap."""
    if size <= 0 or required_capital_per_set <= 0:
        return 0.0
    max_sets = max_capital_per_opportunity / required_capital_per_set
    return min(size, max_sets)


def _market_for_yes_token(event: Event, yes_token_id: str) -> Market | None:
    """Find event market by YES token identifier."""
    for market in event.markets:
        if market.yes_token_id == yes_token_id:
            return market
    return None


def _violation_to_opportunity(
    violation: CorrelationViolation,
    event_map: dict[str, Event],
    event_views: dict[str, EventQuoteView],
    books: dict[str, OrderBook],
    gas_cost_usd: float,
    max_theoretical_roi_pct: float,
    max_capital_per_opportunity: float,
) -> Opportunity | None:
    """Convert a violation into a tradeable opportunity."""
    rel = violation.relation
    source = event_map.get(rel.source_event_id)
    target = event_map.get(rel.target_event_id)
    source_view = event_views.get(rel.source_event_id)
    target_view = event_views.get(rel.target_event_id)
    if not source or not target:
        return None
    if source_view is None or target_view is None:
        return None

    if rel.relation_type == RelationType.PARENT_CHILD:
        return _parent_child_opportunity(
            source,
            target,
            source_view,
            target_view,
            violation,
            books,
            gas_cost_usd,
            max_theoretical_roi_pct=max_theoretical_roi_pct,
            max_capital_per_opportunity=max_capital_per_opportunity,
        )
    elif rel.relation_type == RelationType.TEMPORAL:
        return _temporal_opportunity(
            source,
            target,
            source_view,
            target_view,
            violation,
            books,
            gas_cost_usd,
            max_theoretical_roi_pct=max_theoretical_roi_pct,
            max_capital_per_opportunity=max_capital_per_opportunity,
        )
    elif rel.relation_type == RelationType.COMPLEMENT:
        return _complement_opportunity(
            source,
            target,
            source_view,
            target_view,
            violation,
            gas_cost_usd,
            max_theoretical_roi_pct=max_theoretical_roi_pct,
            max_capital_per_opportunity=max_capital_per_opportunity,
        )
    return None


def _complement_opportunity(
    source: Event,
    target: Event,
    source_view: EventQuoteView,
    target_view: EventQuoteView,
    violation: CorrelationViolation,
    gas_cost_usd: float,
    *,
    max_theoretical_roi_pct: float,
    max_capital_per_opportunity: float,
) -> Opportunity | None:
    """
    Handle complement violations (P(A) + P(B) != 1.0).
    If sum < 1.0 -> BUY both (actionable).
    If sum > 1.0 -> SELL both (requires inventory).
    """
    source_book = source_view.representative_book
    target_book = target_view.representative_book

    total_prob = source_view.implied_prob + target_view.implied_prob

    if total_prob < 1.0:
        # BUY arb
        if not source_book.best_ask or not target_book.best_ask:
            return None
        size = min(source_book.best_ask.size, target_book.best_ask.size)
        side = Side.BUY
        price_s = source_book.best_ask.price
        price_t = target_book.best_ask.price
        payout = 1.0
        required_capital_per_set = price_s + price_t
    else:
        # SELL arb
        if not source_book.best_bid or not target_book.best_bid:
            return None
        size = min(source_book.best_bid.size, target_book.best_bid.size)
        side = Side.SELL
        price_s = source_book.best_bid.price
        price_t = target_book.best_bid.price
        payout = 1.0  # Selling both for > $1.0
        required_capital_per_set = payout

    size = _cap_size_by_capital(size, required_capital_per_set, max_capital_per_opportunity)

    if size <= 0:
        return None

    legs = (
        LegOrder(token_id=source_book.token_id, side=side, price=price_s, size=size),
        LegOrder(token_id=target_book.token_id, side=side, price=price_t, size=size),
    )

    total_gas = gas_cost_usd * len(legs)
    if side == Side.BUY:
        cost = (price_s + price_t) * size
        net_profit = (1.0 - (price_s + price_t)) * size - total_gas
        required_capital = cost
    else:
        proceeds = (price_s + price_t) * size
        net_profit = (proceeds - (1.0 * size)) - total_gas
        required_capital = 1.0 * size

    if net_profit <= 0:
        return None

    roi_pct = (net_profit / required_capital * 100) if required_capital > 0 else 0.0
    if roi_pct > max_theoretical_roi_pct:
        return None

    risk_flags = ("sell_inventory_required",) if side == Side.SELL else ()

    return Opportunity(
        type=OpportunityType.CORRELATION_ARB,
        event_id=violation.relation.source_event_id,
        legs=legs,
        expected_profit_per_set=violation.violation_magnitude,
        net_profit_per_set=violation.violation_magnitude,
        max_sets=size,
        gross_profit=violation.violation_magnitude * size,
        estimated_gas_cost=total_gas,
        net_profit=net_profit,
        roi_pct=roi_pct,
        required_capital=required_capital,
        reason_code=f"corr_complement_{side.value.lower()}_{source_view.aggregation}",
        risk_flags=risk_flags,
    )


def _parent_child_opportunity(
    source: Event,
    target: Event,
    source_view: EventQuoteView,
    target_view: EventQuoteView,
    violation: CorrelationViolation,
    books: dict[str, OrderBook],
    gas_cost_usd: float,
    *,
    max_theoretical_roi_pct: float,
    max_capital_per_opportunity: float,
) -> Opportunity | None:
    """Prefer long-only parent/child structure; fallback to inventory-dependent hedge."""
    source_book = source_view.representative_book
    target_book = target_view.representative_book
    if source_view.sample_count <= 0 or target_view.sample_count <= 0:
        return None

    # Long-only conversion:
    # P(parent) >= P(child)  <=>  P(parent) + P(not child) >= 1
    # If violated, buy parent YES + child NO when tradable.
    source_market = _market_for_yes_token(source, source_book.token_id)
    target_market = _market_for_yes_token(target, target_book.token_id)
    source_ask = source_book.best_ask
    target_no_book = books.get(target_market.no_token_id) if target_market else None
    target_no_ask = target_no_book.best_ask if target_no_book else None
    if source_ask and target_no_ask:
        size = min(source_ask.size, target_no_ask.size)
        required_capital_per_set = source_ask.price + target_no_ask.price
        size = _cap_size_by_capital(
            size,
            required_capital_per_set,
            max_capital_per_opportunity,
        )
        if size > 0 and required_capital_per_set < 1.0:
            legs = (
                LegOrder(
                    token_id=source_book.token_id,
                    side=Side.BUY,
                    price=source_ask.price,
                    size=size,
                ),
                LegOrder(
                    token_id=target_no_book.token_id,
                    side=Side.BUY,
                    price=target_no_ask.price,
                    size=size,
                ),
            )
            total_gas = gas_cost_usd * len(legs)
            net_profit_per_set = 1.0 - required_capital_per_set
            net_profit = net_profit_per_set * size - total_gas
            if net_profit > 0:
                required_capital = required_capital_per_set * size
                roi_pct = (net_profit / required_capital * 100) if required_capital > 0 else 0.0
                if roi_pct <= max_theoretical_roi_pct:
                    return Opportunity(
                        type=OpportunityType.CORRELATION_ARB,
                        event_id=violation.relation.source_event_id,
                        legs=legs,
                        expected_profit_per_set=net_profit_per_set,
                        net_profit_per_set=net_profit_per_set,
                        max_sets=size,
                        gross_profit=net_profit_per_set * size,
                        estimated_gas_cost=total_gas,
                        net_profit=net_profit,
                        roi_pct=roi_pct,
                        required_capital=required_capital,
                        reason_code=f"corr_parent_child_buy_{source_view.aggregation}",
                    )

    if not source_book.best_ask or not target_book.best_bid:
        return None

    size = min(source_book.best_ask.size, target_book.best_bid.size)
    size = _cap_size_by_capital(
        size,
        source_book.best_ask.price,
        max_capital_per_opportunity,
    )
    if size <= 0:
        return None

    legs = (
        LegOrder(
            token_id=source_book.token_id,
            side=Side.BUY,
            price=source_book.best_ask.price,
            size=size,
        ),
        LegOrder(
            token_id=target_book.token_id,
            side=Side.SELL,
            price=target_book.best_bid.price,
            size=size,
        ),
    )

    required_capital = source_book.best_ask.price * size
    total_gas = gas_cost_usd * len(legs)
    net_profit = violation.violation_magnitude * size - total_gas

    if net_profit <= 0:
        return None

    roi_pct = (net_profit / required_capital * 100) if required_capital > 0 else 0.0
    if roi_pct > max_theoretical_roi_pct:
        return None

    return Opportunity(
        type=OpportunityType.CORRELATION_ARB,
        event_id=violation.relation.source_event_id,
        legs=legs,
        expected_profit_per_set=violation.violation_magnitude,
        net_profit_per_set=violation.violation_magnitude,
        max_sets=size,
        gross_profit=violation.violation_magnitude * size,
        estimated_gas_cost=total_gas,
        net_profit=net_profit,
        roi_pct=roi_pct,
        required_capital=required_capital,
        reason_code=f"corr_parent_child_{source_view.aggregation}",
        risk_flags=("sell_inventory_required",),
    )


def _temporal_opportunity(
    source: Event,
    target: Event,
    source_view: EventQuoteView,
    target_view: EventQuoteView,
    violation: CorrelationViolation,
    books: dict[str, OrderBook],
    gas_cost_usd: float,
    *,
    max_theoretical_roi_pct: float,
    max_capital_per_opportunity: float,
) -> Opportunity | None:
    """Prefer long-only temporal structure; fallback to inventory-dependent hedge."""
    source_book = source_view.representative_book
    target_book = target_view.representative_book
    if source_view.sample_count <= 0 or target_view.sample_count <= 0:
        return None

    # Long-only conversion:
    # P(earlier) <= P(later)  <=>  P(not earlier) + P(later) >= 1
    # If violated, buy earlier NO + later YES when tradable.
    source_market = _market_for_yes_token(source, source_book.token_id)
    target_ask = target_book.best_ask
    source_no_book = books.get(source_market.no_token_id) if source_market else None
    source_no_ask = source_no_book.best_ask if source_no_book else None
    if source_no_ask and target_ask:
        size = min(source_no_ask.size, target_ask.size)
        required_capital_per_set = source_no_ask.price + target_ask.price
        size = _cap_size_by_capital(
            size,
            required_capital_per_set,
            max_capital_per_opportunity,
        )
        if size > 0 and required_capital_per_set < 1.0:
            legs = (
                LegOrder(
                    token_id=source_no_book.token_id,
                    side=Side.BUY,
                    price=source_no_ask.price,
                    size=size,
                ),
                LegOrder(
                    token_id=target_book.token_id,
                    side=Side.BUY,
                    price=target_ask.price,
                    size=size,
                ),
            )
            total_gas = gas_cost_usd * len(legs)
            net_profit_per_set = 1.0 - required_capital_per_set
            net_profit = net_profit_per_set * size - total_gas
            if net_profit > 0:
                required_capital = required_capital_per_set * size
                roi_pct = (net_profit / required_capital * 100) if required_capital > 0 else 0.0
                if roi_pct <= max_theoretical_roi_pct:
                    return Opportunity(
                        type=OpportunityType.CORRELATION_ARB,
                        event_id=violation.relation.source_event_id,
                        legs=legs,
                        expected_profit_per_set=net_profit_per_set,
                        net_profit_per_set=net_profit_per_set,
                        max_sets=size,
                        gross_profit=net_profit_per_set * size,
                        estimated_gas_cost=total_gas,
                        net_profit=net_profit,
                        roi_pct=roi_pct,
                        required_capital=required_capital,
                        reason_code=f"corr_temporal_buy_{source_view.aggregation}",
                    )

    if not source_book.best_bid or not target_book.best_ask:
        return None

    size = min(source_book.best_bid.size, target_book.best_ask.size)
    size = _cap_size_by_capital(
        size,
        target_book.best_ask.price,
        max_capital_per_opportunity,
    )
    if size <= 0:
        return None

    legs = (
        LegOrder(
            token_id=source_book.token_id,
            side=Side.SELL,
            price=source_book.best_bid.price,
            size=size,
        ),
        LegOrder(
            token_id=target_book.token_id,
            side=Side.BUY,
            price=target_book.best_ask.price,
            size=size,
        ),
    )

    required_capital = target_book.best_ask.price * size
    total_gas = gas_cost_usd * len(legs)
    net_profit = violation.violation_magnitude * size - total_gas

    if net_profit <= 0:
        return None

    roi_pct = (net_profit / required_capital * 100) if required_capital > 0 else 0.0
    if roi_pct > max_theoretical_roi_pct:
        return None

    return Opportunity(
        type=OpportunityType.CORRELATION_ARB,
        event_id=violation.relation.source_event_id,
        legs=legs,
        expected_profit_per_set=violation.violation_magnitude,
        net_profit_per_set=violation.violation_magnitude,
        max_sets=size,
        gross_profit=violation.violation_magnitude * size,
        estimated_gas_cost=total_gas,
        net_profit=net_profit,
        roi_pct=roi_pct,
        required_capital=required_capital,
        reason_code=f"corr_temporal_{source_view.aggregation}",
        risk_flags=("sell_inventory_required",),
    )
