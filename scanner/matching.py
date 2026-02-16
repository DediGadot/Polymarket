"""
Cross-platform event and contract matching. Maps events between Polymarket and external platforms.

Three matching strategies (in priority order):
1. Manual JSON map (confidence = 1.0)
2. Verified fuzzy matches from verified_matches.json (confidence preserved)
3. Fuzzy text matching via rapidfuzz (logged but NOT auto-traded; confidence set to 0)

Settlement mismatch is the #1 risk in cross-platform arb, so we require
high confidence (configurable, default 90%) before allowing execution.

After event-level matching, contract-level matching validates settlement equivalence:
- Detects numeric threshold mismatches (e.g., "Over 2.5" != "Over 3.5")
- Filters low-confidence contract matches before trading

Generalized for N platforms: each MatchedEvent contains PlatformMatch entries
for every external platform that matches a PM event. Backward-compat properties
preserved for existing Kalshi-specific code paths.
"""

from __future__ import annotations

import json
import logging
import re
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

from rapidfuzz import fuzz

from scanner.models import Event, Market

logger = logging.getLogger(__name__)

# Fuzzy matching threshold: below this score, no match
FUZZY_THRESHOLD = 95.0

# Year pattern for detecting year mismatches (e.g., 2024 vs 2028)
_YEAR_PATTERN = re.compile(r"\b(20\d{2})\b")

# Keywords that indicate different settlement criteria
_SETTLEMENT_KEYWORDS = frozenset({
    "popular vote",
    "electoral",
    "inauguration",
    "sworn in",
    "resign",
    "impeach",
    "conviction",
    "indictment",
    "before",
    "by end of",
    "by march",
    "by june",
    "by december",
    "first term",
    "second term",
})

# Keywords that indicate numeric thresholds (for contract-level matching)
_SETTLEMENT_THRESHOLD_KEYWORDS = frozenset({
    "over",
    "under",
    "above",
    "below",
    "exceeds",
    "goals",
    "points",
    "wins",
    "$",  # Dollar amounts with numbers
})


@dataclass(frozen=True)
class PlatformMatch:
    """A single platform's match to a PM event."""
    platform: str  # "kalshi", "fanatics", etc.
    event_ticker: str
    tickers: tuple[str, ...]
    confidence: float  # 0.0 - 1.0
    match_method: str  # "manual", "verified", or "fuzzy"


@dataclass(frozen=True)
class MatchedEvent:
    """A matched event across Polymarket and one or more external platforms."""
    pm_event_id: str
    pm_markets: tuple[Market, ...]
    platform_matches: tuple[PlatformMatch, ...]

    # Backward-compat: first platform match (Kalshi) accessors
    @property
    def kalshi_event_ticker(self) -> str:
        for m in self.platform_matches:
            if m.platform == "kalshi":
                return m.event_ticker
        return self.platform_matches[0].event_ticker if self.platform_matches else ""

    @property
    def kalshi_tickers(self) -> tuple[str, ...]:
        for m in self.platform_matches:
            if m.platform == "kalshi":
                return m.tickers
        return self.platform_matches[0].tickers if self.platform_matches else ()

    @property
    def confidence(self) -> float:
        """Max confidence across all platform matches."""
        if not self.platform_matches:
            return 0.0
        return max(m.confidence for m in self.platform_matches)

    @property
    def match_method(self) -> str:
        """Method of highest-confidence match."""
        if not self.platform_matches:
            return ""
        best = max(self.platform_matches, key=lambda m: m.confidence)
        return best.match_method


@dataclass(frozen=True)
class ContractMatch:
    """A matched contract (market) between Polymarket and an external platform."""
    pm_market: Market
    ext_market: object  # KalshiMarket or similar (has .ticker, .title attrs)
    confidence: float  # 0.0 - 1.0
    match_method: str  # "manual", "verified", or "fuzzy"


def _year_mismatch(pm_title: str, other_title: str) -> bool:
    """Reject matches where the year differs (e.g., 2024 vs 2028)."""
    pm_years = set(_YEAR_PATTERN.findall(pm_title))
    other_years = set(_YEAR_PATTERN.findall(other_title))
    if pm_years and other_years and pm_years != other_years:
        return True
    return False


def _settlement_mismatch_risk(pm_title: str, other_title: str) -> bool:
    """Check if titles differ on settlement-relevant keywords."""
    pm_lower = pm_title.lower()
    other_lower = other_title.lower()
    for kw in _SETTLEMENT_KEYWORDS:
        pm_has = kw in pm_lower
        other_has = kw in other_lower
        if pm_has != other_has:
            return True
    return False


def _numeric_threshold_mismatch(title1: str, title2: str) -> bool:
    """
    Check if titles differ on numeric thresholds.

    E.g., "Over 2.5 goals" vs "Over 3.5 goals" should mismatch.
    Uses fuzzy text matching for numbers but requires exact match on threshold values.
    """
    import re

    # Pattern for numeric thresholds (including decimals like 2.5, 3.5, and dollar amounts)
    number_pattern = re.compile(r'\$?(\d+\.?\d*)')

    def extract_thresholds(s: str) -> set[str]:
        """Extract numeric thresholds that follow threshold keywords."""
        s_lower = s.lower()
        thresholds = set()

        # Find all numbers
        for match in number_pattern.finditer(s):
            num = match.group(1)
            start, end = match.span()

            # Check if this number follows a threshold keyword within ~20 chars
            before = s_lower[max(0, start - 20):start]
            if any(kw in before for kw in ["over", "under", "above", "below", "exceeds"]):
                thresholds.add(num)

        return thresholds

    thresholds1 = extract_thresholds(title1)
    thresholds2 = extract_thresholds(title2)

    # If both have thresholds but they differ, it's a mismatch
    if thresholds1 and thresholds2 and thresholds1 != thresholds2:
        return True

    return False


class EventMatcher:
    """
    Matches events between Polymarket and external platforms using manual mappings,
    verified fuzzy matches, and new fuzzy text matching.

    Generalized: accepts platform_markets dict keyed by platform name.
    Each value is a list of market objects with .event_ticker, .ticker, .title attrs.
    """

    def __init__(
        self,
        manual_map_path: str = "cross_platform_map.json",
        fuzzy_threshold: float = FUZZY_THRESHOLD,
        verified_path: str = "verified_matches.json",
        negative_ttl_sec: float = 300.0,
    ) -> None:
        self._manual_map = self._load_json_map(manual_map_path)
        self._fuzzy_threshold = fuzzy_threshold
        self._verified = self._load_json_map(verified_path)
        self._negative_ttl_sec = max(0.0, float(negative_ttl_sec))
        self._negative_cache: dict[tuple[str, str], float] = {}

    @staticmethod
    def _load_json_map(path: str) -> dict:
        """
        Load a JSON mapping file.

        Backward-compatible format:
          {"pm_event_id": "kalshi_event_ticker", ...}  (old: string values = Kalshi)
        Multi-platform format:
          {"pm_event_id": {"kalshi": "ticker", "fanatics": "ticker"}, ...}
        """
        p = Path(path)
        if not p.exists():
            logger.info("No map file at %s, using empty map", path)
            return {}
        raw = json.loads(p.read_text())
        if not isinstance(raw, dict):
            raise ValueError(f"Map file must be a JSON object, got {type(raw).__name__}")
        return raw

    def _resolve_manual_map(self, pm_event_id: str, platform: str) -> str | None:
        """
        Look up manual mapping for a PM event -> platform event ticker.

        Handles both old format (string value = Kalshi) and new format (dict value).
        """
        val = self._manual_map.get(pm_event_id)
        if val is None:
            return None
        if isinstance(val, str):
            # Old format: string = Kalshi ticker
            return val if platform == "kalshi" else None
        if isinstance(val, dict):
            return val.get(platform)
        return None

    def _resolve_verified(self, pm_event_id: str, platform: str) -> str | None:
        """Look up verified mapping. Same format handling as manual map."""
        val = self._verified.get(pm_event_id)
        if val is None:
            return None
        if isinstance(val, str):
            return val if platform == "kalshi" else None
        if isinstance(val, dict):
            return val.get(platform)
        return None

    @staticmethod
    def _has_tradeable_mapping_for_platform(mapping: dict, platform: str) -> bool:
        """Return True if mapping dict has at least one entry for platform."""
        for value in mapping.values():
            if isinstance(value, str):
                if platform == "kalshi" and value:
                    return True
            elif isinstance(value, dict):
                mapped = value.get(platform)
                if isinstance(mapped, str) and mapped:
                    return True
        return False

    def has_tradeable_mappings(self, platforms: list[str] | None = None) -> bool:
        """
        True when manual/verified maps contain at least one tradeable mapping
        for any selected platform.
        """
        if platforms is None:
            platforms = ["kalshi", "fanatics"]
        for platform in platforms:
            if self._has_tradeable_mapping_for_platform(self._manual_map, platform):
                return True
            if self._has_tradeable_mapping_for_platform(self._verified, platform):
                return True
        return False

    def _negative_cache_hit(self, pm_event_id: str, platform: str) -> bool:
        """True when a recent non-tradeable fuzzy result is still cooling down."""
        if self._negative_ttl_sec <= 0:
            return False
        key = (pm_event_id, platform)
        expiry = self._negative_cache.get(key, 0.0)
        now = time.time()
        if expiry <= now:
            self._negative_cache.pop(key, None)
            return False
        return True

    def _remember_negative(self, pm_event_id: str, platform: str) -> None:
        """Cache a non-tradeable fuzzy outcome to avoid repeating heavy matching."""
        if self._negative_ttl_sec <= 0:
            return
        self._negative_cache[(pm_event_id, platform)] = time.time() + self._negative_ttl_sec

    def match_events(
        self,
        pm_events: list[Event],
        platform_markets: dict[str, list],
        *,
        include_fuzzy: bool = True,
        should_stop: Callable[[], bool] | None = None,
    ) -> list[MatchedEvent]:
        """
        Match Polymarket events to external platform markets.

        Args:
            pm_events: List of Polymarket Event objects.
            platform_markets: Dict of platform_name -> list of market objects.
                Each market must have .event_ticker, .ticker, .title attributes.

        Returns list of MatchedEvent sorted by confidence descending.
        Only manual and verified matches are tradeable (confidence > 0).
        """
        matches: list[MatchedEvent] = []

        # Build per-platform indexes
        platform_indexes: dict[str, tuple[dict[str, list], dict[str, str]]] = {}
        for platform_name, markets in platform_markets.items():
            by_event: dict[str, list] = {}
            for m in markets:
                by_event.setdefault(m.event_ticker, []).append(m)
            titles: dict[str, str] = {}
            for et, mks in by_event.items():
                titles[et] = mks[0].title
            platform_indexes[platform_name] = (by_event, titles)

        for pm_event in pm_events:
            if should_stop and should_stop():
                break
            platform_match_list: list[PlatformMatch] = []

            for platform_name, (by_event, titles) in platform_indexes.items():
                if should_stop and should_stop():
                    break
                pm_match = self._match_single_platform(
                    pm_event,
                    platform_name,
                    by_event,
                    titles,
                    include_fuzzy=include_fuzzy,
                    should_stop=should_stop,
                )
                if pm_match is not None:
                    platform_match_list.append(pm_match)

            if platform_match_list:
                matches.append(MatchedEvent(
                    pm_event_id=pm_event.event_id,
                    pm_markets=pm_event.markets,
                    platform_matches=tuple(platform_match_list),
                ))

        # Sort by max confidence descending
        matches.sort(key=lambda m: m.confidence, reverse=True)
        return matches

    def _match_single_platform(
        self,
        pm_event: Event,
        platform: str,
        by_event: dict[str, list],
        titles: dict[str, str],
        *,
        include_fuzzy: bool,
        should_stop: Callable[[], bool] | None,
    ) -> PlatformMatch | None:
        """Try to match a PM event to one external platform. Returns PlatformMatch or None."""

        # Strategy 1: Manual mapping (highest confidence)
        manual_ticker = self._resolve_manual_map(pm_event.event_id, platform)
        if manual_ticker and manual_ticker in by_event:
            mks = by_event[manual_ticker]
            logger.info(
                "Manual match: PM %s -> %s %s (%d markets)",
                pm_event.event_id, platform, manual_ticker, len(mks),
            )
            return PlatformMatch(
                platform=platform,
                event_ticker=manual_ticker,
                tickers=tuple(m.ticker for m in mks),
                confidence=1.0,
                match_method="manual",
            )

        # Strategy 2: Verified mapping
        verified_ticker = self._resolve_verified(pm_event.event_id, platform)
        if verified_ticker:
            mks = by_event.get(verified_ticker)
            if not mks:
                logger.warning(
                    "Verified mapping missing on %s: PM %s -> %s",
                    platform, pm_event.event_id, verified_ticker,
                )
                return None

            ext_title = mks[0].title
            if _year_mismatch(pm_event.title, ext_title):
                logger.info(
                    "Verified match REJECTED (year mismatch): PM '%s' vs %s '%s'",
                    pm_event.title[:50], platform, ext_title[:50],
                )
                return None

            if _settlement_mismatch_risk(pm_event.title, ext_title):
                logger.info(
                    "Verified match REJECTED (settlement keyword): PM '%s' vs %s '%s'",
                    pm_event.title[:50], platform, ext_title[:50],
                )
                return None

            best_score = fuzz.token_set_ratio(pm_event.title, ext_title)
            if best_score < self._fuzzy_threshold:
                logger.info(
                    "Verified match below threshold: PM '%s' -> %s '%s' (score=%.1f%% < %.1f%%)",
                    pm_event.title[:50], platform, ext_title[:50], best_score, self._fuzzy_threshold,
                )
                return None

            logger.info(
                "Verified match: PM '%s' -> %s '%s' (score=%.1f%%)",
                pm_event.title[:50], platform, ext_title[:50], best_score,
            )
            return PlatformMatch(
                platform=platform,
                event_ticker=verified_ticker,
                tickers=tuple(m.ticker for m in mks),
                confidence=best_score / 100.0,
                match_method="verified",
            )

        # Strategy 3: Fuzzy text matching
        if not include_fuzzy:
            return None
        if self._negative_cache_hit(pm_event.event_id, platform):
            return None

        best_score = 0.0
        best_ticker = ""
        for et, ext_title in titles.items():
            if should_stop and should_stop():
                return None
            score = fuzz.token_set_ratio(pm_event.title, ext_title)
            if score > best_score:
                best_score = score
                best_ticker = et

        if best_score < self._fuzzy_threshold or not best_ticker:
            self._remember_negative(pm_event.event_id, platform)
            return None

        ext_title = titles[best_ticker]

        if _year_mismatch(pm_event.title, ext_title):
            logger.info(
                "Fuzzy match REJECTED (year mismatch): PM '%s' vs %s '%s'",
                pm_event.title[:50], platform, ext_title[:50],
            )
            self._remember_negative(pm_event.event_id, platform)
            return None

        if _settlement_mismatch_risk(pm_event.title, ext_title):
            logger.info(
                "Fuzzy match REJECTED (settlement keyword): PM '%s' vs %s '%s'",
                pm_event.title[:50], platform, ext_title[:50],
            )
            self._remember_negative(pm_event.event_id, platform)
            return None

        mks = by_event[best_ticker]

        logger.warning(
            "UNVERIFIED fuzzy match: PM '%s' -> %s '%s' (score=%.1f%%). "
            "Add to verified_matches.json to enable trading.",
            pm_event.title[:50], platform, ext_title[:50], best_score,
        )
        self._remember_negative(pm_event.event_id, platform)
        return PlatformMatch(
            platform=platform,
            event_ticker=best_ticker,
            tickers=tuple(m.ticker for m in mks),
            confidence=0.0,  # Blocked from execution by confidence filter
            match_method="fuzzy",
        )


def match_contracts(
    pm_markets: tuple[Market, ...],
    ext_markets: list,
) -> list[ContractMatch]:
    """
    Match individual contracts (markets) after event-level matching.

    Performs a second pass of fuzzy matching on contract titles to detect
    settlement-equivalence issues like "Over 2.5" vs "Over 3.5".

    Args:
        pm_markets: Polymarket markets in the matched event
        ext_markets: External platform markets in the matched event

    Returns:
        List of ContractMatch with confidence scores. Low-confidence matches
        should be filtered out before trading.
    """
    from rapidfuzz import fuzz

    matches: list[ContractMatch] = []

    for pm_market in pm_markets:
        best_score = 0.0
        best_ext_market = None

        for ext_market in ext_markets:
            # Check for numeric threshold mismatch first
            if _numeric_threshold_mismatch(pm_market.question, ext_market.title):
                logger.info(
                    "Contract-level mismatch (threshold): PM '%s' vs ext '%s'",
                    pm_market.question[:50], ext_market.title[:50],
                )
                continue

            # Event-level settlement check still applies
            if _settlement_mismatch_risk(pm_market.question, ext_market.title):
                logger.info(
                    "Contract-level mismatch (settlement keyword): PM '%s' vs ext '%s'",
                    pm_market.question[:50], ext_market.title[:50],
                )
                continue

            # Fuzzy match score
            score = fuzz.token_set_ratio(pm_market.question, ext_market.title)
            if score > best_score:
                best_score = score
                best_ext_market = ext_market

        if best_ext_market and best_score >= FUZZY_THRESHOLD:
            matches.append(ContractMatch(
                pm_market=pm_market,
                ext_market=best_ext_market,
                confidence=best_score / 100.0,
                match_method="fuzzy",
            ))
        elif best_ext_market:
            # Below threshold but log it
            logger.warning(
                "Contract match below threshold: PM '%s' vs ext '%s' (score=%.1f%% < %.1f%%)",
                pm_market.question[:50], best_ext_market.title[:50],
                best_score, FUZZY_THRESHOLD,
            )

    return matches


def filter_by_confidence(
    contract_matches: list[ContractMatch],
    min_confidence: float = 0.90,
) -> list[ContractMatch]:
    """
    Filter contract matches by minimum confidence threshold.

    Args:
        contract_matches: List of ContractMatch to filter
        min_confidence: Minimum confidence (0.0 - 1.0) to allow

    Returns:
        Filtered list of ContractMatch with confidence >= min_confidence
    """
    return [cm for cm in contract_matches if cm.confidence >= min_confidence]
