"""
Cross-platform event matching. Maps events between Polymarket and Kalshi.

Three matching strategies (in priority order):
1. Manual JSON map (confidence = 1.0)
2. Verified fuzzy matches from verified_matches.json (confidence preserved)
3. Fuzzy text matching via rapidfuzz (logged but NOT auto-traded; confidence set to 0)

Settlement mismatch is the #1 risk in cross-platform arb, so we require
high confidence (configurable, default 90%) before allowing execution.
"""

from __future__ import annotations

import json
import logging
import re
from dataclasses import dataclass
from pathlib import Path

from rapidfuzz import fuzz

from scanner.models import Event, Market
from client.kalshi import KalshiMarket

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


@dataclass(frozen=True)
class MatchedEvent:
    """A matched event across Polymarket and Kalshi."""
    pm_event_id: str
    kalshi_event_ticker: str
    pm_markets: tuple[Market, ...]
    kalshi_tickers: tuple[str, ...]
    confidence: float  # 0.0 - 1.0
    match_method: str  # "manual", "verified", or "fuzzy"


def _year_mismatch(pm_title: str, kalshi_title: str) -> bool:
    """Reject matches where the year differs (e.g., 2024 vs 2028)."""
    pm_years = set(_YEAR_PATTERN.findall(pm_title))
    kalshi_years = set(_YEAR_PATTERN.findall(kalshi_title))
    if pm_years and kalshi_years and pm_years != kalshi_years:
        return True
    return False


def _settlement_mismatch_risk(pm_title: str, kalshi_title: str) -> bool:
    """Check if titles differ on settlement-relevant keywords."""
    pm_lower = pm_title.lower()
    kalshi_lower = kalshi_title.lower()
    for kw in _SETTLEMENT_KEYWORDS:
        pm_has = kw in pm_lower
        kalshi_has = kw in kalshi_lower
        if pm_has != kalshi_has:
            return True
    return False


class EventMatcher:
    """
    Matches events between Polymarket and Kalshi using manual mappings,
    verified fuzzy matches, and new fuzzy text matching.
    """

    def __init__(
        self,
        manual_map_path: str = "cross_platform_map.json",
        fuzzy_threshold: float = FUZZY_THRESHOLD,
        verified_path: str = "verified_matches.json",
    ) -> None:
        self._manual_map = self._load_json_map(manual_map_path)
        self._fuzzy_threshold = fuzzy_threshold
        self._verified = self._load_json_map(verified_path)

    @staticmethod
    def _load_json_map(path: str) -> dict[str, str]:
        """
        Load a JSON mapping file.
        Format: {"pm_event_id": "kalshi_event_ticker", ...}
        """
        p = Path(path)
        if not p.exists():
            logger.info("No map file at %s, using empty map", path)
            return {}
        raw = json.loads(p.read_text())
        if not isinstance(raw, dict):
            raise ValueError(f"Map file must be a JSON object, got {type(raw).__name__}")
        return raw

    def match_events(
        self,
        pm_events: list[Event],
        kalshi_markets: list[KalshiMarket],
    ) -> list[MatchedEvent]:
        """
        Match Polymarket events to Kalshi markets.

        Returns list of MatchedEvent sorted by confidence descending.
        Only manual and verified matches are tradeable (confidence > 0).
        """
        # Build Kalshi event index: event_ticker -> list of market tickers
        kalshi_by_event: dict[str, list[KalshiMarket]] = {}
        for km in kalshi_markets:
            kalshi_by_event.setdefault(km.event_ticker, []).append(km)

        # Build Kalshi title index for fuzzy matching
        kalshi_titles: dict[str, str] = {}
        for event_ticker, kms in kalshi_by_event.items():
            kalshi_titles[event_ticker] = kms[0].title

        matches: list[MatchedEvent] = []

        for pm_event in pm_events:
            # Strategy 1: Manual mapping (highest confidence)
            kalshi_ticker = self._manual_map.get(pm_event.event_id)
            if kalshi_ticker and kalshi_ticker in kalshi_by_event:
                kms = kalshi_by_event[kalshi_ticker]
                match = MatchedEvent(
                    pm_event_id=pm_event.event_id,
                    kalshi_event_ticker=kalshi_ticker,
                    pm_markets=pm_event.markets,
                    kalshi_tickers=tuple(km.ticker for km in kms),
                    confidence=1.0,
                    match_method="manual",
                )
                matches.append(match)
                logger.info(
                    "Manual match: PM %s -> Kalshi %s (%d markets)",
                    pm_event.event_id, kalshi_ticker, len(kms),
                )
                continue

            # Strategy 2: Verified mapping (pinned event ticker, score only for confidence)
            verified_ticker = self._verified.get(pm_event.event_id)
            if verified_ticker:
                kms = kalshi_by_event.get(verified_ticker)
                if not kms:
                    logger.warning(
                        "Verified mapping missing on Kalshi: PM %s -> %s",
                        pm_event.event_id, verified_ticker,
                    )
                    continue

                kalshi_title = kms[0].title
                if _year_mismatch(pm_event.title, kalshi_title):
                    logger.info(
                        "Verified match REJECTED (year mismatch): PM '%s' vs Kalshi '%s'",
                        pm_event.title[:50], kalshi_title[:50],
                    )
                    continue

                if _settlement_mismatch_risk(pm_event.title, kalshi_title):
                    logger.info(
                        "Verified match REJECTED (settlement keyword): PM '%s' vs Kalshi '%s'",
                        pm_event.title[:50], kalshi_title[:50],
                    )
                    continue

                best_score = fuzz.token_set_ratio(pm_event.title, kalshi_title)
                if best_score < self._fuzzy_threshold:
                    logger.info(
                        "Verified match below threshold: PM '%s' -> Kalshi '%s' (score=%.1f%% < %.1f%%)",
                        pm_event.title[:50], kalshi_title[:50], best_score, self._fuzzy_threshold,
                    )
                    continue

                matches.append(MatchedEvent(
                    pm_event_id=pm_event.event_id,
                    kalshi_event_ticker=verified_ticker,
                    pm_markets=pm_event.markets,
                    kalshi_tickers=tuple(km.ticker for km in kms),
                    confidence=best_score / 100.0,
                    match_method="verified",
                ))
                logger.info(
                    "Verified match: PM '%s' -> Kalshi '%s' (score=%.1f%%)",
                    pm_event.title[:50], kalshi_title[:50], best_score,
                )
                continue

            # Strategy 2+3: Fuzzy text matching
            best_score = 0.0
            best_kalshi_ticker = ""
            for kt, kalshi_title in kalshi_titles.items():
                score = fuzz.token_set_ratio(pm_event.title, kalshi_title)
                if score > best_score:
                    best_score = score
                    best_kalshi_ticker = kt

            if best_score < self._fuzzy_threshold or not best_kalshi_ticker:
                continue

            kalshi_title = kalshi_titles[best_kalshi_ticker]

            # Safety checks: reject matches with year or settlement mismatches
            if _year_mismatch(pm_event.title, kalshi_title):
                logger.info(
                    "Fuzzy match REJECTED (year mismatch): PM '%s' vs Kalshi '%s'",
                    pm_event.title[:50], kalshi_title[:50],
                )
                continue

            if _settlement_mismatch_risk(pm_event.title, kalshi_title):
                logger.info(
                    "Fuzzy match REJECTED (settlement keyword): PM '%s' vs Kalshi '%s'",
                    pm_event.title[:50], kalshi_title[:50],
                )
                continue

            kms = kalshi_by_event[best_kalshi_ticker]

            # Strategy 3: New unverified fuzzy match — log but block trading
            logger.warning(
                "UNVERIFIED fuzzy match: PM '%s' -> Kalshi '%s' (score=%.1f%%). "
                "Add to verified_matches.json to enable trading.",
                pm_event.title[:50], kalshi_title[:50], best_score,
            )
            match = MatchedEvent(
                pm_event_id=pm_event.event_id,
                kalshi_event_ticker=best_kalshi_ticker,
                pm_markets=pm_event.markets,
                kalshi_tickers=tuple(km.ticker for km in kms),
                confidence=0.0,  # Blocked from execution by confidence filter
                match_method="fuzzy",
            )
            matches.append(match)

        # Sort by confidence descending
        matches.sort(key=lambda m: m.confidence, reverse=True)
        return matches
