"""
Unit tests for scanner/matching.py -- cross-platform event matching.
"""

import json
import tempfile

from scanner.matching import (
    EventMatcher,
    _year_mismatch,
    _settlement_mismatch_risk,
    FUZZY_THRESHOLD,
)
from scanner.models import Market, Event
from client.kalshi import KalshiMarket


def _pm_market(cid: str, yes_id: str, no_id: str, eid: str = "e1") -> Market:
    return Market(
        condition_id=cid,
        question=f"Market {cid}?",
        yes_token_id=yes_id,
        no_token_id=no_id,
        neg_risk=False,
        event_id=eid,
        min_tick_size="0.01",
        active=True,
    )


def _pm_event(eid: str, title: str, markets: list[Market] | None = None) -> Event:
    if markets is None:
        markets = [_pm_market("c1", "y1", "n1", eid)]
    return Event(
        event_id=eid,
        title=title,
        markets=tuple(markets),
        neg_risk=False,
    )


def _kalshi_market(ticker: str, event_ticker: str, title: str) -> KalshiMarket:
    return KalshiMarket(
        ticker=ticker,
        event_ticker=event_ticker,
        title=title,
        subtitle="",
        yes_sub_title="Yes",
        no_sub_title="No",
        status="open",
        result="",
    )


class TestManualMapping:
    def test_exact_match_from_file(self):
        """Manual map should produce confidence=1.0 match."""
        map_data = {"pm-evt-1": "KALSHI-EVT-1"}
        with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
            json.dump(map_data, f)
            map_path = f.name

        matcher = EventMatcher(manual_map_path=map_path)

        pm_events = [_pm_event("pm-evt-1", "Will it rain tomorrow?")]
        kalshi_markets = [
            _kalshi_market("KALSHI-EVT-1-YES", "KALSHI-EVT-1", "Rain tomorrow?"),
        ]

        matches = matcher.match_events(pm_events, kalshi_markets)
        assert len(matches) == 1
        assert matches[0].confidence == 1.0
        assert matches[0].match_method == "manual"
        assert matches[0].pm_event_id == "pm-evt-1"
        assert matches[0].kalshi_event_ticker == "KALSHI-EVT-1"

    def test_manual_map_missing_kalshi_event(self):
        """If manual map points to nonexistent Kalshi event, no match."""
        map_data = {"pm-evt-1": "NONEXISTENT"}
        with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
            json.dump(map_data, f)
            map_path = f.name

        matcher = EventMatcher(manual_map_path=map_path)
        pm_events = [_pm_event("pm-evt-1", "Test event")]
        kalshi_markets = [_kalshi_market("OTHER-YES", "OTHER", "Other event")]

        matches = matcher.match_events(pm_events, kalshi_markets)
        # Should fall through to fuzzy matching (which may or may not match)
        for m in matches:
            assert m.match_method != "manual"

    def test_missing_map_file_uses_empty(self):
        """Missing map file should not crash, just use empty map."""
        matcher = EventMatcher(manual_map_path="/nonexistent/map.json")
        pm_events = [_pm_event("e1", "Test event")]
        kalshi_markets = [_kalshi_market("T1", "E1", "Different event")]

        # Should not crash
        matches = matcher.match_events(pm_events, kalshi_markets)
        # May or may not have fuzzy matches, but shouldn't crash
        assert isinstance(matches, list)


class TestFuzzyMatching:
    def test_high_similarity_unverified_gets_zero_confidence(self):
        """Unverified fuzzy matches get confidence=0.0 (blocked from trading)."""
        matcher = EventMatcher(manual_map_path="/nonexistent/map.json", fuzzy_threshold=85.0)

        pm_events = [_pm_event("e1", "Will Donald Trump win the 2028 presidential election?")]
        kalshi_markets = [
            _kalshi_market("PRES-2028-T", "PRES-2028", "Donald Trump wins 2028 presidential election"),
        ]

        matches = matcher.match_events(pm_events, kalshi_markets)
        assert len(matches) == 1
        assert matches[0].match_method == "fuzzy"
        assert matches[0].confidence == 0.0  # Blocked: not verified

    def test_verified_fuzzy_match_preserves_confidence(self):
        """Verified fuzzy matches preserve their score-based confidence."""
        # Create verified_matches.json with our event
        verified = {"e1": "PRES-2028"}
        with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
            json.dump(verified, f)
            verified_path = f.name

        matcher = EventMatcher(
            manual_map_path="/nonexistent/map.json",
            verified_path=verified_path,
            fuzzy_threshold=85.0,
        )

        pm_events = [_pm_event("e1", "Will Donald Trump win the 2028 presidential election?")]
        kalshi_markets = [
            _kalshi_market("PRES-2028-T", "PRES-2028", "Donald Trump wins 2028 presidential election"),
        ]

        matches = matcher.match_events(pm_events, kalshi_markets)
        assert len(matches) == 1
        assert matches[0].match_method == "verified"
        assert matches[0].confidence > 0.85  # score/100, ~88.7% for these titles

    def test_verified_mapping_enforces_mapped_event_ticker(self):
        """Verified mapping must pin to the configured Kalshi event ticker."""
        verified = {"e1": "EVENT-A"}
        with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
            json.dump(verified, f)
            verified_path = f.name

        matcher = EventMatcher(
            manual_map_path="/nonexistent/map.json",
            verified_path=verified_path,
            fuzzy_threshold=1.0,
        )

        pm_events = [_pm_event("e1", "Will BTC be above 100k by end of 2026?")]
        kalshi_markets = [
            _kalshi_market("A-1", "EVENT-A", "Will BTC close above 100k by end of 2026?"),
            _kalshi_market("B-1", "EVENT-B", "Will BTC be above 100k by end of 2026?"),
        ]

        matches = matcher.match_events(pm_events, kalshi_markets)
        assert len(matches) == 1
        assert matches[0].match_method == "verified"
        assert matches[0].kalshi_event_ticker == "EVENT-A"
        assert set(matches[0].kalshi_tickers) == {"A-1"}

    def test_low_similarity_no_match(self):
        """Very different titles should not match."""
        matcher = EventMatcher(manual_map_path="/nonexistent/map.json")

        pm_events = [_pm_event("e1", "Will it rain in New York tomorrow?")]
        kalshi_markets = [
            _kalshi_market("BTC-100K", "BTC-PRICE", "Bitcoin above $100,000 by end of year"),
        ]

        matches = matcher.match_events(pm_events, kalshi_markets)
        assert len(matches) == 0

    def test_custom_threshold(self):
        """Higher threshold should require closer match."""
        matcher_low = EventMatcher(manual_map_path="/nonexistent/map.json", fuzzy_threshold=50.0)
        matcher_high = EventMatcher(manual_map_path="/nonexistent/map.json", fuzzy_threshold=95.0)

        pm_events = [_pm_event("e1", "Will Biden run in 2028?")]
        kalshi_markets = [
            _kalshi_market("BIDEN-2028", "BIDEN", "Biden runs for president in 2028"),
        ]

        matches_low = matcher_low.match_events(pm_events, kalshi_markets)
        matches_high = matcher_high.match_events(pm_events, kalshi_markets)

        # Lower threshold should find more matches
        assert len(matches_low) >= len(matches_high)

    def test_default_threshold_is_95(self):
        """Default fuzzy threshold should be 95%."""
        assert FUZZY_THRESHOLD == 95.0


class TestYearMismatch:
    def test_same_year_no_mismatch(self):
        assert not _year_mismatch("Trump wins 2028", "Trump wins 2028 election")

    def test_different_years_mismatch(self):
        assert _year_mismatch("Trump wins 2024 election", "Trump wins 2028 election")

    def test_no_years_no_mismatch(self):
        assert not _year_mismatch("Will it rain?", "Rain tomorrow?")

    def test_one_has_year_no_mismatch(self):
        """If only one title has a year, it's not a year mismatch."""
        assert not _year_mismatch("Will Trump win?", "Trump 2028 election")

    def test_multiple_years_partial_overlap(self):
        """If years partially overlap, should still be ok."""
        assert not _year_mismatch("2024-2028 election cycle", "2024-2028 results")


class TestSettlementMismatch:
    def test_popular_vote_mismatch(self):
        assert _settlement_mismatch_risk(
            "Will Trump win the 2028 popular vote?",
            "Will Trump win 2028?",
        )

    def test_electoral_mismatch(self):
        assert _settlement_mismatch_risk(
            "Will Biden win 2028 electoral college?",
            "Will Biden win 2028?",
        )

    def test_resignation_mismatch(self):
        assert _settlement_mismatch_risk(
            "Will X resign by March 2026?",
            "Will X leave office?",
        )

    def test_no_keywords_no_mismatch(self):
        assert not _settlement_mismatch_risk(
            "Will it rain tomorrow?",
            "Rain tomorrow in NYC?",
        )

    def test_same_keyword_both_sides_no_mismatch(self):
        assert not _settlement_mismatch_risk(
            "Will X resign from office?",
            "Will X resign next month?",
        )


class TestFuzzyMatchFiltering:
    def test_year_mismatch_rejected(self):
        """Fuzzy match with year mismatch should be rejected."""
        matcher = EventMatcher(
            manual_map_path="/nonexistent/map.json",
            fuzzy_threshold=50.0,  # Low threshold to ensure fuzzy match happens
        )

        pm_events = [_pm_event("e1", "Will Trump win the 2024 presidential election?")]
        kalshi_markets = [
            _kalshi_market("PRES-2028", "PRES-2028", "Will Trump win the 2028 presidential election?"),
        ]

        matches = matcher.match_events(pm_events, kalshi_markets)
        assert len(matches) == 0

    def test_settlement_keyword_rejected(self):
        """Fuzzy match with settlement keyword mismatch should be rejected."""
        matcher = EventMatcher(
            manual_map_path="/nonexistent/map.json",
            fuzzy_threshold=50.0,
        )

        pm_events = [_pm_event("e1", "Will Biden win the 2028 popular vote?")]
        kalshi_markets = [
            _kalshi_market("PRES-2028-B", "PRES-2028", "Will Biden win 2028?"),
        ]

        matches = matcher.match_events(pm_events, kalshi_markets)
        assert len(matches) == 0


class TestMultipleMatches:
    def test_best_match_selected(self):
        """When multiple Kalshi events exist, should select best fuzzy match."""
        matcher = EventMatcher(manual_map_path="/nonexistent/map.json", fuzzy_threshold=50.0)

        pm_events = [_pm_event("e1", "Will Trump win 2028 election?")]
        kalshi_markets = [
            _kalshi_market("RAIN-YES", "RAIN", "Will it rain tomorrow?"),
            _kalshi_market("PRES-T", "PRES-2028", "Trump wins 2028 election"),
        ]

        matches = matcher.match_events(pm_events, kalshi_markets)
        if matches:
            # Should match to PRES-2028, not RAIN
            assert matches[0].kalshi_event_ticker == "PRES-2028"

    def test_multiple_pm_events_matched(self):
        """Multiple PM events should each get their own match."""
        map_data = {"e1": "K-E1"}
        with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
            json.dump(map_data, f)
            map_path = f.name

        matcher = EventMatcher(manual_map_path=map_path, fuzzy_threshold=50.0)

        pm_events = [
            _pm_event("e1", "Event one"),
            _pm_event("e2", "Will something happen?"),
        ]
        kalshi_markets = [
            _kalshi_market("K-E1-YES", "K-E1", "Event one on Kalshi"),
            _kalshi_market("K-E2-YES", "K-E2", "Will something happen on Kalshi?"),
        ]

        matches = matcher.match_events(pm_events, kalshi_markets)
        assert len(matches) == 2


class TestMatchedEventDataclass:
    def test_kalshi_tickers_populated(self):
        """MatchedEvent should list all Kalshi market tickers for the event."""
        map_data = {"e1": "K-E1"}
        with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
            json.dump(map_data, f)
            map_path = f.name

        matcher = EventMatcher(manual_map_path=map_path)

        pm_events = [_pm_event("e1", "Test")]
        kalshi_markets = [
            _kalshi_market("K-E1-A", "K-E1", "Test A"),
            _kalshi_market("K-E1-B", "K-E1", "Test B"),
        ]

        matches = matcher.match_events(pm_events, kalshi_markets)
        assert len(matches) == 1
        assert set(matches[0].kalshi_tickers) == {"K-E1-A", "K-E1-B"}


class TestSortedByConfidence:
    def test_results_sorted_desc(self):
        """Matches should be sorted by confidence descending."""
        map_data = {"e1": "K-E1"}
        with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
            json.dump(map_data, f)
            map_path = f.name

        matcher = EventMatcher(manual_map_path=map_path, fuzzy_threshold=50.0)

        pm_events = [
            _pm_event("e1", "Manually mapped event"),
            _pm_event("e2", "Will Trump win 2028 presidential election?"),
        ]
        kalshi_markets = [
            _kalshi_market("K-E1-YES", "K-E1", "Manually mapped"),
            _kalshi_market("PRES-T", "PRES-2028", "Trump wins 2028 presidential election"),
        ]

        matches = matcher.match_events(pm_events, kalshi_markets)
        if len(matches) >= 2:
            # Manual (confidence=1.0) should come first
            assert matches[0].confidence >= matches[1].confidence
