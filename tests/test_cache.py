"""
Unit tests for client/cache.py -- Gamma API caching.

Tests thread-safe TTL cache behavior, hit/miss patterns, and
GammaClient wrapper integration.
"""

import time
from unittest.mock import MagicMock, patch

import pytest

from client.cache import GammaCache, GammaClient, CachedValue, _GAMMA_API_TTL, _EVENT_MARKET_COUNTS_TTL
from scanner.models import Market, Event


def _make_market(
    condition_id: str = "0x123",
    question: str = "Test market?",
    yes_token_id: str = "yes_123",
    no_token_id: str = "no_123",
    neg_risk: bool = False,
) -> Market:
    return Market(
        condition_id=condition_id,
        question=question,
        yes_token_id=yes_token_id,
        no_token_id=no_token_id,
        neg_risk=neg_risk,
        event_id="evt_1",
        min_tick_size="0.01",
        active=True,
        volume=1000.0,
        end_date="2025-12-31",
        closed=False,
        neg_risk_market_id="" if not neg_risk else "nrm_1",
    )


def _make_event(
    event_id: str = "evt_1",
    title: str = "Test Event",
    markets: tuple[Market, ...] = (),
    neg_risk: bool = False,
) -> Event:
    return Event(
        event_id=event_id,
        title=title,
        markets=markets,
        neg_risk=neg_risk,
        neg_risk_market_id="" if not neg_risk else "nrm_1",
    )


class TestCachedValue:
    def test_is_stale_when_fresh(self):
        """Fresh value (timestamp = now) should not be stale."""
        cv = CachedValue("value", time.time())
        assert not cv.is_stale(ttl=60.0)

    def test_is_stale_when_old(self):
        """Value older than TTL should be stale."""
        cv = CachedValue("value", time.time() - 120)
        assert cv.is_stale(ttl=60.0)

    def test_is_stale_boundary(self):
        """Value exactly at TTL boundary should not be stale."""
        cv = CachedValue("value", time.time() - 60.0)
        assert not cv.is_stale(ttl=60.1)
        assert cv.is_stale(ttl=59.9)


class TestGammaCacheThreadSafety:
    def test_get_markets_returns_copy(self):
        """get_markets should return a copy, not internal list."""
        cache = GammaCache(gamma_host="https://test.com")
        with patch("client.cache.get_all_markets") as mock_fetch:
            mock_fetch.return_value = [_make_market()]

            markets1 = cache.get_markets()
            markets2 = cache.get_markets()

            # Should be equal but different objects
            assert markets1 == markets2
            assert markets1 is not markets2

    def test_get_events_returns_copy(self):
        """get_events should return a copy, not internal list."""
        cache = GammaCache(gamma_host="https://test.com")
        with patch("client.cache.get_all_markets") as mock_markets:
            mock_markets.return_value = [_make_market()]
            with patch("client.cache.build_events") as mock_build:
                mock_build.return_value = [_make_event()]

                events1 = cache.get_events()
                events2 = cache.get_events()

                assert events1 == events2
                assert events1 is not events2

    def test_get_event_market_counts_returns_copy(self):
        """get_event_market_counts should return a copy, not internal dict."""
        cache = GammaCache(gamma_host="https://test.com")
        with patch("client.cache.get_event_market_counts") as mock_fetch:
            mock_fetch.return_value = {"nrm_1": 10}

            counts1 = cache.get_event_market_counts()
            counts1["nrm_1"] = 999  # Modify copy
            counts2 = cache.get_event_market_counts()

            # Original should be unchanged
            assert counts2["nrm_1"] == 10


class TestGammaCacheTTL:
    def test_markets_cached_within_ttl(self):
        """Markets should be cached and returned within TTL."""
        cache = GammaCache(gamma_host="https://test.com")
        with patch("client.cache.get_all_markets") as mock_fetch:
            mock_fetch.return_value = [_make_market()]

            # First call - fetches
            markets1 = cache.get_markets()
            assert mock_fetch.call_count == 1

            # Second call within TTL - cached
            markets2 = cache.get_markets()
            assert mock_fetch.call_count == 1  # No new call
            assert len(markets2) == len(markets1)

    def test_markets_refetch_after_ttl(self):
        """Markets should be refetched after TTL expires."""
        cache = GammaCache(gamma_host="https://test.com")

        # Use a very short TTL for testing
        original_ttl = _GAMMA_API_TTL
        with patch("client.cache._GAMMA_API_TTL", 0.1):
            with patch("client.cache.get_all_markets") as mock_fetch:
                mock_fetch.return_value = [_make_market()]

                # First call
                cache.get_markets()
                assert mock_fetch.call_count == 1

                # Wait for TTL to expire
                time.sleep(0.15)

                # Second call - should refetch
                cache.get_markets()
                assert mock_fetch.call_count == 2

    def test_event_market_counts_cached_within_ttl(self):
        """Event market counts should be cached within TTL."""
        cache = GammaCache(gamma_host="https://test.com")
        with patch("client.cache.get_event_market_counts") as mock_fetch:
            mock_fetch.return_value = {"nrm_1": 10, "nrm_2": 15}

            # First call - fetches
            counts1 = cache.get_event_market_counts()
            assert mock_fetch.call_count == 1
            assert counts1["nrm_1"] == 10

            # Second call within TTL - cached
            counts2 = cache.get_event_market_counts()
            assert mock_fetch.call_count == 1
            assert counts2["nrm_1"] == 10

    def test_event_market_counts_refetch_after_ttl(self):
        """Event market counts should be refetched after TTL expires."""
        cache = GammaCache(gamma_host="https://test.com")

        # Use a very short TTL for testing
        with patch("client.cache._EVENT_MARKET_COUNTS_TTL", 0.1):
            with patch("client.cache.get_event_market_counts") as mock_fetch:
                mock_fetch.return_value = {"nrm_1": 10}

                # First call
                cache.get_event_market_counts()
                assert mock_fetch.call_count == 1

                # Wait for TTL to expire
                time.sleep(0.15)

                # Second call - should refetch
                cache.get_event_market_counts()
                assert mock_fetch.call_count == 2


class TestGammaCacheRefresh:
    def test_force_refresh_bypasses_cache(self):
        """force_refresh=True should bypass cache and fetch fresh data."""
        cache = GammaCache(gamma_host="https://test.com")
        with patch("client.cache.get_all_markets") as mock_fetch:
            mock_fetch.return_value = [_make_market()]

            # First call - caches
            cache.get_markets()
            assert mock_fetch.call_count == 1

            # Second call with force_refresh - refetches
            cache.get_markets(force_refresh=True)
            assert mock_fetch.call_count == 2

    def test_force_refresh_on_events(self):
        """force_refresh on events should rebuild from fresh markets."""
        cache = GammaCache(gamma_host="https://test.com")
        with patch("client.cache.get_all_markets") as mock_markets:
            mock_markets.return_value = [_make_market()]
            with patch("client.cache.build_events") as mock_build:
                mock_build.return_value = [_make_event()]

                # First call
                cache.get_events()
                assert mock_build.call_count == 1

                # Force refresh - should rebuild
                cache.get_events(force_refresh=True)
                assert mock_build.call_count == 2

    def test_force_refresh_on_counts(self):
        """force_refresh on counts should refetch from API."""
        cache = GammaCache(gamma_host="https://test.com")
        with patch("client.cache.get_event_market_counts") as mock_fetch:
            mock_fetch.return_value = {"nrm_1": 10}

            # First call
            cache.get_event_market_counts()
            assert mock_fetch.call_count == 1

            # Force refresh
            cache.get_event_market_counts(force_refresh=True)
            assert mock_fetch.call_count == 2


class TestGammaCacheClear:
    def test_clear_resets_all_caches(self):
        """clear() should reset all cached data and timestamps."""
        cache = GammaCache(gamma_host="https://test.com")
        with patch("client.cache.get_all_markets") as mock_markets:
            mock_markets.return_value = [_make_market()]
            with patch("client.cache.get_event_market_counts") as mock_counts:
                mock_counts.return_value = {"nrm_1": 10}

                # Populate caches
                cache.get_markets()
                cache.get_event_market_counts()
                assert cache._markets_timestamp > 0
                assert cache._event_market_counts_timestamp > 0

                # Clear
                cache.clear()
                assert cache._markets_timestamp == 0.0
                assert cache._events_timestamp == 0.0
                assert cache._event_market_counts_timestamp == 0.0
                assert len(cache._markets) == 0
                assert len(cache._events) == 0
                assert len(cache._event_market_counts) == 0


class TestGammaCacheEvents:
    def test_events_built_from_markets(self):
        """Events should be built from markets using build_events()."""
        cache = GammaCache(gamma_host="https://test.com")

        markets = [
            _make_market(condition_id="0x123"),
            _make_market(condition_id="0x456"),
        ]

        with patch("client.cache.get_all_markets") as mock_fetch:
            mock_fetch.return_value = markets
            with patch("client.cache.build_events") as mock_build:
                expected_events = [_make_event(event_id="evt_1")]
                mock_build.return_value = expected_events

                events = cache.get_events()

                mock_build.assert_called_once_with(markets)
                assert events == expected_events

    def test_events_rebuild_when_markets_change(self):
        """Events should rebuild if markets are force-refreshed."""
        cache = GammaCache(gamma_host="https://test.com")

        markets1 = [_make_market(condition_id="0x123")]
        markets2 = [_make_market(condition_id="0x456")]

        with patch("client.cache.get_all_markets") as mock_fetch:
            mock_fetch.return_value = markets1
            with patch("client.cache.build_events") as mock_build:
                mock_build.return_value = [_make_event()]

                # First call - builds events
                events1 = cache.get_events()
                assert mock_build.call_count == 1

                # Update markets (simulate new data)
                mock_fetch.return_value = markets2

                # Get events with force_refresh to trigger fresh markets fetch
                events2 = cache.get_events(force_refresh=True)
                assert mock_build.call_count == 2


class TestMarketsTimestamp:
    """Verify markets_timestamp property for cache-change detection."""

    def test_timestamp_zero_before_any_fetch(self):
        """Timestamp should be 0.0 before any markets are fetched."""
        cache = GammaCache(gamma_host="https://test.com")
        assert cache.markets_timestamp == 0.0

    def test_timestamp_set_after_fetch(self):
        """Timestamp should be set after fetching markets."""
        cache = GammaCache(gamma_host="https://test.com")
        with patch("client.cache.get_all_markets") as mock_fetch:
            mock_fetch.return_value = [_make_market()]
            cache.get_markets()
            assert cache.markets_timestamp > 0.0

    def test_timestamp_stable_on_cache_hit(self):
        """Timestamp should not change on cache hit (within TTL)."""
        cache = GammaCache(gamma_host="https://test.com")
        with patch("client.cache.get_all_markets") as mock_fetch:
            mock_fetch.return_value = [_make_market()]
            cache.get_markets()
            ts1 = cache.markets_timestamp
            cache.get_markets()  # cache hit
            ts2 = cache.markets_timestamp
            assert ts1 == ts2

    def test_timestamp_changes_on_refresh(self):
        """Timestamp should change when cache refreshes."""
        cache = GammaCache(gamma_host="https://test.com")
        with patch("client.cache.get_all_markets") as mock_fetch:
            mock_fetch.return_value = [_make_market()]
            cache.get_markets()
            ts1 = cache.markets_timestamp
            # Force refresh
            time.sleep(0.01)
            cache.get_markets(force_refresh=True)
            ts2 = cache.markets_timestamp
            assert ts2 > ts1

    def test_gamma_client_exposes_timestamp(self):
        """GammaClient should expose markets_timestamp from cache."""
        client = GammaClient(gamma_host="https://test.com")
        assert client.markets_timestamp == 0.0
        with patch("client.cache.get_all_markets") as mock_fetch:
            mock_fetch.return_value = [_make_market()]
            client.get_markets()
            assert client.markets_timestamp > 0.0


class TestGammaClient:
    def test_init_creates_cache(self):
        """GammaClient should create a GammaCache instance."""
        client = GammaClient(gamma_host="https://test.com")
        assert client.gamma_host == "https://test.com"
        assert client._cache is not None
        assert isinstance(client._cache, GammaCache)

    def test_get_markets_delegates_to_cache(self):
        """get_markets should delegate to cache."""
        client = GammaClient(gamma_host="https://test.com")
        with patch.object(client._cache, "get_markets") as mock_get:
            mock_get.return_value = [_make_market()]

            markets = client.get_markets()

            mock_get.assert_called_once_with(False)
            assert len(markets) == 1

    def test_get_markets_with_force_refresh(self):
        """get_markets(force_refresh=True) should pass through."""
        client = GammaClient(gamma_host="https://test.com")
        with patch.object(client._cache, "get_markets") as mock_get:
            client.get_markets(force_refresh=True)
            mock_get.assert_called_once_with(True)

    def test_get_events_delegates_to_cache(self):
        """get_events should delegate to cache."""
        client = GammaClient(gamma_host="https://test.com")
        with patch.object(client._cache, "get_events") as mock_get:
            mock_get.return_value = [_make_event()]

            events = client.get_events()

            mock_get.assert_called_once_with(False)
            assert len(events) == 1

    def test_get_event_market_counts_delegates_to_cache(self):
        """get_event_market_counts should delegate to cache."""
        client = GammaClient(gamma_host="https://test.com")
        with patch.object(client._cache, "get_event_market_counts") as mock_get:
            mock_get.return_value = {"nrm_1": 10}

            counts = client.get_event_market_counts()

            mock_get.assert_called_once_with(False)
            assert counts["nrm_1"] == 10

    def test_clear_cache_delegates_to_cache(self):
        """clear_cache should delegate to cache.clear()."""
        client = GammaClient(gamma_host="https://test.com")
        with patch.object(client._cache, "clear") as mock_clear:
            client.clear_cache()
            mock_clear.assert_called_once_with()

    def test_default_gamma_host(self):
        """GammaClient should use default host if not specified."""
        client = GammaClient()
        assert "api.gamma.io" in client.gamma_host
