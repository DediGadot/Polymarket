"""
Gamma API caching client.

Implements generic TTL cache for Gamma API responses with:
- Thread-safe cache access
- TTL tracking (60s for Gamma API, 300s for event market counts)
- Configurable cache paths and sizes
- Logging for debugging
"""

from __future__ import annotations

import json
import logging
import threading
import time
from typing import Any

from client.gamma import get_all_markets, build_events, get_event_market_counts
from scanner.models import Market, Event


logger = logging.getLogger(__name__)

# Default TTL settings
_GAMMA_API_TTL = 60.0  # 60 seconds for market/event data
_EVENT_MARKET_COUNTS_TTL = 300.0  # 5 minutes for event market counts


class CachedValue:
    """Represents a cached value with its timestamp."""

    __slots__ = ("value", "timestamp")

    def __init__(self, value: Any, timestamp: float):
        self.value = value
        self.timestamp = timestamp

    def is_stale(self, ttl: float) -> bool:
        """Check if this cached value is stale (older than TTL)."""
        return time.time() - self.timestamp > ttl


class GammaCache:
    """
    Thread-safe in-memory cache for Gamma API data.

    Cache keys:
    - "markets": list[Market] - Active markets (last fetched)
    - "events": list[Event] - All events (last fetched)
    - "event_market_counts": dict[str, int] - Market counts by neg_risk_market_id (last fetched)
    """

    def __init__(self, gamma_host: str):
        self.gamma_host = gamma_host
        self._cache_lock = threading.Lock()
        self._markets: list[Market] = []
        self._markets_timestamp: float = 0.0
        self._events: list[Event] = []
        self._events_timestamp: float = 0.0
        self._event_market_counts: dict[str, int] = {}
        self._event_market_counts_timestamp: float = 0.0

    def get_markets(self, force_refresh: bool = False) -> list[Market]:
        """Get markets from cache or API.

        Args:
            force_refresh: Bypass cache and fetch from API
        """
        with self._cache_lock:
            if not force_refresh and not self._is_markets_stale_unlocked():
                logger.debug("Markets cache hit, returning %d markets", len(self._markets))
                return self._markets.copy()

        # Fetch outside lock to avoid holding during API call
        logger.info("Markets cache miss or stale, fetching from API")
        markets = get_all_markets(self.gamma_host)

        with self._cache_lock:
            self._markets = markets
            self._markets_timestamp = time.time()

        return markets.copy()

    def get_events(self, force_refresh: bool = False) -> list[Event]:
        """Get events from cache or API.

        Args:
            force_refresh: Bypass cache and fetch from API
        """
        # Events are built from markets - need markets first
        markets = self.get_markets(force_refresh)

        with self._cache_lock:
            if not force_refresh and not self._is_events_stale_unlocked():
                # Check if markets have changed since last event build
                if self._events and self._events_timestamp >= self._markets_timestamp:
                    logger.debug("Events cache hit, returning %d events", len(self._events))
                    return self._events.copy()

        # Build events from markets
        logger.info("Events cache miss or stale, building from markets")
        events = build_events(markets)

        with self._cache_lock:
            self._events = events
            self._events_timestamp = time.time()

        return events.copy()

    def get_event_market_counts(self, force_refresh: bool = False) -> dict[str, int]:
        """Get event market counts from cache or API.

        Returns dict mapping neg_risk_market_id -> total markets.

        Args:
            force_refresh: Bypass cache and fetch from API
        """
        with self._cache_lock:
            if not force_refresh and not self._is_event_market_counts_stale_unlocked():
                logger.debug(
                    "Event market counts cache hit, returning counts for %d neg_risk IDs",
                    len(self._event_market_counts),
                )
                return self._event_market_counts.copy()

        # Fetch outside lock
        logger.info("Event market counts cache miss or stale, fetching from API")
        counts = get_event_market_counts(self.gamma_host)

        with self._cache_lock:
            self._event_market_counts = counts
            self._event_market_counts_timestamp = time.time()

        return counts.copy()

    def clear(self) -> None:
        """Clear all cache."""
        with self._cache_lock:
            self._markets.clear()
            self._markets_timestamp = 0.0
            self._events.clear()
            self._events_timestamp = 0.0
            self._event_market_counts.clear()
            self._event_market_counts_timestamp = 0.0
        logger.debug("Gamma cache cleared")

    @property
    def markets_timestamp(self) -> float:
        """Timestamp of the last markets fetch (0.0 if never fetched).

        Callers can compare across cycles to detect cache refreshes:
        if timestamp hasn't changed, the underlying market data is identical.
        """
        with self._cache_lock:
            return self._markets_timestamp

    def _is_markets_stale_unlocked(self) -> bool:
        """Check if markets cache is stale. Caller must hold lock."""
        return time.time() - self._markets_timestamp > _GAMMA_API_TTL

    def _is_events_stale_unlocked(self) -> bool:
        """Check if events cache is stale. Caller must hold lock."""
        return time.time() - self._events_timestamp > _GAMMA_API_TTL

    def _is_event_market_counts_stale_unlocked(self) -> bool:
        """Check if event market counts cache is stale. Caller must hold lock."""
        return time.time() - self._event_market_counts_timestamp > _EVENT_MARKET_COUNTS_TTL


class GammaClient:
    """
    Gamma API client with caching.

    Usage:
        client = GammaClient(host="https://api.gamma.io")
        markets = client.get_markets()  # Get markets (cached for 60s)
        events = client.get_events()  # Get events (derived from markets)
        counts = client.get_event_market_counts()  # Get market counts (cached for 5min)

    Attributes:
        gamma_host: Base API host URL
        cache: GammaCache instance
    """

    def __init__(self, gamma_host: str = "https://api.gamma.io"):
        self.gamma_host = gamma_host
        self._cache = GammaCache(gamma_host)

    @property
    def markets_timestamp(self) -> float:
        """Timestamp of the last markets fetch from the underlying cache."""
        return self._cache.markets_timestamp

    def get_markets(self, force_refresh: bool = False) -> list[Market]:
        """Get markets from cache or API.

        Args:
            force_refresh: Bypass cache and fetch from API
        """
        return self._cache.get_markets(force_refresh)

    def get_events(self, force_refresh: bool = False) -> list[Event]:
        """Get events from cache or API.

        Args:
            force_refresh: Bypass cache and fetch from API
        """
        return self._cache.get_events(force_refresh)

    def get_event_market_counts(self, force_refresh: bool = False) -> dict[str, int]:
        """Get event market counts from cache or API.

        Returns dict mapping neg_risk_market_id -> total markets.

        Args:
            force_refresh: Bypass cache and fetch from API
        """
        return self._cache.get_event_market_counts(force_refresh)

    def clear_cache(self) -> None:
        """Clear all cached data."""
        self._cache.clear()
