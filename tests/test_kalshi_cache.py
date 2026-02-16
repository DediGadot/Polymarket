"""
Unit tests for client/kalshi_cache.py -- background Kalshi market snapshot cache.

Tests immutable snapshots, background refresh, stale-while-error,
event grouping, and thread lifecycle.
"""

import time
from unittest.mock import MagicMock

import pytest

from client.kalshi import KalshiClient, KalshiMarket
from client.kalshi_cache import KalshiMarketCache, KalshiMarketSnapshot


def _make_kalshi_market(
    ticker: str, event_ticker: str, title: str = "Test",
) -> KalshiMarket:
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


def _mock_client(markets: list[KalshiMarket] | None = None) -> MagicMock:
    """Create a mock KalshiClient with get_all_markets returning given markets."""
    client = MagicMock(spec=KalshiClient)
    client.get_all_markets.return_value = markets or []
    return client


class TestSnapshotBeforeStart:
    def test_snapshot_returns_none_before_start(self):
        """Cache created but not started should return None snapshot."""
        client = _mock_client()
        cache = KalshiMarketCache(client, refresh_sec=60.0)
        assert cache.snapshot() is None
        client.get_all_markets.assert_not_called()


class TestWarmPopulatesSnapshot:
    def test_warm_populates_snapshot(self):
        """start() warms the cache with markets from the client."""
        markets = [
            _make_kalshi_market("PRES-GOP", "PRES-2028", "GOP Nominee"),
            _make_kalshi_market("PRES-DEM", "PRES-2028", "DEM Nominee"),
        ]
        client = _mock_client(markets)
        cache = KalshiMarketCache(client, refresh_sec=300.0)
        try:
            cache.start()
            snap = cache.snapshot()

            assert snap is not None
            assert snap.version == 1
            assert len(snap.markets) == 2
            assert snap.markets[0].ticker == "PRES-GOP"
            assert snap.markets[1].ticker == "PRES-DEM"
            client.get_all_markets.assert_called_with(status="open")
        finally:
            cache.stop()


class TestSnapshotImmutability:
    def test_snapshot_is_immutable(self):
        """Snapshot is a frozen dataclass with tuple markets."""
        markets = [_make_kalshi_market("M1", "E1")]
        client = _mock_client(markets)
        cache = KalshiMarketCache(client, refresh_sec=300.0)
        try:
            cache.start()
            snap = cache.snapshot()

            assert snap is not None
            assert isinstance(snap, KalshiMarketSnapshot)
            assert isinstance(snap.markets, tuple)

            # Frozen dataclass should reject attribute assignment
            with pytest.raises(AttributeError):
                snap.version = 99  # type: ignore[misc]
        finally:
            cache.stop()


class TestByEventIndex:
    def test_by_event_groups_markets_correctly(self):
        """by_event should group markets by event_ticker."""
        markets = [
            _make_kalshi_market("PRES-GOP", "PRES-2028", "GOP Nominee"),
            _make_kalshi_market("PRES-DEM", "PRES-2028", "DEM Nominee"),
            _make_kalshi_market("FED-RATE", "FED-2026", "Rate Decision"),
        ]
        client = _mock_client(markets)
        cache = KalshiMarketCache(client, refresh_sec=300.0)
        try:
            cache.start()
            snap = cache.snapshot()

            assert snap is not None
            assert len(snap.by_event) == 2

            assert "PRES-2028" in snap.by_event
            assert len(snap.by_event["PRES-2028"]) == 2
            tickers = [m.ticker for m in snap.by_event["PRES-2028"]]
            assert "PRES-GOP" in tickers
            assert "PRES-DEM" in tickers

            assert "FED-2026" in snap.by_event
            assert len(snap.by_event["FED-2026"]) == 1
            assert snap.by_event["FED-2026"][0].ticker == "FED-RATE"
        finally:
            cache.stop()


class TestTitlesIndex:
    def test_titles_maps_event_ticker_to_first_title(self):
        """titles should map each event_ticker to the first market's title."""
        markets = [
            _make_kalshi_market("PRES-GOP", "PRES-2028", "GOP Nominee"),
            _make_kalshi_market("PRES-DEM", "PRES-2028", "DEM Nominee"),
            _make_kalshi_market("FED-RATE", "FED-2026", "Rate Decision"),
        ]
        client = _mock_client(markets)
        cache = KalshiMarketCache(client, refresh_sec=300.0)
        try:
            cache.start()
            snap = cache.snapshot()

            assert snap is not None
            assert len(snap.titles) == 2
            # First market in PRES-2028 group determines the title
            assert snap.titles["PRES-2028"] == "GOP Nominee"
            assert snap.titles["FED-2026"] == "Rate Decision"
        finally:
            cache.stop()


class TestBackgroundRefresh:
    def test_background_refresh_increments_version(self):
        """Background thread should refresh and increment version."""
        markets = [_make_kalshi_market("M1", "E1")]
        client = _mock_client(markets)
        cache = KalshiMarketCache(client, refresh_sec=0.3)
        try:
            cache.start()
            snap_v1 = cache.snapshot()
            assert snap_v1 is not None
            assert snap_v1.version == 1

            # Wait for at least one background refresh cycle
            time.sleep(0.5)

            snap_v2 = cache.snapshot()
            assert snap_v2 is not None
            assert snap_v2.version >= 2
        finally:
            cache.stop()


class TestStaleWhileError:
    def test_stale_data_preserved_on_refresh_error(self):
        """On refresh failure, snapshot should keep the last good data."""
        markets = [_make_kalshi_market("M1", "E1")]
        client = _mock_client(markets)

        # First call succeeds (warm), subsequent calls raise
        call_count = 0
        original_markets = markets[:]

        def _side_effect(**kwargs):
            nonlocal call_count
            call_count += 1
            if call_count == 1:
                return original_markets
            raise ConnectionError("Kalshi API down")

        client.get_all_markets.side_effect = _side_effect

        cache = KalshiMarketCache(client, refresh_sec=0.2)
        try:
            cache.start()
            snap_v1 = cache.snapshot()
            assert snap_v1 is not None
            assert snap_v1.version == 1
            assert len(snap_v1.markets) == 1

            # Wait for a failed refresh attempt
            time.sleep(0.5)

            # Snapshot should still have the good data
            snap_after = cache.snapshot()
            assert snap_after is not None
            assert snap_after.version == 1  # No version bump on failure
            assert len(snap_after.markets) == 1
            assert snap_after.markets[0].ticker == "M1"
        finally:
            cache.stop()


class TestStopJoinsThread:
    def test_stop_joins_thread(self):
        """After stop(), the daemon thread should no longer be alive."""
        markets = [_make_kalshi_market("M1", "E1")]
        client = _mock_client(markets)
        cache = KalshiMarketCache(client, refresh_sec=300.0)

        cache.start()
        # Thread should be alive after start
        assert cache._thread is not None
        assert cache._thread.is_alive()

        cache.stop()
        # After stop, _thread is set to None (joined)
        assert cache._thread is None


class TestWarmFailure:
    def test_warm_failure_returns_none(self):
        """If warm fetch fails, start() doesn't crash and snapshot is None."""
        client = _mock_client()
        client.get_all_markets.side_effect = ConnectionError("Kalshi unreachable")

        cache = KalshiMarketCache(client, refresh_sec=300.0)
        try:
            # start() should not raise despite warm failure
            cache.start()
            assert cache.snapshot() is None
        finally:
            cache.stop()


class TestNonBlockingWarm:
    def test_start_returns_immediately(self):
        """start() should return before warm-up completes (non-blocking)."""
        import threading

        warm_started = threading.Event()
        warm_release = threading.Event()

        def _slow_warm(**kwargs):
            warm_started.set()
            warm_release.wait(timeout=5.0)
            return [_make_kalshi_market("M1", "E1")]

        client = _mock_client()
        client.get_all_markets.side_effect = _slow_warm

        cache = KalshiMarketCache(client, refresh_sec=300.0)
        try:
            start_time = time.time()
            cache.start()
            elapsed = time.time() - start_time

            # start() should return in under 0.5s (not wait for warm-up)
            assert elapsed < 0.5, f"start() took {elapsed:.2f}s — should be non-blocking"

            # Snapshot should be None while warming
            assert cache.snapshot() is None

            # Let warm-up finish
            warm_release.set()
            # Wait for the warm thread to populate snapshot
            for _ in range(50):
                if cache.snapshot() is not None:
                    break
                time.sleep(0.05)

            snap = cache.snapshot()
            assert snap is not None
            assert len(snap.markets) == 1
            assert snap.version == 1
        finally:
            warm_release.set()  # Ensure thread isn't stuck
            cache.stop()

    def test_start_snapshot_available_after_warm(self):
        """After background warm completes, snapshot becomes available."""
        markets = [
            _make_kalshi_market("A1", "E1"),
            _make_kalshi_market("A2", "E1"),
        ]
        client = _mock_client(markets)
        cache = KalshiMarketCache(client, refresh_sec=300.0)
        try:
            cache.start()
            # Poll until snapshot available (fast mock should be instant)
            for _ in range(50):
                if cache.snapshot() is not None:
                    break
                time.sleep(0.05)

            snap = cache.snapshot()
            assert snap is not None
            assert len(snap.markets) == 2
        finally:
            cache.stop()


class TestMultipleSnapshots:
    def test_consecutive_snapshots_are_same_object(self):
        """Two snapshot() calls without refresh return the same reference."""
        markets = [_make_kalshi_market("M1", "E1")]
        client = _mock_client(markets)
        cache = KalshiMarketCache(client, refresh_sec=300.0)
        try:
            cache.start()
            snap1 = cache.snapshot()
            snap2 = cache.snapshot()

            # Same immutable object — no copy needed
            assert snap1 is snap2
        finally:
            cache.stop()

    def test_snapshot_changes_after_refresh(self):
        """After a refresh, snapshot() returns a different object."""
        markets_v1 = [_make_kalshi_market("M1", "E1")]
        markets_v2 = [_make_kalshi_market("M1", "E1"), _make_kalshi_market("M2", "E2")]

        client = _mock_client()
        client.get_all_markets.side_effect = [markets_v1, markets_v2]

        cache = KalshiMarketCache(client, refresh_sec=0.2)
        try:
            cache.start()
            snap1 = cache.snapshot()
            assert snap1 is not None
            assert len(snap1.markets) == 1

            time.sleep(0.4)

            snap2 = cache.snapshot()
            assert snap2 is not None
            assert snap2 is not snap1
            assert len(snap2.markets) == 2
            assert snap2.version == 2
        finally:
            cache.stop()
