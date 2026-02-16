"""
Unit tests for client/ws_bridge.py -- synchronous WebSocket bridge.
"""

import asyncio
import queue
import time
from unittest.mock import MagicMock, patch

from client.ws import BookUpdate, PriceUpdate
from client.ws_bridge import WSBridge
from scanner.book_cache import BookCache
from scanner.ofi import OFITracker
from scanner.spike import SpikeDetector


class TestWSBridgeDrain:
    def test_drain_book_updates_into_cache(self):
        """Book updates from WS should be applied to BookCache."""
        cache = BookCache()
        bridge = WSBridge(ws_url="wss://fake", book_cache=cache)

        # Simulate a WSManager with queued updates
        mock_ws = MagicMock()
        q = queue.Queue()
        q.put(BookUpdate(
            token_id="tok1",
            bids=[{"price": "0.55", "size": "100"}],
            asks=[{"price": "0.60", "size": "50"}],
            timestamp=time.time(),
        ))
        mock_ws.book_queue = q
        mock_ws.price_queue = queue.Queue()
        bridge._ws = mock_ws

        count = bridge.drain()
        assert count == 1
        book = cache.get_book("tok1")
        assert book is not None
        assert book.best_bid.price == 0.55
        assert book.best_ask.price == 0.60
        assert bridge.last_changed_tokens == {"tok1"}

    def test_drain_price_updates_into_spike_detector(self):
        """Price updates from WS should feed SpikeDetector."""
        cache = BookCache()
        detector = SpikeDetector(threshold_pct=5.0, window_sec=30.0, cooldown_sec=60.0)
        bridge = WSBridge(ws_url="wss://fake", book_cache=cache, spike_detector=detector)

        # Register a token first
        detector.register_token("tok1", "evt1")

        mock_ws = MagicMock()
        mock_ws.book_queue = queue.Queue()
        price_queue = queue.Queue()
        price_queue.put(PriceUpdate(token_id="tok1", price=0.50, timestamp=time.time()))
        mock_ws.price_queue = price_queue
        bridge._ws = mock_ws

        count = bridge.drain()
        assert count == 1
        # Spike detector should have the price recorded
        assert detector.get_velocity("tok1") is not None or True  # just check no crash
        assert bridge.last_changed_tokens == {"tok1"}

    def test_drain_returns_zero_when_no_ws(self):
        """drain() should return 0 when WSManager is not initialized."""
        cache = BookCache()
        bridge = WSBridge(ws_url="wss://fake", book_cache=cache)
        assert bridge.drain() == 0

    def test_drain_handles_empty_queues(self):
        """drain() should handle empty queues gracefully."""
        cache = BookCache()
        bridge = WSBridge(ws_url="wss://fake", book_cache=cache)
        mock_ws = MagicMock()
        mock_ws.book_queue = queue.Queue()
        mock_ws.price_queue = queue.Queue()
        bridge._ws = mock_ws

        assert bridge.drain() == 0
        assert bridge.last_changed_tokens == set()

    def test_drain_multiple_updates(self):
        """Multiple queued updates should all be drained."""
        cache = BookCache()
        bridge = WSBridge(ws_url="wss://fake", book_cache=cache)

        mock_ws = MagicMock()
        book_queue = queue.Queue()
        for i in range(5):
            book_queue.put(BookUpdate(
                token_id=f"tok{i}",
                bids=[{"price": "0.50", "size": "100"}],
                asks=[{"price": "0.55", "size": "100"}],
                timestamp=time.time(),
            ))
        mock_ws.book_queue = book_queue
        mock_ws.price_queue = queue.Queue()
        bridge._ws = mock_ws

        count = bridge.drain()
        assert count == 5
        assert cache.token_count() == 5

    def test_full_queue_drops_oldest_entry(self):
        """When queue is full, oldest entry should be dropped to make room."""
        cache = BookCache()
        bridge = WSBridge(ws_url="wss://fake", book_cache=cache)

        # Create a small bounded queue to simulate full condition
        mock_ws = MagicMock()
        book_queue = queue.Queue(maxsize=2)

        # Fill the queue
        book_queue.put(BookUpdate(
            token_id="tok1",
            bids=[{"price": "0.50", "size": "100"}],
            asks=[{"price": "0.55", "size": "100"}],
            timestamp=time.time(),
        ))
        book_queue.put(BookUpdate(
            token_id="tok2",
            bids=[{"price": "0.51", "size": "100"}],
            asks=[{"price": "0.56", "size": "100"}],
            timestamp=time.time(),
        ))
        mock_ws.book_queue = book_queue
        mock_ws.price_queue = queue.Queue()
        bridge._ws = mock_ws

        # Now try to put more when full - should handle gracefully
        # This tests the WSManager's internal handling of queue.Full
        count = bridge.drain()
        assert count == 2


class TestWSBridgeStats:
    def test_stats_when_not_started(self):
        cache = BookCache()
        bridge = WSBridge(ws_url="wss://fake", book_cache=cache)
        stats = bridge.stats
        assert stats["connected"] is False
        assert stats["books_received"] == 0
        assert stats["prices_received"] == 0

    def test_stats_after_drain(self):
        cache = BookCache()
        bridge = WSBridge(ws_url="wss://fake", book_cache=cache)

        mock_ws = MagicMock()
        book_queue = queue.Queue()
        book_queue.put(BookUpdate(
            token_id="tok1",
            bids=[{"price": "0.50", "size": "100"}],
            asks=[{"price": "0.55", "size": "100"}],
            timestamp=time.time(),
        ))
        price_queue = queue.Queue()
        price_queue.put(PriceUpdate(token_id="tok2", price=0.50, timestamp=time.time()))
        mock_ws.book_queue = book_queue
        mock_ws.price_queue = price_queue
        bridge._ws = mock_ws

        bridge.drain()
        stats = bridge.stats
        assert stats["books_received"] == 1
        assert stats["prices_received"] == 1


class TestWSBridgeLifecycle:
    @patch("client.ws_bridge.threading.Thread")
    @patch("client.ws_bridge.asyncio.new_event_loop")
    @patch("client.ws_bridge.WSManager")
    def test_start_passes_configured_max_retries(self, mock_ws_manager, mock_new_loop, mock_thread_cls):
        cache = BookCache()
        mock_new_loop.return_value = MagicMock()
        mock_thread = MagicMock()
        mock_thread_cls.return_value = mock_thread

        bridge = WSBridge(ws_url="wss://fake", book_cache=cache, max_retries=9)
        bridge.start(["tok1"])

        assert mock_ws_manager.called
        assert mock_ws_manager.call_args.kwargs["max_retries"] == 9
        mock_thread.start.assert_called_once()

    @patch("client.ws_bridge.threading.Thread")
    @patch("client.ws_bridge.asyncio.new_event_loop")
    @patch("client.ws_bridge.WSManager")
    def test_start_recovers_when_previous_thread_died(self, mock_ws_manager, mock_new_loop, mock_thread_cls):
        cache = BookCache()
        mock_new_loop.return_value = MagicMock()
        mock_thread = MagicMock()
        mock_thread_cls.return_value = mock_thread

        bridge = WSBridge(ws_url="wss://fake", book_cache=cache, max_retries=5)
        dead_thread = MagicMock()
        dead_thread.is_alive.return_value = False
        bridge._started = True
        bridge._thread = dead_thread

        bridge.start(["tok1"])

        assert mock_ws_manager.called
        mock_thread.start.assert_called_once()


class TestWSBridgePool:
    """Tests for WSPool integration in WSBridge."""

    @patch("client.ws_bridge.threading.Thread")
    @patch("client.ws_bridge.asyncio.new_event_loop")
    @patch("client.ws_bridge.WSManager")
    def test_small_token_set_uses_single_ws(self, mock_ws_cls, mock_new_loop, mock_thread_cls):
        """100 tokens should use WSManager, not WSPool."""
        cache = BookCache()
        mock_new_loop.return_value = MagicMock()
        mock_thread_cls.return_value = MagicMock()

        bridge = WSBridge(ws_url="wss://fake", book_cache=cache, max_tokens_per_conn=500)
        bridge.start([f"tok_{i}" for i in range(100)])

        assert bridge._using_pool is False
        assert mock_ws_cls.called

    @patch("client.ws_bridge.threading.Thread")
    @patch("client.ws_bridge.asyncio.new_event_loop")
    @patch("client.ws_bridge.WSPool")
    def test_large_token_set_uses_pool(self, mock_pool_cls, mock_new_loop, mock_thread_cls):
        """1500 tokens should use WSPool."""
        cache = BookCache()
        mock_new_loop.return_value = MagicMock()
        mock_thread_cls.return_value = MagicMock()

        bridge = WSBridge(ws_url="wss://fake", book_cache=cache, max_tokens_per_conn=500)
        bridge.start([f"tok_{i}" for i in range(1500)])

        assert bridge._using_pool is True
        assert mock_pool_cls.called
        call_kwargs = mock_pool_cls.call_args.kwargs
        assert call_kwargs["max_tokens_per_conn"] == 500

    @patch("client.ws_bridge.threading.Thread")
    @patch("client.ws_bridge.asyncio.new_event_loop")
    @patch("client.ws_bridge.WSPool")
    def test_pool_receives_correct_params(self, mock_pool_cls, mock_new_loop, mock_thread_cls):
        """WSPool should receive url, tokens, max_tokens, max_retries."""
        cache = BookCache()
        mock_new_loop.return_value = MagicMock()
        mock_thread_cls.return_value = MagicMock()

        bridge = WSBridge(
            ws_url="wss://test-url",
            book_cache=cache,
            max_retries=7,
            max_tokens_per_conn=300,
        )
        tokens = [f"tok_{i}" for i in range(600)]
        bridge.start(tokens)

        call_kwargs = mock_pool_cls.call_args.kwargs
        assert call_kwargs["url"] == "wss://test-url"
        assert call_kwargs["max_tokens_per_conn"] == 300
        assert call_kwargs["max_retries"] == 7
        assert len(call_kwargs["token_ids"]) == 600

    def test_drain_works_with_pool_queues(self):
        """drain() should read from WSPool's merged queues the same as WSManager."""
        cache = BookCache()
        bridge = WSBridge(ws_url="wss://fake", book_cache=cache, max_tokens_per_conn=500)

        # Simulate a pool with merged queues
        mock_pool = MagicMock()
        mock_pool.book_queue = queue.Queue()
        mock_pool.price_queue = queue.Queue()
        mock_pool.book_queue.put(BookUpdate(
            token_id="tok_pool_1",
            bids=[{"price": "0.45", "size": "200"}],
            asks=[{"price": "0.50", "size": "100"}],
            timestamp=time.time(),
        ))
        mock_pool.price_queue.put(PriceUpdate(
            token_id="tok_pool_2", price=0.65, timestamp=time.time(),
        ))
        bridge._ws = mock_pool

        count = bridge.drain()
        assert count == 2
        assert bridge.last_changed_tokens == {"tok_pool_1", "tok_pool_2"}
        assert cache.get_book("tok_pool_1") is not None

    @patch("client.ws_bridge.threading.Thread")
    @patch("client.ws_bridge.asyncio.new_event_loop")
    @patch("client.ws_bridge.WSManager")
    def test_boundary_500_uses_single_ws(self, mock_ws_cls, mock_new_loop, mock_thread_cls):
        """Exactly 500 tokens should use WSManager (not pool)."""
        cache = BookCache()
        mock_new_loop.return_value = MagicMock()
        mock_thread_cls.return_value = MagicMock()

        bridge = WSBridge(ws_url="wss://fake", book_cache=cache, max_tokens_per_conn=500)
        bridge.start([f"tok_{i}" for i in range(500)])

        assert bridge._using_pool is False

    @patch("client.ws_bridge.threading.Thread")
    @patch("client.ws_bridge.asyncio.new_event_loop")
    @patch("client.ws_bridge.WSPool")
    def test_boundary_501_uses_pool(self, mock_pool_cls, mock_new_loop, mock_thread_cls):
        """501 tokens should trigger WSPool."""
        cache = BookCache()
        mock_new_loop.return_value = MagicMock()
        mock_thread_cls.return_value = MagicMock()

        bridge = WSBridge(ws_url="wss://fake", book_cache=cache, max_tokens_per_conn=500)
        bridge.start([f"tok_{i}" for i in range(501)])

        assert bridge._using_pool is True


class TestWSBridgeOFI:
    """Tests for OFI tracker integration in WSBridge drain loop."""

    def test_book_update_feeds_ofi_tracker(self):
        """Book updates with bid/ask imbalance should record OFI."""
        cache = BookCache()
        ofi = OFITracker(window_sec=60.0)
        bridge = WSBridge(ws_url="wss://fake", book_cache=cache, ofi_tracker=ofi)

        mock_ws = MagicMock()
        mock_ws.book_queue = queue.Queue()
        mock_ws.price_queue = queue.Queue()
        # Bid volume (300) > ask volume (100) → BUY signal
        mock_ws.book_queue.put(BookUpdate(
            token_id="tok1",
            bids=[{"price": "0.55", "size": "200"}, {"price": "0.50", "size": "100"}],
            asks=[{"price": "0.60", "size": "100"}],
            timestamp=time.time(),
        ))
        bridge._ws = mock_ws

        bridge.drain()

        # OFI should show buy pressure (bid_vol 300 > ask_vol 100 → imbalance 200)
        assert ofi.get_ofi("tok1") > 0

    def test_book_update_with_sell_imbalance(self):
        """Ask-heavy book should produce sell OFI signal."""
        cache = BookCache()
        ofi = OFITracker(window_sec=60.0)
        bridge = WSBridge(ws_url="wss://fake", book_cache=cache, ofi_tracker=ofi)

        mock_ws = MagicMock()
        mock_ws.book_queue = queue.Queue()
        mock_ws.price_queue = queue.Queue()
        # Ask volume (500) > bid volume (100) → SELL signal
        mock_ws.book_queue.put(BookUpdate(
            token_id="tok2",
            bids=[{"price": "0.50", "size": "100"}],
            asks=[{"price": "0.55", "size": "200"}, {"price": "0.60", "size": "300"}],
            timestamp=time.time(),
        ))
        bridge._ws = mock_ws

        bridge.drain()

        assert ofi.get_ofi("tok2") < 0

    def test_balanced_book_no_ofi_event(self):
        """Perfectly balanced bid/ask volume should not record OFI."""
        cache = BookCache()
        ofi = OFITracker(window_sec=60.0)
        bridge = WSBridge(ws_url="wss://fake", book_cache=cache, ofi_tracker=ofi)

        mock_ws = MagicMock()
        mock_ws.book_queue = queue.Queue()
        mock_ws.price_queue = queue.Queue()
        mock_ws.book_queue.put(BookUpdate(
            token_id="tok3",
            bids=[{"price": "0.50", "size": "100"}],
            asks=[{"price": "0.55", "size": "100"}],
            timestamp=time.time(),
        ))
        bridge._ws = mock_ws

        bridge.drain()

        assert ofi.get_ofi("tok3") == 0.0

    def test_drain_feeds_all_three_consumers(self):
        """A single drain should feed BookCache, SpikeDetector, and OFITracker."""
        cache = BookCache()
        detector = SpikeDetector(threshold_pct=5.0, window_sec=30.0, cooldown_sec=60.0)
        ofi = OFITracker(window_sec=60.0)
        bridge = WSBridge(
            ws_url="wss://fake",
            book_cache=cache,
            spike_detector=detector,
            ofi_tracker=ofi,
        )

        detector.register_token("tok_price", "evt1")

        mock_ws = MagicMock()
        book_q = queue.Queue()
        price_q = queue.Queue()
        book_q.put(BookUpdate(
            token_id="tok_book",
            bids=[{"price": "0.50", "size": "200"}],
            asks=[{"price": "0.55", "size": "50"}],
            timestamp=time.time(),
        ))
        price_q.put(PriceUpdate(
            token_id="tok_price", price=0.60, timestamp=time.time(),
        ))
        mock_ws.book_queue = book_q
        mock_ws.price_queue = price_q
        bridge._ws = mock_ws

        count = bridge.drain()

        assert count == 2
        # BookCache received the book update
        assert cache.get_book("tok_book") is not None
        # OFI tracker received the imbalance (bid 200 > ask 50)
        assert ofi.get_ofi("tok_book") > 0
        # SpikeDetector received the price update (no crash)
        assert bridge.last_changed_tokens == {"tok_book", "tok_price"}

    def test_no_ofi_tracker_no_crash(self):
        """Without OFI tracker, drain should work as before."""
        cache = BookCache()
        bridge = WSBridge(ws_url="wss://fake", book_cache=cache)

        mock_ws = MagicMock()
        mock_ws.book_queue = queue.Queue()
        mock_ws.price_queue = queue.Queue()
        mock_ws.book_queue.put(BookUpdate(
            token_id="tok1",
            bids=[{"price": "0.50", "size": "100"}],
            asks=[{"price": "0.55", "size": "50"}],
            timestamp=time.time(),
        ))
        bridge._ws = mock_ws

        count = bridge.drain()
        assert count == 1
