"""
Unit tests for client/ws.py -- WebSocket manager with health checking.
"""

import time
from unittest.mock import MagicMock

from client.ws import WSManager, PriceUpdate, BookUpdate


class TestWSManagerHealth:
    """Tests for WebSocket health monitoring (task #21)."""

    def test_is_healthy_when_not_started(self):
        """is_healthy should return False when WS not started."""
        wsm = WSManager(url="wss://fake", token_ids=[])
        assert wsm.is_healthy() is False

    def test_is_healthy_when_running_with_ws(self):
        """is_healthy should return True when running with WS."""
        wsm = WSManager(url="wss://fake", token_ids=[])
        wsm._running = True
        wsm._ws = MagicMock()  # Mock WS connection
        wsm._last_message_time = time.time()

        assert wsm.is_healthy() is True

    def test_is_unhealthy_when_no_ws(self):
        """is_healthy should return False when running but no WS."""
        wsm = WSManager(url="wss://fake", token_ids=[])
        wsm._running = True
        wsm._ws = None

        assert wsm.is_healthy() is False

    def test_is_unhealthy_when_silent_too_long(self):
        """is_healthy should return False when no messages for too long."""
        wsm = WSManager(url="wss://fake", token_ids=[])
        wsm._running = True
        wsm._ws = MagicMock()  # Mock WS connection
        wsm._last_message_time = time.time() - 40  # Last message 40s ago

        # With 30s default max_silence, should be unhealthy
        assert wsm.is_healthy() is False

    def test_is_healthy_when_recent_messages(self):
        """is_healthy should return True when messages received recently."""
        wsm = WSManager(url="wss://fake", token_ids=[])
        wsm._running = True
        wsm._ws = MagicMock()  # Mock WS connection
        wsm._last_message_time = time.time() - 10  # Message 10s ago

        # With 30s default max_silence, should be healthy
        assert wsm.is_healthy() is True

    def test_is_unhealthy_custom_max_silence(self):
        """is_healthy should respect custom max_silence_sec."""
        wsm = WSManager(url="wss://fake", token_ids=[])
        wsm._running = True
        wsm._ws = MagicMock()
        wsm._last_message_time = time.time() - 5  # 5 seconds ago

        # With 3s threshold, 5s silence = unhealthy
        assert wsm.is_healthy(max_silence_sec=3.0) is False

        # With 10s threshold, 5s silence = healthy
        assert wsm.is_healthy(max_silence_sec=10.0) is True

    def test_record_message_updates_timestamp(self):
        """_record_message should update _last_message_time."""
        wsm = WSManager(url="wss://fake", token_ids=[])
        before = time.time()
        wsm._record_message()
        after = time.time()

        assert before <= wsm._last_message_time <= after

    def test_failed_connections_counter(self):
        """Failed connections should be tracked in _failed_connections."""
        wsm = WSManager(url="wss://fake", token_ids=[], max_retries=3)

        # Initially no failures
        assert wsm._failed_connections == 0

    def test_grace_period_after_connection(self):
        """When _last_message_time is 0, use _connect_time for health check."""
        wsm = WSManager(url="wss://fake", token_ids=[])

        now = time.time()
        wsm._connect_time = now
        wsm._running = True
        wsm._ws = MagicMock()
        wsm._last_message_time = 0.0  # No messages yet

        # Should be healthy due to grace period (just connected)
        assert wsm.is_healthy() is True

        # But if connection is old with no messages, unhealthy
        wsm._connect_time = now - 40
        assert wsm.is_healthy() is False
