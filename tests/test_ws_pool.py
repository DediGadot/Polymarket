"""Tests for WebSocket connection pool."""

from __future__ import annotations

import asyncio
import queue
import time
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from client.ws import BookUpdate, PriceUpdate
from client.ws_pool import DEFAULT_MAX_TOKENS_PER_CONN, ShardHealth, WSPool


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_token_ids(n: int) -> list[str]:
    """Generate n unique token IDs."""
    return [f"tok_{i:05d}" for i in range(n)]


def _mock_ws_manager(**kwargs):
    """Create a mock WSManager with real queues."""
    ws = MagicMock()
    ws.book_queue = queue.Queue()
    ws.price_queue = queue.Queue()
    ws.start = AsyncMock()
    ws.stop = AsyncMock()
    ws.subscribe = AsyncMock()
    ws.unsubscribe = AsyncMock()
    ws._running = True
    ws.token_ids = kwargs.get("token_ids", [])
    return ws


# ---------------------------------------------------------------------------
# Sharding logic
# ---------------------------------------------------------------------------


class TestShardingLogic:
    """Tests for initial token assignment across shards."""

    def test_1200_tokens_produce_3_shards(self):
        tokens = _make_token_ids(1200)
        pool = WSPool(url="wss://test", token_ids=tokens, max_tokens_per_conn=500)

        assert len(pool._shard_tokens) == 3
        for shard in pool._shard_tokens:
            assert len(shard) == 400  # 1200 / 3 = 400 each (round-robin)

    def test_500_tokens_produce_1_shard(self):
        tokens = _make_token_ids(500)
        pool = WSPool(url="wss://test", token_ids=tokens, max_tokens_per_conn=500)

        assert len(pool._shard_tokens) == 1
        assert len(pool._shard_tokens[0]) == 500

    def test_501_tokens_produce_2_shards(self):
        tokens = _make_token_ids(501)
        pool = WSPool(url="wss://test", token_ids=tokens, max_tokens_per_conn=500)

        assert len(pool._shard_tokens) == 2
        total = sum(len(s) for s in pool._shard_tokens)
        assert total == 501

    def test_empty_tokens_produce_1_shard(self):
        pool = WSPool(url="wss://test", token_ids=[], max_tokens_per_conn=500)

        assert len(pool._shard_tokens) == 1
        assert len(pool._shard_tokens[0]) == 0

    def test_no_shard_exceeds_limit(self):
        tokens = _make_token_ids(1499)
        pool = WSPool(url="wss://test", token_ids=tokens, max_tokens_per_conn=500)

        for shard in pool._shard_tokens:
            assert len(shard) <= 500

    def test_default_max_tokens(self):
        assert DEFAULT_MAX_TOKENS_PER_CONN == 500


# ---------------------------------------------------------------------------
# Token routing
# ---------------------------------------------------------------------------


class TestTokenRouting:
    """Each token maps to exactly one shard."""

    def test_every_token_has_shard(self):
        tokens = _make_token_ids(1200)
        pool = WSPool(url="wss://test", token_ids=tokens, max_tokens_per_conn=500)

        for tid in tokens:
            assert tid in pool._token_to_shard

    def test_shard_index_valid(self):
        tokens = _make_token_ids(1200)
        pool = WSPool(url="wss://test", token_ids=tokens, max_tokens_per_conn=500)

        for tid in tokens:
            idx = pool._token_to_shard[tid]
            assert 0 <= idx < len(pool._shard_tokens)

    def test_token_in_assigned_shard_list(self):
        tokens = _make_token_ids(100)
        pool = WSPool(url="wss://test", token_ids=tokens, max_tokens_per_conn=30)

        for tid in tokens:
            idx = pool._token_to_shard[tid]
            assert tid in pool._shard_tokens[idx]


# ---------------------------------------------------------------------------
# Subscribe
# ---------------------------------------------------------------------------


class TestSubscribe:
    """Dynamic subscribe places tokens in existing or new shards."""

    @pytest.mark.asyncio
    async def test_subscribe_to_existing_shard(self):
        tokens = _make_token_ids(400)
        pool = WSPool(url="wss://test", token_ids=tokens, max_tokens_per_conn=500)

        with patch("client.ws_pool.WSManager", side_effect=_mock_ws_manager):
            await pool.start()

        new_tokens = _make_token_ids(50)
        # Rename to avoid collision
        new_tokens = [f"new_{i}" for i in range(50)]
        await pool.subscribe(new_tokens)

        assert pool.shard_count == 1  # still 1 shard (400 + 50 = 450 < 500)
        for tid in new_tokens:
            assert tid in pool._token_to_shard
            assert pool._token_to_shard[tid] == 0

        pool._running = False
        await pool.stop()

    @pytest.mark.asyncio
    async def test_subscribe_creates_new_shard_when_full(self):
        tokens = _make_token_ids(500)
        pool = WSPool(url="wss://test", token_ids=tokens, max_tokens_per_conn=500)

        with patch("client.ws_pool.WSManager", side_effect=_mock_ws_manager):
            await pool.start()

        new_tokens = [f"extra_{i}" for i in range(3)]
        with patch("client.ws_pool.WSManager", side_effect=_mock_ws_manager):
            await pool.subscribe(new_tokens)

        assert pool.shard_count == 2  # new shard created
        for tid in new_tokens:
            assert tid in pool._token_to_shard
            assert pool._token_to_shard[tid] == 1

        pool._running = False
        await pool.stop()

    @pytest.mark.asyncio
    async def test_subscribe_ignores_duplicates(self):
        tokens = _make_token_ids(10)
        pool = WSPool(url="wss://test", token_ids=tokens, max_tokens_per_conn=500)

        with patch("client.ws_pool.WSManager", side_effect=_mock_ws_manager):
            await pool.start()

        await pool.subscribe(tokens[:5])  # already tracked
        assert pool.total_tokens == 10  # no change

        pool._running = False
        await pool.stop()


# ---------------------------------------------------------------------------
# Unsubscribe
# ---------------------------------------------------------------------------


class TestUnsubscribe:
    """Remove tokens from tracking."""

    def test_unsubscribe_removes_from_mapping(self):
        tokens = _make_token_ids(10)
        pool = WSPool(url="wss://test", token_ids=tokens, max_tokens_per_conn=500)

        pool.unsubscribe(tokens[:3])

        for tid in tokens[:3]:
            assert tid not in pool._token_to_shard
        for tid in tokens[3:]:
            assert tid in pool._token_to_shard

    def test_unsubscribe_removes_from_shard_list(self):
        tokens = _make_token_ids(10)
        pool = WSPool(url="wss://test", token_ids=tokens, max_tokens_per_conn=500)

        pool.unsubscribe(tokens[:3])

        all_shard_tokens = [t for shard in pool._shard_tokens for t in shard]
        for tid in tokens[:3]:
            assert tid not in all_shard_tokens

    def test_unsubscribe_unknown_token_is_noop(self):
        tokens = _make_token_ids(10)
        pool = WSPool(url="wss://test", token_ids=tokens, max_tokens_per_conn=500)

        pool.unsubscribe(["nonexistent"])
        assert pool.total_tokens == 10

    def test_total_tokens_updates(self):
        tokens = _make_token_ids(10)
        pool = WSPool(url="wss://test", token_ids=tokens, max_tokens_per_conn=500)

        assert pool.total_tokens == 10
        pool.unsubscribe(tokens[:4])
        assert pool.total_tokens == 6


# ---------------------------------------------------------------------------
# Health
# ---------------------------------------------------------------------------


class TestHealth:
    """Health reporting returns correct shard info."""

    @pytest.mark.asyncio
    async def test_health_returns_all_shards(self):
        tokens = _make_token_ids(1200)
        pool = WSPool(url="wss://test", token_ids=tokens, max_tokens_per_conn=500)

        with patch("client.ws_pool.WSManager", side_effect=_mock_ws_manager):
            await pool.start()

        report = pool.health()
        assert len(report) == 3

        for h in report:
            assert isinstance(h, ShardHealth)
            assert h.token_count == 400
            assert h.is_alive is True
            assert h.reconnect_count == 0

        pool._running = False
        await pool.stop()

    def test_shard_health_is_frozen(self):
        h = ShardHealth(shard_id=0, token_count=10, last_message_at=1.0, is_alive=True)
        with pytest.raises(AttributeError):
            h.token_count = 99  # type: ignore[misc]


# ---------------------------------------------------------------------------
# Message merging
# ---------------------------------------------------------------------------


class TestMessageMerging:
    """Drain threads merge shard queues into pool output queues."""

    @pytest.mark.asyncio
    async def test_book_updates_merge(self):
        pool = WSPool(url="wss://test", token_ids=_make_token_ids(10), max_tokens_per_conn=500)

        mock_ws = _mock_ws_manager()
        with patch("client.ws_pool.WSManager", return_value=mock_ws):
            await pool.start()

        update = BookUpdate(token_id="tok_00000", bids=[], asks=[], timestamp=time.time())
        mock_ws.book_queue.put(update)

        # Wait for drain thread to pick it up
        merged = pool.book_queue.get(timeout=2.0)
        assert merged.token_id == "tok_00000"

        pool._running = False
        await pool.stop()

    @pytest.mark.asyncio
    async def test_price_updates_merge(self):
        pool = WSPool(url="wss://test", token_ids=_make_token_ids(10), max_tokens_per_conn=500)

        mock_ws = _mock_ws_manager()
        with patch("client.ws_pool.WSManager", return_value=mock_ws):
            await pool.start()

        update = PriceUpdate(token_id="tok_00001", price=0.55, timestamp=time.time())
        mock_ws.price_queue.put(update)

        merged = pool.price_queue.get(timeout=2.0)
        assert merged.token_id == "tok_00001"
        assert merged.price == 0.55

        pool._running = False
        await pool.stop()

    @pytest.mark.asyncio
    async def test_multiple_shards_merge_into_single_queue(self):
        tokens = _make_token_ids(1000)
        pool = WSPool(url="wss://test", token_ids=tokens, max_tokens_per_conn=500)

        mock_shards = []
        def make_mock(**kw):
            ws = _mock_ws_manager(**kw)
            mock_shards.append(ws)
            return ws

        with patch("client.ws_pool.WSManager", side_effect=make_mock):
            await pool.start()

        assert len(mock_shards) == 2

        # Put one update in each shard
        now = time.time()
        mock_shards[0].book_queue.put(
            BookUpdate(token_id="from_shard_0", bids=[], asks=[], timestamp=now)
        )
        mock_shards[1].book_queue.put(
            BookUpdate(token_id="from_shard_1", bids=[], asks=[], timestamp=now)
        )

        received = set()
        for _ in range(2):
            msg = pool.book_queue.get(timeout=2.0)
            received.add(msg.token_id)

        assert received == {"from_shard_0", "from_shard_1"}

        pool._running = False
        await pool.stop()


# ---------------------------------------------------------------------------
# Properties
# ---------------------------------------------------------------------------


class TestProperties:
    """shard_count and total_tokens track state correctly."""

    def test_shard_count_before_start(self):
        tokens = _make_token_ids(1200)
        pool = WSPool(url="wss://test", token_ids=tokens, max_tokens_per_conn=500)
        # shard_count reflects WSManager instances (empty before start)
        assert pool.shard_count == 0

    @pytest.mark.asyncio
    async def test_shard_count_after_start(self):
        tokens = _make_token_ids(1200)
        pool = WSPool(url="wss://test", token_ids=tokens, max_tokens_per_conn=500)

        with patch("client.ws_pool.WSManager", side_effect=_mock_ws_manager):
            await pool.start()

        assert pool.shard_count == 3

        pool._running = False
        await pool.stop()

    def test_total_tokens(self):
        tokens = _make_token_ids(750)
        pool = WSPool(url="wss://test", token_ids=tokens, max_tokens_per_conn=500)
        assert pool.total_tokens == 750
