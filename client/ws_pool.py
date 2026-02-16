"""
WebSocket connection pool. Shards token subscriptions across multiple
WSManager instances, each limited to max_tokens_per_conn tokens.
Merges all shard queues into single output queues for WSBridge consumption.
"""

from __future__ import annotations

import asyncio
import logging
import math
import queue
import threading
import time
from dataclasses import dataclass

from client.ws import WSManager, BookUpdate, PriceUpdate

logger = logging.getLogger(__name__)

DEFAULT_MAX_TOKENS_PER_CONN = 500


@dataclass(frozen=True)
class ShardHealth:
    """Health status of a single WS shard."""

    shard_id: int
    token_count: int
    last_message_at: float
    is_alive: bool
    reconnect_count: int = 0


class WSPool:
    """
    Manages N WSManager instances, each subscribing to <= max_tokens_per_conn tokens.
    Provides unified book_queue and price_queue that merge from all shards.
    """

    def __init__(
        self,
        url: str,
        token_ids: list[str],
        max_tokens_per_conn: int = DEFAULT_MAX_TOKENS_PER_CONN,
        max_retries: int = 5,
    ) -> None:
        self._url = url
        self._max_tokens = max_tokens_per_conn
        self._max_retries = max_retries

        # Output queues (merged from all shards)
        self.book_queue: queue.Queue[BookUpdate] = queue.Queue()
        self.price_queue: queue.Queue[PriceUpdate] = queue.Queue()

        # Shard state
        self._shards: list[WSManager] = []
        self._shard_tokens: list[list[str]] = []
        self._token_to_shard: dict[str, int] = {}
        self._shard_last_msg: list[float] = []
        self._shard_reconnects: list[int] = []
        self._drain_threads: list[threading.Thread] = []
        self._running = False

        # Initial sharding
        self._assign_tokens(token_ids)

    def _assign_tokens(self, token_ids: list[str]) -> None:
        """Shard token_ids into groups of max_tokens_per_conn."""
        n_shards = max(1, math.ceil(len(token_ids) / self._max_tokens))
        self._shard_tokens = [[] for _ in range(n_shards)]
        self._token_to_shard = {}

        for i, tid in enumerate(token_ids):
            shard_idx = i % n_shards
            self._shard_tokens[shard_idx].append(tid)
            self._token_to_shard[tid] = shard_idx

    async def start(self) -> None:
        """Create and start all WSManager shards."""
        self._running = True
        self._shards = []
        self._shard_last_msg = []
        self._shard_reconnects = []

        now = time.time()
        for i, tokens in enumerate(self._shard_tokens):
            ws = WSManager(
                url=self._url,
                token_ids=list(tokens),
                max_retries=self._max_retries,
            )
            self._shards.append(ws)
            self._shard_last_msg.append(now)
            self._shard_reconnects.append(0)

            t = threading.Thread(
                target=self._drain_shard,
                args=(i, ws),
                daemon=True,
                name=f"ws-pool-drain-{i}",
            )
            self._drain_threads.append(t)
            t.start()

            await ws.start()

        logger.info(
            "WSPool: started %d shards for %d tokens (max %d/conn)",
            len(self._shards),
            len(self._token_to_shard),
            self._max_tokens,
        )

    async def stop(self) -> None:
        """Stop all shards and drain threads."""
        self._running = False
        for ws in self._shards:
            await ws.stop()
        for t in self._drain_threads:
            t.join(timeout=2.0)
        self._drain_threads.clear()
        logger.info("WSPool: stopped %d shards", len(self._shards))

    async def subscribe(self, token_ids: list[str]) -> None:
        """Subscribe to additional tokens, creating new shards if needed."""
        new_tokens = [t for t in token_ids if t not in self._token_to_shard]
        if not new_tokens:
            return

        for tid in new_tokens:
            placed = False
            for i, tokens in enumerate(self._shard_tokens):
                if len(tokens) < self._max_tokens:
                    tokens.append(tid)
                    self._token_to_shard[tid] = i
                    if i < len(self._shards):
                        await self._shards[i].subscribe([tid])
                    placed = True
                    break

            if not placed:
                await self._create_shard([tid])

    async def _create_shard(self, token_ids: list[str]) -> int:
        """Create a new shard for the given tokens. Returns shard index."""
        idx = len(self._shard_tokens)
        self._shard_tokens.append(list(token_ids))
        for tid in token_ids:
            self._token_to_shard[tid] = idx

        ws = WSManager(
            url=self._url,
            token_ids=list(token_ids),
            max_retries=self._max_retries,
        )
        self._shards.append(ws)
        self._shard_last_msg.append(time.time())
        self._shard_reconnects.append(0)

        t = threading.Thread(
            target=self._drain_shard,
            args=(idx, ws),
            daemon=True,
            name=f"ws-pool-drain-{idx}",
        )
        self._drain_threads.append(t)
        t.start()
        await ws.start()
        return idx

    def unsubscribe(self, token_ids: list[str]) -> None:
        """Remove tokens from tracking."""
        for tid in token_ids:
            shard_idx = self._token_to_shard.pop(tid, None)
            if shard_idx is not None and shard_idx < len(self._shard_tokens):
                try:
                    self._shard_tokens[shard_idx].remove(tid)
                except ValueError:
                    pass

    def health(self) -> list[ShardHealth]:
        """Return immutable health snapshot of all shards."""
        result: list[ShardHealth] = []
        for i, ws in enumerate(self._shards):
            result.append(
                ShardHealth(
                    shard_id=i,
                    token_count=len(self._shard_tokens[i]),
                    last_message_at=self._shard_last_msg[i],
                    is_alive=ws._running,
                    reconnect_count=self._shard_reconnects[i],
                )
            )
        return result

    @property
    def shard_count(self) -> int:
        return len(self._shards)

    @property
    def total_tokens(self) -> int:
        return len(self._token_to_shard)

    def _drain_shard(self, shard_idx: int, ws: WSManager) -> None:
        """Background thread: drain a shard's queues into the merged output queues."""
        while self._running:
            drained = False

            try:
                update = ws.book_queue.get(timeout=0.1)
                self.book_queue.put(update)
                self._shard_last_msg[shard_idx] = time.time()
                drained = True
            except queue.Empty:
                pass

            try:
                update = ws.price_queue.get_nowait()
                self.price_queue.put(update)
                self._shard_last_msg[shard_idx] = time.time()
                drained = True
            except queue.Empty:
                pass

            if not drained:
                time.sleep(0.01)
