"""
WebSocket manager for real-time price feeds. Fail-fast: raises after max retries.
"""

from __future__ import annotations

import asyncio
import json
import logging
import queue
import time
from dataclasses import dataclass, field

import websockets
from websockets.asyncio.client import connect

logger = logging.getLogger(__name__)

MAX_RETRIES = 5
BACKOFF_BASE = 1.0  # seconds
BACKOFF_MAX = 30.0


@dataclass
class PriceUpdate:
    token_id: str
    price: float
    timestamp: float


@dataclass
class BookUpdate:
    token_id: str
    bids: list[dict]
    asks: list[dict]
    timestamp: float


@dataclass
class WSManager:
    """
    Manages a WebSocket connection to Polymarket's market data feed.
    Emits price and book updates via bounded synchronous queues.
    """
    url: str
    token_ids: list[str]
    price_queue: queue.Queue = field(default_factory=lambda: queue.Queue(maxsize=10000))
    book_queue: queue.Queue = field(default_factory=lambda: queue.Queue(maxsize=10000))
    max_retries: int = MAX_RETRIES
    _running: bool = False
    _ws: object = None
    _task: asyncio.Task | None = None
    # Health tracking
    _last_message_time: float = field(default_factory=lambda: 0.0)
    _connect_time: float = field(default_factory=lambda: 0.0)
    _failed_connections: int = 0

    async def start(self) -> None:
        """Start the WebSocket listener in a background task."""
        self._running = True
        self._task = asyncio.create_task(self._run_loop())

    async def stop(self) -> None:
        """Stop the WebSocket listener."""
        self._running = False
        if self._task:
            self._task.cancel()
            try:
                await self._task
            except asyncio.CancelledError:
                pass
        if self._ws:
            await self._ws.close()

    async def subscribe(self, token_ids: list[str]) -> None:
        """Dynamically subscribe to additional tokens."""
        self.token_ids.extend(token_ids)
        if self._ws:
            msg = json.dumps({
                "assets_ids": token_ids,
                "type": "market",
            })
            await self._ws.send(msg)

    async def unsubscribe(self, token_ids: list[str]) -> None:
        """Dynamically unsubscribe from tokens."""
        for tid in token_ids:
            if tid in self.token_ids:
                self.token_ids.remove(tid)
        if self._ws:
            msg = json.dumps({
                "assets_ids": token_ids,
                "type": "market",
                "action": "unsubscribe",
            })
            await self._ws.send(msg)

    def is_healthy(self, max_silence_sec: float = 30.0) -> bool:
        """
        Check if WebSocket connection is healthy.

        Args:
            max_silence_sec: Maximum seconds since last message before
                              considering the connection unhealthy.

        Returns:
            True if connected and receiving messages, False if silent or not running.
        """
        if not self._running or not self._ws:
            return False

        # Check if we've received messages recently
        now = time.time()
        time_since_last = now - self._last_message_time

        # Never received a message
        if self._last_message_time == 0.0:
            # Give some grace period after connection
            if now - self._connect_time > max_silence_sec:
                return False
            return True

        return time_since_last <= max_silence_sec

    def _record_message(self) -> None:
        """Update last message time for health tracking."""
        self._last_message_time = time.time()

    async def _run_loop(self) -> None:
        """Connect and listen, with exponential backoff on failures."""
        retries = 0
        while self._running:
            try:
                async with connect(self.url) as ws:
                    self._ws = ws
                    self._connect_time = time.time()
                    self._last_message_time = 0.0  # Reset message time on reconnect
                    retries = 0  # reset on successful connection
                    self._failed_connections = 0
                    logger.info("WebSocket connected to %s", self.url)

                    # Send initial subscription
                    sub_msg = json.dumps({
                        "assets_ids": self.token_ids,
                        "type": "market",
                    })
                    await ws.send(sub_msg)

                    async for raw_msg in ws:
                        if not self._running:
                            break
                        self._record_message()
                        self._handle_message(raw_msg)

            except (websockets.ConnectionClosed, ConnectionError, OSError) as e:
                retries += 1
                self._failed_connections += 1
                if retries > self.max_retries:
                    logger.error(
                        "WebSocket max retries (%d) exceeded. Last error: %s",
                        self.max_retries, e,
                    )
                    raise RuntimeError(
                        f"WebSocket connection failed after {self.max_retries} retries: {e}"
                    ) from e

                backoff = min(BACKOFF_BASE * (2 ** (retries - 1)), BACKOFF_MAX)
                logger.warning(
                    "WebSocket disconnected (retry %d/%d), backoff %.1fs: %s",
                    retries, self.max_retries, backoff, e,
                )
                await asyncio.sleep(backoff)

    def _handle_message(self, raw_msg: str) -> None:
        """Parse a WebSocket message and enqueue updates."""
        try:
            data = json.loads(raw_msg)
        except json.JSONDecodeError:
            logger.warning("Unparseable WebSocket message: %s", raw_msg[:200])
            return

        # The WS sends arrays of events
        events = data if isinstance(data, list) else [data]
        now = time.time()

        for event in events:
            event_type = event.get("event_type", "")
            asset_id = event.get("asset_id", "")

            if event_type == "price_change":
                try:
                    price = float(event["price"])
                    update = PriceUpdate(
                        token_id=asset_id, price=price, timestamp=now
                    )
                    try:
                        self.price_queue.put_nowait(update)
                    except queue.Full:
                        # Drop oldest entry to make room
                        try:
                            self.price_queue.get_nowait()
                            self.price_queue.put_nowait(update)
                        except queue.Empty:
                            pass
                        logger.warning("Price queue full, dropped oldest for %s", asset_id)
                except (KeyError, ValueError) as e:
                    logger.warning("Bad price_change event: %s", e)

            elif event_type == "book":
                update = BookUpdate(
                    token_id=asset_id,
                    bids=event.get("bids", []),
                    asks=event.get("asks", []),
                    timestamp=now,
                )
                try:
                    self.book_queue.put_nowait(update)
                except queue.Full:
                    # Drop oldest entry to make room
                    try:
                        self.book_queue.get_nowait()
                        self.book_queue.put_nowait(update)
                    except queue.Empty:
                        pass
                    logger.warning("Book queue full, dropped oldest for %s", asset_id)
