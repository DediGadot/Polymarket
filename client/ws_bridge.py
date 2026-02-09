"""
Synchronous bridge between async WSManager and the sync main loop.
Runs WSManager in a background thread with its own asyncio event loop.
Feeds book updates into BookCache and price updates into SpikeDetector.
"""

from __future__ import annotations

import asyncio
import logging
import threading

from client.ws import WSManager, BookUpdate, PriceUpdate
from scanner.book_cache import BookCache
from scanner.spike import SpikeDetector

logger = logging.getLogger(__name__)

_DRAIN_BATCH = 100  # max items to drain per call


class WSBridge:
    """
    Synchronous facade over WSManager. Runs WS in a daemon thread.
    Call drain() from the main loop to flush queued updates into BookCache/SpikeDetector.
    """

    def __init__(
        self,
        ws_url: str,
        book_cache: BookCache,
        spike_detector: SpikeDetector | None = None,
        max_retries: int = 5,
    ):
        self._ws_url = ws_url
        self._book_cache = book_cache
        self._spike_detector = spike_detector
        self._max_retries = max_retries

        self._loop: asyncio.AbstractEventLoop | None = None
        self._thread: threading.Thread | None = None
        self._ws: WSManager | None = None
        self._started = False
        self._books_received = 0
        self._prices_received = 0

    def start(self, token_ids: list[str]) -> None:
        """Start the WS background thread and subscribe to token_ids."""
        if self._started:
            if self._thread and self._thread.is_alive():
                return
            logger.warning("WebSocket bridge thread is not alive; restarting")
            self._started = False

        self._ws = WSManager(
            url=self._ws_url,
            token_ids=list(token_ids),
            max_retries=self._max_retries,
        )
        self._loop = asyncio.new_event_loop()
        self._thread = threading.Thread(
            target=self._run_loop,
            daemon=True,
            name="ws-bridge",
        )
        self._started = True
        self._thread.start()
        logger.info("WebSocket bridge started (subscribed to %d tokens)", len(token_ids))

    def stop(self) -> None:
        """Stop the WS background thread."""
        if not self._started:
            return
        self._started = False
        if self._loop and self._ws:
            asyncio.run_coroutine_threadsafe(self._ws.stop(), self._loop)
        if self._loop:
            self._loop.call_soon_threadsafe(self._loop.stop)
        if self._thread:
            self._thread.join(timeout=5.0)
        logger.info(
            "WebSocket bridge stopped (received %d books, %d prices)",
            self._books_received, self._prices_received,
        )

    def subscribe(self, token_ids: list[str]) -> None:
        """Dynamically subscribe to more tokens from the main thread."""
        if not self._started or not self._ws or not self._loop:
            return
        asyncio.run_coroutine_threadsafe(
            self._ws.subscribe(token_ids), self._loop
        )

    def drain(self) -> int:
        """
        Drain queued WS updates into BookCache and SpikeDetector.
        Call this from the main loop at the start of each cycle.
        Returns number of updates processed.
        """
        if not self._ws:
            return 0

        count = 0

        # Drain book updates
        for _ in range(_DRAIN_BATCH):
            try:
                update: BookUpdate = self._ws.book_queue.get_nowait()
                self._book_cache.apply_snapshot(
                    update.token_id, update.bids, update.asks
                )
                self._books_received += 1
                count += 1
            except asyncio.QueueEmpty:
                break

        # Drain price updates
        for _ in range(_DRAIN_BATCH):
            try:
                update: PriceUpdate = self._ws.price_queue.get_nowait()
                if self._spike_detector:
                    self._spike_detector.update(update.token_id, update.price)
                self._prices_received += 1
                count += 1
            except asyncio.QueueEmpty:
                break

        return count

    @property
    def is_connected(self) -> bool:
        return self._started and self._thread is not None and self._thread.is_alive()

    @property
    def stats(self) -> dict:
        return {
            "connected": self.is_connected,
            "books_received": self._books_received,
            "prices_received": self._prices_received,
        }

    def _run_loop(self) -> None:
        """Background thread entry: run asyncio event loop with WSManager."""
        asyncio.set_event_loop(self._loop)
        try:
            self._loop.run_until_complete(self._ws.start())
            self._loop.run_forever()
        except RuntimeError as e:
            if self._started:
                logger.error("WebSocket bridge thread error: %s", e)
        except Exception as e:
            logger.error("WebSocket bridge thread crashed: %s", e)
        finally:
            self._started = False
            self._loop.close()
