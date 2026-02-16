"""
Synchronous bridge between async WSManager/WSPool and the sync main loop.
Runs WS in a background thread with its own asyncio event loop.
Feeds book updates into BookCache, price updates into SpikeDetector,
and order flow imbalance into OFITracker.
Automatically uses WSPool when token count exceeds max_tokens_per_conn.
"""

from __future__ import annotations

import asyncio
import logging
import queue
import threading
from typing import Union

from client.ws import WSManager, BookUpdate, PriceUpdate
from client.ws_pool import WSPool, DEFAULT_MAX_TOKENS_PER_CONN
from scanner.book_cache import BookCache
from scanner.ofi import OFITracker
from scanner.spike import SpikeDetector

logger = logging.getLogger(__name__)

_DRAIN_BATCH = 100  # max items to drain per call


class WSBridge:
    """
    Synchronous facade over WSManager/WSPool. Runs WS in a daemon thread.
    Call drain() from the main loop to flush queued updates into BookCache/SpikeDetector.

    When token count exceeds max_tokens_per_conn, automatically uses WSPool
    to shard subscriptions across multiple connections. Transparent to callers.
    """

    def __init__(
        self,
        ws_url: str,
        book_cache: BookCache,
        spike_detector: SpikeDetector | None = None,
        ofi_tracker: OFITracker | None = None,
        max_retries: int = 5,
        max_tokens_per_conn: int = DEFAULT_MAX_TOKENS_PER_CONN,
    ):
        self._ws_url = ws_url
        self._book_cache = book_cache
        self._spike_detector = spike_detector
        self._ofi_tracker = ofi_tracker
        self._max_retries = max_retries
        self._max_tokens_per_conn = max_tokens_per_conn

        self._loop: asyncio.AbstractEventLoop | None = None
        self._thread: threading.Thread | None = None
        self._ws: Union[WSManager, WSPool, None] = None
        self._using_pool = False
        self._started = False
        self._books_received = 0
        self._prices_received = 0
        self._last_changed_tokens: set[str] = set()
        # token_id -> (bid_px, bid_sz, ask_px, ask_sz)
        self._last_tops: dict[str, tuple[float, float, float, float]] = {}

    def start(self, token_ids: list[str]) -> None:
        """Start the WS background thread and subscribe to token_ids."""
        if self._started:
            if self._thread and self._thread.is_alive():
                return
            logger.warning("WebSocket bridge thread is not alive; restarting")
            self._started = False

        token_list = list(token_ids)
        if len(token_list) > self._max_tokens_per_conn:
            self._ws = WSPool(
                url=self._ws_url,
                token_ids=token_list,
                max_tokens_per_conn=self._max_tokens_per_conn,
                max_retries=self._max_retries,
            )
            self._using_pool = True
            logger.info(
                "WebSocket bridge using pool (%d tokens, %d max/conn)",
                len(token_list), self._max_tokens_per_conn,
            )
        else:
            self._ws = WSManager(
                url=self._ws_url,
                token_ids=token_list,
                max_retries=self._max_retries,
            )
            self._using_pool = False

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
        self._last_tops.clear()
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
        Drain queued WS updates into BookCache, SpikeDetector, and OFITracker.
        Call this from the main loop at the start of each cycle.
        Returns number of updates processed.
        """
        self._last_changed_tokens.clear()
        if not self._ws:
            return 0

        count = 0

        # Drain book updates → BookCache + OFITracker
        for _ in range(_DRAIN_BATCH):
            try:
                update: BookUpdate = self._ws.book_queue.get_nowait()
                if self._ofi_tracker:
                    self._feed_ofi_from_book(update)
                self._book_cache.apply_snapshot(
                    update.token_id, update.bids, update.asks
                )
                self._last_changed_tokens.add(update.token_id)
                self._books_received += 1
                count += 1
            except queue.Empty:
                break

        # Drain price updates → SpikeDetector
        for _ in range(_DRAIN_BATCH):
            try:
                update: PriceUpdate = self._ws.price_queue.get_nowait()
                if self._ofi_tracker and update.side in ("BUY", "SELL") and update.size > 0:
                    # Prefer directional trade-like updates when available.
                    self._ofi_tracker.record(
                        update.token_id,
                        update.side,
                        update.size,
                        timestamp=update.timestamp,
                    )
                if self._spike_detector:
                    self._spike_detector.update(update.token_id, update.price)
                self._last_changed_tokens.add(update.token_id)
                self._prices_received += 1
                count += 1
            except queue.Empty:
                break

        return count

    def _feed_ofi_from_book(self, update: BookUpdate) -> None:
        """
        Approximate aggressor flow from top-of-book transitions.

        Heuristics:
        - Ask depletion / ask price uptick -> aggressive buys.
        - Bid depletion / bid price downtick -> aggressive sells.
        - Fallback to raw depth imbalance when no prior top exists.
        """
        def _best_bid(levels: list[dict]) -> tuple[float, float] | None:
            best: tuple[float, float] | None = None
            for lvl in levels:
                try:
                    px = float(lvl.get("price", 0.0))
                    sz = float(lvl.get("size", 0.0))
                except (TypeError, ValueError):
                    continue
                if sz <= 0:
                    continue
                if best is None or px > best[0]:
                    best = (px, sz)
            return best

        def _best_ask(levels: list[dict]) -> tuple[float, float] | None:
            best: tuple[float, float] | None = None
            for lvl in levels:
                try:
                    px = float(lvl.get("price", 0.0))
                    sz = float(lvl.get("size", 0.0))
                except (TypeError, ValueError):
                    continue
                if sz <= 0:
                    continue
                if best is None or px < best[0]:
                    best = (px, sz)
            return best

        bid = _best_bid(update.bids)
        ask = _best_ask(update.asks)
        if bid is None or ask is None:
            return

        new_bid_px, new_bid_sz = bid
        new_ask_px, new_ask_sz = ask

        prev = self._last_tops.get(update.token_id)
        buy_volume = 0.0
        sell_volume = 0.0

        if prev is None:
            # Cold-start fallback: snapshot imbalance proxy.
            bid_vol = sum(float(b.get("size", 0)) for b in update.bids)
            ask_vol = sum(float(a.get("size", 0)) for a in update.asks)
            imbalance = bid_vol - ask_vol
            if imbalance > 0:
                buy_volume = imbalance
            elif imbalance < 0:
                sell_volume = -imbalance
        else:
            prev_bid_px, prev_bid_sz, prev_ask_px, prev_ask_sz = prev

            # Aggressive BUY proxy: ask consumed.
            if new_ask_px > prev_ask_px + 1e-9:
                buy_volume += max(0.0, prev_ask_sz)
            elif abs(new_ask_px - prev_ask_px) <= 1e-9 and new_ask_sz < prev_ask_sz:
                buy_volume += prev_ask_sz - new_ask_sz

            # Aggressive SELL proxy: bid consumed.
            if new_bid_px < prev_bid_px - 1e-9:
                sell_volume += max(0.0, prev_bid_sz)
            elif abs(new_bid_px - prev_bid_px) <= 1e-9 and new_bid_sz < prev_bid_sz:
                sell_volume += prev_bid_sz - new_bid_sz

            # If transitions are inconclusive, use delta of top imbalance.
            if buy_volume == 0.0 and sell_volume == 0.0:
                prev_imb = prev_bid_sz - prev_ask_sz
                new_imb = new_bid_sz - new_ask_sz
                delta_imb = new_imb - prev_imb
                if delta_imb > 0:
                    buy_volume = delta_imb
                elif delta_imb < 0:
                    sell_volume = -delta_imb

            prev_mid = (prev_bid_px + prev_ask_px) / 2.0
            new_mid = (new_bid_px + new_ask_px) / 2.0
            ofi_signal = buy_volume - sell_volume
            mid_move = new_mid - prev_mid
            if abs(ofi_signal) > 0 and abs(mid_move) > 0:
                self._ofi_tracker.record_quality(ofi_signal, mid_move)

        if buy_volume > 0 or sell_volume > 0:
            self._ofi_tracker.record_aggressor(
                update.token_id,
                buy_volume=buy_volume,
                sell_volume=sell_volume,
                timestamp=update.timestamp,
            )

        self._last_tops[update.token_id] = (new_bid_px, new_bid_sz, new_ask_px, new_ask_sz)

    @property
    def last_changed_tokens(self) -> set[str]:
        """Token IDs touched by the most recent drain() call."""
        return set(self._last_changed_tokens)

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
