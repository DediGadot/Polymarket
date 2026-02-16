"""
Local orderbook cache fed by WebSocket updates.
Maintains in-memory orderbooks per token_id, updated via snapshots and deltas.
Scanners read from cache instead of REST calls per cycle.

Thread-safe for concurrent WS-writer + main-loop-reader pattern via threading.Lock.
"""

from __future__ import annotations

import logging
import threading
import time
from dataclasses import dataclass, field

from scanner.models import BookFetcher, OrderBook, PriceLevel
from scanner.validation import validate_price, validate_size

logger = logging.getLogger(__name__)


@dataclass
class BookCache:
    """
    In-memory orderbook cache. Fed by WS 'book' snapshots and 'price_change' deltas.
    Thread-safe via Lock for concurrent writer (WS thread) + reader (main loop).
    """
    max_age_sec: float = 5.0
    _books: dict[str, OrderBook] = field(default_factory=dict)
    _timestamps: dict[str, float] = field(default_factory=dict)
    _lock: threading.Lock = field(default_factory=threading.Lock)

    def apply_snapshot(self, token_id: str, bids: list[dict], asks: list[dict]) -> None:
        """
        Replace entire book from WS 'book' event.
        bids/asks: [{"price": "0.52", "size": "100"}, ...]
        """
        parsed_bids = tuple(
            PriceLevel(
                price=validate_price(float(b["price"]), context=f"WS snapshot bid ({token_id})"),
                size=validate_size(float(b.get("size", 0)), context=f"WS snapshot bid size ({token_id})"),
            )
            for b in bids if float(b.get("size", 0)) > 0
        )
        parsed_asks = tuple(
            PriceLevel(
                price=validate_price(float(a["price"]), context=f"WS snapshot ask ({token_id})"),
                size=validate_size(float(a.get("size", 0)), context=f"WS snapshot ask size ({token_id})"),
            )
            for a in asks if float(a.get("size", 0)) > 0
        )
        # Sort bids descending (best first), asks ascending (best first)
        sorted_bids = tuple(sorted(parsed_bids, key=lambda p: p.price, reverse=True))
        sorted_asks = tuple(sorted(parsed_asks, key=lambda p: p.price))

        book = OrderBook(
            token_id=token_id,
            bids=sorted_bids,
            asks=sorted_asks,
        )
        with self._lock:
            self._books[token_id] = book
            self._timestamps[token_id] = time.time()

    def apply_delta(self, token_id: str, price_change: dict) -> None:
        """
        Apply incremental update from WS 'price_change' event.
        price_change contains a single price level update with side info.
        Format: {"price": "0.52", "size": "150", "side": "BUY"}
        Size of 0 means remove that level.
        """
        with self._lock:
            book = self._books.get(token_id)
            if not book:
                logger.debug("Delta for unknown token %s, ignoring", token_id)
                return

            price = validate_price(float(price_change["price"]), context=f"WS delta price ({token_id})")
            size = validate_size(float(price_change.get("size", 0)), context=f"WS delta size ({token_id})")
            side = price_change.get("side", "").upper()

            if side == "BUY":
                new_bids = _apply_level_update(book.bids, price, size)
                sorted_bids = tuple(sorted(new_bids, key=lambda p: p.price, reverse=True))
                self._books[token_id] = OrderBook(
                    token_id=token_id, bids=sorted_bids, asks=book.asks,
                )
            elif side == "SELL":
                new_asks = _apply_level_update(book.asks, price, size)
                sorted_asks = tuple(sorted(new_asks, key=lambda p: p.price))
                self._books[token_id] = OrderBook(
                    token_id=token_id, bids=book.bids, asks=sorted_asks,
                )
            else:
                logger.warning("Unknown side in price_change: %s", side)
                return

            self._timestamps[token_id] = time.time()

    def store_book(self, book: OrderBook, timestamp: float | None = None) -> None:
        """Store a fully-formed OrderBook object and mark it fresh."""
        with self._lock:
            self._books[book.token_id] = book
            self._timestamps[book.token_id] = timestamp if timestamp is not None else time.time()

    def store_books(self, books: dict[str, OrderBook], timestamp: float | None = None) -> None:
        """Store multiple fully-formed OrderBook objects."""
        ts = timestamp if timestamp is not None else time.time()
        with self._lock:
            for book in books.values():
                self._books[book.token_id] = book
                self._timestamps[book.token_id] = ts

    def get_book(self, token_id: str) -> OrderBook | None:
        """Return cached book or None if not cached."""
        with self._lock:
            return self._books.get(token_id)

    def get_books(self, token_ids: list[str]) -> dict[str, OrderBook]:
        """Return cached books for multiple tokens. Missing tokens are omitted."""
        with self._lock:
            return {tid: self._books[tid] for tid in token_ids if tid in self._books}

    def get_books_snapshot(self, token_ids: list[str]) -> tuple[dict[str, OrderBook], float]:
        """
        Return a consistent snapshot of multiple books and the snapshot timestamp.
        Holding the lock ensures no interleaved writes during the multi-token read.
        """
        with self._lock:
            now = time.time()
            books = {tid: self._books[tid] for tid in token_ids if tid in self._books}
            return books, now

    def age(self, token_id: str) -> float:
        """Seconds since last update. Returns inf if never cached."""
        ts = self._timestamps.get(token_id)
        if ts is None:
            return float("inf")
        return time.time() - ts

    def is_stale(self, token_id: str) -> bool:
        """True if book is older than max_age_sec or not cached."""
        return self.age(token_id) > self.max_age_sec

    def stale_tokens(self, token_ids: list[str]) -> list[str]:
        """Return subset of token_ids whose cached books are stale or missing.

        Uses a single time.time() call and single lock acquisition instead of
        per-token calls -- significant for 1000+ token batches.
        """
        now = time.time()
        max_age = self.max_age_sec
        with self._lock:
            return [
                tid for tid in token_ids
                if now - self._timestamps.get(tid, 0) > max_age
            ]

    def token_count(self) -> int:
        """Number of tokens currently cached."""
        return len(self._books)

    def prune(self, max_age_sec: float | None = None) -> int:
        """
        Remove cached books older than max_age_sec.

        Prevents memory leak in long-running sessions by pruning
        stale entries that haven't been updated recently.

        Args:
            max_age_sec: Maximum age in seconds. Defaults to self.max_age_sec.

        Returns:
            Number of entries removed.
        """
        if max_age_sec is None:
            max_age_sec = self.max_age_sec

        now = time.time()
        removed = 0

        with self._lock:
            # Find stale token IDs
            stale_tokens = [
                token_id
                for token_id, ts in self._timestamps.items()
                if now - ts > max_age_sec
            ]

            # Remove stale entries
            for token_id in stale_tokens:
                self._books.pop(token_id, None)
                self._timestamps.pop(token_id, None)
                removed += 1

        if removed > 0:
            logger.debug(
                "BookCache: pruned %d stale entries (%d remaining, max_age=%ds)",
                removed, len(self._books), max_age_sec,
            )

        return removed

    def make_caching_fetcher(self, rest_fetcher: BookFetcher) -> BookFetcher:
        """
        Return a BookFetcher callable that:
          1. Checks cache for fresh books (single lock, single time.time())
          2. REST-fetches only stale/missing tokens via rest_fetcher
          3. Stores fresh results in cache
          4. Returns merged dict (cached + fresh) via batch get_books()
        """
        def _caching_fetcher(token_ids: list[str]) -> dict[str, OrderBook]:
            stale = self.stale_tokens(token_ids)
            fresh_from_rest: dict[str, OrderBook] = {}
            if stale:
                fresh_from_rest = rest_fetcher(stale)
                self.store_books(fresh_from_rest)
            # Merge via single lock acquisition for all cached books
            non_stale = [tid for tid in token_ids if tid not in fresh_from_rest]
            cached = self.get_books(non_stale)
            return {**cached, **fresh_from_rest}
        return _caching_fetcher

    def get_all_books(self) -> dict[str, OrderBook]:
        """Return a snapshot of all cached books."""
        with self._lock:
            return dict(self._books)

    def clear(self) -> None:
        """Drop all cached data."""
        with self._lock:
            self._books.clear()
            self._timestamps.clear()


def _apply_level_update(
    levels: tuple[PriceLevel, ...],
    price: float,
    size: float,
) -> list[PriceLevel]:
    """
    Apply a single level update: replace existing level at price, add new, or remove (size=0).
    """
    result = [lvl for lvl in levels if abs(lvl.price - price) > 1e-12]
    if size > 0:
        result.append(PriceLevel(price=price, size=size))
    return result
