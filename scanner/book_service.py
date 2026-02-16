"""
Centralized book data service. Single fetch per cycle, multiple scanner consumers.
Wraps BookCache + REST fetcher into a unified service layer.
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field

from scanner.models import BookFetcher, OrderBook
from scanner.book_cache import BookCache

logger = logging.getLogger(__name__)


@dataclass
class BookService:
    """
    Owns the single book fetch per cycle. All scanners read from this service
    instead of calling get_orderbooks_parallel() directly.

    Usage:
        service = BookService(book_cache=cache, rest_fetcher=clob_fetcher)
        service.prefetch(all_token_ids)  # one REST call per cycle
        books = service.get_books(subset_token_ids)  # from cache, no REST
        fetcher = service.make_fetcher()  # BookFetcher for scanners
    """

    book_cache: BookCache
    rest_fetcher: BookFetcher
    _prefetch_count: int = field(default=0, init=False)
    _last_prefetch_at: float = field(default=0.0, init=False)
    _last_prefetch_tokens: int = field(default=0, init=False)
    _fetcher: BookFetcher | None = field(default=None, init=False)

    def __post_init__(self) -> None:
        self._fetcher = self.book_cache.make_caching_fetcher(self.rest_fetcher)

    def prefetch(self, token_ids: list[str]) -> dict[str, OrderBook]:
        """
        Fetch all requested books in one batch via REST (through caching layer).
        Returns the fetched books. Subsequent get_books() calls will hit cache.
        """
        unique_ids = list(dict.fromkeys(token_ids))
        if not unique_ids:
            return {}
        start = time.monotonic()
        books = self.make_fetcher()(unique_ids)
        elapsed = time.monotonic() - start
        self._prefetch_count += 1
        self._last_prefetch_at = time.time()
        self._last_prefetch_tokens = len(unique_ids)
        logger.debug(
            "BookService: prefetched %d tokens in %.3fs (cycle %d)",
            len(unique_ids),
            elapsed,
            self._prefetch_count,
        )
        return books

    def get_books(self, token_ids: list[str]) -> dict[str, OrderBook]:
        """Get books from cache. No REST calls -- returns only what's cached."""
        return self.book_cache.get_books(token_ids)

    def get_book(self, token_id: str) -> OrderBook | None:
        """Get a single book from cache."""
        return self.book_cache.get_book(token_id)

    def make_fetcher(self) -> BookFetcher:
        """
        Return a BookFetcher callable that reads from cache first,
        falling back to REST for any stale/missing tokens.
        Compatible with existing scanner interfaces.
        """
        if self._fetcher is None:
            self._fetcher = self.book_cache.make_caching_fetcher(self.rest_fetcher)
        return self._fetcher

    @property
    def stats(self) -> dict:
        return {
            "prefetch_count": self._prefetch_count,
            "last_prefetch_at": self._last_prefetch_at,
            "last_prefetch_tokens": self._last_prefetch_tokens,
            "cached_tokens": self.book_cache.token_count(),
        }
