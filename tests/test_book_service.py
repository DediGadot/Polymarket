"""
Unit tests for scanner/book_service.py -- centralized book data service.
"""

from __future__ import annotations

from unittest.mock import MagicMock

from scanner.book_cache import BookCache
from scanner.book_service import BookService
from scanner.models import OrderBook, PriceLevel


def _make_book(token_id: str, bid: float = 0.50, ask: float = 0.55) -> OrderBook:
    return OrderBook(
        token_id=token_id,
        bids=(PriceLevel(price=bid, size=100.0),),
        asks=(PriceLevel(price=ask, size=100.0),),
    )


class TestPrefetch:
    def test_prefetch_fetches_all_tokens(self):
        """prefetch() should call REST for all requested tokens."""
        cache = BookCache(max_age_sec=100.0)
        books = {
            "tok1": _make_book("tok1"),
            "tok2": _make_book("tok2"),
        }
        mock_rest = MagicMock(return_value=books)
        service = BookService(book_cache=cache, rest_fetcher=mock_rest)

        result = service.prefetch(["tok1", "tok2"])

        assert set(result.keys()) == {"tok1", "tok2"}
        mock_rest.assert_called_once_with(["tok1", "tok2"])

    def test_prefetch_empty_list_returns_empty(self):
        """prefetch() with empty list should return {} and skip REST."""
        cache = BookCache(max_age_sec=100.0)
        mock_rest = MagicMock()
        service = BookService(book_cache=cache, rest_fetcher=mock_rest)

        result = service.prefetch([])

        assert result == {}
        mock_rest.assert_not_called()

    def test_prefetch_stores_in_cache(self):
        """prefetch() should store fetched books in BookCache."""
        cache = BookCache(max_age_sec=100.0)
        books = {"tok1": _make_book("tok1")}
        mock_rest = MagicMock(return_value=books)
        service = BookService(book_cache=cache, rest_fetcher=mock_rest)

        service.prefetch(["tok1"])

        # Now cache should have the book
        assert cache.get_book("tok1") is not None
        assert cache.get_book("tok1").token_id == "tok1"


class TestGetBooks:
    def test_get_books_returns_cached(self):
        """get_books() should return previously cached books without REST."""
        cache = BookCache(max_age_sec=100.0)
        book = _make_book("tok1")
        cache.store_book(book)
        mock_rest = MagicMock()
        service = BookService(book_cache=cache, rest_fetcher=mock_rest)

        result = service.get_books(["tok1"])

        assert "tok1" in result
        assert result["tok1"].token_id == "tok1"
        mock_rest.assert_not_called()

    def test_get_books_missing_tokens_omitted(self):
        """get_books() should omit tokens not in cache."""
        cache = BookCache(max_age_sec=100.0)
        mock_rest = MagicMock()
        service = BookService(book_cache=cache, rest_fetcher=mock_rest)

        result = service.get_books(["nonexistent"])

        assert result == {}
        mock_rest.assert_not_called()


class TestGetBook:
    def test_get_book_returns_cached(self):
        """get_book() returns a single cached book."""
        cache = BookCache(max_age_sec=100.0)
        book = _make_book("tok1")
        cache.store_book(book)
        service = BookService(book_cache=cache, rest_fetcher=MagicMock())

        result = service.get_book("tok1")

        assert result is not None
        assert result.token_id == "tok1"

    def test_get_book_returns_none_for_missing(self):
        """get_book() returns None if token not cached."""
        cache = BookCache(max_age_sec=100.0)
        service = BookService(book_cache=cache, rest_fetcher=MagicMock())

        assert service.get_book("missing") is None


class TestMakeFetcher:
    def test_make_fetcher_returns_callable(self):
        """make_fetcher() should return a BookFetcher callable."""
        cache = BookCache(max_age_sec=100.0)
        mock_rest = MagicMock(return_value={})
        service = BookService(book_cache=cache, rest_fetcher=mock_rest)

        fetcher = service.make_fetcher()

        assert callable(fetcher)

    def test_make_fetcher_falls_back_to_rest_for_stale(self):
        """make_fetcher() callable should REST-fetch tokens not in cache."""
        cache = BookCache(max_age_sec=100.0)
        book = _make_book("tok1")
        mock_rest = MagicMock(return_value={"tok1": book})
        service = BookService(book_cache=cache, rest_fetcher=mock_rest)

        fetcher = service.make_fetcher()
        result = fetcher(["tok1"])

        assert "tok1" in result
        mock_rest.assert_called_once()


class TestMultiScannerDedup:
    def test_three_scanners_one_rest_call(self):
        """
        When 3 scanners call make_fetcher(), the REST backend should be called
        only once per unique token set. After prefetch, cache serves all.
        """
        cache = BookCache(max_age_sec=100.0)
        books = {
            "tok1": _make_book("tok1"),
            "tok2": _make_book("tok2"),
            "tok3": _make_book("tok3"),
        }
        mock_rest = MagicMock(return_value=books)
        service = BookService(book_cache=cache, rest_fetcher=mock_rest)

        # One prefetch populates cache
        service.prefetch(["tok1", "tok2", "tok3"])

        # Simulate 3 scanners reading from cache via make_fetcher()
        fetcher = service.make_fetcher()
        scanner1_result = fetcher(["tok1", "tok2"])
        scanner2_result = fetcher(["tok2", "tok3"])
        scanner3_result = fetcher(["tok1", "tok3"])

        # REST was only called once during prefetch
        mock_rest.assert_called_once_with(["tok1", "tok2", "tok3"])

        # All scanners got their books
        assert set(scanner1_result.keys()) == {"tok1", "tok2"}
        assert set(scanner2_result.keys()) == {"tok2", "tok3"}
        assert set(scanner3_result.keys()) == {"tok1", "tok3"}

    def test_fetcher_calls_rest_only_for_stale_after_prefetch(self):
        """
        After prefetch, fetcher should not call REST again for cached tokens.
        Only stale tokens trigger new REST calls.
        """
        cache = BookCache(max_age_sec=100.0)
        book1 = _make_book("tok1")
        mock_rest = MagicMock(return_value={"tok1": book1})
        service = BookService(book_cache=cache, rest_fetcher=mock_rest)

        service.prefetch(["tok1"])
        assert mock_rest.call_count == 1

        # Scanner reads tok1 (cached, fresh) -- no additional REST call
        fetcher = service.make_fetcher()
        fetcher(["tok1"])
        assert mock_rest.call_count == 1

        # Scanner reads tok2 (not cached) -- triggers REST
        book2 = _make_book("tok2")
        mock_rest.return_value = {"tok2": book2}
        result = fetcher(["tok2"])
        assert mock_rest.call_count == 2
        assert "tok2" in result


class TestStats:
    def test_stats_initial(self):
        """stats should return zeroed values before any prefetch."""
        cache = BookCache(max_age_sec=100.0)
        service = BookService(book_cache=cache, rest_fetcher=MagicMock())

        s = service.stats

        assert s["prefetch_count"] == 0
        assert s["last_prefetch_at"] == 0.0
        assert s["last_prefetch_tokens"] == 0
        assert s["cached_tokens"] == 0

    def test_stats_after_prefetch(self):
        """stats should reflect prefetch activity."""
        cache = BookCache(max_age_sec=100.0)
        books = {"tok1": _make_book("tok1"), "tok2": _make_book("tok2")}
        mock_rest = MagicMock(return_value=books)
        service = BookService(book_cache=cache, rest_fetcher=mock_rest)

        service.prefetch(["tok1", "tok2"])
        s = service.stats

        assert s["prefetch_count"] == 1
        assert s["last_prefetch_at"] > 0.0
        assert s["last_prefetch_tokens"] == 2
        assert s["cached_tokens"] == 2

    def test_stats_increment_across_cycles(self):
        """stats.prefetch_count should increment on each prefetch call."""
        cache = BookCache(max_age_sec=0.0)  # always stale -> always fetches
        books = {"tok1": _make_book("tok1")}
        mock_rest = MagicMock(return_value=books)
        service = BookService(book_cache=cache, rest_fetcher=mock_rest)

        service.prefetch(["tok1"])
        service.prefetch(["tok1"])
        service.prefetch(["tok1"])

        assert service.stats["prefetch_count"] == 3
