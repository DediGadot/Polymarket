"""
Unit tests for scanner/book_cache.py -- local orderbook cache.
"""

import time

from scanner.book_cache import BookCache, _apply_level_update
from scanner.models import PriceLevel


class TestBookCacheSnapshot:
    def test_apply_snapshot_creates_book(self):
        cache = BookCache()
        cache.apply_snapshot("tok1", [
            {"price": "0.50", "size": "100"},
            {"price": "0.48", "size": "200"},
        ], [
            {"price": "0.52", "size": "150"},
            {"price": "0.55", "size": "50"},
        ])
        book = cache.get_book("tok1")
        assert book is not None
        assert book.token_id == "tok1"
        # Bids sorted descending
        assert book.bids[0].price == 0.50
        assert book.bids[1].price == 0.48
        # Asks sorted ascending
        assert book.asks[0].price == 0.52
        assert book.asks[1].price == 0.55

    def test_snapshot_replaces_existing(self):
        cache = BookCache()
        cache.apply_snapshot("tok1", [{"price": "0.50", "size": "100"}], [{"price": "0.52", "size": "100"}])
        cache.apply_snapshot("tok1", [{"price": "0.60", "size": "50"}], [{"price": "0.62", "size": "50"}])
        book = cache.get_book("tok1")
        assert len(book.bids) == 1
        assert book.bids[0].price == 0.60

    def test_snapshot_filters_zero_size(self):
        cache = BookCache()
        cache.apply_snapshot("tok1", [
            {"price": "0.50", "size": "100"},
            {"price": "0.48", "size": "0"},
        ], [
            {"price": "0.52", "size": "0"},
            {"price": "0.55", "size": "50"},
        ])
        book = cache.get_book("tok1")
        assert len(book.bids) == 1
        assert len(book.asks) == 1


class TestBookCacheDelta:
    def test_delta_adds_level(self):
        cache = BookCache()
        cache.apply_snapshot("tok1", [{"price": "0.50", "size": "100"}], [{"price": "0.52", "size": "100"}])
        cache.apply_delta("tok1", {"price": "0.48", "size": "200", "side": "BUY"})
        book = cache.get_book("tok1")
        assert len(book.bids) == 2
        assert book.bids[1].price == 0.48
        assert book.bids[1].size == 200.0

    def test_delta_updates_existing_level(self):
        cache = BookCache()
        cache.apply_snapshot("tok1", [{"price": "0.50", "size": "100"}], [{"price": "0.52", "size": "100"}])
        cache.apply_delta("tok1", {"price": "0.50", "size": "250", "side": "BUY"})
        book = cache.get_book("tok1")
        assert len(book.bids) == 1
        assert book.bids[0].size == 250.0

    def test_delta_removes_level_with_zero_size(self):
        cache = BookCache()
        cache.apply_snapshot("tok1", [
            {"price": "0.50", "size": "100"},
            {"price": "0.48", "size": "200"},
        ], [])
        cache.apply_delta("tok1", {"price": "0.48", "size": "0", "side": "BUY"})
        book = cache.get_book("tok1")
        assert len(book.bids) == 1
        assert book.bids[0].price == 0.50

    def test_delta_on_sell_side(self):
        cache = BookCache()
        cache.apply_snapshot("tok1", [], [{"price": "0.52", "size": "100"}])
        cache.apply_delta("tok1", {"price": "0.55", "size": "75", "side": "SELL"})
        book = cache.get_book("tok1")
        assert len(book.asks) == 2
        assert book.asks[0].price == 0.52
        assert book.asks[1].price == 0.55

    def test_delta_ignored_for_unknown_token(self):
        cache = BookCache()
        cache.apply_delta("unknown", {"price": "0.50", "size": "100", "side": "BUY"})
        assert cache.get_book("unknown") is None


class TestBookCacheStaleness:
    def test_age_after_snapshot(self):
        cache = BookCache()
        cache.apply_snapshot("tok1", [], [])
        assert cache.age("tok1") < 1.0

    def test_age_unknown_token_is_inf(self):
        cache = BookCache()
        assert cache.age("unknown") == float("inf")

    def test_is_stale_when_never_cached(self):
        cache = BookCache(max_age_sec=5.0)
        assert cache.is_stale("unknown") is True

    def test_is_stale_when_fresh(self):
        cache = BookCache(max_age_sec=5.0)
        cache.apply_snapshot("tok1", [], [])
        assert cache.is_stale("tok1") is False

    def test_is_stale_when_old(self):
        cache = BookCache(max_age_sec=0.01)
        cache.apply_snapshot("tok1", [], [])
        time.sleep(0.02)
        assert cache.is_stale("tok1") is True

    def test_stale_tokens(self):
        cache = BookCache(max_age_sec=100.0)
        cache.apply_snapshot("tok1", [], [])
        stale = cache.stale_tokens(["tok1", "tok2", "tok3"])
        assert stale == ["tok2", "tok3"]


class TestBookCacheMultiToken:
    def test_get_books(self):
        cache = BookCache()
        cache.apply_snapshot("tok1", [{"price": "0.50", "size": "100"}], [])
        cache.apply_snapshot("tok2", [{"price": "0.60", "size": "100"}], [])
        books = cache.get_books(["tok1", "tok2", "tok3"])
        assert len(books) == 2
        assert "tok1" in books
        assert "tok2" in books
        assert "tok3" not in books

    def test_token_count(self):
        cache = BookCache()
        assert cache.token_count() == 0
        cache.apply_snapshot("tok1", [], [])
        cache.apply_snapshot("tok2", [], [])
        assert cache.token_count() == 2

    def test_clear(self):
        cache = BookCache()
        cache.apply_snapshot("tok1", [], [])
        cache.clear()
        assert cache.token_count() == 0
        assert cache.get_book("tok1") is None


class TestGetBooksSnapshot:
    def test_get_books_snapshot_returns_consistent_view(self):
        cache = BookCache()
        cache.apply_snapshot("tok1", [{"price": "0.50", "size": "100"}], [{"price": "0.52", "size": "100"}])
        cache.apply_snapshot("tok2", [{"price": "0.60", "size": "200"}], [{"price": "0.62", "size": "200"}])

        books, ts = cache.get_books_snapshot(["tok1", "tok2", "tok3"])
        assert len(books) == 2
        assert "tok1" in books
        assert "tok2" in books
        assert "tok3" not in books
        assert isinstance(ts, float)
        assert ts > 0

    def test_get_books_snapshot_empty(self):
        cache = BookCache()
        books, ts = cache.get_books_snapshot(["nonexistent"])
        assert len(books) == 0
        assert ts > 0


class TestStoreBook:
    def test_store_book_direct(self):
        from scanner.models import OrderBook, PriceLevel
        cache = BookCache()
        book = OrderBook(
            token_id="tok1",
            bids=(PriceLevel(0.50, 100),),
            asks=(PriceLevel(0.52, 100),),
        )
        cache.store_book(book)
        assert cache.get_book("tok1") is book
        assert cache.age("tok1") < 1.0

    def test_store_books_multiple(self):
        from scanner.models import OrderBook, PriceLevel
        cache = BookCache()
        b1 = OrderBook(token_id="t1", bids=(PriceLevel(0.50, 100),), asks=())
        b2 = OrderBook(token_id="t2", bids=(), asks=(PriceLevel(0.55, 100),))
        cache.store_books({"t1": b1, "t2": b2})
        assert cache.token_count() == 2
        assert cache.get_book("t1") is b1
        assert cache.get_book("t2") is b2


class TestCachingFetcher:
    def test_fresh_tokens_skip_rest(self):
        """Tokens already fresh in cache should not trigger REST calls."""
        from scanner.models import OrderBook, PriceLevel
        cache = BookCache(max_age_sec=100.0)
        book = OrderBook(token_id="tok1", bids=(PriceLevel(0.50, 100),), asks=(PriceLevel(0.52, 100),))
        cache.store_book(book)

        rest_calls: list[list[str]] = []

        def mock_rest(token_ids: list[str]) -> dict:
            rest_calls.append(token_ids)
            return {}

        fetcher = cache.make_caching_fetcher(mock_rest)
        result = fetcher(["tok1"])
        assert len(rest_calls) == 0  # no REST call needed
        assert "tok1" in result
        assert result["tok1"].best_bid.price == 0.50

    def test_stale_tokens_fetched_from_rest(self):
        """Stale/missing tokens should be fetched via REST."""
        from scanner.models import OrderBook, PriceLevel
        cache = BookCache(max_age_sec=100.0)

        fresh_book = OrderBook(token_id="tok2", bids=(PriceLevel(0.60, 50),), asks=(PriceLevel(0.62, 50),))

        def mock_rest(token_ids: list[str]) -> dict:
            return {tid: fresh_book for tid in token_ids if tid == "tok2"}

        fetcher = cache.make_caching_fetcher(mock_rest)
        result = fetcher(["tok2"])
        assert "tok2" in result
        assert result["tok2"].best_bid.price == 0.60
        # Also stored in cache
        assert cache.get_book("tok2") is not None

    def test_merge_cached_and_fresh(self):
        """Should return merged dict of cached (fresh) + REST-fetched (stale) tokens."""
        from scanner.models import OrderBook, PriceLevel
        cache = BookCache(max_age_sec=100.0)
        cached_book = OrderBook(token_id="tok1", bids=(PriceLevel(0.50, 100),), asks=(PriceLevel(0.52, 100),))
        cache.store_book(cached_book)

        new_book = OrderBook(token_id="tok2", bids=(PriceLevel(0.40, 200),), asks=(PriceLevel(0.42, 200),))

        rest_calls: list[list[str]] = []

        def mock_rest(token_ids: list[str]) -> dict:
            rest_calls.append(token_ids)
            return {"tok2": new_book}

        fetcher = cache.make_caching_fetcher(mock_rest)
        result = fetcher(["tok1", "tok2"])
        assert len(result) == 2
        assert result["tok1"].best_bid.price == 0.50  # from cache
        assert result["tok2"].best_bid.price == 0.40  # from REST
        # REST was only called for stale token
        assert rest_calls == [["tok2"]]


class TestApplyLevelUpdate:
    def test_add_new_level(self):
        levels = (PriceLevel(0.50, 100.0),)
        result = _apply_level_update(levels, 0.48, 200.0)
        assert len(result) == 2

    def test_replace_existing_level(self):
        levels = (PriceLevel(0.50, 100.0),)
        result = _apply_level_update(levels, 0.50, 250.0)
        assert len(result) == 1
        assert result[0].size == 250.0

    def test_remove_level(self):
        levels = (PriceLevel(0.50, 100.0), PriceLevel(0.48, 200.0))
        result = _apply_level_update(levels, 0.50, 0.0)
        assert len(result) == 1
        assert result[0].price == 0.48
