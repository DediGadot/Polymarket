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

    def test_prune_removes_old_entries(self):
        """prune() should remove entries older than max_age_sec."""
        import time
        cache = BookCache(max_age_sec=5.0)

        # Add entries at different times
        now = time.time()
        from scanner.models import OrderBook, PriceLevel

        cache.store_book(OrderBook(token_id="old1", bids=(PriceLevel(0.50, 100),), asks=()),
                     timestamp=now - 400)  # 6m40s ago
        cache.store_book(OrderBook(token_id="old2", bids=(PriceLevel(0.51, 100),), asks=()),
                     timestamp=now - 350)  # 5m50s ago
        cache.store_book(OrderBook(token_id="fresh1", bids=(PriceLevel(0.52, 100),), asks=()),
                     timestamp=now - 100)  # 1m40s ago
        cache.store_book(OrderBook(token_id="fresh2", bids=(PriceLevel(0.53, 100),), asks=()),
                     timestamp=now)  # just now

        # Before prune: 4 entries
        assert cache.token_count() == 4

        # Prune entries older than 5 minutes (300 sec)
        cache.prune(max_age_sec=300)

        # After prune: only 2 fresh entries remain
        assert cache.token_count() == 2
        assert cache.get_book("old1") is None
        assert cache.get_book("old2") is None
        assert cache.get_book("fresh1") is not None
        assert cache.get_book("fresh2") is not None

    def test_prune_with_default_max_age(self):
        """prune() with default max_age should use cache's max_age_sec."""
        import time
        cache = BookCache(max_age_sec=5.0)
        from scanner.models import OrderBook, PriceLevel

        now = time.time()
        cache.store_book(OrderBook(token_id="old", bids=(PriceLevel(0.50, 100),), asks=()),
                     timestamp=now - 400)
        cache.store_book(OrderBook(token_id="fresh", bids=(PriceLevel(0.51, 100),), asks=()),
                     timestamp=now)

        assert cache.token_count() == 2
        cache.prune()  # Should use default max_age_sec=5.0
        assert cache.token_count() == 1
        assert cache.get_book("fresh") is not None

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


class TestPrefetchEliminatesRedundantCalls:
    """Verify that pre-fetching books into cache prevents redundant REST calls
    from multiple concurrent consumers (binary, maker, resolution scanners)."""

    def test_prefetch_prevents_duplicate_rest_calls(self):
        """After one prefetch, subsequent fetcher calls should be cache hits."""
        from scanner.models import OrderBook, PriceLevel
        cache = BookCache(max_age_sec=100.0)

        # Simulate 4 binary markets (8 tokens)
        token_ids = [f"yes_{i}" for i in range(4)] + [f"no_{i}" for i in range(4)]
        books = {
            tid: OrderBook(
                token_id=tid,
                bids=(PriceLevel(0.50, 100),),
                asks=(PriceLevel(0.52, 100),),
            )
            for tid in token_ids
        }

        rest_call_count = 0

        def mock_rest(tids: list[str]) -> dict:
            nonlocal rest_call_count
            rest_call_count += 1
            return {tid: books[tid] for tid in tids if tid in books}

        fetcher = cache.make_caching_fetcher(mock_rest)

        # Pre-fetch all tokens (simulates the new prefetch step)
        fetcher(token_ids)
        assert rest_call_count == 1  # single REST batch

        # Subsequent calls from different scanners should be pure cache hits
        fetcher(token_ids)  # binary scanner
        fetcher(token_ids)  # maker scanner
        fetcher(token_ids)  # resolution scanner
        assert rest_call_count == 1  # still only one REST call total

    def test_prefetch_serves_correct_data(self):
        """Pre-fetched books should have correct data for all consumers."""
        from scanner.models import OrderBook, PriceLevel
        cache = BookCache(max_age_sec=100.0)

        books = {
            "yes_0": OrderBook(token_id="yes_0", bids=(PriceLevel(0.30, 50),), asks=(PriceLevel(0.35, 50),)),
            "no_0": OrderBook(token_id="no_0", bids=(PriceLevel(0.60, 80),), asks=(PriceLevel(0.65, 80),)),
        }

        def mock_rest(tids: list[str]) -> dict:
            return {tid: books[tid] for tid in tids if tid in books}

        fetcher = cache.make_caching_fetcher(mock_rest)
        fetcher(["yes_0", "no_0"])  # prefetch

        # Subsequent fetch returns same data
        result = fetcher(["yes_0", "no_0"])
        assert result["yes_0"].best_ask.price == 0.35
        assert result["no_0"].best_bid.price == 0.60


class TestBatchStalenessCheck:
    """Verify stale_tokens uses efficient batch checking (single time.time())."""

    def test_stale_tokens_batch_returns_correct_results(self):
        """stale_tokens should return stale/missing tokens only."""
        from scanner.models import OrderBook, PriceLevel
        cache = BookCache(max_age_sec=100.0)
        cache.store_book(OrderBook(token_id="fresh", bids=(PriceLevel(0.50, 100),), asks=()))
        stale = cache.stale_tokens(["fresh", "missing1", "missing2"])
        assert "fresh" not in stale
        assert "missing1" in stale
        assert "missing2" in stale

    def test_caching_fetcher_returns_all_fresh_without_rest(self):
        """When 1000 tokens are all fresh, no REST call and all returned."""
        from scanner.models import OrderBook, PriceLevel
        cache = BookCache(max_age_sec=100.0)
        # Store 1000 books
        for i in range(1000):
            cache.store_book(OrderBook(
                token_id=f"tok_{i}",
                bids=(PriceLevel(0.50, 100),),
                asks=(PriceLevel(0.52, 100),),
            ))
        rest_calls = []

        def mock_rest(tids):
            rest_calls.append(tids)
            return {}

        fetcher = cache.make_caching_fetcher(mock_rest)
        result = fetcher([f"tok_{i}" for i in range(1000)])
        assert len(rest_calls) == 0
        assert len(result) == 1000


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
