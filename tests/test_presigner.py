"""
Unit tests for executor/presigner.py -- order pre-signing cache.
"""

from __future__ import annotations

import time
from unittest.mock import MagicMock, call

import pytest

from executor.presigner import OrderPresigner, PresignKey, PresignedOrder


def _make_sign_fn():
    """Return a MagicMock sign_fn that returns unique signed order objects."""
    fn = MagicMock()
    fn.side_effect = lambda **kwargs: {
        "signed": True,
        "token_id": kwargs["token_id"],
        "side": kwargs["side"],
        "price": kwargs["price"],
        "size": kwargs["size"],
    }
    return fn


class TestCacheHit:
    def test_returns_cached_order_on_second_call(self):
        sign_fn = _make_sign_fn()
        ps = OrderPresigner(sign_fn=sign_fn, max_cache_size=10)

        result1 = ps.get_or_sign("tok1", "BUY", 0.50, 10.0)
        result2 = ps.get_or_sign("tok1", "BUY", 0.50, 10.0)

        assert result1 == result2
        assert sign_fn.call_count == 1

    def test_hit_increments_stats(self):
        sign_fn = _make_sign_fn()
        ps = OrderPresigner(sign_fn=sign_fn)

        ps.get_or_sign("tok1", "BUY", 0.50, 10.0)
        ps.get_or_sign("tok1", "BUY", 0.50, 10.0)

        stats = ps.stats
        assert stats["hits"] == 1
        assert stats["misses"] == 1


class TestCacheMiss:
    def test_calls_sign_fn_on_miss(self):
        sign_fn = _make_sign_fn()
        ps = OrderPresigner(sign_fn=sign_fn)

        result = ps.get_or_sign("tok1", "BUY", 0.50, 10.0)

        assert result is not None
        sign_fn.assert_called_once_with(
            token_id="tok1",
            side="BUY",
            price=0.50,
            size=10.0,
            neg_risk=False,
            tick_size="0.01",
        )

    def test_different_keys_are_separate_misses(self):
        sign_fn = _make_sign_fn()
        ps = OrderPresigner(sign_fn=sign_fn)

        ps.get_or_sign("tok1", "BUY", 0.50, 10.0)
        ps.get_or_sign("tok1", "SELL", 0.50, 10.0)
        ps.get_or_sign("tok2", "BUY", 0.50, 10.0)

        assert sign_fn.call_count == 3
        assert ps.stats["misses"] == 3


class TestLRUEviction:
    def test_evicts_oldest_when_full(self):
        sign_fn = _make_sign_fn()
        ps = OrderPresigner(sign_fn=sign_fn, max_cache_size=3)

        # Fill cache to capacity
        ps.get_or_sign("tok1", "BUY", 0.50, 10.0)
        ps.get_or_sign("tok2", "BUY", 0.50, 10.0)
        ps.get_or_sign("tok3", "BUY", 0.50, 10.0)
        assert ps.stats["cache_size"] == 3
        assert ps.stats["evictions"] == 0

        # Add one more -- oldest (tok1) should be evicted
        ps.get_or_sign("tok4", "BUY", 0.50, 10.0)
        assert ps.stats["cache_size"] == 3
        assert ps.stats["evictions"] == 1

        # tok1 should now miss (was evicted)
        sign_fn.reset_mock()
        ps.get_or_sign("tok1", "BUY", 0.50, 10.0)
        assert sign_fn.call_count == 1

    def test_access_refreshes_lru_position(self):
        sign_fn = _make_sign_fn()
        ps = OrderPresigner(sign_fn=sign_fn, max_cache_size=3)

        ps.get_or_sign("tok1", "BUY", 0.50, 10.0)
        ps.get_or_sign("tok2", "BUY", 0.50, 10.0)
        ps.get_or_sign("tok3", "BUY", 0.50, 10.0)

        # Access tok1 to refresh its LRU position
        ps.get_or_sign("tok1", "BUY", 0.50, 10.0)

        # Add tok4 -- tok2 should be evicted (oldest untouched)
        ps.get_or_sign("tok4", "BUY", 0.50, 10.0)

        # tok1 should still be cached (was refreshed)
        sign_fn.reset_mock()
        ps.get_or_sign("tok1", "BUY", 0.50, 10.0)
        assert sign_fn.call_count == 0

        # tok2 should be evicted
        ps.get_or_sign("tok2", "BUY", 0.50, 10.0)
        assert sign_fn.call_count == 1


class TestAgeInvalidation:
    def test_stale_entry_triggers_resign(self):
        sign_fn = _make_sign_fn()
        ps = OrderPresigner(sign_fn=sign_fn, max_age_sec=0.1)

        ps.get_or_sign("tok1", "BUY", 0.50, 10.0)
        assert sign_fn.call_count == 1

        time.sleep(0.15)

        # Should re-sign because entry is stale
        ps.get_or_sign("tok1", "BUY", 0.50, 10.0)
        assert sign_fn.call_count == 2

    def test_fresh_entry_is_cached(self):
        sign_fn = _make_sign_fn()
        ps = OrderPresigner(sign_fn=sign_fn, max_age_sec=5.0)

        ps.get_or_sign("tok1", "BUY", 0.50, 10.0)
        ps.get_or_sign("tok1", "BUY", 0.50, 10.0)

        assert sign_fn.call_count == 1


class TestInvalidateToken:
    def test_removes_all_entries_for_token(self):
        sign_fn = _make_sign_fn()
        ps = OrderPresigner(sign_fn=sign_fn)

        ps.get_or_sign("tokA", "BUY", 0.50, 10.0)
        ps.get_or_sign("tokA", "SELL", 0.50, 10.0)
        ps.get_or_sign("tokB", "BUY", 0.60, 10.0)
        assert ps.stats["cache_size"] == 3

        removed = ps.invalidate_token("tokA")
        assert removed == 2
        assert ps.stats["cache_size"] == 1

    def test_does_not_remove_other_tokens(self):
        sign_fn = _make_sign_fn()
        ps = OrderPresigner(sign_fn=sign_fn)

        ps.get_or_sign("tokA", "BUY", 0.50, 10.0)
        ps.get_or_sign("tokB", "BUY", 0.60, 10.0)

        ps.invalidate_token("tokA")

        # tokB should still be cached
        sign_fn.reset_mock()
        ps.get_or_sign("tokB", "BUY", 0.60, 10.0)
        assert sign_fn.call_count == 0

    def test_invalidate_nonexistent_token_returns_zero(self):
        ps = OrderPresigner(sign_fn=_make_sign_fn())
        assert ps.invalidate_token("nonexistent") == 0


class TestInvalidateStale:
    def test_removes_stale_entries(self):
        sign_fn = _make_sign_fn()
        ps = OrderPresigner(sign_fn=sign_fn, max_age_sec=0.1)

        ps.get_or_sign("tok1", "BUY", 0.50, 10.0)
        time.sleep(0.15)

        # Add a fresh entry
        ps.get_or_sign("tok2", "BUY", 0.60, 10.0)

        removed = ps.invalidate_stale()
        assert removed == 1
        assert ps.stats["cache_size"] == 1

    def test_keeps_fresh_entries(self):
        sign_fn = _make_sign_fn()
        ps = OrderPresigner(sign_fn=sign_fn, max_age_sec=10.0)

        ps.get_or_sign("tok1", "BUY", 0.50, 10.0)
        ps.get_or_sign("tok2", "BUY", 0.60, 10.0)

        removed = ps.invalidate_stale()
        assert removed == 0
        assert ps.stats["cache_size"] == 2


class TestPresignLevels:
    def test_presigns_at_price_levels(self):
        sign_fn = _make_sign_fn()
        ps = OrderPresigner(sign_fn=sign_fn, tick_levels=2)

        count = ps.presign_levels("tok1", 0.50, size=10.0, tick_size="0.01")

        # 5 price levels (0.48, 0.49, 0.50, 0.51, 0.52) x 2 sides = 10
        assert count == 10
        assert ps.stats["cache_size"] == 10

    def test_skips_out_of_range_prices(self):
        sign_fn = _make_sign_fn()
        ps = OrderPresigner(sign_fn=sign_fn, tick_levels=2)

        # Price 0.01: levels would be -0.01, 0.00, 0.01, 0.02, 0.03
        # -0.01 and 0.00 are <= 0, so skipped
        count = ps.presign_levels("tok1", 0.01, size=10.0, tick_size="0.01")

        # Valid prices: 0.01, 0.02, 0.03 = 3 levels x 2 sides = 6
        assert count == 6

    def test_skips_already_cached_entries(self):
        sign_fn = _make_sign_fn()
        ps = OrderPresigner(sign_fn=sign_fn, tick_levels=1)

        # First call signs all levels
        count1 = ps.presign_levels("tok1", 0.50, size=10.0, tick_size="0.01")
        call_count_after_first = sign_fn.call_count

        # Second call should skip all (already cached and fresh)
        count2 = ps.presign_levels("tok1", 0.50, size=10.0, tick_size="0.01")

        assert count2 == 0
        assert sign_fn.call_count == call_count_after_first

    def test_returns_zero_without_sign_fn(self):
        ps = OrderPresigner(sign_fn=None, tick_levels=2)
        count = ps.presign_levels("tok1", 0.50, size=10.0)
        assert count == 0

    def test_handles_sign_failure_gracefully(self):
        sign_fn = MagicMock(side_effect=RuntimeError("signing error"))
        ps = OrderPresigner(sign_fn=sign_fn, tick_levels=1)

        count = ps.presign_levels("tok1", 0.50, size=10.0, tick_size="0.01")
        assert count == 0


class TestStats:
    def test_initial_stats(self):
        ps = OrderPresigner()
        stats = ps.stats
        assert stats["cache_size"] == 0
        assert stats["hits"] == 0
        assert stats["misses"] == 0
        assert stats["evictions"] == 0
        assert stats["hit_rate"] == 0.0

    def test_tracks_hits_misses_evictions(self):
        sign_fn = _make_sign_fn()
        ps = OrderPresigner(sign_fn=sign_fn, max_cache_size=2)

        ps.get_or_sign("tok1", "BUY", 0.50, 10.0)  # miss
        ps.get_or_sign("tok2", "BUY", 0.50, 10.0)  # miss
        ps.get_or_sign("tok1", "BUY", 0.50, 10.0)  # hit
        ps.get_or_sign("tok3", "BUY", 0.50, 10.0)  # miss, evicts tok2

        stats = ps.stats
        assert stats["misses"] == 3
        assert stats["hits"] == 1
        assert stats["evictions"] == 1
        assert stats["hit_rate"] == pytest.approx(0.25)

    def test_hit_rate_computation(self):
        sign_fn = _make_sign_fn()
        ps = OrderPresigner(sign_fn=sign_fn)

        ps.get_or_sign("tok1", "BUY", 0.50, 10.0)  # miss
        ps.get_or_sign("tok1", "BUY", 0.50, 10.0)  # hit
        ps.get_or_sign("tok1", "BUY", 0.50, 10.0)  # hit
        ps.get_or_sign("tok1", "BUY", 0.50, 10.0)  # hit

        assert ps.stats["hit_rate"] == pytest.approx(0.75)


class TestSizeMismatch:
    def test_different_size_triggers_resign(self):
        sign_fn = _make_sign_fn()
        ps = OrderPresigner(sign_fn=sign_fn)

        result1 = ps.get_or_sign("tok1", "BUY", 0.50, 10.0)
        result2 = ps.get_or_sign("tok1", "BUY", 0.50, 20.0)

        assert sign_fn.call_count == 2
        assert result1["size"] == 10.0
        assert result2["size"] == 20.0

    def test_same_size_uses_cache(self):
        sign_fn = _make_sign_fn()
        ps = OrderPresigner(sign_fn=sign_fn)

        ps.get_or_sign("tok1", "BUY", 0.50, 10.0)
        ps.get_or_sign("tok1", "BUY", 0.50, 10.0)

        assert sign_fn.call_count == 1


class TestNoSignFn:
    def test_returns_none_without_sign_fn(self):
        ps = OrderPresigner(sign_fn=None)
        result = ps.get_or_sign("tok1", "BUY", 0.50, 10.0)
        assert result is None

    def test_tracks_miss_without_sign_fn(self):
        ps = OrderPresigner(sign_fn=None)
        ps.get_or_sign("tok1", "BUY", 0.50, 10.0)
        assert ps.stats["misses"] == 1


class TestClear:
    def test_clear_empties_cache(self):
        sign_fn = _make_sign_fn()
        ps = OrderPresigner(sign_fn=sign_fn)

        ps.get_or_sign("tok1", "BUY", 0.50, 10.0)
        ps.get_or_sign("tok2", "BUY", 0.60, 10.0)
        assert ps.stats["cache_size"] == 2

        ps.clear()
        assert ps.stats["cache_size"] == 0


class TestPresignKey:
    def test_frozen_and_hashable(self):
        key = PresignKey("tok1", "BUY", 0.50, "0.01", False)
        assert hash(key)  # hashable
        d = {key: "value"}
        assert d[key] == "value"

    def test_equality(self):
        k1 = PresignKey("tok1", "BUY", 0.50, "0.01", False)
        k2 = PresignKey("tok1", "BUY", 0.50, "0.01", False)
        k3 = PresignKey("tok1", "SELL", 0.50, "0.01", False)
        assert k1 == k2
        assert k1 != k3


class TestPresignedOrder:
    def test_age_sec(self):
        key = PresignKey("tok1", "BUY", 0.50, "0.01", False)
        order = PresignedOrder(
            key=key,
            signed_order={"signed": True},
            size=10.0,
            created_at=time.time() - 5.0,
        )
        assert order.age_sec >= 4.9


class TestThreadSafety:
    def test_concurrent_get_or_sign(self):
        """Multiple threads calling get_or_sign should not corrupt cache."""
        import threading

        sign_fn = _make_sign_fn()
        ps = OrderPresigner(sign_fn=sign_fn, max_cache_size=50)
        errors: list[Exception] = []

        def worker(tid: int):
            try:
                for i in range(20):
                    ps.get_or_sign(f"tok{tid}", "BUY", 0.50, 10.0)
            except Exception as e:
                errors.append(e)

        threads = [threading.Thread(target=worker, args=(i,)) for i in range(10)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        assert errors == []
        assert ps.stats["cache_size"] <= 50
