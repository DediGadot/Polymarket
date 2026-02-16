"""
Order presigner cache. Pre-signs limit orders at common price levels
for hot markets, reducing execution-path signing latency from ~200ms to ~0ms.
"""

from __future__ import annotations

import logging
import threading
import time
from collections import OrderedDict
from dataclasses import dataclass, field
from typing import Any, Callable

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class PresignKey:
    """Cache key for a pre-signed order."""

    token_id: str
    side: str  # "BUY" or "SELL"
    price: float
    tick_size: str
    neg_risk: bool


@dataclass
class PresignedOrder:
    """A cached pre-signed order."""

    key: PresignKey
    signed_order: Any  # The signed order object from py-clob-client
    size: float
    created_at: float

    @property
    def age_sec(self) -> float:
        return time.time() - self.created_at


class OrderPresigner:
    """
    LRU cache of pre-signed orders for hot markets.

    Background thread watches BookCache for price changes and pre-signs
    at common price levels (best +/- 2 ticks). Cache hit = 0ms signing,
    miss = fallback to live signing.
    """

    def __init__(
        self,
        sign_fn: Callable[..., Any] | None = None,
        max_cache_size: int = 200,
        max_age_sec: float = 30.0,
        tick_levels: int = 2,
    ):
        self._sign_fn = sign_fn
        self._max_size = max_cache_size
        self._max_age = max_age_sec
        self._tick_levels = tick_levels
        self._cache: OrderedDict[PresignKey, PresignedOrder] = OrderedDict()
        self._lock = threading.Lock()
        self._hits = 0
        self._misses = 0
        self._evictions = 0

    def get_or_sign(
        self,
        token_id: str,
        side: str,
        price: float,
        size: float,
        neg_risk: bool = False,
        tick_size: str = "0.01",
        **sign_kwargs: Any,
    ) -> Any:
        """
        Return a pre-signed order from cache if available,
        otherwise sign on the fly and cache the result.

        Returns the signed order object, or None if no sign_fn.
        """
        key = PresignKey(
            token_id=token_id,
            side=side,
            price=price,
            tick_size=tick_size,
            neg_risk=neg_risk,
        )

        with self._lock:
            cached = self._cache.get(key)
            if cached is not None and cached.age_sec < self._max_age and cached.size == size:
                # Cache hit -- move to end (most recently used) and return
                self._cache.move_to_end(key)
                self._hits += 1
                return cached.signed_order

        # Cache miss -- sign live
        self._misses += 1
        if self._sign_fn is None:
            return None

        signed = self._sign_fn(
            token_id=token_id,
            side=side,
            price=price,
            size=size,
            neg_risk=neg_risk,
            tick_size=tick_size,
            **sign_kwargs,
        )

        entry = PresignedOrder(
            key=key,
            signed_order=signed,
            size=size,
            created_at=time.time(),
        )
        with self._lock:
            self._cache[key] = entry
            self._cache.move_to_end(key)
            self._evict_if_needed()

        return signed

    def presign_levels(
        self,
        token_id: str,
        best_price: float,
        size: float,
        neg_risk: bool = False,
        tick_size: str = "0.01",
    ) -> int:
        """
        Pre-sign orders at best_price +/- tick_levels ticks for both BUY and SELL.
        Returns number of orders pre-signed.
        """
        if self._sign_fn is None:
            return 0

        tick = float(tick_size)
        count = 0

        for offset in range(-self._tick_levels, self._tick_levels + 1):
            price = round(best_price + offset * tick, 4)
            if price <= 0 or price >= 1.0:
                continue

            for side in ("BUY", "SELL"):
                key = PresignKey(
                    token_id=token_id,
                    side=side,
                    price=price,
                    tick_size=tick_size,
                    neg_risk=neg_risk,
                )

                with self._lock:
                    existing = self._cache.get(key)
                    if (
                        existing is not None
                        and existing.age_sec < self._max_age
                        and existing.size == size
                    ):
                        continue

                try:
                    signed = self._sign_fn(
                        token_id=token_id,
                        side=side,
                        price=price,
                        size=size,
                        neg_risk=neg_risk,
                        tick_size=tick_size,
                    )
                    entry = PresignedOrder(
                        key=key,
                        signed_order=signed,
                        size=size,
                        created_at=time.time(),
                    )
                    with self._lock:
                        self._cache[key] = entry
                        self._cache.move_to_end(key)
                        self._evict_if_needed()
                    count += 1
                except Exception as e:
                    logger.debug(
                        "Presign failed for %s %s@%.4f: %s",
                        side,
                        token_id[:10],
                        price,
                        e,
                    )

        return count

    def invalidate_token(self, token_id: str) -> int:
        """Remove all cached entries for a token. Returns count removed."""
        with self._lock:
            keys_to_remove = [k for k in self._cache if k.token_id == token_id]
            for k in keys_to_remove:
                del self._cache[k]
            return len(keys_to_remove)

    def invalidate_stale(self) -> int:
        """Remove entries older than max_age_sec. Returns count removed."""
        with self._lock:
            keys_to_remove = [
                k for k, v in self._cache.items() if v.age_sec > self._max_age
            ]
            for k in keys_to_remove:
                del self._cache[k]
            return len(keys_to_remove)

    def _evict_if_needed(self) -> None:
        """Evict oldest entries if cache exceeds max size. Must hold lock."""
        while len(self._cache) > self._max_size:
            self._cache.popitem(last=False)
            self._evictions += 1

    @property
    def stats(self) -> dict[str, int | float]:
        with self._lock:
            return {
                "cache_size": len(self._cache),
                "max_size": self._max_size,
                "hits": self._hits,
                "misses": self._misses,
                "evictions": self._evictions,
                "hit_rate": self._hits / max(1, self._hits + self._misses),
            }

    def clear(self) -> None:
        """Clear entire cache."""
        with self._lock:
            self._cache.clear()
