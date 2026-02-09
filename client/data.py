"""
Position and balance fetcher via Polymarket Data API.
Provides inventory checks for sell-side execution.
"""

from __future__ import annotations

import logging
import time

import httpx

logger = logging.getLogger(__name__)

_TIMEOUT = 5.0


class PositionTracker:
    """
    Cached position tracker. Queries Data API for current token holdings.
    Used by safety checks to verify we hold tokens before selling.
    """

    def __init__(
        self,
        data_host: str = "https://data-api.polymarket.com",
        profile_address: str = "",
        cache_sec: float = 5.0,
    ):
        self._data_host = data_host
        self._profile_address = profile_address
        self._cache_sec = cache_sec
        self._positions: dict[str, float] = {}
        self._last_fetch: float = 0.0

    def get_positions(self) -> dict[str, float]:
        """
        Return current positions as {token_id: size}.
        Uses cache if fresh. Returns empty dict if fetch fails or no address.
        """
        if not self._profile_address:
            return {}

        now = time.time()
        if self._positions and (now - self._last_fetch) < self._cache_sec:
            return self._positions

        try:
            resp = httpx.get(
                f"{self._data_host}/positions",
                params={"user": self._profile_address},
                timeout=_TIMEOUT,
            )
            resp.raise_for_status()
            raw = resp.json()

            positions: dict[str, float] = {}
            for pos in raw:
                token_id = pos.get("asset", pos.get("token_id", ""))
                size = float(pos.get("size", pos.get("amount", 0)))
                if token_id and size > 0:
                    positions[token_id] = size

            self._positions = positions
            self._last_fetch = now
            logger.debug("Fetched %d positions", len(positions))
            return positions

        except Exception as e:
            logger.warning("Position fetch failed: %s", e)
            return self._positions  # return stale cache on failure

    def get_position(self, token_id: str) -> float:
        """Return held size for a single token. 0 if not held."""
        positions = self.get_positions()
        return positions.get(token_id, 0.0)
