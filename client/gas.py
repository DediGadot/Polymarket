"""
Real-time gas oracle for Polygon network. Queries RPC for gas price and
CoinGecko for MATIC/USD. Results cached to avoid hammering endpoints.
"""

from __future__ import annotations

import logging
import time

import httpx

logger = logging.getLogger(__name__)

_TIMEOUT = 5.0


class GasOracle:
    """
    Cached gas price + MATIC/USD oracle.
    Queries Polygon RPC and CoinGecko with configurable cache TTL.
    """

    def __init__(
        self,
        rpc_url: str = "https://polygon-rpc.com",
        cache_sec: float = 10.0,
        default_gas_gwei: float = 30.0,
        default_matic_usd: float = 0.50,
        allow_network: bool = False,
    ):
        self._rpc_url = rpc_url
        self._cache_sec = cache_sec
        self._default_gas_gwei = default_gas_gwei
        self._default_matic_usd = default_matic_usd
        self._allow_network = allow_network

        self._cached_gas_gwei: float | None = None
        self._gas_ts: float = 0.0
        self._cached_matic_usd: float | None = None
        self._matic_ts: float = 0.0

    def get_gas_price_gwei(self) -> float:
        """Return current gas price in gwei. Uses cache if fresh."""
        now = time.time()
        if self._cached_gas_gwei is not None and (now - self._gas_ts) < self._cache_sec:
            return self._cached_gas_gwei
        if not self._allow_network:
            return self._default_gas_gwei

        try:
            resp = httpx.post(
                self._rpc_url,
                json={"jsonrpc": "2.0", "method": "eth_gasPrice", "params": [], "id": 1},
                timeout=_TIMEOUT,
            )
            resp.raise_for_status()
            hex_price = resp.json()["result"]
            wei = int(hex_price, 16)
            gwei = wei / 1e9
            self._cached_gas_gwei = gwei
            self._gas_ts = now
            logger.debug("Gas price: %.1f gwei", gwei)
            return gwei
        except Exception as e:
            logger.warning("Gas price fetch failed, using default %.1f gwei: %s", self._default_gas_gwei, e)
            return self._default_gas_gwei

    def get_matic_usd(self) -> float:
        """Return current MATIC/USD price. Uses cache if fresh."""
        now = time.time()
        if self._cached_matic_usd is not None and (now - self._matic_ts) < self._cache_sec:
            return self._cached_matic_usd
        if not self._allow_network:
            return self._default_matic_usd

        try:
            resp = httpx.get(
                "https://api.coingecko.com/api/v3/simple/price",
                params={"ids": "polygon-ecosystem-token", "vs_currencies": "usd"},
                timeout=_TIMEOUT,
            )
            resp.raise_for_status()
            price = resp.json()["polygon-ecosystem-token"]["usd"]
            self._cached_matic_usd = float(price)
            self._matic_ts = now
            logger.debug("MATIC/USD: $%.4f", self._cached_matic_usd)
            return self._cached_matic_usd
        except Exception as e:
            logger.warning("MATIC/USD fetch failed, using default $%.2f: %s", self._default_matic_usd, e)
            return self._default_matic_usd

    def estimate_cost_usd(self, n_orders: int, gas_per_order: int) -> float:
        """Estimate total gas cost in USD for n_orders."""
        gas_gwei = self.get_gas_price_gwei()
        matic_usd = self.get_matic_usd()
        total_gas = n_orders * gas_per_order
        cost_matic = (total_gas * gas_gwei * 1e9) / 1e18
        return cost_matic * matic_usd
