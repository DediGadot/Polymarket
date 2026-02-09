"""
Unit tests for client/gas.py -- gas oracle with caching.
"""

import time
from unittest.mock import patch, MagicMock

import httpx

from client.gas import GasOracle


class TestGetGasPriceGwei:
    @patch("client.gas.httpx.post")
    def test_fetches_from_rpc(self, mock_post):
        """Should parse hex gas price from RPC response."""
        # 30 gwei = 30e9 wei = 0x6FC23AC00
        mock_resp = MagicMock()
        mock_resp.json.return_value = {"result": "0x6FC23AC00"}
        mock_resp.raise_for_status = MagicMock()
        mock_post.return_value = mock_resp

        oracle = GasOracle(cache_sec=0)
        gwei = oracle.get_gas_price_gwei()
        assert abs(gwei - 30.0) < 0.1
        mock_post.assert_called_once()

    @patch("client.gas.httpx.post")
    def test_uses_cache_when_fresh(self, mock_post):
        """Should not re-fetch if cache is fresh."""
        mock_resp = MagicMock()
        mock_resp.json.return_value = {"result": "0x6FC23AC00"}
        mock_resp.raise_for_status = MagicMock()
        mock_post.return_value = mock_resp

        oracle = GasOracle(cache_sec=60.0)
        oracle.get_gas_price_gwei()
        oracle.get_gas_price_gwei()
        assert mock_post.call_count == 1

    @patch("client.gas.httpx.post")
    def test_falls_back_to_default_on_error(self, mock_post):
        """Should return default on RPC failure."""
        mock_post.side_effect = httpx.ConnectError("Connection refused")

        oracle = GasOracle(default_gas_gwei=50.0)
        gwei = oracle.get_gas_price_gwei()
        assert gwei == 50.0


class TestGetMaticUsd:
    @patch("client.gas.httpx.get")
    def test_fetches_from_coingecko(self, mock_get):
        """Should parse MATIC/USD from CoinGecko."""
        mock_resp = MagicMock()
        mock_resp.json.return_value = {"polygon-ecosystem-token": {"usd": 0.85}}
        mock_resp.raise_for_status = MagicMock()
        mock_get.return_value = mock_resp

        oracle = GasOracle(cache_sec=0)
        price = oracle.get_matic_usd()
        assert abs(price - 0.85) < 0.01

    @patch("client.gas.httpx.get")
    def test_uses_cache_when_fresh(self, mock_get):
        """Should not re-fetch if cache is fresh."""
        mock_resp = MagicMock()
        mock_resp.json.return_value = {"polygon-ecosystem-token": {"usd": 0.85}}
        mock_resp.raise_for_status = MagicMock()
        mock_get.return_value = mock_resp

        oracle = GasOracle(cache_sec=60.0)
        oracle.get_matic_usd()
        oracle.get_matic_usd()
        assert mock_get.call_count == 1

    @patch("client.gas.httpx.get")
    def test_falls_back_to_default_on_error(self, mock_get):
        """Should return default on API failure."""
        mock_get.side_effect = httpx.ConnectError("Connection refused")

        oracle = GasOracle(default_matic_usd=0.60)
        price = oracle.get_matic_usd()
        assert price == 0.60


class TestEstimateCostUsd:
    @patch("client.gas.httpx.get")
    @patch("client.gas.httpx.post")
    def test_estimate_calculation(self, mock_post, mock_get):
        """Gas cost = n_orders * gas_per_order * gwei * 1e9 / 1e18 * matic_usd."""
        mock_gas = MagicMock()
        mock_gas.json.return_value = {"result": "0x6FC23AC00"}  # 30 gwei
        mock_gas.raise_for_status = MagicMock()
        mock_post.return_value = mock_gas

        mock_matic = MagicMock()
        mock_matic.json.return_value = {"polygon-ecosystem-token": {"usd": 0.50}}
        mock_matic.raise_for_status = MagicMock()
        mock_get.return_value = mock_matic

        oracle = GasOracle(cache_sec=0)
        cost = oracle.estimate_cost_usd(n_orders=2, gas_per_order=150_000)
        # 2 * 150000 * 30 * 1e9 / 1e18 * 0.50 = 0.0045
        assert abs(cost - 0.0045) < 0.001

    def test_estimate_with_cached_values(self):
        """Should use cached values when available."""
        oracle = GasOracle(default_gas_gwei=30.0, default_matic_usd=0.50, cache_sec=60.0)
        # Pre-populate cache
        oracle._cached_gas_gwei = 30.0
        oracle._gas_ts = time.time()
        oracle._cached_matic_usd = 0.50
        oracle._matic_ts = time.time()
        cost = oracle.estimate_cost_usd(n_orders=3, gas_per_order=150_000)
        # 3 * 150000 * 30 * 1e9 / 1e18 * 0.50 = 0.00675
        assert abs(cost - 0.00675) < 0.001
