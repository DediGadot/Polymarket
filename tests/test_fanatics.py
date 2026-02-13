"""
Tests for client/fanatics.py and client/fanatics_auth.py.
All trading stubs raise NotImplementedError.
"""

from __future__ import annotations

import pytest

from client.fanatics_auth import FanaticsAuth
from client.fanatics import FanaticsClient, FanaticsMarket


class TestFanaticsAuth:
    def test_sign_request_deterministic(self):
        """Same inputs produce same signature."""
        auth = FanaticsAuth(api_key="testkey", api_secret="testsecret")
        result1 = auth.sign_request("private/get-balance", "req1", {"foo": "bar"}, nonce=12345)
        result2 = auth.sign_request("private/get-balance", "req1", {"foo": "bar"}, nonce=12345)
        assert result1["sig"] == result2["sig"]

    def test_sign_request_fields(self):
        """Signed request has all required fields."""
        auth = FanaticsAuth(api_key="mykey", api_secret="mysecret")
        result = auth.sign_request("private/get-balance", "req42", nonce=100)
        assert result["id"] == "req42"
        assert result["method"] == "private/get-balance"
        assert result["api_key"] == "mykey"
        assert result["nonce"] == 100
        assert "sig" in result
        assert len(result["sig"]) == 64  # SHA-256 hex digest

    def test_sign_request_different_nonces(self):
        """Different nonces produce different signatures."""
        auth = FanaticsAuth(api_key="key", api_secret="secret")
        sig1 = auth.sign_request("method", "id1", nonce=1)["sig"]
        sig2 = auth.sign_request("method", "id1", nonce=2)["sig"]
        assert sig1 != sig2

    def test_params_sorted_for_signing(self):
        """Parameters must be sorted by key for deterministic signatures."""
        auth = FanaticsAuth(api_key="k", api_secret="s")
        # Same params in different order should produce same sig
        sig1 = auth.sign_request("m", "id", {"b": "2", "a": "1"}, nonce=0)["sig"]
        sig2 = auth.sign_request("m", "id", {"a": "1", "b": "2"}, nonce=0)["sig"]
        assert sig1 == sig2


class TestFanaticsMarket:
    def test_frozen_dataclass(self):
        m = FanaticsMarket(ticker="T1", event_ticker="E1", title="Test", status="open")
        assert m.ticker == "T1"
        with pytest.raises(AttributeError):
            m.ticker = "T2"  # type: ignore[misc]


class TestFanaticsClient:
    def _make_client(self) -> FanaticsClient:
        auth = FanaticsAuth(api_key="key", api_secret="secret")
        return FanaticsClient(auth=auth)

    def test_platform_name(self):
        assert self._make_client().platform_name == "fanatics"

    def test_get_all_markets_raises(self):
        with pytest.raises(NotImplementedError, match="not yet available"):
            self._make_client().get_all_markets()

    def test_get_orderbook_raises(self):
        with pytest.raises(NotImplementedError, match="not yet available"):
            self._make_client().get_orderbook("ticker")

    def test_get_orderbooks_raises(self):
        with pytest.raises(NotImplementedError, match="not yet available"):
            self._make_client().get_orderbooks(["t1", "t2"])

    def test_place_order_raises(self):
        with pytest.raises(NotImplementedError, match="not yet available"):
            self._make_client().place_order(ticker="t", side="yes")

    def test_cancel_order_raises(self):
        with pytest.raises(NotImplementedError, match="not yet available"):
            self._make_client().cancel_order("oid")

    def test_get_order_raises(self):
        with pytest.raises(NotImplementedError, match="not yet available"):
            self._make_client().get_order("oid")

    def test_get_positions_raises(self):
        with pytest.raises(NotImplementedError, match="not yet available"):
            self._make_client().get_positions()

    def test_get_balance_raises(self):
        with pytest.raises(NotImplementedError, match="not yet available"):
            self._make_client().get_balance()

    def test_book_fetcher_raises(self):
        """book_fetcher returns get_orderbooks which raises."""
        client = self._make_client()
        fetcher = client.book_fetcher
        with pytest.raises(NotImplementedError):
            fetcher(["t1"])
