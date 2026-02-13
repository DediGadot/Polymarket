"""
Tests for platform protocol conformance and active_platforms() helper.
"""

from __future__ import annotations

import pytest

from client.platform import PlatformClient
from scanner.platform_fees import PlatformFeeModel
from config import Config, active_platforms


class TestPlatformClientProtocol:
    def test_kalshi_satisfies_protocol(self):
        """KalshiClient should satisfy PlatformClient protocol structurally."""
        from client.kalshi import KalshiClient
        # runtime_checkable protocols with non-method members don't support issubclass(),
        # so we check structurally via hasattr
        assert hasattr(KalshiClient, "platform_name")
        assert hasattr(KalshiClient, "book_fetcher")
        assert hasattr(KalshiClient, "get_all_markets")
        assert hasattr(KalshiClient, "get_orderbook")
        assert hasattr(KalshiClient, "get_orderbooks")
        assert hasattr(KalshiClient, "place_order")
        assert hasattr(KalshiClient, "cancel_order")
        assert hasattr(KalshiClient, "get_order")
        assert hasattr(KalshiClient, "get_positions")
        assert hasattr(KalshiClient, "get_balance")

    def test_fanatics_satisfies_protocol(self):
        """FanaticsClient should satisfy PlatformClient protocol."""
        from client.fanatics_auth import FanaticsAuth
        from client.fanatics import FanaticsClient
        auth = FanaticsAuth(api_key="test", api_secret="test")
        client = FanaticsClient(auth=auth)
        assert isinstance(client, PlatformClient)
        assert client.platform_name == "fanatics"

    def test_fanatics_platform_name(self):
        from client.fanatics_auth import FanaticsAuth
        from client.fanatics import FanaticsClient
        auth = FanaticsAuth(api_key="k", api_secret="s")
        client = FanaticsClient(auth=auth)
        assert client.platform_name == "fanatics"


class TestPlatformFeeModelProtocol:
    def test_kalshi_fee_model_satisfies_protocol(self):
        """KalshiFeeModel should satisfy PlatformFeeModel protocol."""
        from scanner.kalshi_fees import KalshiFeeModel
        model = KalshiFeeModel()
        assert isinstance(model, PlatformFeeModel)
        assert model.platform_name == "kalshi"
        assert model.has_resolution_fee is False

    def test_fanatics_fee_model_satisfies_protocol(self):
        """FanaticsFeeModel should satisfy PlatformFeeModel protocol."""
        from scanner.fanatics_fees import FanaticsFeeModel
        model = FanaticsFeeModel()
        assert isinstance(model, PlatformFeeModel)
        assert model.platform_name == "fanatics"
        assert model.has_resolution_fee is False


class TestActivePlatforms:
    def test_no_credentials_returns_empty(self):
        cfg = Config(
            kalshi_api_key_id="",
            kalshi_private_key_path="",
            fanatics_api_key="",
            fanatics_api_secret="",
        )
        assert active_platforms(cfg) == []

    def test_kalshi_credentials_detected(self):
        cfg = Config(
            kalshi_api_key_id="key123",
            kalshi_private_key_path="/path/to/key.pem",
            fanatics_api_key="",
            fanatics_api_secret="",
        )
        result = active_platforms(cfg)
        assert "kalshi" in result
        assert "fanatics" not in result

    def test_fanatics_credentials_detected(self):
        cfg = Config(
            kalshi_api_key_id="",
            kalshi_private_key_path="",
            fanatics_api_key="fkey",
            fanatics_api_secret="fsecret",
        )
        result = active_platforms(cfg)
        assert "fanatics" in result
        assert "kalshi" not in result

    def test_both_platforms_detected(self):
        cfg = Config(
            kalshi_api_key_id="key",
            kalshi_private_key_path="/key.pem",
            fanatics_api_key="fkey",
            fanatics_api_secret="fsecret",
        )
        result = active_platforms(cfg)
        assert "kalshi" in result
        assert "fanatics" in result
        # Kalshi comes first
        assert result.index("kalshi") < result.index("fanatics")

    def test_partial_credentials_not_detected(self):
        """Both key and secret required for Fanatics."""
        cfg = Config(
            kalshi_api_key_id="",
            kalshi_private_key_path="",
            fanatics_api_key="fkey",
            fanatics_api_secret="",  # missing secret
        )
        assert "fanatics" not in active_platforms(cfg)
