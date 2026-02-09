"""
Unit tests for config.py.
"""

import pytest
from pydantic import ValidationError

from config import Config, load_config


class TestConfig:
    def test_defaults(self):
        """Required fields present, defaults applied for optional."""
        cfg = Config(
            private_key="abc123",
            polymarket_profile_address="0x1234",
        )
        assert cfg.private_key == "abc123"
        assert cfg.signature_type == 1
        assert cfg.min_profit_usd == 0.50
        assert cfg.min_roi_pct == 2.0
        assert cfg.max_exposure_per_trade == 500.0
        assert cfg.max_total_exposure == 5000.0
        assert cfg.paper_trading is True
        assert cfg.chain_id == 137
        assert cfg.scan_interval_sec == 1.0

    def test_empty_credentials_allowed_for_dry_run(self, monkeypatch):
        """Credentials default to empty string (dry-run mode needs no wallet)."""
        monkeypatch.delenv("PRIVATE_KEY", raising=False)
        monkeypatch.delenv("POLYMARKET_PROFILE_ADDRESS", raising=False)
        cfg = Config(_env_file=None)
        assert cfg.private_key == ""
        assert cfg.polymarket_profile_address == ""

    def test_credentials_set_when_provided(self):
        cfg = Config(private_key="abc123", polymarket_profile_address="0x1234")
        assert cfg.private_key == "abc123"
        assert cfg.polymarket_profile_address == "0x1234"

    def test_invalid_signature_type(self):
        with pytest.raises(ValidationError):
            Config(
                private_key="abc123",
                polymarket_profile_address="0x1234",
                signature_type=5,
            )

    def test_negative_min_profit_raises(self):
        with pytest.raises(ValidationError):
            Config(
                private_key="abc123",
                polymarket_profile_address="0x1234",
                min_profit_usd=-1.0,
            )

    def test_custom_values(self):
        cfg = Config(
            private_key="mykey",
            polymarket_profile_address="0xaddr",
            signature_type=0,
            min_profit_usd=1.0,
            min_roi_pct=5.0,
            max_exposure_per_trade=1000.0,
            paper_trading=False,
            log_level="DEBUG",
        )
        assert cfg.signature_type == 0
        assert cfg.min_profit_usd == 1.0
        assert cfg.paper_trading is False
        assert cfg.log_level == "DEBUG"

    def test_non_polymarket_apis_enabled_by_default(self):
        cfg = Config()
        assert cfg.allow_non_polymarket_apis is True

    def test_non_polymarket_apis_can_be_disabled_by_env(self, monkeypatch):
        monkeypatch.setenv("ALLOW_NON_POLYMARKET_APIS", "false")
        cfg = Config(_env_file=None)
        assert cfg.allow_non_polymarket_apis is False

    def test_load_config_from_env(self, monkeypatch):
        """load_config should read from environment variables."""
        monkeypatch.setenv("PRIVATE_KEY", "envkey")
        monkeypatch.setenv("POLYMARKET_PROFILE_ADDRESS", "0xenvaddr")
        monkeypatch.setenv("MIN_PROFIT_USD", "2.0")
        monkeypatch.setenv("PAPER_TRADING", "false")

        # Prevent .env file from interfering
        monkeypatch.setenv("ENV_FILE", "/dev/null")
        cfg = load_config()
        assert cfg.private_key == "envkey"
        assert cfg.polymarket_profile_address == "0xenvaddr"
        assert cfg.min_profit_usd == 2.0
        assert cfg.paper_trading is False
