"""
Configuration loaded from environment variables. Fail-fast on missing required values.
"""

from pydantic_settings import BaseSettings
from pydantic import Field


class Config(BaseSettings):
    model_config = {"env_file": ".env", "env_file_encoding": "utf-8"}

    # Credentials (required for paper/live trading, optional for dry-run)
    private_key: str = Field(default="", description="Polygon wallet private key (hex)")
    polymarket_profile_address: str = Field(default="", description="Polymarket proxy address")
    signature_type: int = Field(default=1, ge=0, le=2)

    # API endpoints
    clob_host: str = "https://clob.polymarket.com"
    gamma_host: str = "https://gamma-api.polymarket.com"
    data_host: str = "https://data-api.polymarket.com"
    ws_market_url: str = "wss://ws-subscriptions-clob.polymarket.com/ws/market"
    ws_user_url: str = "wss://ws-subscriptions-clob.polymarket.com/ws/user"
    chain_id: int = 137  # Polygon mainnet

    # Trading thresholds
    min_profit_usd: float = Field(default=0.50, gt=0)
    min_roi_pct: float = Field(default=2.0, gt=0)
    max_exposure_per_trade: float = Field(default=500.0, gt=0)
    max_total_exposure: float = Field(default=5000.0, gt=0)

    # Circuit breakers
    max_loss_per_hour: float = Field(default=50.0, gt=0)
    max_loss_per_day: float = Field(default=200.0, gt=0)
    max_consecutive_failures: int = Field(default=5, gt=0)

    # Timing
    scan_interval_sec: float = Field(default=1.0, gt=0)
    order_timeout_sec: float = Field(default=5.0, gt=0)

    # Modes
    paper_trading: bool = True
    log_level: str = "INFO"

    # Gas estimation
    gas_per_order: int = 150_000
    gas_price_gwei: float = 30.0  # default, overridden at runtime


def load_config() -> Config:
    """Load and validate config from environment. Raises on missing required fields."""
    return Config()
