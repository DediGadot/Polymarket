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

    # WebSocket + Book Cache (Iteration 1)
    ws_enabled: bool = True
    ws_reconnect_max: int = 5
    book_cache_max_age_sec: float = 5.0
    use_fak_orders: bool = True

    # Gas + Fee Intelligence (Iteration 2)
    polygon_rpc_url: str = "https://polygon-rpc.com"
    gas_cache_sec: float = 10.0
    max_gas_profit_ratio: float = 0.50
    fee_model_enabled: bool = True

    # Execution hardening
    max_legs_per_opportunity: int = 15  # skip opportunities with more legs (one batch max)
    book_fetch_workers: int = Field(default=8, ge=1, le=32)

    # Volume pre-filter
    min_volume_filter: float = Field(default=0.0, ge=0)

    # Depth Sweep + Latency Arb (Iteration 3)
    target_size_usd: float = 100.0
    latency_enabled: bool = True
    spot_price_cache_sec: float = 2.0
    latency_min_edge_pct: float = 5.0

    # Spike Detection (Iteration 4)
    spike_threshold_pct: float = 5.0
    spike_window_sec: float = 30.0
    spike_cooldown_sec: float = 60.0

    # Kalshi Cross-Platform (Iteration 5)
    kalshi_api_key_id: str = ""
    kalshi_private_key_path: str = ""
    kalshi_host: str = "https://trading-api.kalshi.com/trade-api/v2"
    kalshi_demo: bool = False
    cross_platform_enabled: bool = False
    cross_platform_min_confidence: float = 0.90
    cross_platform_manual_map: str = "cross_platform_map.json"
    kalshi_position_limit: float = 25000.0
    cross_platform_deadline_sec: float = 5.0
    cross_platform_verified_path: str = "verified_matches.json"


def load_config() -> Config:
    """Load and validate config from environment. Raises on missing required fields."""
    return Config()
