"""
Configuration loaded from environment variables. Fail-fast on missing required values.
"""

from pydantic_settings import BaseSettings
from pydantic import Field


class Config(BaseSettings):
    model_config = {"env_file": ".env", "env_file_encoding": "utf-8", "frozen": True, "extra": "ignore"}

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
    # Maximum exposure per trade ($) — scale up as confidence builds
    max_exposure_per_trade: float = Field(default=5000.0, gt=0)
    # Maximum total portfolio exposure ($)
    max_total_exposure: float = Field(default=50000.0, gt=0)

    # Kelly sizing odds (probability of success for sizing calculation)
    # For confirmed arbs (YES+NO < $1), fill probability is ~85-95%.
    # 0.65 odds = 1.54:1 implied = ~65% win probability for Kelly sizing
    kelly_odds_confirmed: float = Field(default=0.65, gt=0, le=1.0)  # Confirmed arb (high fill probability)
    # Cross-platform arbs have higher execution risk (partial fills, platform failures)
    kelly_odds_cross_platform: float = Field(default=0.40, gt=0, le=1.0)  # Cross-platform (higher execution risk)

    # Circuit breakers
    max_loss_per_hour: float = Field(default=50.0, gt=0)
    max_loss_per_day: float = Field(default=200.0, gt=0)
    max_consecutive_failures: int = Field(default=5, gt=0)

    # Timing
    scan_interval_sec: float = Field(default=1.0, gt=0)
    order_timeout_sec: float = Field(default=5.0, gt=0)
    # Session-level duplicate suppression window for scan-only metrics.
    scan_tracker_dedup_window_sec: float = Field(default=30.0, ge=0.0)
    # Safety cap for dry-run scans to avoid CLOB REST rate-limit storms.
    # Set to 0 to disable the auto-cap.
    dry_run_default_limit: int = Field(default=1200, ge=0)
    # In dry-run, allow longer cache reuse to avoid repeatedly fetching the
    # entire book universe every cycle.
    dry_run_book_cache_max_age_sec: float = Field(default=90.0, ge=1.0)

    # Modes
    paper_trading: bool = True
    log_level: str = "INFO"
    # Default behavior: allow external market data integrations (Binance/CoinGecko/Kalshi).
    # Set false to force strict Polymarket-only mode.
    allow_non_polymarket_apis: bool = True

    # Gas estimation
    gas_per_order: int = 150_000
    gas_price_gwei: float = 30.0  # default, overridden at runtime

    # WebSocket + Book Cache (Iteration 1)
    ws_enabled: bool = True
    ws_reconnect_max: int = 5
    book_cache_max_age_sec: float = 5.0
    use_fak_orders: bool = True
    # Event-driven scanning: when WS is healthy and markets are unchanged,
    # skip full scans unless fresh WS updates arrive. Periodically force a
    # full rescan to avoid starvation if updates are sparse.
    ws_event_driven_scan: bool = True
    ws_force_rescan_sec: float = Field(default=15.0, ge=1.0)

    # Gas + Fee Intelligence (Iteration 2)
    polygon_rpc_url: str = "https://polygon-rpc.com"
    gas_cache_sec: float = 10.0
    max_gas_profit_ratio: float = 0.50
    fee_model_enabled: bool = True

    # Execution hardening
    max_legs_per_opportunity: int = 15  # skip opportunities with more legs (one batch max)
    book_fetch_workers: int = Field(default=8, ge=1, le=32)

    # Pre-filters
    min_volume_filter: float = Field(default=0.0, ge=0)
    # Minimum hours until resolution (0.0 = allow near-resolution markets)
    min_hours_to_resolution: float = Field(default=0.0, ge=0)

    # Slippage ceiling for depth scanning
    # Fraction of edge to allow as slippage (0.4 = accept up to 40% of edge as slippage)
    slippage_fraction: float = Field(default=0.4, ge=0, le=1.0)
    # Maximum slippage percentage regardless of edge
    max_slippage_pct: float = Field(default=3.0, ge=0.1, le=10.0)

    # Depth Sweep + Latency Arb (Iteration 3)
    target_size_usd: float = 100.0
    # Requires allow_non_polymarket_apis=true (uses Binance spot API).
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
    kalshi_host: str = "https://api.elections.kalshi.com/trade-api/v2"
    kalshi_demo: bool = False
    # Cross-platform arb enabled by default -- all configured platforms scanned.
    cross_platform_enabled: bool = True
    cross_platform_min_confidence: float = 0.90
    cross_platform_manual_map: str = "cross_platform_map.json"
    # Keep fuzzy matches as research-only by default; they are high-risk for
    # settlement mismatch and can create heavy matching churn.
    cross_platform_allow_unverified_fuzzy: bool = False
    # In dry-run, skip cross-platform scan when no tradeable (manual/verified)
    # mappings exist.
    cross_platform_skip_without_tradeable_map_in_dry_run: bool = True
    # Cooldown for repeated non-tradeable fuzzy match attempts.
    cross_platform_matching_negative_ttl_sec: float = Field(default=300.0, ge=0.0)
    kalshi_position_limit: float = 25000.0
    cross_platform_deadline_sec: float = 5.0
    cross_platform_verified_path: str = "verified_matches.json"
    kalshi_market_refresh_sec: float = 300.0
    kalshi_market_warm_timeout_sec: float = 180.0

    # Value scanner (partial negrisk)
    # Disabled: assumes uniform 1/N probability across outcomes, producing 100%
    # false positives on markets with known favorites (e.g. PSG top 4 at $0.98).
    value_scanner_enabled: bool = False
    value_min_edge_pct: float = Field(default=10.0, ge=1.0)
    value_max_exposure: float = Field(default=10000.0, ge=0)

    # Stale-quote sniping
    stale_quote_enabled: bool = True
    stale_quote_min_move_pct: float = Field(default=3.0, ge=1.0)
    stale_quote_max_staleness_ms: float = Field(default=500.0, ge=100.0)
    stale_quote_cooldown_sec: float = Field(default=5.0, ge=1.0)

    # Maker scanner filters (phantom arb suppression)
    maker_min_depth_sets: float = Field(default=15.0, ge=1.0)
    maker_min_leg_price: float = Field(default=0.05, ge=0.01, le=0.50)
    maker_min_volume: float = Field(default=500.0, ge=0)
    # Realism gate: reject maker candidates that require too much spread
    # capture vs current taker crossing cost.
    maker_max_taker_cost: float = Field(default=1.08, ge=1.0, le=1.5)
    maker_max_spread_ticks: int = Field(default=8, ge=1)
    maker_min_persistence_cycles: int = Field(default=3, ge=1)
    # Execution-quality gates (queue-aware paired fill + toxicity).
    maker_min_pair_fill_prob: float = Field(default=0.55, ge=0.0, le=1.0)
    maker_max_toxicity_score: float = Field(default=0.70, ge=0.0, le=1.0)
    maker_min_expected_ev_usd: float = Field(default=0.20, ge=0.0)
    # Maker lifecycle execution controls
    maker_order_max_age_sec: float = Field(default=30.0, ge=1.0)
    maker_order_max_drift_ticks: int = Field(default=2, ge=1)
    maker_max_active_pairs: int = Field(default=10, ge=1)
    maker_hedge_timeout_sec: float = Field(default=8.0, ge=0.5)
    maker_post_only: bool = True
    maker_use_gtd: bool = True
    maker_quote_ttl_sec: float = Field(default=8.0, ge=1.0)

    # Pre-execution confidence gate (rejects first-seen thin-book arbs)
    min_confidence_gate: float = Field(default=0.50, ge=0.0, le=1.0)
    # Correlation BUY opportunities can be structurally valid yet appear for a
    # short window; keep a separate actionable-now confidence/fill gate.
    correlation_actionable_min_confidence: float = Field(default=0.30, ge=0.0, le=1.0)
    correlation_actionable_min_fill_score: float = Field(default=0.35, ge=0.0, le=1.0)
    # Rank opportunities by fill-risk-adjusted EV instead of raw theoretical
    # profit in the scorer's profit component.
    risk_ranked_ev_enabled: bool = True
    # Allow long-only transformed correlation structures (e.g. parent+child_no,
    # earlier_no+later) into the actionable taker BUY bucket.
    correlation_actionable_allow_structural_buy: bool = True

    # Resolution sniping
    resolution_sniping_enabled: bool = True
    resolution_max_minutes: float = Field(default=60.0, ge=0)
    resolution_min_edge_pct: float = Field(default=3.0, ge=0)

    # Correlation scanner (cross-event probability violations)
    correlation_scanner_enabled: bool = True
    correlation_min_edge_pct: float = Field(default=3.0, ge=0.5)
    correlation_min_confidence: float = Field(default=0.7, ge=0.0, le=1.0)
    correlation_aggregation: str = "liquidity_weighted"  # liquidity_weighted | median | top_liquidity
    correlation_max_markets_per_event: int = Field(default=5, ge=1, le=20)
    correlation_min_market_volume: float = Field(default=500.0, ge=0.0)
    correlation_min_book_depth: float = Field(default=50.0, ge=0.0)
    correlation_max_theoretical_roi_pct: float = Field(default=250.0, ge=1.0)
    # Ignore "too-good-to-be-true" complement BUY sums from dead/stale books.
    correlation_min_buy_total_prob: float = Field(default=0.08, ge=0.0, le=1.0)
    # Require persistence before correlation opportunities are emitted.
    correlation_min_persistence_cycles: int = Field(default=1, ge=1)
    # Prevent oversized notional in correlation opportunities from dominating
    # scan-only metrics and executable lane accounting.
    correlation_max_capital_per_opp_usd: float = Field(default=1000.0, ge=1.0)
    # Hard cap to keep research-only correlation volume from dominating cycles.
    # 0 disables capping.
    correlation_max_opps_per_cycle: int = Field(default=120, ge=0)
    # When capping, preserve at least this many BUY-side correlation
    # opportunities so actionable taker candidates are not starved by SELL-heavy
    # rankings.
    correlation_cap_min_buy_opps_per_cycle: int = Field(default=40, ge=0)
    # Per-event diversity cap for BUY-side correlation opportunities when
    # applying correlation_max_opps_per_cycle.
    correlation_cap_max_buy_per_event: int = Field(default=3, ge=1)
    # By default, correlation opportunities stay in the research lane and are
    # excluded from taker execution queueing.
    correlation_execute_enabled: bool = True
    research_lane_enabled: bool = True
    executable_lane_zero_streak_warn_cycles: int = Field(default=3, ge=1)

    # ML scorer (augments hand-tuned scorer when trained)
    ml_scorer_enabled: bool = False
    ml_scorer_min_samples: int = 100
    ml_scorer_retrain_cycles: int = 50
    ml_scorer_model_path: str = "ml_scorer.joblib"
    ml_scorer_blend_weight: float = Field(default=0.15, ge=0.0, le=0.5)

    # Order presigning (reduces execution latency)
    presigner_enabled: bool = True
    presigner_max_cache_size: int = 200
    presigner_max_age_sec: float = 30.0
    presigner_tick_levels: int = 2
    presigner_prewarm_top_n: int = Field(default=10, ge=0, le=100)

    # State checkpointing (crash recovery)
    state_checkpoint_enabled: bool = True
    state_checkpoint_interval: int = Field(default=10, ge=1)
    state_checkpoint_db: str = "state.db"

    # Runtime recorder for replay/backtest datasets
    recording_enabled: bool = False
    recording_dir: str = "recordings"
    recording_max_mb: int = Field(default=500, ge=10)

    # Cross-platform safety semantics
    cross_platform_max_signal_age_sec: float = Field(default=1.5, ge=0.1, le=30.0)
    cross_platform_inventory_pm_only: bool = True

    # Large-event negRisk basket mode (> max_legs outcome sets)
    negrisk_large_event_subset_enabled: bool = True
    negrisk_large_event_max_subset: int = Field(default=15, ge=2, le=15)
    negrisk_large_event_tail_max_prob: float = Field(default=0.05, ge=0.0, le=0.5)

    # Fanatics Markets (Iteration 6)
    fanatics_api_key: str = ""
    fanatics_api_secret: str = ""
    fanatics_host: str = ""
    fanatics_position_limit: float = 25000.0
    fanatics_enabled: bool = False


def active_platforms(cfg: Config) -> list[str]:
    """Auto-detect external platforms with valid credentials."""
    platforms: list[str] = []
    if cfg.kalshi_api_key_id and cfg.kalshi_private_key_path:
        platforms.append("kalshi")
    if cfg.fanatics_api_key and cfg.fanatics_api_secret:
        platforms.append("fanatics")
    return platforms


def load_config() -> Config:
    """Load and validate config from environment. Raises on missing required fields."""
    return Config()
