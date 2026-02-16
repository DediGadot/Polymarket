#!/usr/bin/env python3
"""
Polymarket Arbitrage Bot -- Single pipeline script.

Combines all modules into one working pipeline:
  1. Auth + connect
  2. Fetch markets
  3. Scan for arbitrage (binary + negRisk)
  4. Size + safety check + execute
  5. Track P&L
  6. Repeat

Usage:
  uv run python run.py --dry-run        # no wallet needed, scan real markets
  uv run python run.py --scan-only      # detect only, no execution (needs wallet)
  uv run python run.py                  # paper trading (default)
  uv run python run.py --live           # live trading
"""

from __future__ import annotations

import argparse
import json
import logging
import signal
import sys
import time
from collections import Counter
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, replace
from pathlib import Path

from config import load_config, Config, active_platforms
from functools import partial

from client.auth import build_clob_client
from client.clob import (
    cancel_all,
    cancel_order,
    create_limit_order,
    create_market_order,
    get_orderbooks,
    get_orderbooks_parallel,
    post_order,
    post_orders,
)
from client.gamma import build_events
from client.cache import GammaClient
from client.gas import GasOracle
from client.platform import PlatformClient
from client.kalshi_cache import KalshiMarketCache
from scanner.binary import scan_binary_markets
from scanner.negrisk import scan_negrisk_events
from scanner.book_cache import BookCache
from scanner.book_service import BookService
from scanner.fees import MarketFeeModel
from scanner.latency import LatencyScanner, scan_latency_markets
from scanner.spike import SpikeDetector, scan_spike_opportunities
from scanner.depth import sweep_depth
from scanner.scorer import rank_opportunities, ScoringContext, ScoredOpportunity
from scanner.strategy import StrategySelector, MarketState
from scanner.models import Opportunity, OpportunityType, Side, TradeResult
from scanner.cross_platform import scan_cross_platform
from scanner.filters import apply_pre_filters
from scanner.kalshi_fees import KalshiFeeModel
from scanner.maker import MakerPersistenceGate, MakerExecutionModel
from scanner.matching import EventMatcher, MatchedEvent
from scanner.platform_fees import PlatformFeeModel
from scanner.realized_ev import RealizedEVTracker
from scanner.ofi import OFITracker
from scanner.correlation import CorrelationScanner
from scanner.ml_scorer import MLScorer, MLScorerConfig
from executor.sizing import compute_position_size
from executor.safety import (
    CircuitBreaker,
    CircuitBreakerTripped,
    SafetyCheckFailed,
    verify_prices_fresh,
    verify_depth,
    verify_gas_reasonable,
    verify_max_legs,
    verify_opportunity_ttl,
    verify_edge_intact,
    verify_inventory,
    verify_cross_platform_books,
    verify_platform_limits,
    verify_min_confidence,
)
from client.data import PositionTracker
from client.ws_bridge import WSBridge
from executor.maker_lifecycle import MakerLifecycle
from executor.engine import execute_opportunity, UnwindFailed
from executor.presigner import OrderPresigner
from executor.cross_platform import CrossPlatformUnwindFailed
from monitor.pnl import PnLTracker
from monitor.scan_tracker import ScanTracker
from scanner.confidence import ArbTracker
from state.checkpoint import CheckpointManager
from benchmark.recorder import create_recorder
from monitor.status import StatusWriter
from monitor.logger import setup_logging
from monitor.display import print_startup, print_cycle_header, print_scan_result, print_cycle_error, print_cycle_footer
from py_clob_client.clob_types import OrderType

logger = logging.getLogger(__name__)


_BANNER = r"""
 ____       _                            _        _
|  _ \ ___ | |_   _ _ __ ___   __ _ _ __| | _____| |_
| |_) / _ \| | | | | '_ ` _ \ / _` | '__| |/ / _ \ __|
|  __/ (_) | | |_| | | | | | | (_| | |  |   <  __/ |_
|_|   \___/|_|\__, |_| |_| |_|\__,_|_|  |_|\_\___|\__|
              |___/            Arbitrage Bot v0.1
"""


_EXECUTION_SUPPORTED_TYPES = {
    OpportunityType.BINARY_REBALANCE,
    OpportunityType.NEGRISK_REBALANCE,
    OpportunityType.LATENCY_ARB,
    OpportunityType.SPIKE_LAG,
    OpportunityType.CROSS_PLATFORM_ARB,
    OpportunityType.CORRELATION_ARB,
}
_EXECUTABLE_NOW_MIN_FILL_SCORE = 0.50
_CORRELATION_ACTIONABLE_BUY_PREFIXES = (
    "corr_complement_buy",
    "corr_parent_child_buy",
    "corr_temporal_buy",
)


@dataclass
class MakerPairState:
    """Tracks a posted maker YES/NO pair across scan cycles."""

    pair_id: str
    opportunity: Opportunity
    score: float
    yes_order_id: str
    no_order_id: str
    created_at: float
    yes_filled_size: float = 0.0
    no_filled_size: float = 0.0
    closed: bool = False

    @property
    def active(self) -> bool:
        return not self.closed

    @property
    def both_filled(self) -> bool:
        return self.yes_filled_size > 0 and self.no_filled_size > 0

    @property
    def one_leg_filled(self) -> bool:
        return (self.yes_filled_size > 0) ^ (self.no_filled_size > 0)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Polymarket Arbitrage Bot")
    parser.add_argument("--live", action="store_true", help="Enable live trading (disables paper mode)")
    parser.add_argument("--scan-only", action="store_true", help="Only scan for opportunities, do not execute")
    parser.add_argument("--dry-run", action="store_true", help="No wallet needed. Scan real markets using public APIs only")
    parser.add_argument("--limit", type=int, default=0, help="Max markets to scan (0 = all). Useful for dry-run testing")
    parser.add_argument("--json-log", type=str, default=None, help="Path to JSON log file for machine-readable output")
    parser.add_argument("--report", action="store_true", help="Enable pipeline dashboard")
    parser.add_argument("--report-host", type=str, default="0.0.0.0", help="Dashboard bind address (default: 0.0.0.0)")
    parser.add_argument("--report-port", type=int, default=8787, help="Dashboard server port (default: 8787)")
    return parser.parse_args()


def _mode_label(args: argparse.Namespace, cfg: Config) -> str:
    if args.dry_run:
        return "DRY-RUN (public APIs only, no execution)"
    if args.scan_only:
        return "SCAN-ONLY (detect opportunities, no execution)"
    if cfg.paper_trading:
        return "PAPER TRADING (simulated fills, no real orders)"
    return "LIVE TRADING"


def _enforce_polymarket_only_mode(cfg: Config) -> Config:
    """
    Disable strategies/integrations that require non-Polymarket APIs unless
    explicitly opted in via config.

    Returns a new Config with appropriate flags disabled (immutable).
    """
    if cfg.allow_non_polymarket_apis:
        return cfg

    updates = {}
    if cfg.latency_enabled:
        logger.warning(
            "LATENCY_ENABLED=true ignored because ALLOW_NON_POLYMARKET_APIS=false "
            "(Binance spot feed is external)."
        )
        updates["latency_enabled"] = False

    if cfg.cross_platform_enabled:
        logger.warning(
            "CROSS_PLATFORM_ENABLED=true ignored because ALLOW_NON_POLYMARKET_APIS=false "
            "(Kalshi API is external)."
        )
        updates["cross_platform_enabled"] = False

    if updates:
        return cfg.model_copy(update=updates)
    return cfg


def _format_duration(seconds: float) -> str:
    """Format seconds into a human-readable duration string."""
    if seconds < 60:
        return f"{seconds:.1f}s"
    minutes = int(seconds) // 60
    secs = seconds - minutes * 60
    if minutes < 60:
        return f"{minutes}m {secs:.0f}s"
    hours = minutes // 60
    mins = minutes % 60
    return f"{hours}h {mins}m"


def _is_execution_supported_type(opp_type: OpportunityType) -> bool:
    """True if executor + safety stack can currently handle this opportunity type."""
    return opp_type in _EXECUTION_SUPPORTED_TYPES


def _is_research_opportunity(opp: Opportunity, cfg: Config) -> bool:
    """
    Classify opportunities into research vs executable lanes.

    Research lane defaults:
    - Correlation opportunities unless explicitly enabled for execution.
    - Any strategy type not handled by the taker execution stack, except maker
      opportunities (which are handled by maker lifecycle separately).
    """
    if opp.type == OpportunityType.MAKER_REBALANCE:
        return False
    if opp.type == OpportunityType.CORRELATION_ARB:
        if not cfg.correlation_execute_enabled:
            return cfg.research_lane_enabled
        # Allow buy-only correlation arbs into executable lane
        return not opp.is_buy_arb
    if not _is_execution_supported_type(opp.type):
        return cfg.research_lane_enabled
    return False


def _is_executable_now(scored: ScoredOpportunity, cfg: Config) -> bool:
    """
    Conservative gate for "actionable now" taker opportunities.

    Excludes maker/resting strategies, unsupported strategy types, SELL-side
    inventory-dependent opportunities, and thin-book candidates.
    """
    return _executable_reject_reason(scored, cfg) is None


def _executable_reject_reason(scored: ScoredOpportunity, cfg: Config) -> str | None:
    """Return rejection reason for actionable-now gate, or None if accepted."""
    opp = scored.opportunity
    if opp.type == OpportunityType.MAKER_REBALANCE:
        return "maker"
    if not _is_execution_supported_type(opp.type):
        return "unsupported_type"
    if not opp.is_buy_arb:
        return "not_buy_arb"
    if not opp.is_profitable:
        return "not_profitable"

    # Correlation is only actionable for approved buy-side structures.
    if opp.type == OpportunityType.CORRELATION_ARB:
        allowed_prefixes = ("corr_complement_buy",)
        if cfg.correlation_actionable_allow_structural_buy:
            allowed_prefixes = _CORRELATION_ACTIONABLE_BUY_PREFIXES
        if not any(opp.reason_code.startswith(prefix) for prefix in allowed_prefixes):
            return "correlation_not_actionable_buy"

    min_fill = _EXECUTABLE_NOW_MIN_FILL_SCORE
    min_persistence = cfg.min_confidence_gate
    if opp.type == OpportunityType.CORRELATION_ARB:
        min_fill = cfg.correlation_actionable_min_fill_score
        min_persistence = cfg.correlation_actionable_min_confidence

    if scored.fill_score < min_fill:
        return "low_fill_score"
    if scored.persistence_score < min_persistence:
        return "low_persistence"
    return None



def _print_scan_summary(tracker: ScanTracker) -> None:
    """Print a rich shutdown summary for scan-only mode."""
    s = tracker.summary()
    logger.info("")
    logger.info("=" * 70)
    logger.info("  SESSION SUMMARY (scan-only)")
    logger.info("=" * 70)
    logger.info("  %-30s %d", "Total cycles:", s["total_cycles"])
    logger.info("  %-30s %s", "Session duration:", _format_duration(s["duration_sec"]))
    logger.info("  %-30s %s", "Markets scanned (cumulative):", f"{s['markets_scanned']:,}")
    logger.info("  %-30s %d", "Opportunities found:", s["opportunities_found"])
    logger.info("  %-30s %d", "Unique events:", s["unique_events"])
    if s["opportunities_found"] > 0:
        logger.info("  %-30s %.2f%%", "Best ROI:", s["best_roi_pct"])
        logger.info("  %-30s $%.2f", "Best profit:", s["best_profit_usd"])
        logger.info("  %-30s $%.2f", "Total theoretical profit:", s["total_theoretical_profit_usd"])
        logger.info(
            "    %-28s $%.2f (%d opps)",
            "Executable lane:",
            s.get("executable_opp_profit_usd", 0.0),
            s.get("executable_opp_count", 0),
        )
        logger.info(
            "    %-28s $%.2f (%d opps)",
            "Research lane:",
            s.get("research_opp_profit_usd", 0.0),
            s.get("research_opp_count", 0),
        )
        logger.info(
            "    %-28s $%.2f (%d opps)",
            "Actionable now (taker BUY):",
            s["actionable_now_profit_usd"],
            s["actionable_now_count"],
        )
        logger.info(
            "    %-28s $%.2f (%d opps)",
            "Maker candidates (resting):",
            s["maker_candidate_profit_usd"],
            s["maker_candidate_count"],
        )
        logger.info("    %-28s $%.2f (%d opps)", "Requires inventory (sell):", s["sell_arb_profit_usd"], s["sell_arb_count"])
        logger.info("  %-30s %.2f%%", "Avg ROI:", s["avg_roi_pct"])
        logger.info("  %-30s $%.2f", "Avg profit:", s["avg_profit_usd"])
        if s["by_type"]:
            logger.info("  Breakdown by type:")
            for k, v in s["by_type"].items():
                logger.info("    %-28s %d", k, v)
    logger.info("=" * 70)


def _print_pnl_summary(pnl: PnLTracker) -> None:
    """Print a rich shutdown summary for trading mode."""
    s = pnl.summary()
    logger.info("")
    logger.info("=" * 70)
    logger.info("  SESSION SUMMARY (trading)")
    logger.info("=" * 70)
    logger.info("  %-30s %s", "Session duration:", _format_duration(s["session_duration_sec"]))
    logger.info("  %-30s %d", "Total trades:", s["total_trades"])
    logger.info("  %-30s %d W / %d L (%.1f%% win rate)", "Results:", s["winning_trades"], s["losing_trades"], s["win_rate_pct"])
    logger.info("  %-30s $%.2f", "Net P&L:", s["total_pnl"])
    logger.info("  %-30s $%.2f", "Avg P&L per trade:", s["avg_pnl"])
    logger.info("  %-30s $%.2f", "Total volume:", s["total_volume"])
    logger.info("  %-30s $%.2f", "Current exposure:", s["current_exposure"])
    logger.info("=" * 70)


def main() -> None:
    args = parse_args()
    cfg = load_config()
    dry_run_auto_limit_applied = False

    # Override paper trading based on CLI flag
    if args.live:
        cfg = cfg.model_copy(update={"paper_trading": False})
    if args.dry_run:
        args.scan_only = True  # dry-run implies scan-only
        if args.limit <= 0 and cfg.dry_run_default_limit > 0:
            args.limit = cfg.dry_run_default_limit
            dry_run_auto_limit_applied = True

    cfg = _enforce_polymarket_only_mode(cfg)

    # Validate credentials for non-dry-run modes
    if not args.dry_run and (not cfg.private_key or not cfg.polymarket_profile_address):
        logger.error("PRIVATE_KEY and POLYMARKET_PROFILE_ADDRESS required for paper/live trading.")
        logger.error("Use --dry-run to scan without a wallet.")
        sys.exit(1)

    log_file_path = setup_logging(cfg.log_level, json_log_file=args.json_log)
    logger.info(_BANNER.strip())
    logger.info("  Log file: %s", log_file_path)
    print_startup(cfg, args)
    if dry_run_auto_limit_applied:
        logger.warning(
            "Dry-run auto-limit enabled: scanning at most %d markets per cycle "
            "(set DRY_RUN_DEFAULT_LIMIT=0 to disable).",
            args.limit,
        )

    # Initialize client
    if args.dry_run:
        logger.debug("Initializing unauthenticated CLOB client (dry-run mode)...")
        from py_clob_client.client import ClobClient
        client = ClobClient(host=cfg.clob_host)
        logger.debug("Client ready -- using public endpoints only")
    else:
        logger.debug("Authenticating with Polymarket CLOB...")
        logger.debug("  Deriving L2 API credentials from wallet...")
        client = build_clob_client(cfg)
        logger.debug("Authentication successful -- client ready for %s",
                      "paper trading" if cfg.paper_trading else "LIVE trading")

    # Initialize components
    pnl = PnLTracker()
    tracker = ScanTracker()
    breaker = CircuitBreaker(
        max_loss_per_hour=cfg.max_loss_per_hour,
        max_loss_per_day=cfg.max_loss_per_day,
        max_consecutive_failures=cfg.max_consecutive_failures,
    )
    book_cache_age = cfg.book_cache_max_age_sec
    if args.dry_run:
        book_cache_age = max(book_cache_age, cfg.dry_run_book_cache_max_age_sec)
    book_cache = BookCache(max_age_sec=book_cache_age)
    gas_oracle = GasOracle(
        rpc_url=cfg.polygon_rpc_url,
        cache_sec=cfg.gas_cache_sec,
        default_gas_gwei=cfg.gas_price_gwei,
        allow_network=cfg.allow_non_polymarket_apis,
    )
    gamma_client = GammaClient(gamma_host=cfg.gamma_host)
    logger.debug("Gamma client initialized with caching (host=%s)", cfg.gamma_host)
    fee_model = MarketFeeModel(enabled=cfg.fee_model_enabled)
    latency_scanner = LatencyScanner(
        min_edge_pct=cfg.latency_min_edge_pct,
        spot_cache_sec=cfg.spot_price_cache_sec,
        fee_model=fee_model,
    ) if cfg.latency_enabled else None
    spike_detector = SpikeDetector(
        threshold_pct=cfg.spike_threshold_pct,
        window_sec=cfg.spike_window_sec,
        cooldown_sec=cfg.spike_cooldown_sec,
    )
    position_tracker = PositionTracker(
        profile_address=cfg.polymarket_profile_address if not args.dry_run else "",
    )
    logger.debug("Safety systems initialized (circuit breaker active)")
    logger.debug("Book cache initialized (max_age=%.1fs)", book_cache_age)
    if args.dry_run and book_cache_age > cfg.book_cache_max_age_sec:
        logger.debug(
            "Dry-run cache extension active (book_cache_max_age %.1fs -> %.1fs)",
            cfg.book_cache_max_age_sec,
            book_cache_age,
        )
    logger.debug("Gas oracle initialized (rpc=%s, cache=%.0fs)", cfg.polygon_rpc_url, cfg.gas_cache_sec)
    logger.debug("Fee model %s", "enabled" if cfg.fee_model_enabled else "disabled")
    logger.debug("Latency scanner %s (min_edge=%.1f%%)", "enabled" if cfg.latency_enabled else "disabled", cfg.latency_min_edge_pct)
    strategy = StrategySelector(
        base_min_profit=cfg.min_profit_usd,
        base_min_roi=cfg.min_roi_pct,
        base_target_size=cfg.target_size_usd,
    )
    logger.debug("Spike detector initialized (threshold=%.1f%%, window=%.0fs, cooldown=%.0fs)",
                 cfg.spike_threshold_pct, cfg.spike_window_sec, cfg.spike_cooldown_sec)
    logger.debug("Strategy selector initialized (adaptive mode)")

    # ArbTracker for persistent arb confidence scoring
    arb_tracker = ArbTracker()
    logger.debug("ArbTracker initialized for persistence confidence")

    # Maker-specific state
    maker_persistence_gate = MakerPersistenceGate(
        min_consecutive_cycles=cfg.maker_min_persistence_cycles,
    )
    maker_execution_model = MakerExecutionModel()
    maker_ev_tracker = RealizedEVTracker()

    # OFI tracker
    ofi_tracker = OFITracker()

    # Correlation scanner
    correlation_scanner: CorrelationScanner | None = None
    if cfg.correlation_scanner_enabled:
        correlation_scanner = CorrelationScanner(
            min_edge_pct=cfg.correlation_min_edge_pct,
            min_confidence=cfg.correlation_min_confidence,
            aggregation=cfg.correlation_aggregation,
            max_markets_per_event=cfg.correlation_max_markets_per_event,
            min_market_volume=cfg.correlation_min_market_volume,
            min_book_depth=cfg.correlation_min_book_depth,
            max_theoretical_roi_pct=cfg.correlation_max_theoretical_roi_pct,
            min_buy_total_prob=cfg.correlation_min_buy_total_prob,
            min_persistence_cycles=cfg.correlation_min_persistence_cycles,
            max_capital_per_opportunity=cfg.correlation_max_capital_per_opp_usd,
        )
        logger.debug(
            "Correlation scanner initialized (min_edge=%.1f%%, min_confidence=%.2f, min_buy_total=%.2f, persistence=%d, cap=$%.2f)",
            cfg.correlation_min_edge_pct,
            cfg.correlation_min_confidence,
            cfg.correlation_min_buy_total_prob,
            cfg.correlation_min_persistence_cycles,
            cfg.correlation_max_capital_per_opp_usd,
        )

    # Optional ML reranker (gated; deterministic scorer remains primary fallback)
    ml_scorer: MLScorer | None = None
    if cfg.ml_scorer_enabled:
        model_path = Path(cfg.ml_scorer_model_path)
        if model_path.exists():
            try:
                ml_scorer = MLScorer.load(model_path)
            except Exception as e:
                logger.warning("Failed to load ML scorer model (%s): %s", model_path, e)
        if ml_scorer is None:
            ml_scorer = MLScorer(
                config=MLScorerConfig(
                    min_samples=cfg.ml_scorer_min_samples,
                    retrain_every_cycles=cfg.ml_scorer_retrain_cycles,
                    model_path=cfg.ml_scorer_model_path,
                )
            )
        logger.debug(
            "ML scorer initialized (enabled=%s, trained=%s, samples=%d)",
            cfg.ml_scorer_enabled,
            ml_scorer.is_trained if ml_scorer is not None else False,
            ml_scorer.sample_count if ml_scorer is not None else 0,
        )

    # Checkpoint: restore tracker state from previous session
    checkpoint_mgr: CheckpointManager | None = None
    if cfg.state_checkpoint_enabled:
        checkpoint_mgr = CheckpointManager(
            db_path=cfg.state_checkpoint_db,
            auto_save_interval=cfg.state_checkpoint_interval,
        )
        _tracker_registry: dict[str, tuple[str, type, object]] = {
            "arb_tracker": ("arb_tracker", ArbTracker, arb_tracker),
            "spike_detector": ("spike_detector", SpikeDetector, spike_detector),
            "maker_persistence_gate": ("maker_persistence_gate", MakerPersistenceGate, maker_persistence_gate),
            "maker_ev_tracker": ("maker_ev_tracker", RealizedEVTracker, maker_ev_tracker),
            "ofi_tracker": ("ofi_tracker", OFITracker, ofi_tracker),
        }
        for ckpt_name, (_, tracker_cls, _) in _tracker_registry.items():
            restored = checkpoint_mgr.load(ckpt_name, tracker_cls)
            if restored is not None:
                if ckpt_name == "arb_tracker":
                    arb_tracker = restored
                elif ckpt_name == "spike_detector":
                    spike_detector = restored
                elif ckpt_name == "maker_persistence_gate":
                    maker_persistence_gate = restored
                elif ckpt_name == "maker_ev_tracker":
                    maker_ev_tracker = restored
                elif ckpt_name == "ofi_tracker":
                    ofi_tracker = restored

        # Register live trackers for auto-save
        checkpoint_mgr.register("arb_tracker", arb_tracker)
        checkpoint_mgr.register("spike_detector", spike_detector)
        checkpoint_mgr.register("maker_persistence_gate", maker_persistence_gate)
        checkpoint_mgr.register("maker_ev_tracker", maker_ev_tracker)
        checkpoint_mgr.register("ofi_tracker", ofi_tracker)
        logger.debug(
            "Checkpoint manager initialized (db=%s, interval=%d cycles, %d checkpoints found)",
            cfg.state_checkpoint_db, cfg.state_checkpoint_interval,
            len(checkpoint_mgr.list_checkpoints()),
        )
    maker_max_age_sec = min(cfg.maker_order_max_age_sec, cfg.maker_quote_ttl_sec) if cfg.maker_use_gtd else cfg.maker_order_max_age_sec
    maker_lifecycle = MakerLifecycle(
        max_age_sec=maker_max_age_sec,
        max_orders=max(2, cfg.maker_max_active_pairs * 2),
        max_drift_ticks=cfg.maker_order_max_drift_ticks,
    )
    maker_pairs: dict[str, MakerPairState] = {}
    maker_order_to_pair: dict[str, str] = {}
    last_full_scan_at = 0.0
    logger.debug(
        "Maker realism+execution initialized (persist=%d cycles, max_pairs=%d, max_taker_cost=%.3f, "
        "max_spread_ticks=%d, min_fill_p=%.2f, max_tox=%.2f, min_ev=$%.2f, post_only=%s, gtd=%s ttl=%.1fs)",
        cfg.maker_min_persistence_cycles,
        cfg.maker_max_active_pairs,
        cfg.maker_max_taker_cost,
        cfg.maker_max_spread_ticks,
        cfg.maker_min_pair_fill_prob,
        cfg.maker_max_toxicity_score,
        cfg.maker_min_expected_ev_usd,
        cfg.maker_post_only,
        cfg.maker_use_gtd,
        cfg.maker_quote_ttl_sec,
    )

    # Report dashboard (optional)
    from report import create_collector
    collector = create_collector(enabled=args.report)
    if args.report:
        from report.server import start_server, notify_cycle
        collector._sse_callback = notify_cycle
        start_server(collector._store, host=args.report_host, port=args.report_port)
        _secret_keys = {"private_key", "kalshi_api_key_id", "kalshi_private_key_path", "fanatics_api_key", "fanatics_api_secret"}
        safe_config = {k: ("***" if k in _secret_keys and v else v) for k, v in cfg.model_dump().items()}
        collector.start_session(
            mode=_mode_label(args, cfg),
            config_json=json.dumps(safe_config),
            cli_args_json=json.dumps({"dry_run": args.dry_run, "scan_only": args.scan_only, "live": args.live, "limit": args.limit}),
        )

    # Cross-platform initialization: build platform registry from credentials
    platform_clients: dict[str, PlatformClient] = {}
    platform_fee_models: dict[str, PlatformFeeModel] = {}
    event_matcher = None
    kalshi_market_cache: KalshiMarketCache | None = None

    if cfg.cross_platform_enabled:
        detected = active_platforms(cfg)
        if detected:
            event_matcher = EventMatcher(
                manual_map_path=cfg.cross_platform_manual_map,
                verified_path=cfg.cross_platform_verified_path,
                negative_ttl_sec=cfg.cross_platform_matching_negative_ttl_sec,
            )

        for pname in detected:
            if pname == "kalshi":
                from client.kalshi_auth import KalshiAuth
                from client.kalshi import KalshiClient
                kalshi_auth = KalshiAuth(
                    api_key_id=cfg.kalshi_api_key_id,
                    private_key_path=cfg.kalshi_private_key_path,
                )
                platform_clients["kalshi"] = KalshiClient(
                    auth=kalshi_auth,
                    host=cfg.kalshi_host,
                    demo=cfg.kalshi_demo,
                )
                platform_fee_models["kalshi"] = KalshiFeeModel()
                kalshi_market_cache = KalshiMarketCache(
                    client=platform_clients["kalshi"],
                    refresh_sec=cfg.kalshi_market_refresh_sec,
                    warm_timeout_sec=cfg.kalshi_market_warm_timeout_sec,
                )
                kalshi_market_cache.start()
                logger.debug("Kalshi client initialized (host=%s, demo=%s)", cfg.kalshi_host, cfg.kalshi_demo)

            elif pname == "fanatics":
                from client.fanatics_auth import FanaticsAuth
                from client.fanatics import FanaticsClient
                from scanner.fanatics_fees import FanaticsFeeModel
                fanatics_auth = FanaticsAuth(
                    api_key=cfg.fanatics_api_key,
                    api_secret=cfg.fanatics_api_secret,
                )
                platform_clients["fanatics"] = FanaticsClient(
                    auth=fanatics_auth,
                    host=cfg.fanatics_host,
                )
                platform_fee_models["fanatics"] = FanaticsFeeModel()
                logger.debug("Fanatics client initialized (host=%s)", cfg.fanatics_host)

        if platform_clients:
            logger.debug("Cross-platform enabled: %s (min_confidence=%.2f, map=%s)",
                         ", ".join(platform_clients.keys()),
                         cfg.cross_platform_min_confidence, cfg.cross_platform_manual_map)
        else:
            logger.debug("Cross-platform enabled but no platform credentials configured")
    else:
        logger.debug("Cross-platform arbitrage disabled")

    # WebSocket bridge for real-time book/price feeds
    ws_bridge: WSBridge | None = None
    if cfg.ws_enabled and not args.dry_run:
        ws_bridge = WSBridge(
            ws_url=cfg.ws_market_url,
            book_cache=book_cache,
            spike_detector=spike_detector,
            ofi_tracker=ofi_tracker,
            max_retries=cfg.ws_reconnect_max,
        )
        logger.debug("WebSocket bridge initialized (url=%s)", cfg.ws_market_url)
    elif cfg.ws_enabled:
        logger.debug("WebSocket disabled in dry-run mode (REST only)")

    # Runtime recorder for replay/backtesting datasets
    recorder = create_recorder(
        enabled=cfg.recording_enabled,
        output_dir=cfg.recording_dir,
        max_mb=cfg.recording_max_mb,
    )

    # Presigner for low-latency execution path (live/paper execution modes only)
    presigner: OrderPresigner | None = None
    if cfg.presigner_enabled and not args.scan_only and not args.dry_run:
        def _presign_fn(
            *,
            token_id: str,
            side: str,
            price: float,
            size: float,
            neg_risk: bool = False,
            tick_size: str = "0.01",
            **_: object,
        ) -> object:
            side_enum = Side.BUY if side.upper() == "BUY" else Side.SELL
            return create_limit_order(
                client,
                token_id=token_id,
                side=side_enum,
                price=price,
                size=size,
                neg_risk=neg_risk,
                tick_size=tick_size,
            )

        presigner = OrderPresigner(
            sign_fn=_presign_fn,
            max_cache_size=cfg.presigner_max_cache_size,
            max_age_sec=cfg.presigner_max_age_sec,
            tick_levels=cfg.presigner_tick_levels,
        )
        logger.debug(
            "Presigner initialized (cache=%d age=%.1fs levels=%d)",
            cfg.presigner_max_cache_size,
            cfg.presigner_max_age_sec,
            cfg.presigner_tick_levels,
        )

    status_writer = StatusWriter(file_path="status.md")
    status_writer.write_cycle(
        cycle=0,
        mode=_mode_label(args, cfg),
        markets_scanned=0,
        scored_opps=[],
        event_questions={},
        market_questions={},
        total_opps_found=0,
        total_trades_executed=0,
        total_pnl=0.0,
        current_exposure=0.0,
        scan_only=args.scan_only,
        executable_lane_count=0,
        research_lane_count=0,
        executable_lane_profit=0.0,
        research_lane_profit=0.0,
    )
    logger.debug("Status writer initialized (status.md, last %d cycles)", status_writer.max_history)
    logger.debug("")

    # Graceful shutdown handler
    shutdown_requested = False
    shutdown_signal: int | None = None

    def handle_signal(signum, frame):
        nonlocal shutdown_requested, shutdown_signal
        shutdown_requested = True
        if shutdown_signal is None:
            shutdown_signal = signum

    signal.signal(signal.SIGINT, handle_signal)
    signal.signal(signal.SIGTERM, handle_signal)

    # Main loop
    cycle = 0
    session_start = time.time()
    total_opps_found = 0
    total_trades_executed = 0
    has_crypto_momentum = False
    best_profit_ever = 0.0
    best_roi_ever = 0.0
    rate_limit_streak = 0
    executable_zero_streak = 0

    # Cached processed market data -- reused across cycles when the Gamma
    # market cache hasn't refreshed (59 out of every 60 cycles).
    _last_markets_ts: float = 0.0
    _cached_all_markets: list[Market] | None = None
    _cached_events: list[Event] | None = None
    _cached_binary_markets: list[Market] | None = None
    _cached_negrisk_events: list[Event] | None = None
    _cached_event_questions: dict[str, str] | None = None
    _cached_market_questions: dict[str, str] | None = None

    # Hoist invariant fetcher creation outside the loop.
    poly_rest_fetcher = partial(get_orderbooks_parallel, client, max_workers=cfg.book_fetch_workers)
    book_service = BookService(book_cache=book_cache, rest_fetcher=poly_rest_fetcher)
    poly_book_fetcher = book_service.make_fetcher()

    while not shutdown_requested:
        cycle += 1
        cycle_start = time.time()
        collector.begin_cycle(cycle)

        try:
            print_cycle_header(cycle)

            # Step 1: Fetch active markets (with caching)
            fetch_start = time.time()
            logger.debug("[1/3] Fetching active markets from Gamma API...")

            # On the very first cycle, pre-warm event_market_counts in
            # parallel with the market fetch.  Both hit the Gamma API
            # independently; overlapping saves ~10s on cold start.  After
            # cycle 1 the EMC cache is warm (300s TTL) so the parallel
            # fetch would be a no-op.
            if cycle == 1:
                emc_executor = ThreadPoolExecutor(max_workers=1)
                emc_future = emc_executor.submit(gamma_client.get_event_market_counts)
            else:
                emc_future = None

            # Trigger market fetch (returns from 60s cache on most cycles)
            all_markets_raw = gamma_client.get_markets()
            collector.record_funnel("raw_markets", len(all_markets_raw))
            current_markets_ts = gamma_client.markets_timestamp
            markets_changed = not (
                current_markets_ts == _last_markets_ts and _cached_all_markets is not None
            )

            # Wait for EMC pre-warm on first cycle
            if emc_future is not None:
                try:
                    emc_future.result(timeout=30.0)
                except Exception:
                    logger.debug("      Event market counts pre-warm failed (negrisk will retry)")
                emc_executor.shutdown(wait=False)

            # Skip expensive reprocessing when market data hasn't changed.
            # The Gamma cache refreshes every 60s; we reuse the previous
            # cycle's filtered/grouped data for the other ~59 cycles.
            if current_markets_ts == _last_markets_ts and _cached_all_markets is not None:
                all_markets = _cached_all_markets
                events = _cached_events
                binary_markets = _cached_binary_markets
                negrisk_events = _cached_negrisk_events
                event_questions = _cached_event_questions
                market_questions = _cached_market_questions
                fetch_elapsed = time.time() - fetch_start
                logger.debug(
                    "      Markets unchanged (cache hit), reusing %s markets in %.3fs",
                    f"{len(all_markets):,}", fetch_elapsed,
                )
            else:
                # Markets changed -- reprocess
                all_markets = all_markets_raw
                if args.limit > 0:
                    # Never truncate negRisk markets -- they require complete
                    # outcome sets.  Apply limit to non-negRisk only, then
                    # include complete sibling groups for any negRisk markets
                    # that fell within the initial truncated window.
                    non_nr = [m for m in all_markets if not m.neg_risk]
                    nr_all = [m for m in all_markets if m.neg_risk]
                    non_nr = non_nr[:args.limit]

                    # Determine which negRisk groups to keep: those whose
                    # neg_risk_market_id appeared in the first `limit` markets.
                    initial_truncated = all_markets[:args.limit]
                    nr_ids = {
                        m.neg_risk_market_id or m.event_id
                        for m in initial_truncated
                        if m.neg_risk
                    }

                    if nr_ids:
                        nr_complete = [
                            m for m in nr_all
                            if (m.neg_risk_market_id or m.event_id) in nr_ids
                        ]
                        extra = len(nr_complete) - sum(
                            1 for m in initial_truncated if m.neg_risk
                        )
                        if extra > 0:
                            logger.debug(
                                "      --limit: included %d extra negRisk markets "
                                "for event completeness",
                                extra,
                            )
                        all_markets = non_nr + nr_complete
                    else:
                        all_markets = non_nr
                pre_filter_count = len(all_markets)
                all_markets = apply_pre_filters(
                    all_markets,
                    min_volume=cfg.min_volume_filter,
                    min_hours=cfg.min_hours_to_resolution,
                )
                if len(all_markets) < pre_filter_count:
                    logger.debug(
                        "      Pre-filtered %d → %d markets (volume≥%.0f, hours≥%.1f)",
                        pre_filter_count, len(all_markets),
                        cfg.min_volume_filter, cfg.min_hours_to_resolution,
                    )
                events = build_events(all_markets)
                event_questions = {e.event_id: e.title for e in events}
                market_questions = {
                    token_id: market.question
                    for market in all_markets
                    for token_id in (market.yes_token_id, market.no_token_id)
                }
                binary_markets = [m for m in all_markets if not m.neg_risk]
                negrisk_events = [e for e in events if e.neg_risk]

                # Cache for subsequent cycles
                _last_markets_ts = current_markets_ts
                _cached_all_markets = all_markets
                _cached_events = events
                _cached_binary_markets = binary_markets
                _cached_negrisk_events = negrisk_events
                _cached_event_questions = event_questions
                _cached_market_questions = market_questions

                fetch_elapsed = time.time() - fetch_start
                logger.debug(
                    "      Received %s markets in %.1fs",
                    f"{len(all_markets):,}", fetch_elapsed,
                )
            # Record funnel stages (always, even on cache-hit cycles)
            collector.record_funnel("after_filter", len(all_markets))
            logger.debug(
                "      %s binary  |  %d negRisk events (%s markets)",
                f"{len(binary_markets):,}",
                len(negrisk_events),
                f"{sum(len(e.markets) for e in negrisk_events):,}",
            )

            # Start WebSocket on first cycle (now we know which tokens to subscribe to)
            if ws_bridge and not ws_bridge.is_connected:
                all_token_ids = []
                for m in all_markets:
                    all_token_ids.append(m.yes_token_id)
                    all_token_ids.append(m.no_token_id)
                ws_bridge.start(all_token_ids)

            # Drain any queued WebSocket updates into caches
            ws_updates = 0
            ws_changed_tokens: set[str] = set()
            if ws_bridge:
                ws_updates = ws_bridge.drain()
                ws_changed_tokens = ws_bridge.last_changed_tokens
                if ws_updates > 0:
                    logger.debug("      WebSocket: drained %d updates into cache", ws_updates)

            # Step 2: Scan for opportunities (strategy-tuned)
            scan_start = time.time()
            force_rescan_due = (time.time() - last_full_scan_at) >= cfg.ws_force_rescan_sec
            skip_full_scan = (
                cfg.ws_event_driven_scan
                and ws_bridge is not None
                and cycle > 1
                and not markets_changed
                and ws_updates == 0
                and not force_rescan_due
            )
            if skip_full_scan:
                logger.debug(
                    "      Event-driven scan skip (no WS deltas, markets unchanged, force_rescan_in=%.1fs)",
                    max(0.0, cfg.ws_force_rescan_sec - (time.time() - last_full_scan_at)),
                )
            else:
                last_full_scan_at = time.time()

            # Select adaptive strategy for this cycle
            market_state = MarketState(
                gas_price_gwei=gas_oracle.get_gas_price_gwei(),
                active_spike_count=len(spike_detector.detect_spikes()) if cycle > 1 else 0,
                has_crypto_momentum=has_crypto_momentum,
                recent_win_rate=pnl.win_rate / 100.0 if not args.scan_only else 0.50,
            )
            scan_params = strategy.select(market_state)
            logger.debug("[2/3] Scanning for arbitrage (strategy=%s)...", scan_params.mode.value)
            from report.collector import StrategySnapshot
            collector.record_strategy(StrategySnapshot(
                cycle=cycle, mode=scan_params.mode.value,
                gas_price_gwei=market_state.gas_price_gwei,
                active_spike_count=market_state.active_spike_count,
                has_crypto_momentum=market_state.has_crypto_momentum,
                recent_win_rate=market_state.recent_win_rate,
            ))

            # Centralized prefetch: fetch once per cycle, let scanners read cached books.
            if not skip_full_scan:
                prefetch_ids: list[str] = []
                if scan_params.binary_enabled:
                    for m in binary_markets:
                        if shutdown_requested:
                            break
                        prefetch_ids.append(m.yes_token_id)
                        prefetch_ids.append(m.no_token_id)
                if scan_params.negrisk_enabled or cfg.value_scanner_enabled:
                    for event in negrisk_events:
                        if shutdown_requested:
                            break
                        for m in event.markets:
                            if m.active:
                                prefetch_ids.append(m.yes_token_id)
                if cfg.correlation_scanner_enabled and correlation_scanner:
                    for event in events:
                        if shutdown_requested:
                            break
                        for m in event.markets:
                            if m.active:
                                prefetch_ids.append(m.yes_token_id)
                if prefetch_ids:
                    book_service.prefetch(prefetch_ids)
                    logger.debug(
                        "      BookService prefetched %d tokens (cached=%d)",
                        len(set(prefetch_ids)),
                        book_service.stats["cached_tokens"],
                    )

            # Parallelize independent scanners (binary + negrisk) using ThreadPoolExecutor
            # These scanners have no dependencies on each other and can run concurrently
            binary_opps: list[Opportunity] = []
            negrisk_opps: list[Opportunity] = []

            def _run_binary_scan() -> list[Opportunity]:
                if shutdown_requested:
                    return []
                if not scan_params.binary_enabled:
                    return []
                logger.debug("      Scanning %s binary markets...", f"{len(binary_markets):,}")
                return scan_binary_markets(
                    poly_book_fetcher, binary_markets,
                    scan_params.min_profit_usd, scan_params.min_roi_pct,
                    cfg.gas_per_order, cfg.gas_price_gwei,
                    gas_oracle=gas_oracle, fee_model=fee_model, book_cache=book_cache,
                    min_volume=cfg.min_volume_filter,
                    slippage_fraction=cfg.slippage_fraction,
                    max_slippage_pct=cfg.max_slippage_pct,
                )

            def _run_negrisk_scan() -> list[Opportunity]:
                if shutdown_requested:
                    return []
                if not scan_params.negrisk_enabled:
                    return []
                logger.debug("      Scanning %d negRisk events...", len(negrisk_events))
                # Fetch expected market counts for event completeness validation (with caching).
                # This prevents false arbs from incomplete outcome sets.
                event_market_counts = gamma_client.get_event_market_counts()
                candidate_events = negrisk_events
                if event_market_counts:
                    candidate_events = []
                    for event in negrisk_events:
                        nrm_key = event.neg_risk_market_id or event.event_id
                        expected_total = event_market_counts.get(nrm_key, 0)
                        if expected_total == 0:
                            continue
                        if (
                            cfg.max_legs_per_opportunity > 0
                            and expected_total > cfg.max_legs_per_opportunity
                            and not cfg.negrisk_large_event_subset_enabled
                        ):
                            continue
                        candidate_events.append(event)
                    if len(candidate_events) < len(negrisk_events):
                        logger.debug(
                            "      negRisk prefilter kept %d/%d events",
                            len(candidate_events),
                            len(negrisk_events),
                        )
                return scan_negrisk_events(
                    poly_book_fetcher, candidate_events,
                    scan_params.min_profit_usd, scan_params.min_roi_pct,
                    cfg.gas_per_order, cfg.gas_price_gwei,
                    gas_oracle=gas_oracle, fee_model=fee_model, book_cache=book_cache,
                    min_volume=cfg.min_volume_filter,
                    max_legs=cfg.max_legs_per_opportunity,
                    event_market_counts=event_market_counts,
                    slippage_fraction=cfg.slippage_fraction,
                    max_slippage_pct=cfg.max_slippage_pct,
                    large_event_subset_enabled=cfg.negrisk_large_event_subset_enabled,
                    large_event_max_subset=cfg.negrisk_large_event_max_subset,
                    large_event_tail_max_prob=cfg.negrisk_large_event_tail_max_prob,
                    should_stop=lambda: shutdown_requested,
                )

            # Define scanner functions for parallel execution
            def _run_latency_scan() -> tuple[list[Opportunity], bool]:
                """Run latency arb scanner. Returns (opportunities, has_crypto_momentum)."""
                from scanner.latency import scan_latency_markets
                opps: list[Opportunity] = []
                has_momentum = False
                if shutdown_requested:
                    return opps, has_momentum
                if latency_scanner:
                    crypto_markets = latency_scanner.identify_crypto_markets(all_markets)
                    has_momentum = bool(crypto_markets)
                    if crypto_markets:
                        logger.debug("      Scanning %d crypto 15-min markets for latency arb...", len(crypto_markets))
                        # Fetch spot prices
                        symbols_seen: set[str] = set()
                        for _, sym, _ in crypto_markets:
                            if sym not in symbols_seen:
                                latency_scanner.get_spot_price(sym)
                                symbols_seen.add(sym)
                        # Fetch books for crypto markets (YES + NO tokens)
                        crypto_token_ids = [m.yes_token_id for m, _, _ in crypto_markets]
                        crypto_no_token_ids = [m.no_token_id for m, _, _ in crypto_markets]
                        crypto_books = get_orderbooks(client, crypto_token_ids)
                        crypto_no_books = get_orderbooks(client, crypto_no_token_ids)
                        book_cache.store_books(crypto_books)
                        book_cache.store_books(crypto_no_books)
                        gas_cost = gas_oracle.estimate_cost_usd(1, cfg.gas_per_order)
                        opps = scan_latency_markets(
                            latency_scanner, crypto_markets, crypto_books, gas_cost,
                            no_books=crypto_no_books,
                        )
                return opps, has_momentum

            def _run_spike_scan() -> list[Opportunity]:
                    """Run spike detection scanner. Returns opportunities list."""
                    from scanner.spike import scan_spike_opportunities
                    opps: list[Opportunity] = []
                    if shutdown_requested:
                        return opps
                    # Feed current midpoints into spike detector
                    for m in all_markets:
                        if shutdown_requested:
                            return opps
                        book = book_cache.get_book(m.yes_token_id)
                        if book and book.midpoint is not None:
                            spike_detector.register_token(m.yes_token_id, m.event_id)
                            spike_detector.update(m.yes_token_id, book.midpoint)
                    spikes = spike_detector.detect_spikes()
                    if spikes:
                        logger.debug("      Detected %d price spikes, checking siblings...", len(spikes))
                        # Build event lookup
                        event_map = {e.event_id: e for e in events}
                        gas_cost = gas_oracle.estimate_cost_usd(1, cfg.gas_per_order)
                        for spike in spikes:
                            ev = event_map.get(spike.event_id)
                            if ev:
                                opps.extend(scan_spike_opportunities(
                                    spike, ev, book_cache, fee_model,
                                    gas_cost_usd=gas_cost,
                                    min_profit_usd=scan_params.min_profit_usd,
                                ))
                    return opps

            def _run_cross_platform_scan() -> list[Opportunity]:
                """Run cross-platform arbitrage scanner. Returns opportunities list."""
                from scanner.cross_platform import scan_cross_platform

                opps: list[Opportunity] = []
                if shutdown_requested:
                    return opps
                if not (cfg.cross_platform_enabled and platform_clients and event_matcher):
                    return opps

                if args.dry_run and cfg.cross_platform_skip_without_tradeable_map_in_dry_run:
                    if not event_matcher.has_tradeable_mappings(list(platform_clients.keys())):
                        logger.debug(
                            "      Skipping cross-platform scan in dry-run (no manual/verified mappings)"
                        )
                        return opps

                logger.debug("      Scanning cross-platform arbitrage (%s)...", ", ".join(platform_clients.keys()))

                # Fetch markets from all platforms (use cache for Kalshi)
                all_platform_markets: dict[str, list] = {}
                for pname, pclient in platform_clients.items():
                    if shutdown_requested:
                        return opps
                    if pname == "kalshi" and kalshi_market_cache is not None:
                        snap = kalshi_market_cache.snapshot()
                        if snap is not None:
                            all_platform_markets[pname] = list(snap.markets)
                            logger.debug(
                                "      Using cached %d %s markets (v%d, %.0fs old)",
                                len(snap.markets),
                                pname,
                                snap.version,
                                time.time() - snap.timestamp,
                            )
                        else:
                            logger.warning("      Kalshi market cache not ready, skipping")
                    else:
                        try:
                            mkts = pclient.get_all_markets(status="open")
                            all_platform_markets[pname] = mkts
                            logger.debug("      Fetched %d active %s markets", len(mkts), pname)
                        except NotImplementedError:
                            logger.debug("      Skipping %s (API not yet available)", pname)
                        except Exception as e:
                            logger.warning("      Failed to fetch %s markets: %s", pname, e)

                if not all_platform_markets:
                    return opps

                include_fuzzy = cfg.cross_platform_allow_unverified_fuzzy and not args.dry_run
                matched_events = event_matcher.match_events(
                    events,
                    all_platform_markets,
                    include_fuzzy=include_fuzzy,
                    should_stop=lambda: shutdown_requested,
                )
                if shutdown_requested:
                    return opps

                tradeable_events: list[MatchedEvent] = []
                dropped_matches = 0
                for match in matched_events:
                    tradeable_platform_matches = tuple(
                        pm for pm in match.platform_matches
                        if pm.confidence >= cfg.cross_platform_min_confidence
                    )
                    dropped_matches += len(match.platform_matches) - len(tradeable_platform_matches)
                    if tradeable_platform_matches:
                        tradeable_events.append(
                            MatchedEvent(
                                pm_event_id=match.pm_event_id,
                                pm_markets=match.pm_markets,
                                platform_matches=tradeable_platform_matches,
                            )
                        )

                logger.debug(
                    "      Matched %d events across platforms (%d tradeable, dropped=%d)",
                    len(matched_events),
                    len(tradeable_events),
                    dropped_matches,
                )
                if not tradeable_events:
                    return opps

                # Collect all token IDs needed from PM and each platform
                pm_token_ids: list[str] = []
                platform_tickers: dict[str, list[str]] = {}
                for match in tradeable_events:
                    for pm_mkt in match.pm_markets:
                        pm_token_ids.append(pm_mkt.yes_token_id)
                        pm_token_ids.append(pm_mkt.no_token_id)
                    for pm in match.platform_matches:
                        platform_tickers.setdefault(pm.platform, []).extend(pm.tickers)
                for platform, tickers in platform_tickers.items():
                    platform_tickers[platform] = list(dict.fromkeys(tickers))

                if not platform_tickers:
                    return opps

                # Fetch orderbooks from PM and each platform in parallel
                pm_cross_books = poly_book_fetcher(pm_token_ids) if pm_token_ids else {}
                book_cache.store_books(pm_cross_books)

                all_platform_books: dict[str, dict] = {}
                with ThreadPoolExecutor(max_workers=len(platform_tickers) or 1) as executor:
                    def _fetch_platform_books(pname: str) -> tuple[str, dict] | None:
                        if shutdown_requested:
                            return None
                        pclient = platform_clients.get(pname)
                        tickers = platform_tickers.get(pname, [])
                        if pclient and tickers:
                            try:
                                try:
                                    return pname, pclient.get_orderbooks(tickers, max_workers=4)
                                except TypeError:
                                    # Some platform clients may not expose max_workers.
                                    return pname, pclient.get_orderbooks(tickers)
                            except NotImplementedError:
                                logger.debug("      Skipping %s orderbooks (API not available)", pname)
                            except Exception as e:
                                logger.warning("      Failed to fetch %s orderbooks: %s", pname, e)
                        return None

                    futures = {
                        executor.submit(_fetch_platform_books, pname): pname
                        for pname in platform_tickers.keys()
                    }
                    for future in as_completed(futures):
                        if shutdown_requested:
                            break
                        result = future.result()
                        if result:
                            pname, books = result
                            all_platform_books[pname] = books

                if shutdown_requested:
                    return opps

                opps = scan_cross_platform(
                    tradeable_events,
                    pm_cross_books,
                    platform_books=all_platform_books,
                    platform_markets=all_platform_markets,
                    min_profit_usd=scan_params.min_profit_usd,
                    min_roi_pct=scan_params.min_roi_pct,
                    gas_per_order=cfg.gas_per_order,
                    gas_oracle=gas_oracle,
                    pm_fee_model=fee_model,
                    platform_fee_models=platform_fee_models,
                    min_confidence=cfg.cross_platform_min_confidence,
                    should_stop=lambda: shutdown_requested,
                )
                if opps:
                    logger.debug("      Found %d cross-platform opportunities", len(opps))
                return opps

            def _run_value_scan() -> list[Opportunity]:
                """Run partial negrisk value scanner."""
                from scanner.value import scan_value_opportunities
                if shutdown_requested:
                    return []
                if not cfg.value_scanner_enabled:
                    return []
                logger.debug("      Scanning negrisk events for value opportunities...")
                return scan_value_opportunities(
                    poly_book_fetcher, negrisk_events,
                    scan_params.min_profit_usd, scan_params.min_roi_pct,
                    cfg.gas_per_order, cfg.gas_price_gwei,
                    gas_oracle=gas_oracle, fee_model=fee_model, book_cache=book_cache,
                    min_volume=cfg.min_volume_filter,
                    min_edge_pct=cfg.value_min_edge_pct,
                )

            def _run_maker_scan() -> list[Opportunity]:
                """Run maker spread capture scanner on binary markets."""
                from scanner.maker import scan_maker_opportunities
                if shutdown_requested:
                    return []
                if not scan_params.binary_enabled:
                    return []
                logger.debug("      Scanning %s binary markets for maker spreads...", f"{len(binary_markets):,}")
                # Use cached books where available
                token_ids = []
                for m in binary_markets:
                    if not m.neg_risk:
                        token_ids.append(m.yes_token_id)
                        token_ids.append(m.no_token_id)
                books = poly_book_fetcher(token_ids) if token_ids else {}
                gas_cost = gas_oracle.estimate_cost_usd(1, cfg.gas_per_order) if gas_oracle else 0.005
                return scan_maker_opportunities(
                    binary_markets, books,
                    fee_model=fee_model,
                    min_edge_usd=scan_params.min_profit_usd,
                    gas_cost_per_order=gas_cost,
                    min_leg_price=cfg.maker_min_leg_price,
                    min_depth_sets=cfg.maker_min_depth_sets,
                    min_volume=cfg.maker_min_volume,
                    max_taker_cost=cfg.maker_max_taker_cost,
                    max_spread_ticks=cfg.maker_max_spread_ticks,
                    persistence_gate=maker_persistence_gate,
                    execution_model=maker_execution_model,
                    min_pair_fill_prob=cfg.maker_min_pair_fill_prob,
                    max_toxicity_score=cfg.maker_max_toxicity_score,
                    min_expected_ev_usd=cfg.maker_min_expected_ev_usd,
                )

            def _run_resolution_scan() -> list[Opportunity]:
                """Run resolution sniping scanner."""
                from scanner.resolution import scan_resolution_opportunities
                from scanner.outcome_oracle import OutcomeOracle
                if shutdown_requested:
                    return []
                if not cfg.resolution_sniping_enabled:
                    return []
                logger.debug("      Scanning for resolution sniping opportunities...")
                oracle = OutcomeOracle(allow_network=cfg.allow_non_polymarket_apis)
                # Fetch books for all binary markets
                token_ids = []
                for m in binary_markets:
                    token_ids.append(m.yes_token_id)
                    token_ids.append(m.no_token_id)
                books = poly_book_fetcher(token_ids) if token_ids else {}
                gas_cost = gas_oracle.estimate_cost_usd(1, cfg.gas_per_order) if gas_oracle else 0.005
                return scan_resolution_opportunities(
                    binary_markets, books, oracle.check_outcome,
                    fee_model=fee_model,
                    max_minutes_to_resolution=cfg.resolution_max_minutes,
                    min_edge_pct=cfg.resolution_min_edge_pct,
                    gas_cost_per_order=gas_cost,
                )

            def _run_correlation_scan() -> list[Opportunity]:
                """Run correlation scanner for cross-event probability violations."""
                if shutdown_requested:
                    return []
                if not correlation_scanner:
                    return []
                logger.debug("      Scanning %d events for correlation violations...", len(events))
                # Collect token IDs for all event markets
                corr_token_ids = []
                for e in events:
                    if shutdown_requested:
                        return []
                    for m in e.markets:
                        if m.active:
                            corr_token_ids.append(m.yes_token_id)
                            corr_token_ids.append(m.no_token_id)
                corr_books = poly_book_fetcher(corr_token_ids) if corr_token_ids else {}
                gas_cost = gas_oracle.estimate_cost_usd(1, cfg.gas_per_order) if gas_oracle else 0.005
                return correlation_scanner.scan(events, corr_books, gas_cost_usd=gas_cost)

            # Run all scanners in parallel
            binary_opps: list[Opportunity] = []
            negrisk_opps: list[Opportunity] = []
            latency_opps: list[Opportunity] = []
            spike_opps: list[Opportunity] = []
            cross_platform_opps: list[Opportunity] = []
            value_opps: list[Opportunity] = []
            maker_opps: list[Opportunity] = []
            resolution_opps: list[Opportunity] = []
            correlation_opps: list[Opportunity] = []
            has_crypto_momentum = False

            # Build list of enabled scanners
            scanner_futures = {}
            scanner_started: dict[object, float] = {}
            scanner_elapsed: dict[str, float] = {}
            with ThreadPoolExecutor(max_workers=10) as executor:
                # Submit enabled scanners
                if scan_params.binary_enabled and not skip_full_scan:
                    fut = executor.submit(_run_binary_scan)
                    scanner_futures[fut] = "binary"
                    scanner_started[fut] = time.monotonic()
                if scan_params.negrisk_enabled and not skip_full_scan:
                    fut = executor.submit(_run_negrisk_scan)
                    scanner_futures[fut] = "negrisk"
                    scanner_started[fut] = time.monotonic()
                if scan_params.latency_enabled and not skip_full_scan:
                    fut = executor.submit(_run_latency_scan)
                    scanner_futures[fut] = "latency"
                    scanner_started[fut] = time.monotonic()
                if scan_params.spike_enabled and not skip_full_scan:
                    fut = executor.submit(_run_spike_scan)
                    scanner_futures[fut] = "spike"
                    scanner_started[fut] = time.monotonic()
                if cfg.cross_platform_enabled and platform_clients and event_matcher and not skip_full_scan:
                    fut = executor.submit(_run_cross_platform_scan)
                    scanner_futures[fut] = "cross_platform"
                    scanner_started[fut] = time.monotonic()
                if cfg.value_scanner_enabled and not skip_full_scan:
                    fut = executor.submit(_run_value_scan)
                    scanner_futures[fut] = "value"
                    scanner_started[fut] = time.monotonic()
                if scan_params.binary_enabled and not skip_full_scan:
                    fut = executor.submit(_run_maker_scan)
                    scanner_futures[fut] = "maker"
                    scanner_started[fut] = time.monotonic()
                if cfg.resolution_sniping_enabled and not skip_full_scan:
                    fut = executor.submit(_run_resolution_scan)
                    scanner_futures[fut] = "resolution"
                    scanner_started[fut] = time.monotonic()
                if cfg.correlation_scanner_enabled and correlation_scanner and not skip_full_scan:
                    fut = executor.submit(_run_correlation_scan)
                    scanner_futures[fut] = "correlation"
                    scanner_started[fut] = time.monotonic()

                # Collect results as they complete
                for future in as_completed(scanner_futures):
                    scanner_type = scanner_futures[future]
                    try:
                        result = future.result()
                        started = scanner_started.get(future)
                        if started is not None:
                            scanner_elapsed[scanner_type] = time.monotonic() - started
                        if scanner_type == "binary":
                            binary_opps = result
                            logger.debug("      binary scanner completed in %.2fs (%d opps)", scanner_elapsed.get(scanner_type, 0.0), len(binary_opps))
                        elif scanner_type == "negrisk":
                            negrisk_opps = result
                            logger.debug("      negrisk scanner completed in %.2fs (%d opps)", scanner_elapsed.get(scanner_type, 0.0), len(negrisk_opps))
                        elif scanner_type == "latency":
                            latency_opps, has_crypto_momentum = result
                            logger.debug("      latency scanner completed in %.2fs (%d opps)", scanner_elapsed.get(scanner_type, 0.0), len(latency_opps))
                        elif scanner_type == "spike":
                            spike_opps = result
                            logger.debug("      spike scanner completed in %.2fs (%d opps)", scanner_elapsed.get(scanner_type, 0.0), len(spike_opps))
                        elif scanner_type == "cross_platform":
                            cross_platform_opps = result
                            logger.debug("      cross-platform scanner completed in %.2fs (%d opps)", scanner_elapsed.get(scanner_type, 0.0), len(cross_platform_opps))
                        elif scanner_type == "value":
                            value_opps = result
                            logger.debug("      value scanner completed in %.2fs (%d opps)", scanner_elapsed.get(scanner_type, 0.0), len(value_opps))
                        elif scanner_type == "maker":
                            maker_opps = result
                            logger.debug("      maker scanner completed in %.2fs (%d opps)", scanner_elapsed.get(scanner_type, 0.0), len(maker_opps))
                        elif scanner_type == "resolution":
                            resolution_opps = result
                            logger.debug("      resolution scanner completed in %.2fs (%d opps)", scanner_elapsed.get(scanner_type, 0.0), len(resolution_opps))
                        elif scanner_type == "correlation":
                            correlation_opps = result
                            logger.debug("      correlation scanner completed in %.2fs (%d opps)", scanner_elapsed.get(scanner_type, 0.0), len(correlation_opps))
                    except Exception as e:
                        logger.error("      %s scanner failed: %s", scanner_type.replace("_", " ").capitalize(), e)

            if scanner_elapsed:
                telemetry = ", ".join(
                    f"{name}={elapsed:.2f}s"
                    for name, elapsed in sorted(scanner_elapsed.items())
                )
                logger.debug("      Scanner telemetry: %s", telemetry)

            if cfg.correlation_max_opps_per_cycle > 0 and len(correlation_opps) > cfg.correlation_max_opps_per_cycle:
                before = len(correlation_opps)
                cap = cfg.correlation_max_opps_per_cycle
                min_buy = min(max(0, cfg.correlation_cap_min_buy_opps_per_cycle), cap)
                max_buy_per_event = max(1, cfg.correlation_cap_max_buy_per_event)
                buy_corr = [o for o in correlation_opps if o.is_buy_arb]
                selected: list[Opportunity] = []
                buy_event_counts: Counter[str] = Counter()
                if min_buy > 0 and buy_corr:
                    buy_sorted = sorted(buy_corr, key=lambda o: o.net_profit, reverse=True)
                    for opp in buy_sorted:
                        if buy_event_counts[opp.event_id] >= max_buy_per_event:
                            continue
                        selected.append(opp)
                        buy_event_counts[opp.event_id] += 1
                        if len(selected) >= min_buy:
                            break
                selected_ids = {id(o) for o in selected}
                remaining_pool = sorted(
                    (o for o in correlation_opps if id(o) not in selected_ids),
                    key=lambda o: o.net_profit,
                    reverse=True,
                )
                for opp in remaining_pool:
                    if len(selected) >= cap:
                        break
                    if opp.is_buy_arb and buy_event_counts[opp.event_id] >= max_buy_per_event:
                        continue
                    selected.append(opp)
                    if opp.is_buy_arb:
                        buy_event_counts[opp.event_id] += 1
                correlation_opps = selected
                kept_buy = sum(1 for o in correlation_opps if o.is_buy_arb)
                logger.debug(
                    "      Correlation cap applied: kept %d/%d opportunities (buy_kept=%d buy_total=%d min_buy=%d max_buy_per_event=%d)",
                    len(correlation_opps),
                    before,
                    kept_buy,
                    len(buy_corr),
                    min_buy,
                    max_buy_per_event,
                )

            # Stale-quote sniping: check book cache for recent price moves
            stale_quote_opps: list[Opportunity] = []
            if cfg.stale_quote_enabled and book_cache and not args.dry_run and not skip_full_scan:
                try:
                    from scanner.stale_quote import StaleQuoteDetector
                    if not hasattr(main, '_stale_detector'):
                        main._stale_detector = StaleQuoteDetector(
                            min_move_pct=cfg.stale_quote_min_move_pct,
                            max_staleness_ms=cfg.stale_quote_max_staleness_ms,
                            cooldown_sec=cfg.stale_quote_cooldown_sec,
                        )
                    # Build token→Market lookup for stale-quote signal generation
                    token_market_map: dict[str, object] = {}
                    for m in all_markets:
                        token_market_map[m.yes_token_id] = m
                        token_market_map[m.no_token_id] = m

                    # Feed recent book cache updates to the stale detector
                    cached_books = book_cache.get_all_books()
                    for token_id, book in cached_books.items():
                        if book.best_ask:
                            token_market = token_market_map.get(token_id)
                            stale_signal = main._stale_detector.on_price_update(
                                token_id, book.best_ask.price, time.time(),
                                market=token_market,
                            )
                            if stale_signal:
                                opp = main._stale_detector.check_complementary_book(
                                    stale_signal,
                                    cached_books,
                                    fee_model=fee_model,
                                    gas_per_order=cfg.gas_per_order,
                                    gas_price_gwei=cfg.gas_price_gwei,
                                    min_profit_usd=scan_params.min_profit_usd,
                                    min_roi_pct=scan_params.min_roi_pct,
                                )
                                if opp:
                                    stale_quote_opps.append(opp)
                except Exception as e:
                    logger.debug("      Stale-quote scan error: %s", e)

            all_opps = binary_opps + negrisk_opps + latency_opps + spike_opps + cross_platform_opps + value_opps + stale_quote_opps + maker_opps + resolution_opps + correlation_opps

            # Defense-in-depth: drop ALL negrisk_value opps when value scanner is disabled.
            # The value scanner assumes uniform 1/N probability, producing 100% false
            # positives on markets with known favorites. When disabled, any negrisk_value
            # opps that leak through are phantom arbs.
            if not cfg.value_scanner_enabled:
                nv_count = sum(1 for o in all_opps if o.type == OpportunityType.NEGRISK_VALUE)
                if nv_count > 0:
                    all_opps = [o for o in all_opps if o.type != OpportunityType.NEGRISK_VALUE]
                    logger.warning("Filtered %d negrisk_value opps (value scanner disabled)", nv_count)

            # Update realized-EV candidate history before scoring.
            maker_ev_tracker.observe_candidates(all_opps)

            # Record opportunities in ArbTracker for persistence tracking
            arb_tracker.record(cycle, all_opps)

            # In scan-only mode, filter out opportunities that safety would block
            if args.scan_only:
                pre_filter_count = len(all_opps)
                all_opps = [o for o in all_opps if len(o.legs) <= cfg.max_legs_per_opportunity]
                filtered_count = pre_filter_count - len(all_opps)
                if filtered_count > 0:
                    logger.debug(
                        "      Filtered %d opportunities with >%d legs",
                        filtered_count, cfg.max_legs_per_opportunity,
                    )

            # Build real ScoringContext per opportunity with ArbTracker confidence.
            # Inventory here reflects existing holdings only (not whether any BUY
            # opportunities are present in this cycle).
            has_inventory = pnl.current_exposure > 0
            contexts = _build_scoring_contexts(
                all_opps, book_cache, all_markets, cfg.target_size_usd,
                arb_tracker=arb_tracker,
                has_inventory=has_inventory,
                realized_ev_tracker=maker_ev_tracker,
                ofi_tracker=ofi_tracker,
            )
            # Composite scoring with real context data
            scored_opps = rank_opportunities(all_opps, contexts=contexts)
            if ml_scorer is not None:
                scored_opps = _rerank_with_ml(
                    scored_opps=scored_opps,
                    opportunities=all_opps,
                    contexts=contexts,
                    ml_scorer=ml_scorer,
                    blend_weight=cfg.ml_scorer_blend_weight,
                )

            ctx_by_opp_id = {id(opp): ctx for opp, ctx in zip(all_opps, contexts)}
            context_by_scored_id: dict[int, ScoringContext] = {}
            for scored in scored_opps:
                ctx = ctx_by_opp_id.get(id(scored.opportunity))
                if ctx is not None:
                    context_by_scored_id[id(scored)] = ctx

            recorder.record_cycle(
                cycle=cycle,
                books=book_cache.get_all_books(),
                opportunities=all_opps,
                scoring_contexts=contexts,
                strategy_mode=scan_params.mode.value,
                config=cfg.model_dump() if cycle == 1 else None,
            )

            research_scored: list[ScoredOpportunity] = []
            executable_scored: list[ScoredOpportunity] = []
            for scored in scored_opps:
                if _is_research_opportunity(scored.opportunity, cfg):
                    research_scored.append(scored)
                else:
                    executable_scored.append(scored)

            actionable_now_scored: list[ScoredOpportunity] = []
            actionable_reject_reasons: Counter[str] = Counter()
            for scored in executable_scored:
                reject_reason = _executable_reject_reason(scored, cfg)
                if reject_reason is None:
                    actionable_now_scored.append(scored)
                else:
                    actionable_reject_reasons[reject_reason] += 1
            if actionable_reject_reasons:
                reject_detail = ", ".join(f"{k}={v}" for k, v in sorted(actionable_reject_reasons.items()))
                logger.debug("      Actionable gate rejects: %s", reject_detail)
            maker_candidate_scored = [
                s for s in executable_scored if s.opportunity.type == OpportunityType.MAKER_REBALANCE
            ]
            if executable_scored:
                executable_zero_streak = 0
            else:
                executable_zero_streak += 1
                warn_every = cfg.executable_lane_zero_streak_warn_cycles
                if executable_zero_streak == warn_every or executable_zero_streak % warn_every == 0:
                    logger.warning(
                        "Executable lane empty for %d consecutive cycles (research=%d total=%d).",
                        executable_zero_streak,
                        len(research_scored),
                        len(scored_opps),
                    )
            total_opps_found += len(all_opps)
            collector.record_funnel("opps_found", len(all_opps))
            collector.record_funnel("opps_scored", len(scored_opps))
            collector.record_funnel("opps_executable_lane", len(executable_scored))
            collector.record_funnel("opps_research_lane", len(research_scored))
            collector.record_funnel("opps_actionable_now", len(actionable_now_scored))
            collector.record_funnel("maker_candidates", len(maker_candidate_scored))
            collector.record_scored_opps(
                scored_opps,
                contexts,
                event_questions=event_questions,
                market_questions=market_questions,
            )

            scan_elapsed = time.time() - scan_start

            scanner_counts = {
                "binary": len(binary_opps),
                "negrisk": len(negrisk_opps),
                "latency": len(latency_opps),
                "spike": len(spike_opps),
                "cross_platform": len(cross_platform_opps),
                "value": len(value_opps),
                "stale_quote": len(stale_quote_opps),
                "maker": len(maker_opps),
                "resolution": len(resolution_opps),
                "correlation": len(correlation_opps),
            }
            collector.record_scanner_counts(scanner_counts)

            # Track best-ever stats for scan-only footer
            for opp in all_opps:
                if opp.net_profit > best_profit_ever:
                    best_profit_ever = opp.net_profit
                if opp.roi_pct > best_roi_ever:
                    best_roi_ever = opp.roi_pct

            print_scan_result(
                scored_opps=scored_opps,
                event_questions=event_questions,
                market_questions=market_questions,
                scanner_counts=scanner_counts,
                scan_elapsed=scan_elapsed,
                fetch_elapsed=fetch_elapsed,
                markets_count=len(all_markets),
                binary_count=len(binary_markets),
                negrisk_event_count=len(negrisk_events),
                negrisk_market_count=sum(len(e.markets) for e in negrisk_events),
                strategy_name=scan_params.mode.value,
                actionable_now_count=len(actionable_now_scored),
                maker_candidate_count=len(maker_candidate_scored),
                executable_lane_count=len(executable_scored),
                research_lane_count=len(research_scored),
            )
            if args.dry_run:
                logger.debug("      OFI inactive in dry-run (WS disabled)")
            else:
                logger.debug(
                    "      OFI quality corr=%.3f tracked_tokens=%d",
                    ofi_tracker.quality_correlation,
                    ofi_tracker.tracked_tokens,
                )

            if args.scan_only:
                tracker.record_cycle(
                    cycle,
                    len(all_markets),
                    all_opps,
                    executable_opps=[s.opportunity for s in executable_scored],
                    research_opps=[s.opportunity for s in research_scored],
                    actionable_now=[s.opportunity for s in actionable_now_scored],
                    maker_candidates=[s.opportunity for s in maker_candidate_scored],
                )
                status_writer.write_cycle(
                    cycle=cycle,
                    mode=_mode_label(args, cfg),
                    markets_scanned=len(all_markets),
                    scored_opps=scored_opps,
                    event_questions=event_questions,
                    market_questions=market_questions,
                    total_opps_found=total_opps_found,
                    total_trades_executed=0,
                    total_pnl=0.0,
                    current_exposure=0.0,
                    scan_only=True,
                    executable_lane_count=len(executable_scored),
                    research_lane_count=len(research_scored),
                    executable_lane_profit=sum(s.opportunity.net_profit for s in executable_scored),
                    research_lane_profit=sum(s.opportunity.net_profit for s in research_scored),
                )
                collector.record_funnel("executed", 0)
                collector.end_cycle()
                rate_limit_streak = 0
                print_cycle_footer(
                    cycle=cycle,
                    cycle_elapsed=time.time() - cycle_start,
                    total_opps=total_opps_found,
                    total_trades=0,
                    total_pnl=0.0,
                    best_profit=best_profit_ever,
                    best_roi=best_roi_ever,
                    scan_only=True,
                )
                _sleep_remaining(cycle_start, cfg.scan_interval_sec, shutdown_requested)
                continue

            # Step 3: Execute supported opportunities (ordered by composite score)
            unsupported_for_execution = [
                s for s in executable_scored if not _is_execution_supported_type(s.opportunity.type)
            ]
            if unsupported_for_execution:
                unsupported_counts = Counter(s.opportunity.type.value for s in unsupported_for_execution)
                details = ", ".join(f"{k}={v}" for k, v in unsupported_counts.items())
                logger.info(
                    "      Deferring %d non-taker opportunities this cycle: %s",
                    len(unsupported_for_execution),
                    details,
                )

            execution_queue = [
                s for s in executable_scored if _is_execution_supported_type(s.opportunity.type)
            ]
            if presigner is not None and execution_queue:
                prewarm_count = 0
                for scored in execution_queue[: cfg.presigner_prewarm_top_n]:
                    opp = scored.opportunity
                    prewarm_size = max(1.0, min(opp.max_sets, cfg.target_size_usd))
                    for leg in opp.legs:
                        if leg.side == Side.BUY:
                            neg_risk = opp.type == OpportunityType.NEGRISK_REBALANCE
                            prewarm_count += presigner.presign_levels(
                                token_id=leg.token_id,
                                best_price=leg.price,
                                size=prewarm_size,
                                neg_risk=neg_risk,
                                tick_size=leg.tick_size,
                            )
                logger.debug("      Presigner prewarmed %d templates", prewarm_count)

            logger.info("[3/3] Executing %d opportunities...", len(execution_queue))
            executed_this_cycle = 0
            for i, scored in enumerate(execution_queue, 1):
                if shutdown_requested:
                    break
                opp = scored.opportunity

                try:
                    # Pre-execution confidence gate
                    opp_confidence = scored.persistence_score
                    verify_min_confidence(opp_confidence, cfg.min_confidence_gate, opp.event_id)

                    logger.info(
                        "      [%d/%d] Executing %s  event=%s  score=%.2f  profit=$%.2f...",
                        i, len(execution_queue), opp.type.value, opp.event_id[:14], scored.total_score, opp.net_profit,
                    )
                    result = _execute_single(
                        client,
                        cfg,
                        opp,
                        pnl,
                        breaker,
                        gas_oracle=gas_oracle,
                        position_tracker=position_tracker,
                        platform_clients=platform_clients,
                        presigner=presigner,
                    )
                    if result is not None:
                        collector.record_trade(result, scored.total_score)
                        if ml_scorer is not None:
                            ctx = context_by_scored_id.get(id(scored))
                            if ctx is not None:
                                ml_scorer.add_sample(
                                    opp,
                                    ctx,
                                    profitable=(result.net_pnl > 0 and result.fully_filled),
                                )
                        total_trades_executed += 1
                        executed_this_cycle += 1
                    logger.info(
                        "      [%d/%d] Trade complete  |  session P&L: $%.2f (%d trades)",
                        i, len(execution_queue), pnl.total_pnl, pnl.total_trades,
                    )
                except SafetyCheckFailed as e:
                    logger.warning("      [%d/%d] Safety check failed, skipping: %s", i, len(execution_queue), e)
                    collector.record_safety_rejection(opp, _extract_safety_check_name(str(e)), str(e))
                    arb_tracker.record_failure(opp.event_id)
                    continue
                except UnwindFailed as e:
                    logger.error("      [%d/%d] UNWIND FAILED -- stuck positions: %s", i, len(execution_queue), e)
                    breaker.record_trade(-opp.net_profit)  # count as a loss
                    continue
                except CrossPlatformUnwindFailed as e:
                    logger.error("      [%d/%d] CROSS-PLATFORM UNWIND FAILED -- stuck Kalshi positions: %s", i, len(execution_queue), e)
                    breaker.record_trade(-opp.net_profit)
                    continue
                except CircuitBreakerTripped as e:
                    logger.error("CIRCUIT BREAKER TRIPPED: %s", e)
                    shutdown_requested = True
                    break

            if ml_scorer is not None and ml_scorer.maybe_retrain():
                try:
                    ml_scorer.save(cfg.ml_scorer_model_path)
                except Exception as e:
                    logger.warning("Failed to persist ML scorer model: %s", e)

            maker_results = _run_maker_lifecycle_cycle(
                client=client,
                cfg=cfg,
                maker_candidates=maker_candidate_scored,
                maker_lifecycle=maker_lifecycle,
                maker_pairs=maker_pairs,
                maker_order_to_pair=maker_order_to_pair,
                pnl=pnl,
                breaker=breaker,
                book_cache=book_cache,
                realized_ev_tracker=maker_ev_tracker,
            )
            if maker_results:
                for result, score in maker_results:
                    collector.record_trade(result, score)
                total_trades_executed += len(maker_results)
                executed_this_cycle += len(maker_results)

            # Write status after execution
            status_writer.write_cycle(
                cycle=cycle,
                mode=_mode_label(args, cfg),
                markets_scanned=len(all_markets),
                scored_opps=scored_opps,
                event_questions=event_questions,
                market_questions=market_questions,
                total_opps_found=total_opps_found,
                total_trades_executed=total_trades_executed,
                total_pnl=pnl.total_pnl,
                current_exposure=pnl.current_exposure,
                scan_only=False,
                executable_lane_count=len(executable_scored),
                research_lane_count=len(research_scored),
                executable_lane_profit=sum(s.opportunity.net_profit for s in executable_scored),
                research_lane_profit=sum(s.opportunity.net_profit for s in research_scored),
            )
            if presigner is not None:
                stats = presigner.stats
                logger.debug(
                    "      Presigner stats: hit_rate=%.1f%% hits=%d misses=%d cache=%d/%d",
                    100.0 * float(stats.get("hit_rate", 0.0)),
                    int(stats.get("hits", 0)),
                    int(stats.get("misses", 0)),
                    int(stats.get("cache_size", 0)),
                    int(stats.get("max_size", 0)),
                )
            collector.record_funnel("executed", executed_this_cycle)
            collector.end_cycle()
            rate_limit_streak = 0

        except KeyboardInterrupt:
            break
        except Exception as e:
            # Full traceback goes to the debug log file; console gets clean summary
            is_api_error = "Request exception" in str(e) or "PolyApiException" in type(e).__name__
            if is_api_error:
                logger.debug("Cycle %d failed: %s", cycle, e, exc_info=True)
            else:
                logger.error("Cycle %d failed: %s", cycle, e, exc_info=True)
            print_cycle_error(e)
            # Keep scan-only summary cycle count accurate even when a cycle errors.
            if args.scan_only and tracker.total_cycles < cycle:
                tracker.record_cycle(cycle, 0, [])
            if _is_rate_limit_error(str(e)):
                rate_limit_streak += 1
                backoff_sec = min(20.0, 2.0 * (2 ** (rate_limit_streak - 1)))
                logger.warning(
                    "Rate limit detected; cooling down %.1fs before next cycle (streak=%d).",
                    backoff_sec, rate_limit_streak,
                )
                time.sleep(backoff_sec)
            else:
                rate_limit_streak = 0
            collector.end_cycle()  # flush whatever telemetry we captured before the error

        print_cycle_footer(
            cycle=cycle,
            cycle_elapsed=time.time() - cycle_start,
            total_opps=total_opps_found,
            total_trades=total_trades_executed if not args.scan_only else 0,
            total_pnl=pnl.total_pnl if not args.scan_only else 0.0,
            best_profit=best_profit_ever if args.scan_only else 0.0,
            best_roi=best_roi_ever if args.scan_only else 0.0,
            scan_only=args.scan_only,
        )
        # Checkpoint: auto-save trackers every N cycles
        if checkpoint_mgr is not None:
            checkpoint_mgr.tick()

        _sleep_remaining(cycle_start, cfg.scan_interval_sec, shutdown_requested)

    # Shutdown
    logger.info("")
    logger.info("Shutting down gracefully after %s (%d cycles)",
                _format_duration(time.time() - session_start), cycle)

    # Save all tracker state before exiting
    if checkpoint_mgr is not None:
        saved = checkpoint_mgr.save_all(cycle_num=cycle)
        logger.info("Checkpoint: saved %d tracker(s) on shutdown (cycle %d)", saved, cycle)
        checkpoint_mgr.close()

    if kalshi_market_cache is not None:
        kalshi_market_cache.stop()
    if ws_bridge:
        ws_bridge.stop()
    for pclient in platform_clients.values():
        if hasattr(pclient, "close"):
            pclient.close()
    if not args.scan_only and not args.dry_run and not cfg.paper_trading:
        maker_lifecycle.cancel_all(lambda order_id: _safe_cancel_order(client, order_id))
    _cancel_open_orders_on_shutdown(client, args, cfg)
    if ml_scorer is not None:
        try:
            ml_scorer.save(cfg.ml_scorer_model_path)
        except Exception as e:
            logger.warning("Failed to save ML scorer on shutdown: %s", e)
    recorder_stats = recorder.stats
    recorder.close()
    if recorder_stats.get("enabled"):
        logger.info(
            "Recorder: wrote %d cycles to %s",
            recorder_stats.get("cycles_recorded", 0),
            recorder_stats.get("current_file", ""),
        )
    collector.end_session()

    if args.scan_only:
        _print_scan_summary(tracker)
    else:
        _print_pnl_summary(pnl)


def _build_scoring_contexts(
    opps: list[Opportunity],
    book_cache: BookCache,
    all_markets: list,
    target_size: float,
    arb_tracker: ArbTracker | None = None,
    has_inventory: bool = True,
    realized_ev_tracker: RealizedEVTracker | None = None,
    ofi_tracker: OFITracker | None = None,
) -> list[ScoringContext]:
    """Build real ScoringContext for each opportunity using cached book data."""
    from datetime import datetime, timezone

    # Build market lookup by token_id
    market_by_token: dict[str, object] = {}
    for m in all_markets:
        market_by_token[m.yes_token_id] = m
        market_by_token[m.no_token_id] = m

    contexts: list[ScoringContext] = []
    for opp in opps:
        # Depth ratio: min(available_depth / target_sets) across all legs.
        # Convert target_size (USD) into a target contract count using per-set
        # capital so depth ratio remains unit-consistent.
        min_depth_ratio = float("inf") if opp.legs else 0.0
        if target_size > 0 and opp.max_sets > 0 and opp.required_capital > 0:
            per_set_capital = opp.required_capital / opp.max_sets
            target_sets = (target_size / per_set_capital) if per_set_capital > 0 else opp.max_sets
        else:
            target_sets = opp.max_sets if opp.max_sets > 0 else 1.0

        for leg in opp.legs:
            book = book_cache.get_book(leg.token_id)
            if book:
                available = sweep_depth(book, leg.side, max_price=leg.price * 1.005 if leg.side == Side.BUY else leg.price * 0.995)
                ratio = available / target_sets if target_sets > 0 else 1.0
                min_depth_ratio = min(min_depth_ratio, ratio)
            else:
                min_depth_ratio = 0.0
                break

        if min_depth_ratio == float("inf"):
            min_depth_ratio = 0.0

        # Market volume: use first leg's market volume
        volume = 0.0
        first_market = market_by_token.get(opp.legs[0].token_id) if opp.legs else None
        if first_market and hasattr(first_market, "volume"):
            volume = first_market.volume

        # Time to resolution: from Market.end_date
        time_to_resolution_hours = 720.0  # default 30 days
        if first_market and hasattr(first_market, "end_date") and first_market.end_date:
            try:
                dt_str = first_market.end_date.replace("Z", "+00:00")
                if "T" not in dt_str:
                    dt_str += "T23:59:59+00:00"
                end_dt = datetime.fromisoformat(dt_str)
                now = datetime.now(timezone.utc)
                hours = (end_dt - now).total_seconds() / 3600.0
                if hours > 0:
                    time_to_resolution_hours = hours
            except (ValueError, TypeError):
                pass

        is_spike = opp.type in (OpportunityType.SPIKE_LAG, OpportunityType.LATENCY_ARB)

        # Get confidence from ArbTracker if available
        confidence = 0.5  # default
        if arb_tracker:
            has_inventory_for_opp = has_inventory if opp.is_sell_arb else True
            confidence = arb_tracker.confidence(
                opp.event_id,
                depth_ratio=min_depth_ratio,
                has_inventory=has_inventory_for_opp,
            )
        if opp.type == OpportunityType.MAKER_REBALANCE:
            execution_conf = max(
                0.01,
                opp.pair_fill_prob * (1.0 - 0.60 * opp.toxicity_score),
            )
            confidence = min(confidence, execution_conf)
        realized_ev_score = 0.5
        if realized_ev_tracker:
            realized_ev_score = realized_ev_tracker.score(opp)

        # OFI divergence: max pairwise divergence across legs
        ofi_divergence = 0.0
        if ofi_tracker and len(opp.legs) >= 2:
            for i, leg_a in enumerate(opp.legs):
                for leg_b in opp.legs[i + 1:]:
                    div = ofi_tracker.get_divergence(leg_a.token_id, leg_b.token_id)
                    ofi_divergence = max(ofi_divergence, div)

        contexts.append(ScoringContext(
            market_volume=volume,
            recent_trade_count=0,
            time_to_resolution_hours=time_to_resolution_hours,
            is_spike=is_spike,
            book_depth_ratio=min_depth_ratio,
            confidence=confidence,
            realized_ev_score=realized_ev_score,
            ofi_divergence=ofi_divergence,
        ))

    return contexts


def _rerank_with_ml(
    *,
    scored_opps: list[ScoredOpportunity],
    opportunities: list[Opportunity],
    contexts: list[ScoringContext],
    ml_scorer: MLScorer,
    blend_weight: float,
) -> list[ScoredOpportunity]:
    """
    Blend deterministic score with ML profit probability.
    Keeps deterministic scorer as fallback unless ML model is trained.
    """
    if not scored_opps or not ml_scorer.is_trained or blend_weight <= 0:
        return scored_opps

    ctx_by_opp_id = {id(opp): ctx for opp, ctx in zip(opportunities, contexts)}
    blended: list[ScoredOpportunity] = []
    for scored in scored_opps:
        ctx = ctx_by_opp_id.get(id(scored.opportunity), ScoringContext())
        ml_prob = ml_scorer.predict(scored.opportunity, ctx)
        total = (1.0 - blend_weight) * scored.total_score + blend_weight * ml_prob
        blended.append(replace(scored, total_score=total))

    blended.sort(key=lambda s: s.total_score, reverse=True)
    return blended


def _run_maker_lifecycle_cycle(
    *,
    client,
    cfg: Config,
    maker_candidates: list[ScoredOpportunity],
    maker_lifecycle: MakerLifecycle,
    maker_pairs: dict[str, MakerPairState],
    maker_order_to_pair: dict[str, str],
    pnl: PnLTracker,
    breaker: CircuitBreaker,
    book_cache: BookCache,
    realized_ev_tracker: RealizedEVTracker,
) -> list[tuple[TradeResult, float]]:
    """
    Manage posted maker orders and post new maker candidates.

    Returns:
        List of (TradeResult, score) for realized maker outcomes this cycle.
    """
    results: list[tuple[TradeResult, float]] = []
    now = time.time()

    if cfg.paper_trading:
        # Paper mode: simulate only top maker candidates; no live order lifecycle.
        for scored in maker_candidates[: min(3, cfg.maker_max_active_pairs)]:
            opp = scored.opportunity
            size = _size_maker_opportunity(
                opp=opp,
                cfg=cfg,
                pnl=pnl,
                maker_exposure=0.0,
            )
            if size <= 0:
                continue
            sized_opp = _with_sized_legs(opp, size)
            result = execute_opportunity(
                client,
                sized_opp,
                size,
                paper_trading=True,
                use_fak=False,
                order_timeout_sec=cfg.order_timeout_sec,
            )
            pnl.record(result)
            breaker.record_trade(result.net_pnl)
            realized_ev_tracker.record_full_fill(sized_opp, result.net_pnl)
            results.append((result, scored.total_score))
        return results

    status_cache: dict[str, dict] = {}

    def _status_reader(order_id: str) -> dict | None:
        tracked = maker_lifecycle._orders.get(order_id)
        default_size = tracked.size if tracked is not None else 0.0
        snap = _maker_order_snapshot(client, order_id, default_size=default_size)
        status_cache[order_id] = snap
        return {"filled": snap["filled"], "cancelled": snap["cancelled"]}

    # 1) Update fill statuses.
    filled_orders = maker_lifecycle.check_fills(_status_reader)
    for filled in filled_orders:
        pair_id = maker_order_to_pair.get(filled.order_id)
        if not pair_id:
            continue
        pair = maker_pairs.get(pair_id)
        if not pair or pair.closed:
            continue
        snap = status_cache.get(filled.order_id, {})
        filled_size = _maker_extract_filled_size(snap, default_size=filled.size)
        if filled.order_id == pair.yes_order_id:
            pair.yes_filled_size = max(pair.yes_filled_size, filled_size)
        elif filled.order_id == pair.no_order_id:
            pair.no_filled_size = max(pair.no_filled_size, filled_size)

    # 2) Cancel stale and drifted orders.
    cancel_fn = lambda order_id: _safe_cancel_order(client, order_id)
    maker_lifecycle.cancel_stale(cancel_fn)
    maker_lifecycle.cancel_if_price_moved(
        cancel_fn,
        book_cache.get_all_books(),
        max_drift_ticks=cfg.maker_order_max_drift_ticks,
    )

    active_order_ids = {
        oid for oid, ord_state in maker_lifecycle._orders.items() if ord_state.status == "active"
    }

    # 3) Close completed/failed maker pairs.
    for pair_id, pair in list(maker_pairs.items()):
        if pair.closed:
            continue

        if pair.both_filled:
            result = _build_maker_full_fill_result(pair)
            pnl.record(result)
            breaker.record_trade(result.net_pnl)
            realized_ev_tracker.record_full_fill(pair.opportunity, result.net_pnl)
            results.append((result, pair.score))
            pair.closed = True
            continue

        if pair.one_leg_filled and (now - pair.created_at) >= cfg.maker_hedge_timeout_sec:
            hedge_result = _hedge_orphan_maker_leg(
                client=client,
                pair=pair,
                active_order_ids=active_order_ids,
                book_cache=book_cache,
                gas_cost=max(0.0, pair.opportunity.estimated_gas_cost),
            )
            pnl.record(hedge_result)
            breaker.record_trade(hedge_result.net_pnl)
            realized_ev_tracker.record_orphan_hedge(pair.opportunity, hedge_result.net_pnl)
            results.append((hedge_result, pair.score))
            pair.closed = True
            continue

        # No remaining active orders and no fills => fully cancelled pair.
        if (
            pair.yes_order_id not in active_order_ids
            and pair.no_order_id not in active_order_ids
            and not pair.one_leg_filled
        ):
            pair.closed = True

    for pair_id, pair in list(maker_pairs.items()):
        if not pair.closed:
            continue
        maker_order_to_pair.pop(pair.yes_order_id, None)
        maker_order_to_pair.pop(pair.no_order_id, None)
        del maker_pairs[pair_id]

    maker_lifecycle.prune_filled_and_cancelled()

    # 4) Post new maker pairs.
    active_pair_signatures = {
        _maker_pair_signature(p.opportunity) for p in maker_pairs.values() if p.active
    }
    for scored in maker_candidates:
        if len(maker_pairs) >= cfg.maker_max_active_pairs:
            break
        opp = scored.opportunity
        signature = _maker_pair_signature(opp)
        if signature in active_pair_signatures:
            continue
        if len(opp.legs) != 2:
            continue

        size = _size_maker_opportunity(
            opp=opp,
            cfg=cfg,
            pnl=pnl,
            maker_exposure=maker_lifecycle.active_exposure,
        )
        if size <= 0:
            continue
        sized_opp = _with_sized_legs(opp, size)
        yes_leg, no_leg = sized_opp.legs
        maker_order_type = OrderType.GTD if cfg.maker_use_gtd else OrderType.GTC
        expiration_ts = int(time.time() + cfg.maker_quote_ttl_sec) if cfg.maker_use_gtd else 0

        try:
            signed_yes = create_limit_order(
                client,
                token_id=yes_leg.token_id,
                side=yes_leg.side,
                price=yes_leg.price,
                size=size,
                neg_risk=False,
                tick_size=yes_leg.tick_size,
                expiration=expiration_ts,
            )
            signed_no = create_limit_order(
                client,
                token_id=no_leg.token_id,
                side=no_leg.side,
                price=no_leg.price,
                size=size,
                neg_risk=False,
                tick_size=no_leg.tick_size,
                expiration=expiration_ts,
            )
            post_batch: list[tuple[object, OrderType] | tuple[object, OrderType, bool]] = [
                (signed_yes, maker_order_type, cfg.maker_post_only),
                (signed_no, maker_order_type, cfg.maker_post_only),
            ]
            try:
                responses = post_orders(client, post_batch)
            except Exception:
                # Some venues/API versions can reject GTD if expiration semantics differ.
                # Retry as GTC while preserving post-only and lifecycle TTL safeguards.
                if not cfg.maker_use_gtd:
                    raise
                logger.warning("      Maker GTD post rejected, retrying as GTC for %s", opp.event_id[:14])
                responses = post_orders(
                    client,
                    [
                        (signed_yes, OrderType.GTC, cfg.maker_post_only),
                        (signed_no, OrderType.GTC, cfg.maker_post_only),
                    ],
                )
        except Exception as e:
            logger.warning("      Maker post failed for %s: %s", opp.event_id[:14], e)
            continue

        if len(responses) != 2:
            logger.warning("      Maker pair post returned %d responses, expected 2", len(responses))
            continue

        yes_order_id = responses[0].get("orderID", responses[0].get("order_id", ""))
        no_order_id = responses[1].get("orderID", responses[1].get("order_id", ""))
        if not yes_order_id or not no_order_id:
            logger.warning("      Maker pair missing order IDs, skipping")
            continue

        pair_id = f"{sized_opp.event_id}:{yes_order_id}:{no_order_id}"
        pair = MakerPairState(
            pair_id=pair_id,
            opportunity=sized_opp,
            score=scored.total_score,
            yes_order_id=yes_order_id,
            no_order_id=no_order_id,
            created_at=time.time(),
        )

        yes_snap = _maker_order_snapshot_from_payload(responses[0], default_size=size)
        no_snap = _maker_order_snapshot_from_payload(responses[1], default_size=size)
        if yes_snap["filled"]:
            pair.yes_filled_size = _maker_extract_filled_size(yes_snap, default_size=size)
        if no_snap["filled"]:
            pair.no_filled_size = _maker_extract_filled_size(no_snap, default_size=size)

        maker_pairs[pair_id] = pair
        maker_order_to_pair[yes_order_id] = pair_id
        maker_order_to_pair[no_order_id] = pair_id
        maker_lifecycle.post_order(
            order_id=yes_order_id,
            token_id=yes_leg.token_id,
            side=yes_leg.side,
            price=yes_leg.price,
            size=size,
        )
        maker_lifecycle.post_order(
            order_id=no_order_id,
            token_id=no_leg.token_id,
            side=no_leg.side,
            price=no_leg.price,
            size=size,
        )
        active_pair_signatures.add(signature)
        logger.info(
            "      Maker pair posted event=%s size=%.2f score=%.2f ids=(%s,%s)",
            sized_opp.event_id[:14],
            size,
            scored.total_score,
            yes_order_id[:10],
            no_order_id[:10],
        )

    return results


def _size_maker_opportunity(
    *,
    opp: Opportunity,
    cfg: Config,
    pnl: PnLTracker,
    maker_exposure: float,
) -> float:
    if opp.max_sets <= 0 or opp.required_capital <= 0:
        return 0.0
    size = compute_position_size(
        opp,
        bankroll=cfg.max_total_exposure,
        max_exposure_per_trade=cfg.max_exposure_per_trade,
        max_total_exposure=cfg.max_total_exposure,
        current_exposure=pnl.current_exposure + maker_exposure,
        kelly_odds_confirmed=cfg.kelly_odds_confirmed,
        kelly_odds_cross_platform=cfg.kelly_odds_cross_platform,
    )
    if size <= 0:
        return 0.0

    per_set_capital = opp.required_capital / opp.max_sets
    if per_set_capital <= 0:
        return 0.0
    free_capital = max(0.0, cfg.max_total_exposure - (pnl.current_exposure + maker_exposure))
    max_sets_by_capital = free_capital / per_set_capital
    size = min(size, opp.max_sets, max_sets_by_capital)
    return max(0.0, size)


def _maker_pair_signature(opp: Opportunity) -> str:
    token_ids = ",".join(sorted(leg.token_id for leg in opp.legs))
    return f"{opp.event_id}:{token_ids}"


def _build_maker_full_fill_result(pair: MakerPairState) -> TradeResult:
    yes_size = pair.yes_filled_size
    no_size = pair.no_filled_size
    sets = min(yes_size, no_size)
    net_pnl = pair.opportunity.net_profit_per_set * sets - pair.opportunity.estimated_gas_cost
    return TradeResult(
        opportunity=pair.opportunity,
        order_ids=(pair.yes_order_id, pair.no_order_id),
        fill_prices=(pair.opportunity.legs[0].price, pair.opportunity.legs[1].price),
        fill_sizes=(yes_size, no_size),
        fees=0.0,
        gas_cost=pair.opportunity.estimated_gas_cost,
        net_pnl=net_pnl,
        execution_time_ms=max(0.0, (time.time() - pair.created_at) * 1000.0),
        fully_filled=True,
    )


def _hedge_orphan_maker_leg(
    *,
    client,
    pair: MakerPairState,
    active_order_ids: set[str],
    book_cache: BookCache,
    gas_cost: float,
) -> TradeResult:
    yes_leg, no_leg = pair.opportunity.legs
    if pair.yes_filled_size > 0:
        filled_leg = yes_leg
        filled_size = pair.yes_filled_size
        unfilled_order_id = pair.no_order_id
        fill_sizes = (filled_size, 0.0)
        fill_prices = (yes_leg.price, 0.0)
    else:
        filled_leg = no_leg
        filled_size = pair.no_filled_size
        unfilled_order_id = pair.yes_order_id
        fill_sizes = (0.0, filled_size)
        fill_prices = (0.0, no_leg.price)

    if unfilled_order_id in active_order_ids:
        _safe_cancel_order(client, unfilled_order_id)

    hedge_price = 0.0
    hedge_side = Side.SELL if filled_leg.side == Side.BUY else Side.BUY
    hedge_filled = False
    try:
        book = book_cache.get_book(filled_leg.token_id)
        if hedge_side == Side.SELL:
            hedge_price = book.best_bid.price if book and book.best_bid else 0.0
            amount = filled_size
        else:
            hedge_price = book.best_ask.price if book and book.best_ask else 0.0
            amount = max(0.01, filled_size * hedge_price)

        if amount > 0:
            hedge_order = create_market_order(
                client,
                token_id=filled_leg.token_id,
                side=hedge_side,
                amount=amount,
                neg_risk=False,
                tick_size=filled_leg.tick_size,
            )
            hedge_resp = post_order(client, hedge_order, OrderType.FOK)
            hedge_snap = _maker_order_snapshot_from_payload(hedge_resp, default_size=filled_size)
            hedge_filled = hedge_snap["filled"]
            if hedge_price <= 0 and book:
                if hedge_side == Side.SELL and book.best_bid:
                    hedge_price = book.best_bid.price
                elif hedge_side == Side.BUY and book.best_ask:
                    hedge_price = book.best_ask.price
    except Exception as e:
        logger.warning("      Maker orphan hedge failed for %s: %s", filled_leg.token_id, e)

    entry_cost = filled_leg.price * filled_size
    if hedge_filled and hedge_price > 0:
        if hedge_side == Side.SELL:
            hedge_value = hedge_price * filled_size
            net_pnl = hedge_value - entry_cost - gas_cost
        else:
            hedge_cost = hedge_price * filled_size
            net_pnl = entry_cost - hedge_cost - gas_cost
    else:
        # Conservative fallback when hedge fails or no book is available.
        net_pnl = -(entry_cost + gas_cost)

    return TradeResult(
        opportunity=pair.opportunity,
        order_ids=(pair.yes_order_id, pair.no_order_id),
        fill_prices=fill_prices,
        fill_sizes=fill_sizes,
        fees=0.0,
        gas_cost=gas_cost,
        net_pnl=net_pnl,
        execution_time_ms=max(0.0, (time.time() - pair.created_at) * 1000.0),
        fully_filled=False,
    )


def _safe_cancel_order(client, order_id: str) -> bool:
    try:
        cancel_order(client, order_id)
        return True
    except Exception as e:
        logger.warning("      Failed to cancel maker order %s: %s", order_id, e)
        return False


def _maker_extract_filled_size(payload: dict, default_size: float) -> float:
    for key in (
        "filled_size",
        "filledSize",
        "size_filled",
        "sizeFilled",
        "matched_size",
        "matchedSize",
        "size_matched",
        "filled",
        "fill_size",
        "filledQuantity",
        "quantity_filled",
    ):
        if key not in payload:
            continue
        try:
            val = float(payload[key])
            if val > 0:
                return min(val, default_size) if default_size > 0 else val
        except (TypeError, ValueError):
            continue
    return 0.0


def _maker_order_snapshot(client, order_id: str, default_size: float) -> dict:
    try:
        payload = client.get_order(order_id) or {}
    except Exception as e:
        logger.warning("      Failed to fetch maker order %s: %s", order_id, e)
        payload = {}
    snap = _maker_order_snapshot_from_payload(payload, default_size=default_size)
    snap["order_id"] = order_id
    return snap


def _maker_order_snapshot_from_payload(payload: dict, default_size: float) -> dict:
    status = str(payload.get("status", "")).lower()
    filled_size = _maker_extract_filled_size(payload, default_size=default_size)
    if status in ("matched", "filled"):
        if filled_size <= 0:
            filled_size = max(0.0, default_size)
        filled = True
    elif status in ("partial", "partially_filled"):
        filled = filled_size >= max(0.0, default_size) and default_size > 0
    else:
        filled = False
    cancelled = status in ("cancelled", "canceled", "expired", "rejected")
    return {
        "status": status,
        "filled": filled,
        "cancelled": cancelled,
        "filled_size": filled_size,
    }


def _execute_single(
    client,
    cfg: Config,
    opp: Opportunity,
    pnl: PnLTracker,
    breaker: CircuitBreaker,
    gas_oracle: GasOracle | None = None,
    position_tracker: PositionTracker | None = None,
    platform_clients: dict[str, PlatformClient] | None = None,
    presigner: OrderPresigner | None = None,
):
    """Execute a single opportunity with full safety checks."""
    if not _is_execution_supported_type(opp.type):
        raise SafetyCheckFailed(f"execution_support: unsupported type {opp.type.value}")

    # Opportunity TTL check (reject stale opportunities)
    logger.info("        Verifying opportunity freshness (TTL)...")
    verify_opportunity_ttl(opp)

    # Max legs check (reject multi-batch opportunities)
    logger.info("        Verifying leg count...")
    verify_max_legs(opp, cfg.max_legs_per_opportunity)

    # Safety checks independent of size
    pm_books: dict[str, object] = {}
    ext_books_nested: dict[str, dict[str, object]] = {}
    if opp.type == OpportunityType.CROSS_PLATFORM_ARB:
        if not platform_clients:
            raise SafetyCheckFailed("platform_clients required for cross-platform opportunity")
        signal_age_sec = time.time() - opp.timestamp
        if signal_age_sec > cfg.cross_platform_max_signal_age_sec:
            raise SafetyCheckFailed(
                f"fill_gap: signal age {signal_age_sec:.3f}s exceeds "
                f"{cfg.cross_platform_max_signal_age_sec:.3f}s"
            )
        pm_token_ids = [leg.token_id for leg in opp.legs if leg.platform in ("polymarket", "")]
        ext_by_platform: dict[str, list[str]] = {}
        for leg in opp.legs:
            if leg.platform not in ("polymarket", ""):
                ext_by_platform.setdefault(leg.platform, []).append(leg.token_id)
        missing_clients = [p for p in ext_by_platform if p not in platform_clients]
        if missing_clients:
            raise SafetyCheckFailed(
                f"venue_preflight: missing platform client(s): {', '.join(sorted(missing_clients))}"
            )
        pm_books = get_orderbooks(client, pm_token_ids) if pm_token_ids else {}
        for pname, tids in ext_by_platform.items():
            pclient = platform_clients.get(pname)
            if pclient:
                ext_books_nested[pname] = pclient.get_orderbooks(tids)
        verify_cross_platform_books(opp, pm_books, platform_books=ext_books_nested, min_depth=1.0)
        # Merge all books into flat dict for verify_prices_fresh / verify_edge_intact
        books = dict(pm_books)
        for pbooks in ext_books_nested.values():
            books.update(pbooks)
    else:
        token_ids = [leg.token_id for leg in opp.legs]
        books = get_orderbooks(client, token_ids)

    logger.info("        Verifying price freshness...")
    verify_prices_fresh(opp, books)

    # Edge revalidation with fresh books
    logger.info("        Verifying edge still intact...")
    verify_edge_intact(opp, books)

    # Position sizing
    bankroll = cfg.max_total_exposure  # simplified: use max exposure as bankroll
    size = compute_position_size(
        opp,
        bankroll=bankroll,
        max_exposure_per_trade=cfg.max_exposure_per_trade,
        max_total_exposure=cfg.max_total_exposure,
        current_exposure=pnl.current_exposure,
        kelly_odds_confirmed=cfg.kelly_odds_confirmed,
        kelly_odds_cross_platform=cfg.kelly_odds_cross_platform,
    )
    if size <= 0:
        logger.info("        Position size = 0 (insufficient capital or edge), skipping")
        return

    execution_size = size
    if opp.type == OpportunityType.CROSS_PLATFORM_ARB:
        execution_size = float(int(size))
        if execution_size <= 0:
            logger.info("        Cross-platform size rounds to 0 contracts, skipping")
            return

    sized_opp = _with_sized_legs(opp, execution_size)

    # Inventory check for sell legs
    if position_tracker:
        if cfg.cross_platform_inventory_pm_only:
            has_sell_legs = any(
                leg.side == Side.SELL and (leg.platform in ("", "polymarket"))
                for leg in opp.legs
            )
            platform_filter = {"polymarket"}
        else:
            has_sell_legs = any(leg.side == Side.SELL for leg in opp.legs)
            platform_filter = None
        if has_sell_legs:
            logger.info("        Verifying inventory for sell legs...")
            verify_inventory(position_tracker, opp, execution_size, platform_filter=platform_filter)

    if opp.type == OpportunityType.CROSS_PLATFORM_ARB:
        verify_cross_platform_books(sized_opp, pm_books, platform_books=ext_books_nested, min_depth=execution_size)
        # Verify platform position limits
        for leg in opp.legs:
            if leg.platform and leg.platform not in ("polymarket", ""):
                position_value = leg.price * execution_size
                try:
                    verify_platform_limits(
                        leg.platform,
                        position_value,
                        kalshi_limit=cfg.kalshi_position_limit,
                        fanatics_limit=cfg.fanatics_position_limit,
                    )
                except SafetyCheckFailed:
                    logger.info("        Platform limit check failed for %s: %s", leg.platform, position_value)
                    raise

    logger.info("        Verifying orderbook depth...")
    verify_depth(sized_opp, books)
    if gas_oracle:
        logger.info("        Verifying gas reasonableness...")
        verify_gas_reasonable(
            gas_oracle, opp, cfg.gas_per_order, cfg.max_gas_profit_ratio, size=execution_size,
        )

    logger.info(
        "        Sized %.1f sets ($%.2f capital) via half-Kelly",
        execution_size, execution_size * opp.required_capital / opp.max_sets,
    )

    # Execute
    result = execute_opportunity(
        client, sized_opp, execution_size,
        paper_trading=cfg.paper_trading,
        use_fak=cfg.use_fak_orders,
        order_timeout_sec=cfg.order_timeout_sec,
        platform_clients=platform_clients,
        cross_platform_deadline_sec=cfg.cross_platform_deadline_sec,
        presigner=presigner,
    )

    # Record
    pnl.record(result)
    breaker.record_trade(result.net_pnl)
    return result


def _with_sized_legs(opp: Opportunity, size: float) -> Opportunity:
    """Return a copy of the opportunity where each leg size matches the chosen execution size."""
    return replace(opp, legs=tuple(replace(leg, size=size) for leg in opp.legs))


def _cancel_open_orders_on_shutdown(client, args: argparse.Namespace, cfg: Config) -> None:
    """Attempt to cancel all open orders on shutdown for live trading sessions."""
    if args.scan_only or args.dry_run or cfg.paper_trading:
        return
    try:
        resp = cancel_all(client)
        logger.info("Shutdown cleanup: cancel_all response=%s", resp)
    except Exception as e:
        logger.error("Shutdown cleanup failed while canceling open orders: %s", e)


def _extract_safety_check_name(msg: str) -> str:
    """Extract safety check name from exception message."""
    checks = [
        "ttl",
        "price_fresh",
        "depth",
        "gas",
        "edge",
        "max_legs",
        "inventory",
        "platform_limit",
        "execution_support",
        "fill_gap",
        "venue_preflight",
    ]
    msg_lower = msg.lower()
    for name in checks:
        if name.replace("_", " ") in msg_lower or name in msg_lower:
            return name
    return "unknown"


def _is_rate_limit_error(msg: str) -> bool:
    """Return True if an error string indicates API rate limiting."""
    text = msg.lower()
    return (
        "429" in text
        or "rate limit" in text
        or "too many requests" in text
    )


def _sleep_remaining(cycle_start: float, interval: float, shutdown: bool) -> None:
    """Sleep for the remaining scan interval, respecting shutdown."""
    if shutdown:
        return
    elapsed = time.time() - cycle_start
    remaining = interval - elapsed
    if remaining > 0:
        logger.debug("Sleeping %.1fs until next cycle...", remaining)
        time.sleep(remaining)


if __name__ == "__main__":
    main()
