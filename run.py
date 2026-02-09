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
import logging
import signal
import sys
import time
from dataclasses import replace

from config import load_config, Config
from functools import partial

from client.auth import build_clob_client
from client.clob import cancel_all, get_orderbooks, get_orderbooks_parallel
from client.gamma import get_all_markets, build_events, get_event_market_counts
from client.gas import GasOracle
from scanner.binary import scan_binary_markets
from scanner.negrisk import scan_negrisk_events
from scanner.book_cache import BookCache
from scanner.fees import MarketFeeModel
from scanner.latency import LatencyScanner, scan_latency_markets
from scanner.spike import SpikeDetector, scan_spike_opportunities
from scanner.depth import sweep_depth
from scanner.scorer import rank_opportunities, ScoringContext
from scanner.strategy import StrategySelector, MarketState
from scanner.models import Opportunity, OpportunityType, Side
from scanner.cross_platform import scan_cross_platform
from scanner.kalshi_fees import KalshiFeeModel
from scanner.matching import EventMatcher
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
)
from client.data import PositionTracker
from client.ws_bridge import WSBridge
from executor.engine import execute_opportunity, UnwindFailed
from executor.cross_platform import CrossPlatformUnwindFailed
from monitor.pnl import PnLTracker
from monitor.scan_tracker import ScanTracker
from monitor.status import StatusWriter
from monitor.logger import setup_logging

logger = logging.getLogger(__name__)


_BANNER = r"""
 ____       _                            _        _
|  _ \ ___ | |_   _ _ __ ___   __ _ _ __| | _____| |_
| |_) / _ \| | | | | '_ ` _ \ / _` | '__| |/ / _ \ __|
|  __/ (_) | | |_| | | | | | | (_| | |  |   <  __/ |_
|_|   \___/|_|\__, |_| |_| |_|\__,_|_|  |_|\_\___|\__|
              |___/            Arbitrage Bot v0.1
"""


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Polymarket Arbitrage Bot")
    parser.add_argument("--live", action="store_true", help="Enable live trading (disables paper mode)")
    parser.add_argument("--scan-only", action="store_true", help="Only scan for opportunities, do not execute")
    parser.add_argument("--dry-run", action="store_true", help="No wallet needed. Scan real markets using public APIs only")
    parser.add_argument("--limit", type=int, default=0, help="Max markets to scan (0 = all). Useful for dry-run testing")
    parser.add_argument("--json-log", type=str, default=None, help="Path to JSON log file for machine-readable output")
    return parser.parse_args()


def _mode_label(args: argparse.Namespace, cfg: Config) -> str:
    if args.dry_run:
        return "DRY-RUN (public APIs only, no execution)"
    if args.scan_only:
        return "SCAN-ONLY (detect opportunities, no execution)"
    if cfg.paper_trading:
        return "PAPER TRADING (simulated fills, no real orders)"
    return "LIVE TRADING"


def _print_startup(args: argparse.Namespace, cfg: Config) -> None:
    """Print the startup banner and full configuration summary."""
    mode = _mode_label(args, cfg)

    logger.info(_BANNER.strip())
    logger.info("")
    logger.info("%-20s %s", "Mode:", mode)
    logger.info("%-20s $%.2f", "Min profit:", cfg.min_profit_usd)
    logger.info("%-20s %.1f%%", "Min ROI:", cfg.min_roi_pct)
    logger.info("%-20s $%.0f per trade / $%.0f total", "Max exposure:", cfg.max_exposure_per_trade, cfg.max_total_exposure)
    logger.info("%-20s $%.0f/hr  $%.0f/day  %d consecutive", "Circuit breakers:", cfg.max_loss_per_hour, cfg.max_loss_per_day, cfg.max_consecutive_failures)
    logger.info("%-20s %.1fs", "Scan interval:", cfg.scan_interval_sec)
    if args.limit > 0:
        logger.info("%-20s %d markets", "Scan limit:", args.limit)
    logger.info("%-20s %s", "Order type:", "FAK (Fill-and-Kill)" if cfg.use_fak_orders else "GTC")
    logger.info("%-20s %s", "WebSocket:", "enabled" if cfg.ws_enabled else "disabled")
    logger.info("%-20s %s", "Cross-platform:", "enabled" if cfg.cross_platform_enabled else "disabled")
    logger.info("%-20s %s", "CLOB endpoint:", cfg.clob_host)
    logger.info("%-20s %s", "Gamma endpoint:", cfg.gamma_host)
    if cfg.cross_platform_enabled:
        logger.info("%-20s %s", "Kalshi endpoint:", cfg.kalshi_host)
        logger.info("%-20s %.0f%%", "Min confidence:", cfg.cross_platform_min_confidence * 100)
    logger.info("")


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


def _print_opportunity(opp: Opportunity, index: int) -> None:
    """Print a single opportunity in a structured format."""
    type_label = opp.type.value
    if opp.is_sell_arb:
        type_label = f"[SELL] {type_label}"
    logger.info(
        "  #%-2d  %-25s  event=%-14s  profit=$%.2f  roi=%.2f%%  legs=%d  capital=$%.2f",
        index,
        type_label,
        opp.event_id[:14],
        opp.net_profit,
        opp.roi_pct,
        len(opp.legs),
        opp.required_capital,
    )


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
        logger.info("    %-28s $%.2f (%d opps)", "Actionable (buy arbs):", s["buy_arb_profit_usd"], s["buy_arb_count"])
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

    # Override paper trading based on CLI flag
    if args.live:
        cfg.paper_trading = False
    if args.dry_run:
        args.scan_only = True  # dry-run implies scan-only

    # Validate credentials for non-dry-run modes
    if not args.dry_run and (not cfg.private_key or not cfg.polymarket_profile_address):
        logger.error("PRIVATE_KEY and POLYMARKET_PROFILE_ADDRESS required for paper/live trading.")
        logger.error("Use --dry-run to scan without a wallet.")
        sys.exit(1)

    setup_logging(cfg.log_level, json_log_file=args.json_log)
    _print_startup(args, cfg)

    # Initialize client
    if args.dry_run:
        logger.info("Initializing unauthenticated CLOB client (dry-run mode)...")
        from py_clob_client.client import ClobClient
        client = ClobClient(host=cfg.clob_host)
        logger.info("Client ready -- using public endpoints only")
    else:
        logger.info("Authenticating with Polymarket CLOB...")
        logger.info("  Deriving L2 API credentials from wallet...")
        client = build_clob_client(cfg)
        logger.info("Authentication successful -- client ready for %s",
                     "paper trading" if cfg.paper_trading else "LIVE trading")

    # Initialize components
    pnl = PnLTracker()
    tracker = ScanTracker()
    breaker = CircuitBreaker(
        max_loss_per_hour=cfg.max_loss_per_hour,
        max_loss_per_day=cfg.max_loss_per_day,
        max_consecutive_failures=cfg.max_consecutive_failures,
    )
    book_cache = BookCache(max_age_sec=cfg.book_cache_max_age_sec)
    gas_oracle = GasOracle(
        rpc_url=cfg.polygon_rpc_url,
        cache_sec=cfg.gas_cache_sec,
        default_gas_gwei=cfg.gas_price_gwei,
    )
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
    logger.info("Safety systems initialized (circuit breaker active)")
    logger.info("Book cache initialized (max_age=%.1fs)", cfg.book_cache_max_age_sec)
    logger.info("Gas oracle initialized (rpc=%s, cache=%.0fs)", cfg.polygon_rpc_url, cfg.gas_cache_sec)
    logger.info("Fee model %s", "enabled" if cfg.fee_model_enabled else "disabled")
    logger.info("Latency scanner %s (min_edge=%.1f%%)", "enabled" if cfg.latency_enabled else "disabled", cfg.latency_min_edge_pct)
    strategy = StrategySelector(
        base_min_profit=cfg.min_profit_usd,
        base_min_roi=cfg.min_roi_pct,
        base_target_size=cfg.target_size_usd,
    )
    logger.info("Spike detector initialized (threshold=%.1f%%, window=%.0fs, cooldown=%.0fs)",
                cfg.spike_threshold_pct, cfg.spike_window_sec, cfg.spike_cooldown_sec)
    logger.info("Strategy selector initialized (adaptive mode)")

    # Kalshi cross-platform initialization (gated behind config toggle)
    kalshi_client = None
    event_matcher = None
    kalshi_fee_model = None
    if cfg.cross_platform_enabled:
        if not cfg.kalshi_api_key_id or not cfg.kalshi_private_key_path:
            logger.error("KALSHI_API_KEY_ID and KALSHI_PRIVATE_KEY_PATH required when CROSS_PLATFORM_ENABLED=true")
            sys.exit(1)
        from client.kalshi_auth import KalshiAuth
        from client.kalshi import KalshiClient
        kalshi_auth = KalshiAuth(
            api_key_id=cfg.kalshi_api_key_id,
            private_key_path=cfg.kalshi_private_key_path,
        )
        kalshi_client = KalshiClient(
            auth=kalshi_auth,
            host=cfg.kalshi_host,
            demo=cfg.kalshi_demo,
        )
        event_matcher = EventMatcher(
            manual_map_path=cfg.cross_platform_manual_map,
        )
        kalshi_fee_model = KalshiFeeModel()
        logger.info("Kalshi client initialized (host=%s, demo=%s)", cfg.kalshi_host, cfg.kalshi_demo)
        logger.info("Cross-platform matching enabled (min_confidence=%.2f, map=%s)",
                    cfg.cross_platform_min_confidence, cfg.cross_platform_manual_map)
    else:
        logger.info("Cross-platform arbitrage disabled")

    # WebSocket bridge for real-time book/price feeds
    ws_bridge: WSBridge | None = None
    if cfg.ws_enabled and not args.dry_run:
        ws_bridge = WSBridge(
            ws_url=cfg.ws_market_url,
            book_cache=book_cache,
            spike_detector=spike_detector,
            max_retries=cfg.ws_reconnect_max,
        )
        logger.info("WebSocket bridge initialized (url=%s)", cfg.ws_market_url)
    elif cfg.ws_enabled:
        logger.info("WebSocket disabled in dry-run mode (REST only)")

    status_writer = StatusWriter(file_path="status.md")
    status_writer.write_cycle(
        cycle=0,
        mode=_mode_label(args, cfg),
        markets_scanned=0,
        scored_opps=[],
        event_questions={},
        total_opps_found=0,
        total_trades_executed=0,
        total_pnl=0.0,
        current_exposure=0.0,
        scan_only=args.scan_only,
    )
    logger.info("Status writer initialized (status.md, last %d cycles)", status_writer.max_history)
    logger.info("")

    # Graceful shutdown handler
    shutdown_requested = False

    def handle_signal(signum, frame):
        nonlocal shutdown_requested
        shutdown_requested = True
        # Avoid reentrant logging -- print directly to stderr instead
        import sys
        print(f"\nShutdown signal received (signal {signum}) -- finishing current cycle...",
              file=sys.stderr, flush=True)

    signal.signal(signal.SIGINT, handle_signal)
    signal.signal(signal.SIGTERM, handle_signal)

    # Main loop
    cycle = 0
    session_start = time.time()
    total_opps_found = 0
    total_trades_executed = 0
    has_crypto_momentum = False

    while not shutdown_requested:
        cycle += 1
        cycle_start = time.time()

        try:
            logger.info("── Cycle %d ──────────────────────────────────────────────", cycle)

            # Step 1: Fetch active markets
            step_start = time.time()
            logger.info("[1/3] Fetching active markets from Gamma API...")
            all_markets = get_all_markets(cfg.gamma_host)
            if args.limit > 0:
                # Never truncate negRisk markets -- they require complete outcome
                # sets. Truncating splits multi-outcome events and causes false
                # arbitrage detection (e.g. seeing 6/59 Masters outcomes).
                negrisk_mkts = [m for m in all_markets if m.neg_risk]
                binary_mkts = [m for m in all_markets if not m.neg_risk][:args.limit]
                all_markets = binary_mkts + negrisk_mkts
            events = build_events(all_markets)
            event_questions = {e.event_id: e.title for e in events}

            binary_markets = [m for m in all_markets if not m.neg_risk]
            negrisk_events = [e for e in events if e.neg_risk]

            logger.info(
                "      Received %s markets in %.1fs",
                f"{len(all_markets):,}", time.time() - step_start,
            )
            logger.info(
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
            if ws_bridge:
                ws_updates = ws_bridge.drain()
                if ws_updates > 0:
                    logger.info("      WebSocket: drained %d updates into cache", ws_updates)

            # Step 2: Scan for opportunities (strategy-tuned)
            step_start = time.time()

            # Select adaptive strategy for this cycle
            market_state = MarketState(
                gas_price_gwei=gas_oracle.get_gas_price_gwei(),
                active_spike_count=len(spike_detector.detect_spikes()) if cycle > 1 else 0,
                has_crypto_momentum=has_crypto_momentum,
                recent_win_rate=pnl.win_rate / 100.0 if not args.scan_only else 0.50,
            )
            scan_params = strategy.select(market_state)
            logger.info("[2/3] Scanning for arbitrage (strategy=%s)...", scan_params.mode.value)

            # Create BookFetcher callable: parallel REST wrapped with cache layer
            poly_rest_fetcher = partial(get_orderbooks_parallel, client, max_workers=cfg.book_fetch_workers)
            poly_book_fetcher = book_cache.make_caching_fetcher(poly_rest_fetcher)

            binary_opps: list[Opportunity] = []
            if scan_params.binary_enabled:
                logger.info("      Scanning %s binary markets...", f"{len(binary_markets):,}")
                binary_opps = scan_binary_markets(
                    poly_book_fetcher, binary_markets,
                    scan_params.min_profit_usd, scan_params.min_roi_pct,
                    cfg.gas_per_order, cfg.gas_price_gwei,
                    gas_oracle=gas_oracle, fee_model=fee_model, book_cache=book_cache,
                    min_volume=cfg.min_volume_filter,
                )

            negrisk_opps: list[Opportunity] = []
            if scan_params.negrisk_enabled:
                logger.info("      Scanning %d negRisk events...", len(negrisk_events))
                # Fetch expected market counts for event completeness validation.
                # This prevents false arbs from incomplete outcome sets.
                event_market_counts = get_event_market_counts(cfg.gamma_host)
                negrisk_opps = scan_negrisk_events(
                    poly_book_fetcher, negrisk_events,
                    scan_params.min_profit_usd, scan_params.min_roi_pct,
                    cfg.gas_per_order, cfg.gas_price_gwei,
                    gas_oracle=gas_oracle, fee_model=fee_model, book_cache=book_cache,
                    min_volume=cfg.min_volume_filter,
                    max_legs=cfg.max_legs_per_opportunity,
                    event_market_counts=event_market_counts,
                )

            # Latency arb on 15-min crypto markets
            latency_opps: list[Opportunity] = []
            if latency_scanner and scan_params.latency_enabled:
                crypto_markets = latency_scanner.identify_crypto_markets(all_markets)
                has_crypto_momentum = bool(crypto_markets)
                if crypto_markets:
                    logger.info("      Scanning %d crypto 15-min markets for latency arb...", len(crypto_markets))
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
                    latency_opps = scan_latency_markets(
                        latency_scanner, crypto_markets, crypto_books, gas_cost,
                        no_books=crypto_no_books,
                    )
            else:
                has_crypto_momentum = False

            # Spike detection: update histories and check for lag arbs
            spike_opps: list[Opportunity] = []
            if scan_params.spike_enabled:
                # Feed current midpoints into spike detector
                for m in all_markets:
                    book = book_cache.get_book(m.yes_token_id)
                    if book and book.midpoint is not None:
                        spike_detector.register_token(m.yes_token_id, m.event_id)
                        spike_detector.update(m.yes_token_id, book.midpoint)
                spikes = spike_detector.detect_spikes()
                if spikes:
                    logger.info("      Detected %d price spikes, checking siblings...", len(spikes))
                    # Build event lookup
                    event_map = {e.event_id: e for e in events}
                    gas_cost = gas_oracle.estimate_cost_usd(1, cfg.gas_per_order)
                    for spike in spikes:
                        ev = event_map.get(spike.event_id)
                        if ev:
                            spike_opps.extend(scan_spike_opportunities(
                                spike, ev, book_cache, fee_model,
                                gas_cost_usd=gas_cost,
                                min_profit_usd=scan_params.min_profit_usd,
                            ))

            # Cross-platform arbitrage (Polymarket vs Kalshi)
            cross_platform_opps: list[Opportunity] = []
            if cfg.cross_platform_enabled and kalshi_client and event_matcher:
                logger.info("      Scanning cross-platform arbitrage (Kalshi)...")
                # Fetch Kalshi markets
                kalshi_markets = kalshi_client.get_all_markets(status="open")
                logger.info("      Fetched %d active Kalshi markets", len(kalshi_markets))

                # Match events across platforms
                matched_events = event_matcher.match_events(events, kalshi_markets)
                logger.info("      Matched %d events across platforms", len(matched_events))

                if matched_events:
                    # Collect all token IDs needed from both platforms
                    pm_token_ids: list[str] = []
                    kalshi_tickers: list[str] = []
                    for match in matched_events:
                        for pm_mkt in match.pm_markets:
                            pm_token_ids.append(pm_mkt.yes_token_id)
                            pm_token_ids.append(pm_mkt.no_token_id)
                        kalshi_tickers.extend(match.kalshi_tickers)

                    # Fetch orderbooks from both platforms
                    pm_cross_books = poly_book_fetcher(pm_token_ids) if pm_token_ids else {}
                    kalshi_cross_books = kalshi_client.get_orderbooks(kalshi_tickers) if kalshi_tickers else {}
                    book_cache.store_books(pm_cross_books)

                    cross_platform_opps = scan_cross_platform(
                        matched_events, pm_cross_books, kalshi_cross_books,
                        min_profit_usd=scan_params.min_profit_usd,
                        min_roi_pct=scan_params.min_roi_pct,
                        gas_per_order=cfg.gas_per_order,
                        gas_oracle=gas_oracle,
                        pm_fee_model=fee_model,
                        kalshi_fee_model=kalshi_fee_model,
                        min_confidence=cfg.cross_platform_min_confidence,
                    )
                    if cross_platform_opps:
                        logger.info("      Found %d cross-platform opportunities", len(cross_platform_opps))

            all_opps = binary_opps + negrisk_opps + latency_opps + spike_opps + cross_platform_opps

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

            # Build real ScoringContext per opportunity
            contexts = _build_scoring_contexts(all_opps, book_cache, all_markets, cfg.target_size_usd)
            # Composite scoring with real context data
            scored_opps = rank_opportunities(all_opps, contexts=contexts)
            total_opps_found += len(all_opps)

            scan_elapsed = time.time() - step_start
            if scored_opps:
                best = scored_opps[0]
                logger.info(
                    "      Found %d opportunities in %.1fs (best score: %.2f, best profit: $%.2f)",
                    len(scored_opps), scan_elapsed,
                    best.total_score,
                    max(o.net_profit for o in all_opps),
                )
                for i, scored in enumerate(scored_opps, 1):
                    _print_opportunity(scored.opportunity, i)
            else:
                logger.info("      No opportunities found (%.1fs)", scan_elapsed)

            if args.scan_only:
                tracker.record_cycle(cycle, len(all_markets), all_opps)
                status_writer.write_cycle(
                    cycle=cycle,
                    mode=_mode_label(args, cfg),
                    markets_scanned=len(all_markets),
                    scored_opps=scored_opps,
                    event_questions=event_questions,
                    total_opps_found=total_opps_found,
                    total_trades_executed=0,
                    total_pnl=0.0,
                    current_exposure=0.0,
                    scan_only=True,
                )
                cycle_elapsed = time.time() - cycle_start
                logger.info(
                    "      Cycle %d complete in %.1fs  |  session: %d cycles, %d opps found",
                    cycle, cycle_elapsed, cycle, total_opps_found,
                )
                _sleep_remaining(cycle_start, cfg.scan_interval_sec, shutdown_requested)
                continue

            # Step 3: Execute top opportunities (ordered by composite score)
            logger.info("[3/3] Executing %d opportunities...", len(scored_opps))
            for i, scored in enumerate(scored_opps, 1):
                if shutdown_requested:
                    break
                opp = scored.opportunity

                try:
                    logger.info(
                        "      [%d/%d] Executing %s  event=%s  score=%.2f  profit=$%.2f...",
                        i, len(scored_opps), opp.type.value, opp.event_id[:14], scored.total_score, opp.net_profit,
                    )
                    _execute_single(client, cfg, opp, pnl, breaker, gas_oracle=gas_oracle, position_tracker=position_tracker, kalshi_client=kalshi_client)
                    total_trades_executed += 1
                    logger.info(
                        "      [%d/%d] Trade complete  |  session P&L: $%.2f (%d trades)",
                        i, len(scored_opps), pnl.total_pnl, pnl.total_trades,
                    )
                except SafetyCheckFailed as e:
                    logger.warning("      [%d/%d] Safety check failed, skipping: %s", i, len(scored_opps), e)
                    continue
                except UnwindFailed as e:
                    logger.error("      [%d/%d] UNWIND FAILED -- stuck positions: %s", i, len(scored_opps), e)
                    breaker.record_trade(-opp.net_profit)  # count as a loss
                    continue
                except CrossPlatformUnwindFailed as e:
                    logger.error("      [%d/%d] CROSS-PLATFORM UNWIND FAILED -- stuck Kalshi positions: %s", i, len(scored_opps), e)
                    breaker.record_trade(-opp.net_profit)
                    continue
                except CircuitBreakerTripped as e:
                    logger.error("CIRCUIT BREAKER TRIPPED: %s", e)
                    _print_pnl_summary(pnl)
                    return

            # Write status after execution
            status_writer.write_cycle(
                cycle=cycle,
                mode=_mode_label(args, cfg),
                markets_scanned=len(all_markets),
                scored_opps=scored_opps,
                event_questions=event_questions,
                total_opps_found=total_opps_found,
                total_trades_executed=total_trades_executed,
                total_pnl=pnl.total_pnl,
                current_exposure=pnl.current_exposure,
                scan_only=False,
            )

        except KeyboardInterrupt:
            break
        except Exception as e:
            logger.error("Cycle %d failed: %s", cycle, e, exc_info=True)

        cycle_elapsed = time.time() - cycle_start
        logger.info(
            "      Cycle %d complete in %.1fs  |  session: %d trades, P&L $%.2f",
            cycle, cycle_elapsed, pnl.total_trades, pnl.total_pnl,
        )
        _sleep_remaining(cycle_start, cfg.scan_interval_sec, shutdown_requested)

    # Shutdown
    logger.info("")
    logger.info("Shutting down gracefully after %s (%d cycles)",
                _format_duration(time.time() - session_start), cycle)
    if ws_bridge:
        ws_bridge.stop()
    _cancel_open_orders_on_shutdown(client, args, cfg)

    if args.scan_only:
        _print_scan_summary(tracker)
    else:
        _print_pnl_summary(pnl)


def _build_scoring_contexts(
    opps: list[Opportunity],
    book_cache: BookCache,
    all_markets: list,
    target_size: float,
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
        # Depth ratio: min(available_depth / target_size) across all legs
        min_depth_ratio = 1.0
        for leg in opp.legs:
            book = book_cache.get_book(leg.token_id)
            if book:
                available = sweep_depth(book, leg.side, max_price=leg.price * 1.005 if leg.side == Side.BUY else leg.price * 0.995)
                ratio = available / target_size if target_size > 0 else 1.0
                min_depth_ratio = min(min_depth_ratio, ratio)
            else:
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

        contexts.append(ScoringContext(
            market_volume=volume,
            recent_trade_count=0,
            time_to_resolution_hours=time_to_resolution_hours,
            is_spike=is_spike,
            book_depth_ratio=min_depth_ratio,
        ))

    return contexts


def _execute_single(
    client,
    cfg: Config,
    opp: Opportunity,
    pnl: PnLTracker,
    breaker: CircuitBreaker,
    gas_oracle: GasOracle | None = None,
    position_tracker: PositionTracker | None = None,
    kalshi_client: object | None = None,
) -> None:
    """Execute a single opportunity with full safety checks."""
    # Opportunity TTL check (reject stale opportunities)
    logger.info("        Verifying opportunity freshness (TTL)...")
    verify_opportunity_ttl(opp)

    # Max legs check (reject multi-batch opportunities)
    logger.info("        Verifying leg count...")
    verify_max_legs(opp, cfg.max_legs_per_opportunity)

    # Safety checks independent of size
    pm_books: dict[str, object] = {}
    kalshi_books: dict[str, object] = {}
    if opp.type == OpportunityType.CROSS_PLATFORM_ARB:
        if kalshi_client is None:
            raise SafetyCheckFailed("kalshi_client required for cross-platform opportunity")
        pm_token_ids = [leg.token_id for leg in opp.legs if leg.platform != "kalshi"]
        kalshi_token_ids = [leg.token_id for leg in opp.legs if leg.platform == "kalshi"]
        pm_books = get_orderbooks(client, pm_token_ids) if pm_token_ids else {}
        kalshi_books = kalshi_client.get_orderbooks(kalshi_token_ids) if kalshi_token_ids else {}
        verify_cross_platform_books(opp, pm_books, kalshi_books, min_depth=1.0)
        books = {**pm_books, **kalshi_books}
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
        has_sell_legs = any(leg.side == Side.SELL for leg in opp.legs)
        if has_sell_legs:
            logger.info("        Verifying inventory for sell legs...")
            verify_inventory(position_tracker, opp, execution_size)

    if opp.type == OpportunityType.CROSS_PLATFORM_ARB:
        verify_cross_platform_books(sized_opp, pm_books, kalshi_books, min_depth=execution_size)

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
        client, opp, execution_size,
        paper_trading=cfg.paper_trading,
        use_fak=cfg.use_fak_orders,
        order_timeout_sec=cfg.order_timeout_sec,
        kalshi_client=kalshi_client,
    )

    # Record
    pnl.record(result)
    breaker.record_trade(result.net_pnl)


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
