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

from config import load_config, Config
from client.auth import build_clob_client
from client.gamma import get_all_markets, build_events
from scanner.binary import scan_binary_markets
from scanner.negrisk import scan_negrisk_events
from scanner.models import Opportunity
from executor.sizing import compute_position_size
from executor.safety import (
    CircuitBreaker,
    CircuitBreakerTripped,
    SafetyCheckFailed,
    verify_prices_fresh,
    verify_depth,
)
from executor.engine import execute_opportunity
from monitor.pnl import PnLTracker
from monitor.scan_tracker import ScanTracker
from monitor.logger import setup_logging

logger = logging.getLogger(__name__)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Polymarket Arbitrage Bot")
    parser.add_argument("--live", action="store_true", help="Enable live trading (disables paper mode)")
    parser.add_argument("--scan-only", action="store_true", help="Only scan for opportunities, do not execute")
    parser.add_argument("--dry-run", action="store_true", help="No wallet needed. Scan real markets using public APIs only")
    parser.add_argument("--limit", type=int, default=0, help="Max markets to scan (0 = all). Useful for dry-run testing")
    return parser.parse_args()


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

    setup_logging(cfg.log_level)

    mode_str = "DRY-RUN" if args.dry_run else ("PAPER" if cfg.paper_trading else "LIVE")
    logger.info("=" * 60)
    logger.info("Polymarket Arbitrage Bot starting")
    logger.info("Mode: %s", mode_str)
    logger.info("Min profit: $%.2f | Min ROI: %.1f%%", cfg.min_profit_usd, cfg.min_roi_pct)
    logger.info("Max exposure/trade: $%.0f | Max total: $%.0f", cfg.max_exposure_per_trade, cfg.max_total_exposure)
    logger.info("=" * 60)

    # Initialize client
    if args.dry_run:
        # L0 client: host only, no auth. Public endpoints (orderbooks, prices) work fine.
        from py_clob_client.client import ClobClient
        client = ClobClient(host=cfg.clob_host)
        logger.info("Dry-run mode: using unauthenticated public API")
    else:
        logger.info("Authenticating with Polymarket CLOB...")
        client = build_clob_client(cfg)
        logger.info("Authenticated successfully")

    # Initialize components
    pnl = PnLTracker()
    tracker = ScanTracker()
    breaker = CircuitBreaker(
        max_loss_per_hour=cfg.max_loss_per_hour,
        max_loss_per_day=cfg.max_loss_per_day,
        max_consecutive_failures=cfg.max_consecutive_failures,
    )

    # Graceful shutdown handler
    shutdown_requested = False

    def handle_signal(signum, frame):
        nonlocal shutdown_requested
        logger.info("Shutdown signal received (signal %d)", signum)
        shutdown_requested = True

    signal.signal(signal.SIGINT, handle_signal)
    signal.signal(signal.SIGTERM, handle_signal)

    # Main loop
    cycle = 0
    while not shutdown_requested:
        cycle += 1
        cycle_start = time.time()

        try:
            # 1. Fetch active markets
            logger.info("Cycle %d: fetching markets...", cycle)
            all_markets = get_all_markets(cfg.gamma_host)
            if args.limit > 0:
                all_markets = all_markets[:args.limit]
            events = build_events(all_markets)

            binary_markets = [m for m in all_markets if not m.neg_risk]
            negrisk_events = [e for e in events if e.neg_risk]

            logger.info(
                "Fetched %d markets (%d binary, %d negRisk events with %d markets)",
                len(all_markets),
                len(binary_markets),
                len(negrisk_events),
                sum(len(e.markets) for e in negrisk_events),
            )

            # 2. Scan for opportunities
            binary_opps = scan_binary_markets(
                client, binary_markets,
                cfg.min_profit_usd, cfg.min_roi_pct,
                cfg.gas_per_order, cfg.gas_price_gwei,
            )

            negrisk_opps = scan_negrisk_events(
                client, negrisk_events,
                cfg.min_profit_usd, cfg.min_roi_pct,
                cfg.gas_per_order, cfg.gas_price_gwei,
            )

            all_opps = binary_opps + negrisk_opps
            all_opps.sort(key=lambda o: o.roi_pct, reverse=True)

            if all_opps:
                logger.info(
                    "Found %d opportunities (best ROI: %.2f%%, best profit: $%.2f)",
                    len(all_opps),
                    all_opps[0].roi_pct,
                    max(o.net_profit for o in all_opps),
                )
            else:
                logger.info("No opportunities found")

            if args.scan_only:
                tracker.record_cycle(cycle, len(all_markets), all_opps)
                for opp in all_opps:
                    logger.info(
                        "  [%s] event=%s profit=$%.2f roi=%.2f%% legs=%d",
                        opp.type.value, opp.event_id[:12],
                        opp.net_profit, opp.roi_pct, len(opp.legs),
                    )
                _sleep_remaining(cycle_start, cfg.scan_interval_sec, shutdown_requested)
                continue

            # 3. Execute top opportunities
            for opp in all_opps:
                if shutdown_requested:
                    break

                try:
                    _execute_single(client, cfg, opp, pnl, breaker)
                except SafetyCheckFailed as e:
                    logger.warning("Safety check failed, skipping: %s", e)
                    continue
                except CircuitBreakerTripped as e:
                    logger.error("CIRCUIT BREAKER TRIPPED: %s", e)
                    logger.info("Shutting down. Final PnL: %s", pnl.summary())
                    return

        except KeyboardInterrupt:
            break
        except Exception as e:
            logger.error("Cycle %d error: %s", cycle, e, exc_info=True)

        _sleep_remaining(cycle_start, cfg.scan_interval_sec, shutdown_requested)

    # Shutdown
    logger.info("Shutting down gracefully")
    if args.scan_only:
        logger.info("Scan summary: %s", tracker.summary())
    else:
        logger.info("Final PnL summary: %s", pnl.summary())


def _execute_single(
    client,
    cfg: Config,
    opp: Opportunity,
    pnl: PnLTracker,
    breaker: CircuitBreaker,
) -> None:
    """Execute a single opportunity with full safety checks."""
    # Safety checks
    verify_prices_fresh(client, opp)
    verify_depth(client, opp)

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
        logger.info("Position size = 0, skipping opportunity")
        return

    # Execute
    result = execute_opportunity(
        client, opp, size,
        paper_trading=cfg.paper_trading,
    )

    # Record
    pnl.record(result)
    breaker.record_trade(result.net_pnl)


def _sleep_remaining(cycle_start: float, interval: float, shutdown: bool) -> None:
    """Sleep for the remaining scan interval, respecting shutdown."""
    if shutdown:
        return
    elapsed = time.time() - cycle_start
    remaining = interval - elapsed
    if remaining > 0:
        time.sleep(remaining)


if __name__ == "__main__":
    main()
