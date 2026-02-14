"""
Clean, scannable console output for the arbitrage bot.

Pure formatting functions that emit structured log lines using box-drawing
characters. No side effects beyond logging. All data arrives via arguments.
"""

from __future__ import annotations

import argparse
import logging
import time
from collections import Counter

from config import Config, active_platforms
from scanner.models import OpportunityType
from scanner.scorer import ScoredOpportunity

logger = logging.getLogger(__name__)

# Box-drawing characters
_TOP = "\u250c"  # ┌
_MID = "\u2502"  # │
_BOT = "\u2514"  # └
_DASH = "\u2500"  # ─
_VERT_SEP = "\u2502"  # │ (inline separator)

_MAX_QUESTION_LEN = 50


def _truncate(text: str, length: int = _MAX_QUESTION_LEN) -> str:
    """Truncate text to *length* chars, appending ellipsis if trimmed."""
    if len(text) <= length:
        return text
    return text[: length - 1] + "\u2026"


def _type_label(opp_type: OpportunityType) -> str:
    """Short human-readable label for an opportunity type."""
    return opp_type.value


def _scanner_breakdown(scored_opps: list[ScoredOpportunity]) -> str:
    """Return e.g. '(2 binary, 1 negrisk)' from a list of scored opps."""
    counts: Counter[str] = Counter()
    for s in scored_opps:
        raw = s.opportunity.type.value  # e.g. "binary_rebalance"
        scanner_name = raw.split("_")[0]  # e.g. "binary"
        counts[scanner_name] += 1
    parts = [f"{v} {k}" for k, v in counts.most_common()]
    return f"({', '.join(parts)})"


def _mode_label(args: argparse.Namespace, cfg: Config) -> str:
    if getattr(args, "dry_run", False):
        return "DRY-RUN"
    if getattr(args, "scan_only", False):
        return "SCAN-ONLY"
    if cfg.paper_trading:
        return "PAPER"
    return "LIVE"


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def print_startup(cfg: Config, args: argparse.Namespace) -> None:
    """Compact 4-line config block emitted once after the banner."""
    mode = _mode_label(args, cfg)

    scanners = []
    scanners.append("binary")
    scanners.append("negrisk")
    scanners.append("latency" if cfg.latency_enabled else "~latency~")
    scanners.append("spike")
    scanners.append("cross" if cfg.cross_platform_enabled else "~cross~")
    scanner_str = "  ".join(scanners)

    logger.info(
        "  Mode: %-12s Profit >= $%.2f  ROI >= %.1f%%  Exposure <= $%.0f",
        mode, cfg.min_profit_usd, cfg.min_roi_pct, cfg.max_exposure_per_trade,
    )
    logger.info("  Scanners: %s", scanner_str)
    platforms = ["polymarket"] + active_platforms(cfg)
    platform_str = "  ".join(platforms)
    logger.info("  Platforms: %s", platform_str)
    logger.info(
        "  Interval: %.1fs  Order: %s  WS: %s",
        cfg.scan_interval_sec,
        "FAK" if cfg.use_fak_orders else "GTC",
        "on" if cfg.ws_enabled else "off",
    )


def print_cycle_header(cycle: int) -> None:
    """Emit a horizontal divider with cycle number and wall-clock time."""
    ts = time.strftime("%H:%M:%S")
    label = f" Cycle {cycle} "
    # Pad to ~60 chars total
    left_dashes = _DASH * 2
    right_pad = 60 - len(left_dashes) - len(label) - len(ts) - 3
    if right_pad < 2:
        right_pad = 2
    line = f"{left_dashes}{label}{_DASH * right_pad} {ts} {_DASH * 2}"
    logger.info(line)


def print_scan_result(
    scored_opps: list[ScoredOpportunity],
    event_questions: dict[str, str],
    scanner_counts: dict[str, int] | None,
    scan_elapsed: float,
    fetch_elapsed: float,
    markets_count: int,
    binary_count: int,
    negrisk_event_count: int,
    negrisk_market_count: int,
    strategy_name: str = "",
) -> None:
    """
    Emit the boxed scan summary.

    When *scored_opps* is empty: compact 3-line box.
    When opportunities exist: full table with event names.
    """
    # Market fetch line
    logger.info(
        "  Fetched %s markets (%s binary, %s negrisk in %d events) in %.1fs",
        f"{markets_count:,}",
        f"{binary_count:,}",
        f"{negrisk_market_count:,}",
        negrisk_event_count,
        fetch_elapsed,
    )

    # Scan timing + strategy
    strategy_tag = f"  [strategy={strategy_name}]" if strategy_name else ""
    logger.info("  Scanned in %.1fs%s", scan_elapsed, strategy_tag)

    if not scored_opps:
        logger.info("  %s No opportunities found", _TOP)
        return

    # --- Opportunities found ---
    n = len(scored_opps)
    breakdown = _scanner_breakdown(scored_opps)
    logger.info("  %s %d opportunit%s found %s", _TOP, n, "y" if n == 1 else "ies", breakdown)
    logger.info("  %s", _MID)

    # Table header
    logger.info(
        "  %s  %-3s %-18s %-50s %8s %7s %7s %4s",
        _MID, "#", "Type", "Event", "Profit", "ROI", "Score", "Legs",
    )

    # Table rows
    for idx, scored in enumerate(scored_opps, 1):
        opp = scored.opportunity
        question = event_questions.get(opp.event_id, opp.event_id[:14])
        question = _truncate(question)
        logger.info(
            "  %s  %-3d %-18s %-50s %7s %6.2f%% %7.2f %4d",
            _MID,
            idx,
            _type_label(opp.type),
            question,
            f"${opp.net_profit:.2f}",
            opp.roi_pct,
            scored.total_score,
            len(opp.legs),
        )

    logger.info("  %s", _MID)

    # Summary line
    best_profit = max(s.opportunity.net_profit for s in scored_opps)
    best_roi = max(s.opportunity.roi_pct for s in scored_opps)
    total_capital = sum(s.opportunity.required_capital for s in scored_opps)
    logger.info(
        "  %s  Best: $%.2f profit, %.2f%% ROI %s Total capital needed: $%.2f",
        _MID, best_profit, best_roi, _VERT_SEP, total_capital,
    )


def print_cycle_error(error: Exception) -> None:
    """Emit a compact error box when a cycle fails before scan completes."""
    msg = str(error)
    # Shorten common verbose exception wrappers
    if "error_message=" in msg:
        # e.g. PolyApiException[status_code=None, error_message=Request exception!]
        start = msg.find("error_message=")
        end = msg.find("]", start)
        if start != -1 and end != -1:
            msg = msg[start + len("error_message="):end]
    logger.info("  %s API error: %s", _TOP, msg)


def print_cycle_footer(
    cycle: int,
    cycle_elapsed: float,
    total_opps: int,
    total_trades: int,
    total_pnl: float,
    best_profit: float,
    best_roi: float,
    scan_only: bool,
) -> None:
    """
    Close the box with session-level totals.

    *scan_only* mode shows best-ever stats; trading mode shows P&L.
    """
    if scan_only:
        if best_profit > 0:
            logger.info(
                "  %s Session: %d cycles %s %d opps total %s best $%.2f ROI %.2f%%",
                _BOT, cycle, _VERT_SEP, total_opps, _VERT_SEP, best_profit, best_roi,
            )
        else:
            logger.info(
                "  %s Session: %d cycles %s %d opps total",
                _BOT, cycle, _VERT_SEP, total_opps,
            )
    else:
        logger.info(
            "  %s Session: %d cycles %s %d opps total %s P&L $%.2f (%d trades)",
            _BOT, cycle, _VERT_SEP, total_opps, _VERT_SEP, total_pnl, total_trades,
        )
