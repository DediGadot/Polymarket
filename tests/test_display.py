"""
Unit tests for monitor/display.py -- clean console output formatting.
"""

from __future__ import annotations

import argparse
import logging

from monitor.display import (
    print_startup,
    print_cycle_header,
    print_scan_result,
    print_cycle_error,
    print_cycle_footer,
    _truncate,
    _scanner_breakdown,
)
from scanner.models import (
    Opportunity,
    OpportunityType,
    LegOrder,
    Side,
)
from scanner.scorer import ScoredOpportunity
from config import Config


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_opp(
    opp_type=OpportunityType.BINARY_REBALANCE,
    event_id="evt_abc123",
    net_profit=2.14,
    roi_pct=3.41,
    max_sets=100.0,
    num_legs=2,
):
    legs = tuple(
        LegOrder(f"tok_{i}", Side.BUY, 0.45, max_sets)
        for i in range(num_legs)
    )
    return Opportunity(
        type=opp_type,
        event_id=event_id,
        legs=legs,
        expected_profit_per_set=net_profit / max_sets if max_sets > 0 else 0,
        net_profit_per_set=net_profit / max_sets if max_sets > 0 else 0,
        max_sets=max_sets,
        gross_profit=net_profit + 0.01,
        estimated_gas_cost=0.01,
        net_profit=net_profit,
        roi_pct=roi_pct,
        required_capital=net_profit / (roi_pct / 100) if roi_pct > 0 else 1.0,
    )


def _make_scored(opp, total_score=0.72):
    return ScoredOpportunity(
        opportunity=opp,
        total_score=total_score,
        profit_score=0.5,
        fill_score=0.5,
        efficiency_score=0.5,
        urgency_score=0.5,
        competition_score=0.5,
    )


def _default_args(**overrides):
    ns = argparse.Namespace(dry_run=True, scan_only=True, live=False, limit=0)
    for k, v in overrides.items():
        setattr(ns, k, v)
    return ns


def _collect_logs(caplog, func, *args, **kwargs):
    """Run *func* and return list of captured INFO-level messages."""
    with caplog.at_level(logging.INFO, logger="monitor.display"):
        func(*args, **kwargs)
    return [r.message for r in caplog.records if r.name == "monitor.display"]


# ---------------------------------------------------------------------------
# Tests: print_startup
# ---------------------------------------------------------------------------


class TestPrintStartup:
    def test_logs_mode(self, caplog):
        cfg = Config(min_profit_usd=0.50, min_roi_pct=2.0, max_exposure_per_trade=5000)
        args = _default_args(dry_run=True)
        msgs = _collect_logs(caplog, print_startup, cfg, args)
        assert any("DRY-RUN" in m for m in msgs)

    def test_logs_thresholds(self, caplog):
        cfg = Config(min_profit_usd=1.25, min_roi_pct=3.5, max_exposure_per_trade=200)
        args = _default_args()
        msgs = _collect_logs(caplog, print_startup, cfg, args)
        joined = " ".join(msgs)
        assert "$1.25" in joined
        assert "3.5%" in joined
        assert "$200" in joined

    def test_logs_scanner_status(self, caplog):
        cfg = Config(latency_enabled=False, cross_platform_enabled=True)
        args = _default_args()
        msgs = _collect_logs(caplog, print_startup, cfg, args)
        joined = " ".join(msgs)
        assert "~latency~" in joined
        assert "cross" in joined
        assert "~cross~" not in joined

    def test_logs_interval(self, caplog):
        cfg = Config(scan_interval_sec=5.0, use_fak_orders=True, ws_enabled=True)
        args = _default_args()
        msgs = _collect_logs(caplog, print_startup, cfg, args)
        joined = " ".join(msgs)
        assert "5.0s" in joined
        assert "FAK" in joined
        assert "WS: on" in joined

    def test_paper_mode(self, caplog):
        cfg = Config(paper_trading=True)
        args = _default_args(dry_run=False, scan_only=False)
        msgs = _collect_logs(caplog, print_startup, cfg, args)
        assert any("PAPER" in m for m in msgs)

    def test_no_trailing_blank_line(self, caplog):
        cfg = Config()
        args = _default_args()
        msgs = _collect_logs(caplog, print_startup, cfg, args)
        assert msgs[-1].strip() != ""


# ---------------------------------------------------------------------------
# Tests: print_cycle_error
# ---------------------------------------------------------------------------


class TestPrintCycleError:
    def test_shows_error_message(self, caplog):
        err = RuntimeError("connection timeout")
        msgs = _collect_logs(caplog, print_cycle_error, err)
        assert len(msgs) == 1
        assert "connection timeout" in msgs[0]
        assert "\u250c" in msgs[0]  # ┌

    def test_extracts_poly_api_error_message(self, caplog):
        err = Exception("PolyApiException[status_code=None, error_message=Request exception!]")
        msgs = _collect_logs(caplog, print_cycle_error, err)
        assert "Request exception!" in msgs[0]
        # Should NOT contain the verbose wrapper
        assert "status_code" not in msgs[0]

    def test_plain_exception(self, caplog):
        err = ValueError("bad value")
        msgs = _collect_logs(caplog, print_cycle_error, err)
        assert "bad value" in msgs[0]


# ---------------------------------------------------------------------------
# Tests: print_cycle_header
# ---------------------------------------------------------------------------


class TestPrintCycleHeader:
    def test_includes_cycle_number(self, caplog):
        msgs = _collect_logs(caplog, print_cycle_header, 17)
        assert len(msgs) == 1
        assert "Cycle 17" in msgs[0]

    def test_includes_time(self, caplog):
        msgs = _collect_logs(caplog, print_cycle_header, 1)
        # Time is HH:MM:SS format — contains two colons
        assert msgs[0].count(":") >= 2

    def test_uses_box_drawing_dashes(self, caplog):
        msgs = _collect_logs(caplog, print_cycle_header, 5)
        assert "\u2500" in msgs[0]  # ─


# ---------------------------------------------------------------------------
# Tests: print_scan_result -- no opportunities
# ---------------------------------------------------------------------------


class TestScanResultEmpty:
    def test_no_opps_compact(self, caplog):
        msgs = _collect_logs(
            caplog,
            print_scan_result,
            scored_opps=[],
            event_questions={},
            scanner_counts=None,
            scan_elapsed=3.4,
            fetch_elapsed=1.2,
            markets_count=2847,
            binary_count=2691,
            negrisk_event_count=12,
            negrisk_market_count=156,
            strategy_name="CONSERVATIVE",
        )
        joined = "\n".join(msgs)
        assert "2,847" in joined
        assert "No opportunities found" in joined
        assert "CONSERVATIVE" in joined

    def test_no_opps_no_table_rows(self, caplog):
        msgs = _collect_logs(
            caplog,
            print_scan_result,
            scored_opps=[],
            event_questions={},
            scanner_counts=None,
            scan_elapsed=1.0,
            fetch_elapsed=0.5,
            markets_count=100,
            binary_count=80,
            negrisk_event_count=2,
            negrisk_market_count=20,
        )
        # Should not contain table header or mid-box chars beyond the "no opps" line
        joined = "\n".join(msgs)
        assert "Type" not in joined
        assert "Event" not in joined


# ---------------------------------------------------------------------------
# Tests: print_scan_result -- with opportunities
# ---------------------------------------------------------------------------


class TestScanResultWithOpps:
    def _make_three_opps(self):
        opp1 = _make_opp(event_id="evt1", net_profit=2.14, roi_pct=3.41, num_legs=2)
        opp2 = _make_opp(
            opp_type=OpportunityType.NEGRISK_REBALANCE,
            event_id="evt2", net_profit=1.87, roi_pct=1.22, num_legs=8,
        )
        opp3 = _make_opp(event_id="evt3", net_profit=0.53, roi_pct=1.05, num_legs=2)
        return [
            _make_scored(opp1, 0.72),
            _make_scored(opp2, 0.65),
            _make_scored(opp3, 0.41),
        ]

    def test_shows_event_questions(self, caplog):
        scored = self._make_three_opps()
        questions = {
            "evt1": "Will BTC hit $150K in 2026?",
            "evt2": "Super Bowl LXII Winner",
            "evt3": "Will ETH flip BTC market cap?",
        }
        msgs = _collect_logs(
            caplog,
            print_scan_result,
            scored_opps=scored,
            event_questions=questions,
            scanner_counts=None,
            scan_elapsed=2.8,
            fetch_elapsed=1.1,
            markets_count=2847,
            binary_count=2691,
            negrisk_event_count=12,
            negrisk_market_count=156,
            strategy_name="AGGRESSIVE",
        )
        joined = "\n".join(msgs)
        assert "Will BTC hit $150K in 2026?" in joined
        assert "Super Bowl LXII Winner" in joined
        assert "$2.14" in joined
        assert "$1.87" in joined

    def test_scanner_breakdown(self, caplog):
        scored = self._make_three_opps()
        msgs = _collect_logs(
            caplog,
            print_scan_result,
            scored_opps=scored,
            event_questions={},
            scanner_counts=None,
            scan_elapsed=1.0,
            fetch_elapsed=0.5,
            markets_count=100,
            binary_count=80,
            negrisk_event_count=2,
            negrisk_market_count=20,
        )
        joined = "\n".join(msgs)
        # 2 binary + 1 negrisk
        assert "2 binary" in joined
        assert "1 negrisk" in joined

    def test_summary_line(self, caplog):
        scored = self._make_three_opps()
        msgs = _collect_logs(
            caplog,
            print_scan_result,
            scored_opps=scored,
            event_questions={},
            scanner_counts=None,
            scan_elapsed=1.0,
            fetch_elapsed=0.5,
            markets_count=100,
            binary_count=80,
            negrisk_event_count=2,
            negrisk_market_count=20,
        )
        joined = "\n".join(msgs)
        assert "Best:" in joined
        assert "Total capital needed:" in joined

    def test_shows_scores(self, caplog):
        scored = self._make_three_opps()
        msgs = _collect_logs(
            caplog,
            print_scan_result,
            scored_opps=scored,
            event_questions={},
            scanner_counts=None,
            scan_elapsed=1.0,
            fetch_elapsed=0.5,
            markets_count=100,
            binary_count=80,
            negrisk_event_count=2,
            negrisk_market_count=20,
        )
        joined = "\n".join(msgs)
        assert "0.72" in joined
        assert "0.65" in joined

    def test_leg_count_displayed(self, caplog):
        opp = _make_opp(num_legs=8)
        scored = [_make_scored(opp)]
        msgs = _collect_logs(
            caplog,
            print_scan_result,
            scored_opps=scored,
            event_questions={},
            scanner_counts=None,
            scan_elapsed=1.0,
            fetch_elapsed=0.5,
            markets_count=100,
            binary_count=80,
            negrisk_event_count=2,
            negrisk_market_count=20,
        )
        joined = "\n".join(msgs)
        assert "8" in joined


# ---------------------------------------------------------------------------
# Tests: print_cycle_footer
# ---------------------------------------------------------------------------


class TestPrintCycleFooter:
    def test_scan_only_with_best(self, caplog):
        msgs = _collect_logs(
            caplog,
            print_cycle_footer,
            cycle=14,
            cycle_elapsed=3.4,
            total_opps=3,
            total_trades=0,
            total_pnl=0.0,
            best_profit=1.24,
            best_roi=4.12,
            scan_only=True,
        )
        joined = " ".join(msgs)
        assert "14 cycles" in joined
        assert "3 opps" in joined
        assert "$1.24" in joined
        assert "4.12%" in joined
        assert "\u2514" in msgs[0]  # └

    def test_scan_only_no_opps(self, caplog):
        msgs = _collect_logs(
            caplog,
            print_cycle_footer,
            cycle=5,
            cycle_elapsed=2.0,
            total_opps=0,
            total_trades=0,
            total_pnl=0.0,
            best_profit=0.0,
            best_roi=0.0,
            scan_only=True,
        )
        joined = " ".join(msgs)
        assert "5 cycles" in joined
        assert "0 opps" in joined

    def test_trading_mode(self, caplog):
        msgs = _collect_logs(
            caplog,
            print_cycle_footer,
            cycle=17,
            cycle_elapsed=5.0,
            total_opps=6,
            total_trades=3,
            total_pnl=4.20,
            best_profit=2.14,
            best_roi=3.41,
            scan_only=False,
        )
        joined = " ".join(msgs)
        assert "17 cycles" in joined
        assert "P&L $4.20" in joined
        assert "3 trades" in joined


# ---------------------------------------------------------------------------
# Tests: truncation
# ---------------------------------------------------------------------------


class TestTruncation:
    def test_short_string_unchanged(self):
        assert _truncate("hello") == "hello"

    def test_exact_length_unchanged(self):
        text = "a" * 50
        assert _truncate(text) == text

    def test_long_string_truncated(self):
        text = "a" * 60
        result = _truncate(text)
        assert len(result) == 50
        assert result.endswith("\u2026")  # …

    def test_custom_length(self):
        result = _truncate("abcdefghij", length=5)
        assert len(result) == 5
        assert result.endswith("\u2026")


# ---------------------------------------------------------------------------
# Tests: scanner breakdown
# ---------------------------------------------------------------------------


class TestScannerBreakdown:
    def test_mixed_types(self):
        opps = [
            _make_scored(_make_opp(opp_type=OpportunityType.BINARY_REBALANCE)),
            _make_scored(_make_opp(opp_type=OpportunityType.BINARY_REBALANCE)),
            _make_scored(_make_opp(opp_type=OpportunityType.NEGRISK_REBALANCE)),
        ]
        result = _scanner_breakdown(opps)
        assert "2 binary" in result
        assert "1 negrisk" in result

    def test_single_type(self):
        opps = [
            _make_scored(_make_opp(opp_type=OpportunityType.LATENCY_ARB)),
        ]
        result = _scanner_breakdown(opps)
        assert "1 latency" in result

    def test_empty(self):
        result = _scanner_breakdown([])
        assert result == "()"

    def test_all_types(self):
        opps = [
            _make_scored(_make_opp(opp_type=OpportunityType.BINARY_REBALANCE)),
            _make_scored(_make_opp(opp_type=OpportunityType.NEGRISK_REBALANCE)),
            _make_scored(_make_opp(opp_type=OpportunityType.LATENCY_ARB)),
            _make_scored(_make_opp(opp_type=OpportunityType.SPIKE_LAG)),
            _make_scored(_make_opp(opp_type=OpportunityType.CROSS_PLATFORM_ARB)),
        ]
        result = _scanner_breakdown(opps)
        for name in ["binary", "negrisk", "latency", "spike", "cross"]:
            assert name in result
