"""
Unit tests for benchmark/evs.py -- EVS metric computation from dry-run logs.
"""

from __future__ import annotations

import json
import tempfile
from pathlib import Path

from benchmark.evs import (
    CycleMetrics,
    ParsedOpportunity,
    SessionReport,
    compute_evs,
    parse_log_lines,
    build_session_report,
    classify_confidence,
    main as evs_main,
)


# ── Helpers ──────────────────────────────────────────────────────────────

def _opp_line(
    rank: int = 1,
    opp_type: str = "negrisk_rebalance",
    event_id: str = "12345",
    profit: float = 5.0,
    roi: float = 10.0,
    legs: int = 2,
    capital: float = 50.0,
) -> dict:
    """Build a single opportunity log line (JSON dict as written to NDJSON)."""
    msg = (
        f"  #{rank:<3d} {opp_type:<25s}  "
        f"event={event_id:<14s}  "
        f"profit=${profit:.2f}  roi={roi:.2f}%  "
        f"legs={legs}  capital=${capital:.2f}"
    )
    return {"ts": "2026-02-09T05:58:32", "level": "INFO", "module": "run", "msg": msg}


def _cycle_marker(cycle: int) -> dict:
    return {
        "ts": "2026-02-09T05:58:00",
        "level": "INFO",
        "module": "run",
        "msg": f"\u2500\u2500 Cycle {cycle} " + "\u2500" * 46,
    }


def _cycle_complete(cycle: int, elapsed: float = 8.0, opps: int = 0) -> dict:
    if opps > 0:
        msg = f"      Cycle {cycle} complete in {elapsed:.1f}s  |  session: {cycle} cycles, {opps} opps found"
    else:
        msg = f"      Cycle {cycle} complete in {elapsed:.1f}s  |  session: 0 trades, P&L $0.00"
    return {"ts": "2026-02-09T05:58:08", "level": "INFO", "module": "run", "msg": msg}


def _found_line(count: int, elapsed: float = 4.0, best_score: float = 0.49, best_profit: float = 5.0) -> dict:
    msg = f"      Found {count} opportunities in {elapsed:.1f}s (best score: {best_score:.2f}, best profit: ${best_profit:.2f})"
    return {"ts": "2026-02-09T05:58:05", "level": "INFO", "module": "run", "msg": msg}


def _write_ndjson(lines: list[dict], path: Path) -> None:
    with open(path, "w") as f:
        for line in lines:
            f.write(json.dumps(line) + "\n")


# ── Tests: parse_log_lines ──────────────────────────────────────────────

class TestParseLogLines:
    def test_empty_input(self):
        result = parse_log_lines([])
        assert result == []

    def test_single_cycle_no_opps(self):
        lines = [
            _cycle_marker(1),
            _cycle_complete(1, elapsed=10.0),
        ]
        result = parse_log_lines(lines)
        assert len(result) == 1
        assert result[0].cycle == 1
        assert result[0].opportunities == ()
        assert result[0].scan_latency_sec == 10.0

    def test_single_cycle_with_opps(self):
        lines = [
            _cycle_marker(1),
            _found_line(2, elapsed=4.0, best_profit=8.64),
            _opp_line(1, profit=8.64, roi=6.63, legs=2, capital=130.41),
            _opp_line(2, profit=1.97, roi=7.63, legs=2, capital=25.84),
            _cycle_complete(1, elapsed=8.2, opps=2),
        ]
        result = parse_log_lines(lines)
        assert len(result) == 1
        cycle = result[0]
        assert cycle.cycle == 1
        assert len(cycle.opportunities) == 2
        assert cycle.opportunities[0].profit == 8.64
        assert cycle.opportunities[1].roi_pct == 7.63

    def test_multi_cycle(self):
        lines = [
            _cycle_marker(1),
            _cycle_complete(1, elapsed=10.0),
            _cycle_marker(2),
            _found_line(1, best_profit=5.0),
            _opp_line(1, profit=5.0, roi=10.0),
            _cycle_complete(2, elapsed=8.0, opps=1),
        ]
        result = parse_log_lines(lines)
        assert len(result) == 2
        assert result[0].opportunities == ()
        assert len(result[1].opportunities) == 1


# ── Tests: classify_confidence ──────────────────────────────────────────

class TestClassifyConfidence:
    def test_deep_book_many_legs(self):
        """Multi-leg with high capital → known (1.0)."""
        c = classify_confidence(legs=6, capital=200.0, target_size=100.0)
        assert c == 1.0

    def test_first_seen_deep(self):
        """2-leg with capital >= target_size → first-seen deep (0.7)."""
        c = classify_confidence(legs=2, capital=130.0, target_size=100.0)
        assert c == 0.7

    def test_thin_book(self):
        """Capital < target_size → thin (0.3)."""
        c = classify_confidence(legs=2, capital=10.0, target_size=100.0)
        assert c == 0.3

    def test_zero_capital(self):
        c = classify_confidence(legs=2, capital=0.0, target_size=100.0)
        assert c == 0.3


# ── Tests: compute_evs ─────────────────────────────────────────────────

class TestComputeEvs:
    def test_empty_opps(self):
        evs = compute_evs((), target_size=100.0)
        assert evs == 0.0

    def test_single_opp(self):
        opp = ParsedOpportunity(
            opp_type="negrisk_rebalance",
            event_id="12345",
            profit=5.0,
            roi_pct=10.0,
            legs=2,
            capital=130.0,
            platform="polymarket",
        )
        evs = compute_evs((opp,), target_size=100.0)
        # C=0.7 (first-seen deep), R=min(10/100, 1)=0.1, D=min(130/100, 1)=1.0
        expected = 0.7 * 0.1 * 1.0
        assert abs(evs - expected) < 1e-9

    def test_multi_opps(self):
        opp1 = ParsedOpportunity(
            opp_type="negrisk_rebalance",
            event_id="aaa",
            profit=8.64,
            roi_pct=6.63,
            legs=2,
            capital=130.41,
            platform="polymarket",
        )
        opp2 = ParsedOpportunity(
            opp_type="negrisk_rebalance",
            event_id="bbb",
            profit=1.97,
            roi_pct=29.27,
            legs=2,
            capital=7.96,
            platform="polymarket",
        )
        evs = compute_evs((opp1, opp2), target_size=100.0)
        # opp1: C=0.7, R=0.0663, D=1.0 → 0.04641
        # opp2: C=0.3 (thin), R=0.2927, D=0.0796 → 0.006989...
        c1, r1, d1 = 0.7, min(6.63 / 100.0, 1.0), min(130.41 / 100.0, 1.0)
        c2, r2, d2 = 0.3, min(29.27 / 100.0, 1.0), min(7.96 / 100.0, 1.0)
        expected = c1 * r1 * d1 + c2 * r2 * d2
        assert abs(evs - expected) < 1e-9

    def test_negative_roi_clamped(self):
        opp = ParsedOpportunity(
            opp_type="binary_rebalance",
            event_id="neg",
            profit=-1.0,
            roi_pct=-5.0,
            legs=2,
            capital=50.0,
            platform="polymarket",
        )
        evs = compute_evs((opp,), target_size=100.0)
        assert evs == 0.0

    def test_zero_depth(self):
        opp = ParsedOpportunity(
            opp_type="binary_rebalance",
            event_id="zero",
            profit=5.0,
            roi_pct=10.0,
            legs=2,
            capital=0.0,
            platform="polymarket",
        )
        evs = compute_evs((opp,), target_size=100.0)
        assert evs == 0.0

    def test_roi_capped_at_one(self):
        opp = ParsedOpportunity(
            opp_type="negrisk_rebalance",
            event_id="big",
            profit=500.0,
            roi_pct=200.0,
            legs=6,
            capital=500.0,
            platform="polymarket",
        )
        evs = compute_evs((opp,), target_size=100.0)
        # C=1.0 (known: many legs + deep), R=min(200/100, 1)=1.0, D=min(500/100, 1)=1.0
        assert abs(evs - 1.0) < 1e-9


# ── Tests: build_session_report ─────────────────────────────────────────

class TestBuildSessionReport:
    def test_empty_cycles(self):
        report = build_session_report((), target_size=100.0)
        assert report.total_cycles == 0
        assert report.overall_evs == 0.0
        assert report.cycle_metrics == ()

    def test_dashboard_metrics(self):
        opp1 = ParsedOpportunity("negrisk_rebalance", "aaa", 8.64, 6.63, 2, 130.41, "polymarket")
        opp2 = ParsedOpportunity("negrisk_rebalance", "bbb", 1.97, 7.63, 2, 25.84, "polymarket")
        from benchmark.evs import CycleParsed
        cycles = (
            CycleParsed(cycle=1, opportunities=(), scan_latency_sec=10.0),
            CycleParsed(cycle=2, opportunities=(opp1, opp2), scan_latency_sec=8.2),
        )
        report = build_session_report(cycles, target_size=100.0)
        assert report.total_cycles == 2
        assert len(report.cycle_metrics) == 2

        # Cycle 1: no opps → all zeros
        m1 = report.cycle_metrics[0]
        assert m1.evs == 0.0
        assert m1.arb_count == 0
        assert m1.scan_latency_sec == 10.0

        # Cycle 2: 2 opps
        m2 = report.cycle_metrics[1]
        assert m2.arb_count == 2
        assert m2.mean_roi > 0
        assert m2.mean_confidence > 0
        assert m2.depth_weighted_profit > 0
        assert m2.scan_latency_sec == 8.2
        assert m2.hit_rate == 1.0  # 1 cycle with hits / 1 cycle so far... actually per-cycle hit_rate
        assert m2.platform_count == 1  # only polymarket

    def test_cross_platform_count(self):
        opp_pm = ParsedOpportunity("negrisk_rebalance", "aaa", 5.0, 10.0, 2, 100.0, "polymarket")
        opp_k = ParsedOpportunity("cross_platform_arb", "bbb", 3.0, 8.0, 2, 100.0, "kalshi")
        from benchmark.evs import CycleParsed
        cycles = (
            CycleParsed(cycle=1, opportunities=(opp_pm, opp_k), scan_latency_sec=5.0),
        )
        report = build_session_report(cycles, target_size=100.0)
        assert report.cycle_metrics[0].platform_count == 2


# ── Tests: JSON output format ───────────────────────────────────────────

class TestJsonOutput:
    def test_roundtrip(self, tmp_path):
        opp = ParsedOpportunity("negrisk_rebalance", "12345", 5.0, 10.0, 2, 100.0, "polymarket")
        from benchmark.evs import CycleParsed
        cycles = (
            CycleParsed(cycle=1, opportunities=(opp,), scan_latency_sec=8.0),
        )
        report = build_session_report(cycles, target_size=100.0)
        out_path = tmp_path / "evs_output.json"
        report.to_json(out_path)

        with open(out_path) as f:
            data = json.load(f)

        assert "total_cycles" in data
        assert "overall_evs" in data
        assert "cycles" in data
        assert len(data["cycles"]) == 1
        assert "evs" in data["cycles"][0]
        assert "arb_count" in data["cycles"][0]
        assert "mean_confidence" in data["cycles"][0]
        assert "mean_roi" in data["cycles"][0]
        assert "depth_weighted_profit" in data["cycles"][0]
        assert "scan_latency_sec" in data["cycles"][0]
        assert "hit_rate" in data["cycles"][0]
        assert "platform_count" in data["cycles"][0]


# ── Tests: CLI arg parsing ──────────────────────────────────────────────

class TestCLI:
    def test_end_to_end(self, tmp_path):
        """Full pipeline: write NDJSON → run CLI → check JSON output."""
        input_path = tmp_path / "dry_run.jsonl"
        output_path = tmp_path / "evs_out.json"

        lines = [
            _cycle_marker(1),
            _found_line(1, best_profit=5.0),
            _opp_line(1, profit=5.0, roi=10.0, legs=2, capital=130.0),
            _cycle_complete(1, elapsed=8.0, opps=1),
            _cycle_marker(2),
            _cycle_complete(2, elapsed=10.0),
        ]
        _write_ndjson(lines, input_path)

        evs_main(["--input", str(input_path), "--output", str(output_path), "--target-size", "100"])

        with open(output_path) as f:
            data = json.load(f)

        assert data["total_cycles"] == 2
        assert data["overall_evs"] > 0
        assert len(data["cycles"]) == 2
        assert data["cycles"][0]["arb_count"] == 1
        assert data["cycles"][1]["arb_count"] == 0

    def test_custom_target_size(self, tmp_path):
        input_path = tmp_path / "dry_run.jsonl"
        output_path = tmp_path / "evs_out.json"

        lines = [
            _cycle_marker(1),
            _found_line(1, best_profit=5.0),
            _opp_line(1, profit=5.0, roi=10.0, legs=2, capital=50.0),
            _cycle_complete(1, elapsed=8.0, opps=1),
        ]
        _write_ndjson(lines, input_path)

        # With target_size=50, capital=50 → D=1.0 instead of 0.5
        evs_main(["--input", str(input_path), "--output", str(output_path), "--target-size", "50"])

        with open(output_path) as f:
            data = json.load(f)

        evs_50 = data["overall_evs"]

        # Rerun with target_size=200, capital=50 → D=0.25
        evs_main(["--input", str(input_path), "--output", str(output_path), "--target-size", "200"])

        with open(output_path) as f:
            data = json.load(f)

        evs_200 = data["overall_evs"]
        assert evs_50 > evs_200  # smaller target → higher depth ratio → higher EVS
