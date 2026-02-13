"""
Unit tests for benchmark/cross_platform.py -- cross-platform EVS breakdown.
"""

from __future__ import annotations

import json
import tempfile
from pathlib import Path

from benchmark.cross_platform import (
    LabeledPair,
    PlatformBreakdown,
    ThresholdResult,
    compute_platform_breakdown,
    evaluate_thresholds,
    main as cp_main,
)
from benchmark.evs import CycleParsed, ParsedOpportunity


# ── Helpers (reuse pattern from test_evs.py) ─────────────────────────────

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


def _make_opp(
    opp_type: str = "negrisk_rebalance",
    event_id: str = "aaa",
    profit: float = 5.0,
    roi_pct: float = 10.0,
    legs: int = 2,
    capital: float = 100.0,
    platform: str = "polymarket",
) -> ParsedOpportunity:
    return ParsedOpportunity(
        opp_type=opp_type,
        event_id=event_id,
        profit=profit,
        roi_pct=roi_pct,
        legs=legs,
        capital=capital,
        platform=platform,
    )


def _make_cycle(
    cycle: int,
    opps: tuple[ParsedOpportunity, ...],
    latency: float = 5.0,
) -> CycleParsed:
    return CycleParsed(cycle=cycle, opportunities=opps, scan_latency_sec=latency)


# ── Tests: compute_platform_breakdown ────────────────────────────────────

class TestComputePlatformBreakdown:
    def test_empty_cycles(self):
        """Empty cycles → zero EVS for all platforms."""
        bd = compute_platform_breakdown([], target_size=100.0)
        assert bd.total_evs == 0.0
        assert bd.pm_only_evs == 0.0
        assert bd.cross_platform_evs == 0.0
        assert bd.delta == 0.0
        assert bd.total_arb_count == 0
        assert bd.pm_arb_count == 0
        assert bd.cross_platform_arb_count == 0

    def test_pm_only_cycles(self):
        """PM-only opps → cross_platform_evs = 0, pm_only_evs = total_evs."""
        opp_pm = _make_opp(opp_type="negrisk_rebalance", platform="polymarket", capital=150.0)
        cycles = [_make_cycle(1, (opp_pm,))]
        bd = compute_platform_breakdown(cycles, target_size=100.0)
        assert bd.cross_platform_evs == 0.0
        assert bd.pm_only_evs == bd.total_evs
        assert bd.total_evs > 0.0
        assert bd.delta == 0.0
        assert bd.pm_arb_count == 1
        assert bd.cross_platform_arb_count == 0

    def test_mixed_platform_cycles(self):
        """Correct split between PM-only and cross-platform."""
        opp_pm = _make_opp(
            opp_type="negrisk_rebalance", platform="polymarket",
            roi_pct=10.0, capital=100.0, legs=2,
        )
        opp_xp = _make_opp(
            opp_type="cross_platform_arb", platform="kalshi",
            roi_pct=8.0, capital=100.0, legs=2,
        )
        cycles = [_make_cycle(1, (opp_pm, opp_xp))]
        bd = compute_platform_breakdown(cycles, target_size=100.0)

        assert bd.total_evs > 0.0
        assert bd.pm_only_evs > 0.0
        assert bd.cross_platform_evs > 0.0
        assert abs(bd.total_evs - (bd.pm_only_evs + bd.cross_platform_evs)) < 1e-9
        assert bd.delta == bd.cross_platform_evs
        assert bd.pm_arb_count == 1
        assert bd.cross_platform_arb_count == 1
        assert bd.total_arb_count == 2

    def test_multi_cycle_aggregation(self):
        """EVS is aggregated across multiple cycles."""
        opp1 = _make_opp(opp_type="binary_rebalance", platform="polymarket", roi_pct=5.0, capital=100.0)
        opp2 = _make_opp(opp_type="cross_platform_arb", platform="kalshi", roi_pct=12.0, capital=200.0)
        cycles = [
            _make_cycle(1, (opp1,)),
            _make_cycle(2, (opp2,)),
        ]
        bd = compute_platform_breakdown(cycles, target_size=100.0)
        assert bd.total_arb_count == 2
        assert bd.pm_arb_count == 1
        assert bd.cross_platform_arb_count == 1
        assert bd.total_evs > 0.0

    def test_cross_platform_only(self):
        """Only cross-platform opps → pm_only_evs = 0."""
        opp = _make_opp(opp_type="cross_platform_arb", platform="kalshi", roi_pct=10.0, capital=100.0)
        cycles = [_make_cycle(1, (opp,))]
        bd = compute_platform_breakdown(cycles, target_size=100.0)
        assert bd.pm_only_evs == 0.0
        assert bd.cross_platform_evs == bd.total_evs
        assert bd.cross_platform_evs > 0.0
        # delta is cross_platform_evs when pm_only_evs is 0
        assert bd.delta == bd.cross_platform_evs


# ── Tests: evaluate_thresholds ───────────────────────────────────────────

class TestEvaluateThresholds:
    def test_threshold_evaluation(self):
        """Correct precision/recall at different thresholds."""
        labeled = [
            LabeledPair(pm_title="Will Biden win the election?", kalshi_title="Biden wins election", is_match=True),
            LabeledPair(pm_title="Will Biden win the election?", kalshi_title="Trump wins election", is_match=False),
            LabeledPair(pm_title="Bitcoin above 100k by March?", kalshi_title="BTC over 100k March", is_match=True),
        ]
        thresholds = [50.0, 80.0, 95.0]
        results = evaluate_thresholds(labeled, thresholds)
        assert len(results) == 3
        for r in results:
            assert 0.0 <= r.precision <= 1.0
            assert 0.0 <= r.recall <= 1.0
            assert 0.0 <= r.f1_score <= 1.0

    def test_perfect_matches(self):
        """Identical strings → precision=1.0, recall=1.0 at low threshold."""
        labeled = [
            LabeledPair(pm_title="exact match test", kalshi_title="exact match test", is_match=True),
            LabeledPair(pm_title="something else", kalshi_title="completely different", is_match=False),
        ]
        results = evaluate_thresholds(labeled, [50.0])
        assert len(results) == 1
        r = results[0]
        # "exact match test" == "exact match test" → score 100 >= 50 → predicted match
        # "something else" vs "completely different" → score low → predicted no match
        assert r.precision == 1.0
        assert r.recall == 1.0
        assert r.f1_score == 1.0
        assert r.true_positives == 1
        assert r.false_positives == 0
        assert r.matches_found == 1

    def test_empty_labeled_pairs(self):
        """No labeled pairs → zeros."""
        results = evaluate_thresholds([], [50.0, 80.0])
        assert len(results) == 2
        for r in results:
            assert r.matches_found == 0
            assert r.precision == 0.0
            assert r.recall == 0.0
            assert r.f1_score == 0.0

    def test_high_threshold_rejects_fuzzy(self):
        """High threshold rejects imperfect matches → recall drops."""
        labeled = [
            LabeledPair(
                pm_title="Will the S&P 500 close above 5000?",
                kalshi_title="S&P 500 above 5000 at close",
                is_match=True,
            ),
        ]
        results_low = evaluate_thresholds(labeled, [50.0])
        results_high = evaluate_thresholds(labeled, [99.9])
        # At 50 threshold, fuzzy should match → recall=1
        assert results_low[0].recall == 1.0
        # At 99.9 threshold, likely rejected → recall=0
        assert results_high[0].recall == 0.0


# ── Tests: CLI end-to-end ────────────────────────────────────────────────

class TestCLI:
    def test_cli_end_to_end(self, tmp_path):
        """Write NDJSON → run main() → check JSON output."""
        input_path = tmp_path / "dry_run.jsonl"
        output_path = tmp_path / "cross_platform_report.json"

        lines = [
            _cycle_marker(1),
            _found_line(2, best_profit=8.0),
            _opp_line(1, opp_type="negrisk_rebalance", profit=8.0, roi=6.63, legs=2, capital=130.0),
            _opp_line(2, opp_type="cross_platform_arb", profit=3.0, roi=8.0, legs=2, capital=100.0),
            _cycle_complete(1, elapsed=8.0, opps=2),
            _cycle_marker(2),
            _cycle_complete(2, elapsed=10.0),
        ]
        _write_ndjson(lines, input_path)

        cp_main(["--input", str(input_path), "--output", str(output_path)])

        with open(output_path) as f:
            data = json.load(f)

        assert "platform_breakdown" in data
        bd = data["platform_breakdown"]
        assert bd["total_arb_count"] == 2
        assert bd["pm_arb_count"] == 1
        assert bd["cross_platform_arb_count"] == 1
        assert bd["total_evs"] > 0
        assert bd["pm_only_evs"] > 0
        assert bd["cross_platform_evs"] > 0
        assert abs(bd["total_evs"] - (bd["pm_only_evs"] + bd["cross_platform_evs"])) < 1e-9

    def test_cli_with_target_size(self, tmp_path):
        """Custom target-size flag is respected."""
        input_path = tmp_path / "dry_run.jsonl"
        output_path = tmp_path / "report.json"

        lines = [
            _cycle_marker(1),
            _found_line(1),
            _opp_line(1, profit=5.0, roi=10.0, legs=2, capital=50.0),
            _cycle_complete(1, elapsed=5.0, opps=1),
        ]
        _write_ndjson(lines, input_path)

        # target_size=50 → depth ratio=1.0
        cp_main(["--input", str(input_path), "--output", str(output_path), "--target-size", "50"])
        with open(output_path) as f:
            data_50 = json.load(f)

        # target_size=200 → depth ratio=0.25
        cp_main(["--input", str(input_path), "--output", str(output_path), "--target-size", "200"])
        with open(output_path) as f:
            data_200 = json.load(f)

        assert data_50["platform_breakdown"]["total_evs"] > data_200["platform_breakdown"]["total_evs"]
