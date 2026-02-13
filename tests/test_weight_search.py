"""
Unit tests for benchmark/weight_search.py -- scorer weight grid search.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from benchmark.weight_search import (
    WeightConfig,
    WeightResult,
    generate_weight_grid,
    evaluate_weights,
    main as ws_main,
)
from scanner.models import (
    LegOrder,
    Opportunity,
    OpportunityType,
    Side,
)
from scanner.scorer import ScoringContext


# ── Helpers ──────────────────────────────────────────────────────────────

def _make_opp(
    opp_type: OpportunityType = OpportunityType.BINARY_REBALANCE,
    net_profit: float = 5.0,
    roi_pct: float = 10.0,
    required_capital: float = 100.0,
    num_legs: int = 2,
) -> Opportunity:
    legs = tuple(
        LegOrder(token_id=f"tok_{i}", side=Side.BUY, price=0.50, size=10.0)
        for i in range(num_legs)
    )
    return Opportunity(
        type=opp_type,
        event_id="evt_test",
        legs=legs,
        expected_profit_per_set=net_profit + 1.0,
        net_profit_per_set=net_profit / max(1, num_legs),
        max_sets=float(num_legs),
        gross_profit=net_profit + 1.0,
        estimated_gas_cost=0.50,
        net_profit=net_profit,
        roi_pct=roi_pct,
        required_capital=required_capital,
    )


def _make_ctx(
    book_depth_ratio: float = 1.0,
    confidence: float = 0.5,
    is_spike: bool = False,
    recent_trade_count: int = 0,
) -> ScoringContext:
    return ScoringContext(
        book_depth_ratio=book_depth_ratio,
        confidence=confidence,
        is_spike=is_spike,
        recent_trade_count=recent_trade_count,
    )


def _opp_line(
    rank: int = 1,
    opp_type: str = "negrisk_rebalance",
    event_id: str = "12345",
    profit: float = 5.0,
    roi: float = 10.0,
    legs: int = 2,
    capital: float = 50.0,
) -> dict:
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
    msg = f"      Cycle {cycle} complete in {elapsed:.1f}s  |  session: {cycle} cycles, {opps} opps found"
    return {"ts": "2026-02-09T05:58:08", "level": "INFO", "module": "run", "msg": msg}


def _write_ndjson(lines: list[dict], path: Path) -> None:
    with open(path, "w") as f:
        for line in lines:
            f.write(json.dumps(line) + "\n")


# ── Tests: WeightConfig ─────────────────────────────────────────────────

class TestWeightConfig:
    def test_as_tuple_order(self):
        cfg = WeightConfig(
            profit=0.20, fill=0.20, efficiency=0.15,
            urgency=0.15, competition=0.15, persistence=0.15,
        )
        assert cfg.as_tuple() == (0.20, 0.20, 0.15, 0.15, 0.15, 0.15)

    def test_frozen(self):
        cfg = WeightConfig(
            profit=0.20, fill=0.20, efficiency=0.15,
            urgency=0.15, competition=0.15, persistence=0.15,
        )
        with pytest.raises(AttributeError):
            cfg.profit = 0.30


# ── Tests: generate_weight_grid ──────────────────────────────────────────

class TestGenerateWeightGrid:
    def test_nonempty(self):
        grid = generate_weight_grid()
        assert len(grid) > 0

    def test_all_sum_to_one(self):
        grid = generate_weight_grid()
        for cfg in grid:
            total = sum(cfg.as_tuple())
            assert abs(total - 1.0) < 1e-9, f"Weights sum to {total}, expected 1.0"

    def test_bounds_default(self):
        grid = generate_weight_grid(min_w=0.05, max_w=0.40)
        for cfg in grid:
            for w in cfg.as_tuple():
                assert w >= 0.05 - 1e-9, f"Weight {w} below minimum 0.05"
                assert w <= 0.40 + 1e-9, f"Weight {w} above maximum 0.40"

    def test_each_weight_within_bounds(self):
        """Every individual weight in every config respects [min_w, max_w]."""
        grid = generate_weight_grid(step=0.10, min_w=0.10, max_w=0.40)
        for cfg in grid:
            assert cfg.profit >= 0.10 - 1e-9
            assert cfg.fill >= 0.10 - 1e-9
            assert cfg.efficiency >= 0.10 - 1e-9
            assert cfg.urgency >= 0.10 - 1e-9
            assert cfg.competition >= 0.10 - 1e-9
            assert cfg.persistence >= 0.10 - 1e-9
            assert cfg.profit <= 0.40 + 1e-9
            assert cfg.persistence <= 0.40 + 1e-9

    def test_finer_step_more_combos(self):
        grid_fine = generate_weight_grid(step=0.05)
        grid_coarse = generate_weight_grid(step=0.10)
        assert len(grid_fine) > len(grid_coarse)

    def test_no_duplicates(self):
        grid = generate_weight_grid(step=0.10, min_w=0.10, max_w=0.40)
        tuples = [cfg.as_tuple() for cfg in grid]
        assert len(tuples) == len(set(tuples))


# ── Tests: evaluate_weights ─────────────────────────────────────────────

class TestEvaluateWeights:
    def test_sorted_by_evs_descending(self):
        opps = [
            _make_opp(net_profit=10.0, roi_pct=15.0, required_capital=200.0),
            _make_opp(net_profit=2.0, roi_pct=5.0, required_capital=40.0),
        ]
        ctxs = [
            _make_ctx(book_depth_ratio=1.5, confidence=0.7),
            _make_ctx(book_depth_ratio=0.5, confidence=0.3),
        ]
        configs = generate_weight_grid(step=0.10, min_w=0.10, max_w=0.40)
        results = evaluate_weights(opps, ctxs, configs)
        for i in range(len(results) - 1):
            assert results[i].evs >= results[i + 1].evs

    def test_top_result_has_highest_evs(self):
        opps = [
            _make_opp(net_profit=10.0, roi_pct=15.0, required_capital=200.0),
            _make_opp(net_profit=2.0, roi_pct=5.0, required_capital=40.0),
        ]
        ctxs = [
            _make_ctx(book_depth_ratio=1.5, confidence=0.7),
            _make_ctx(book_depth_ratio=0.5, confidence=0.3),
        ]
        configs = generate_weight_grid(step=0.10, min_w=0.10, max_w=0.40)
        results = evaluate_weights(opps, ctxs, configs)
        max_evs = max(r.evs for r in results)
        assert abs(results[0].evs - max_evs) < 1e-12

    def test_single_opportunity(self):
        opps = [_make_opp(net_profit=5.0, roi_pct=10.0, required_capital=100.0)]
        ctxs = [_make_ctx()]
        configs = generate_weight_grid(step=0.10, min_w=0.10, max_w=0.40)
        results = evaluate_weights(opps, ctxs, configs)
        assert len(results) == len(configs)
        assert all(isinstance(r, WeightResult) for r in results)

    def test_empty_opportunities(self):
        configs = generate_weight_grid(step=0.10, min_w=0.10, max_w=0.40)
        results = evaluate_weights([], [], configs)
        assert all(r.evs == 0.0 for r in results)
        assert all(r.mean_score == 0.0 for r in results)

    def test_result_fields(self):
        opps = [_make_opp()]
        ctxs = [_make_ctx()]
        cfg = WeightConfig(0.22, 0.22, 0.17, 0.17, 0.07, 0.15)
        results = evaluate_weights(opps, ctxs, [cfg])
        assert len(results) == 1
        r = results[0]
        assert r.config == cfg
        assert r.evs >= 0.0
        assert r.mean_score >= 0.0

    def test_different_configs_different_scores(self):
        """Configs with extreme weight differences produce different mean scores."""
        opps = [
            _make_opp(
                opp_type=OpportunityType.SPIKE_LAG,
                net_profit=20.0, roi_pct=50.0, required_capital=200.0,
                num_legs=3,
            ),
            _make_opp(net_profit=1.0, roi_pct=2.0, required_capital=30.0),
        ]
        ctxs = [
            _make_ctx(book_depth_ratio=2.0, confidence=0.9, is_spike=True),
            _make_ctx(book_depth_ratio=0.3, confidence=0.2),
        ]
        # Urgency-heavy vs profit-heavy
        cfg_urgency = WeightConfig(0.05, 0.05, 0.05, 0.40, 0.05, 0.40)
        cfg_profit = WeightConfig(0.40, 0.05, 0.40, 0.05, 0.05, 0.05)
        results = evaluate_weights(opps, ctxs, [cfg_urgency, cfg_profit])
        assert results[0].mean_score != results[1].mean_score


# ── Tests: CLI ───────────────────────────────────────────────────────────

class TestCLI:
    def test_end_to_end(self, tmp_path):
        input_path = tmp_path / "dry_run.jsonl"
        output_path = tmp_path / "results" / "weight_search.json"

        lines = [
            _cycle_marker(1),
            _opp_line(1, profit=8.64, roi=6.63, legs=2, capital=130.41),
            _opp_line(2, profit=1.97, roi=7.63, legs=2, capital=25.84),
            _cycle_complete(1, elapsed=8.0, opps=2),
            _cycle_marker(2),
            _opp_line(1, profit=5.0, roi=10.0, legs=3, capital=200.0),
            _cycle_complete(2, elapsed=6.0, opps=1),
        ]
        _write_ndjson(lines, input_path)

        ws_main([
            "--input", str(input_path),
            "--output", str(output_path),
            "--step", "0.10",
            "--top", "5",
        ])

        assert output_path.exists()
        with open(output_path) as f:
            data = json.load(f)

        assert "meta" in data
        assert data["meta"]["total_opportunities"] == 3
        assert "results" in data
        assert len(data["results"]) <= 5
        assert data["results"][0]["rank"] == 1
        # Top result should have highest EVS
        evs_values = [r["evs"] for r in data["results"]]
        assert evs_values == sorted(evs_values, reverse=True)
