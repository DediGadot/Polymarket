"""
Unit tests for benchmark/replay.py -- replay engine for recorded cycles.
"""

from __future__ import annotations

import json
import math
from pathlib import Path

from benchmark.replay import (
    CycleRecord,
    DEFAULT_WEIGHTS,
    WeightVector,
    WeightReport,
    _parse_opportunity,
    _parse_context,
    _rescore,
    generate_weight_sweep,
    parse_recording,
    replay_cycle,
    replay_with_weights,
    report_to_json,
    run_sweep,
)
from scanner.models import (
    LegOrder,
    Opportunity,
    OpportunityType,
    Side,
)
from scanner.scorer import ScoringContext, score_opportunity


# ── Helpers ──────────────────────────────────────────────────────────────

def _make_opp(
    net_profit: float = 1.0,
    roi_pct: float = 5.0,
    opp_type: OpportunityType = OpportunityType.BINARY_REBALANCE,
    required_capital: float = 20.0,
) -> Opportunity:
    return Opportunity(
        type=opp_type,
        event_id="evt_1",
        legs=(
            LegOrder(token_id="tok_yes", side=Side.BUY, price=0.45, size=20.0),
            LegOrder(token_id="tok_no", side=Side.BUY, price=0.50, size=20.0),
        ),
        expected_profit_per_set=net_profit * 1.1,
        net_profit_per_set=net_profit / 2,
        max_sets=2.0,
        gross_profit=net_profit * 1.2,
        estimated_gas_cost=0.50,
        net_profit=net_profit,
        roi_pct=roi_pct,
        required_capital=required_capital,
    )


def _make_cycle(
    cycle_num: int = 1,
    opps: list[Opportunity] | None = None,
    ctxs: list[ScoringContext] | None = None,
    trade_pnls: list[float] | None = None,
) -> CycleRecord:
    if opps is None:
        opps = [_make_opp()]
    if ctxs is None:
        ctxs = [ScoringContext() for _ in opps]
    return CycleRecord(
        cycle=cycle_num,
        opportunities=tuple(opps),
        contexts=tuple(ctxs),
        trade_pnls=tuple(trade_pnls) if trade_pnls else (),
    )


def _opp_to_dict(opp: Opportunity) -> dict:
    """Serialize an Opportunity to JSON dict (matching recorder format)."""
    return {
        "type": opp.type.value,
        "event_id": opp.event_id,
        "legs": [
            {
                "token_id": leg.token_id,
                "side": leg.side.value,
                "price": leg.price,
                "size": leg.size,
                "platform": leg.platform,
                "tick_size": leg.tick_size,
            }
            for leg in opp.legs
        ],
        "expected_profit_per_set": opp.expected_profit_per_set,
        "net_profit_per_set": opp.net_profit_per_set,
        "max_sets": opp.max_sets,
        "gross_profit": opp.gross_profit,
        "estimated_gas_cost": opp.estimated_gas_cost,
        "net_profit": opp.net_profit,
        "roi_pct": opp.roi_pct,
        "required_capital": opp.required_capital,
        "pair_fill_prob": opp.pair_fill_prob,
        "toxicity_score": opp.toxicity_score,
        "expected_realized_net": opp.expected_realized_net,
        "quote_theoretical_net": opp.quote_theoretical_net,
    }


def _ctx_to_dict(ctx: ScoringContext) -> dict:
    """Serialize a ScoringContext to JSON dict."""
    return {
        "market_volume": ctx.market_volume,
        "recent_trade_count": ctx.recent_trade_count,
        "time_to_resolution_hours": ctx.time_to_resolution_hours,
        "is_spike": ctx.is_spike,
        "book_depth_ratio": ctx.book_depth_ratio,
        "confidence": ctx.confidence,
        "realized_ev_score": ctx.realized_ev_score,
    }


def _write_recording(path: Path, cycles: list[CycleRecord]) -> None:
    """Write a test NDJSON recording file."""
    with open(path, "w") as f:
        # Config line
        f.write(json.dumps({"type": "config", "data": {"test": True}}) + "\n")
        for cycle in cycles:
            record = {
                "type": "cycle",
                "cycle": cycle.cycle,
                "data": {
                    "opportunities": [_opp_to_dict(o) for o in cycle.opportunities],
                    "contexts": [_ctx_to_dict(c) for c in cycle.contexts],
                    "trades": [{"net_pnl": p} for p in cycle.trade_pnls],
                },
            }
            f.write(json.dumps(record) + "\n")


# ── Tests ────────────────────────────────────────────────────────────────

class TestParseOpportunity:
    def test_roundtrip(self):
        """Serialize and deserialize an Opportunity."""
        opp = _make_opp(net_profit=2.50, roi_pct=8.0)
        d = _opp_to_dict(opp)
        parsed = _parse_opportunity(d)

        assert parsed.type == opp.type
        assert parsed.event_id == opp.event_id
        assert parsed.net_profit == opp.net_profit
        assert parsed.roi_pct == opp.roi_pct
        assert len(parsed.legs) == len(opp.legs)
        assert parsed.legs[0].token_id == "tok_yes"
        assert parsed.legs[0].side == Side.BUY

    def test_missing_fields_use_defaults(self):
        """Missing fields should use safe defaults."""
        parsed = _parse_opportunity({})

        assert parsed.type == OpportunityType.BINARY_REBALANCE
        assert parsed.net_profit == 0.0
        assert parsed.legs == ()


class TestParseContext:
    def test_roundtrip(self):
        ctx = ScoringContext(
            market_volume=50000.0,
            recent_trade_count=5,
            book_depth_ratio=1.5,
            confidence=0.8,
            realized_ev_score=0.7,
        )
        d = _ctx_to_dict(ctx)
        parsed = _parse_context(d)

        assert parsed.market_volume == 50000.0
        assert parsed.recent_trade_count == 5
        assert parsed.book_depth_ratio == 1.5
        assert parsed.confidence == 0.8
        assert parsed.realized_ev_score == 0.7


class TestParseRecording:
    def test_parse_ndjson_file(self, tmp_path: Path):
        """Parse a well-formed NDJSON recording file."""
        opp = _make_opp()
        cycle = _make_cycle(cycle_num=1, opps=[opp])
        recording_path = tmp_path / "test.jsonl"
        _write_recording(recording_path, [cycle])

        parsed = parse_recording(recording_path)

        assert len(parsed) == 1
        assert parsed[0].cycle == 1
        assert len(parsed[0].opportunities) == 1
        assert parsed[0].opportunities[0].net_profit == opp.net_profit

    def test_skips_non_cycle_records(self, tmp_path: Path):
        """Non-cycle records (config, etc.) are skipped."""
        recording_path = tmp_path / "test.jsonl"
        with open(recording_path, "w") as f:
            f.write(json.dumps({"type": "config", "data": {}}) + "\n")
            f.write(json.dumps({"type": "unknown", "data": {}}) + "\n")

        parsed = parse_recording(recording_path)
        assert len(parsed) == 0

    def test_skips_malformed_lines(self, tmp_path: Path):
        """Malformed JSON lines are skipped gracefully."""
        recording_path = tmp_path / "test.jsonl"
        opp = _make_opp()
        with open(recording_path, "w") as f:
            f.write("not json\n")
            f.write("\n")
            record = {
                "type": "cycle",
                "cycle": 1,
                "data": {
                    "opportunities": [_opp_to_dict(opp)],
                    "contexts": [_ctx_to_dict(ScoringContext())],
                },
            }
            f.write(json.dumps(record) + "\n")

        parsed = parse_recording(recording_path)
        assert len(parsed) == 1

    def test_pads_short_contexts(self, tmp_path: Path):
        """Contexts shorter than opportunities get padded with defaults."""
        recording_path = tmp_path / "test.jsonl"
        record = {
            "type": "cycle",
            "cycle": 1,
            "data": {
                "opportunities": [_opp_to_dict(_make_opp()), _opp_to_dict(_make_opp())],
                "contexts": [_ctx_to_dict(ScoringContext())],  # only 1 context for 2 opps
            },
        }
        with open(recording_path, "w") as f:
            f.write(json.dumps(record) + "\n")

        parsed = parse_recording(recording_path)
        assert len(parsed[0].opportunities) == 2
        assert len(parsed[0].contexts) == 2

    def test_parses_legacy_flat_cycle_schema(self, tmp_path: Path):
        recording_path = tmp_path / "legacy.jsonl"
        legacy = {
            "cycle": 7,
            "opportunities": [_opp_to_dict(_make_opp())],
            "scoring_contexts": [_ctx_to_dict(ScoringContext())],
        }
        with open(recording_path, "w") as f:
            f.write(json.dumps(legacy) + "\n")

        parsed = parse_recording(recording_path)
        assert len(parsed) == 1
        assert parsed[0].cycle == 7
        assert len(parsed[0].opportunities) == 1


class TestReplayCycle:
    def test_returns_sorted_by_score(self):
        """replay_cycle() should return opportunities sorted by reweighted score."""
        high_profit = _make_opp(net_profit=10.0, roi_pct=20.0)
        low_profit = _make_opp(net_profit=0.50, roi_pct=1.0)
        cycle = _make_cycle(opps=[low_profit, high_profit])

        ranked = replay_cycle(cycle, DEFAULT_WEIGHTS)

        assert len(ranked) == 2
        # Higher profit should rank first
        assert ranked[0].opportunity.net_profit >= ranked[1].opportunity.net_profit


class TestReplayWithWeights:
    def test_empty_cycles(self):
        """Empty cycle list returns zeroed report."""
        report = replay_with_weights([], DEFAULT_WEIGHTS)

        assert report.total_pnl == 0.0
        assert report.cycles_replayed == 0
        assert report.win_rate == 0.0

    def test_single_cycle_with_profit(self):
        """Single profitable cycle."""
        cycle = _make_cycle(opps=[_make_opp(net_profit=5.0)])
        report = replay_with_weights([cycle], DEFAULT_WEIGHTS)

        assert report.total_pnl == 5.0
        assert report.win_count == 1
        assert report.loss_count == 0
        assert report.win_rate == 1.0
        assert report.cycles_replayed == 1

    def test_uses_trade_pnls_when_present(self):
        """When actual trade P&Ls exist, use them instead of hypothetical."""
        cycle = _make_cycle(
            opps=[_make_opp(net_profit=5.0)],
            trade_pnls=[3.50],  # actual fill was less than theoretical
        )
        report = replay_with_weights([cycle], DEFAULT_WEIGHTS)

        assert report.total_pnl == 3.50

    def test_sharpe_computed(self):
        """Sharpe ratio should be non-zero with multiple cycles of varying P&L."""
        cycles = [
            _make_cycle(cycle_num=i, opps=[_make_opp(net_profit=float(i))])
            for i in range(1, 6)
        ]
        report = replay_with_weights(cycles, DEFAULT_WEIGHTS)

        assert report.sharpe != 0.0
        assert report.cycles_replayed == 5


class TestSameWeightsIdenticalRanking:
    def test_record_replay_identical(self):
        """
        Record → replay with same weights → verify identical ranking.
        This is the critical consistency test.
        """
        opps = [
            _make_opp(net_profit=1.0, roi_pct=3.0),
            _make_opp(net_profit=5.0, roi_pct=10.0),
            _make_opp(net_profit=0.20, roi_pct=0.5),
        ]
        ctxs = [ScoringContext() for _ in opps]

        # "Live" scoring with default weights
        live_scored = [score_opportunity(o, c) for o, c in zip(opps, ctxs)]
        live_scored.sort(key=lambda s: s.total_score, reverse=True)
        live_order = [s.opportunity.net_profit for s in live_scored]

        # Replayed scoring
        cycle = _make_cycle(opps=opps, ctxs=ctxs)
        replayed = replay_cycle(cycle, DEFAULT_WEIGHTS)
        replay_order = [s.opportunity.net_profit for s in replayed]

        assert live_order == replay_order

    def test_modified_weights_different_ranking(self):
        """
        Replay with modified weights → verify different ranking from default.
        """
        # One opp is high-profit/low-urgency, other is low-profit/high-urgency
        steady_opp = _make_opp(net_profit=10.0, roi_pct=15.0, opp_type=OpportunityType.BINARY_REBALANCE)
        spike_opp = _make_opp(net_profit=1.0, roi_pct=2.0, opp_type=OpportunityType.SPIKE_LAG)
        ctxs = [
            ScoringContext(),
            ScoringContext(is_spike=True),
        ]

        cycle = _make_cycle(opps=[steady_opp, spike_opp], ctxs=ctxs)

        # Default weights: profit=0.20, urgency=0.15
        default_ranked = replay_cycle(cycle, DEFAULT_WEIGHTS)

        # Urgency-heavy weights: urgency=0.60
        urgency_heavy = WeightVector(
            profit=0.05, fill=0.05, efficiency=0.05,
            urgency=0.60, competition=0.05, persistence=0.10, realized_ev=0.10,
        )
        urgency_ranked = replay_cycle(cycle, urgency_heavy)

        # With urgency-heavy weights, spike should rank higher
        assert urgency_ranked[0].opportunity.type == OpportunityType.SPIKE_LAG


class TestWeightSweep:
    def test_generates_valid_vectors(self):
        """All generated vectors should sum to ~1.0."""
        vectors = generate_weight_sweep(step=0.10)
        assert len(vectors) > 0
        for v in vectors:
            total = v.profit + v.fill + v.efficiency + v.urgency + v.competition + v.persistence + v.realized_ev
            assert abs(total - 1.0) < 1e-9, f"Weights sum to {total}, expected 1.0"

    def test_sweep_returns_results(self):
        """run_sweep returns ranked WeightReports."""
        cycles = [
            _make_cycle(cycle_num=1, opps=[_make_opp(net_profit=2.0)]),
            _make_cycle(cycle_num=2, opps=[_make_opp(net_profit=3.0)]),
        ]
        results = run_sweep(cycles, step=0.10, top_n=5)

        assert len(results) <= 5
        assert all(isinstance(r, WeightReport) for r in results)
        # Results should be sorted by total_pnl descending
        for i in range(len(results) - 1):
            assert results[i].total_pnl >= results[i + 1].total_pnl


class TestReportToJson:
    def test_serializable(self):
        """report_to_json output should be JSON-serializable."""
        report = replay_with_weights(
            [_make_cycle(opps=[_make_opp()])],
            DEFAULT_WEIGHTS,
        )
        output = report_to_json([report], meta={"mode": "test"})

        # Should not raise
        serialized = json.dumps(output)
        parsed = json.loads(serialized)

        assert parsed["meta"]["mode"] == "test"
        assert len(parsed["results"]) == 1
        assert "total_pnl" in parsed["results"][0]
        assert "weights" in parsed["results"][0]


class TestEndToEnd:
    def test_record_parse_replay_roundtrip(self, tmp_path: Path):
        """Full roundtrip: write NDJSON → parse → replay → report."""
        opps = [
            _make_opp(net_profit=3.0, roi_pct=8.0),
            _make_opp(net_profit=1.0, roi_pct=2.0),
        ]
        cycles = [
            _make_cycle(cycle_num=1, opps=opps),
            _make_cycle(cycle_num=2, opps=[_make_opp(net_profit=0.50)]),
            _make_cycle(cycle_num=3, opps=[_make_opp(net_profit=7.0, roi_pct=15.0)]),
        ]

        # Write recording
        recording_path = tmp_path / "recording.jsonl"
        _write_recording(recording_path, cycles)

        # Parse
        parsed = parse_recording(recording_path)
        assert len(parsed) == 3

        # Replay with default weights
        report = replay_with_weights(parsed, DEFAULT_WEIGHTS)
        assert report.cycles_replayed == 3
        assert report.total_pnl > 0

        # Replay with modified weights
        alt_weights = WeightVector(
            profit=0.30, fill=0.10, efficiency=0.10,
            urgency=0.10, competition=0.10, persistence=0.20, realized_ev=0.10,
        )
        alt_report = replay_with_weights(parsed, alt_weights)
        assert alt_report.cycles_replayed == 3

        # Generate report JSON
        output = report_to_json(
            [report, alt_report],
            meta={"cycles": 3},
        )
        assert len(output["results"]) == 2
