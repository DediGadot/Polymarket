"""
Replay engine for recorded pipeline cycles. Re-ranks opportunities through
rank_opportunities() with configurable scorer weights, enabling offline
weight optimization and what-if analysis.

Reads NDJSON recordings (one JSON object per line), replays the scorer with
different weight vectors, and reports per-weight P&L, Sharpe, win rate,
and capital efficiency.

Usage:
    python -m benchmark.replay --input recordings/file.jsonl --sweep scorer_weights
    python -m benchmark.replay --input recordings/file.jsonl --weights 0.20,0.20,0.15,0.15,0.05,0.15,0.10
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from dataclasses import dataclass
from pathlib import Path

from scanner.models import (
    LegOrder,
    Opportunity,
    OpportunityType,
    Side,
)
from scanner.scorer import ScoringContext, score_opportunity, ScoredOpportunity


# ── NDJSON record schema ─────────────────────────────────────────────────
#
# Expected record types in the NDJSON file:
#
# {"type": "config", "data": {...}}           — config snapshot (first line)
# {"type": "cycle",  "cycle": N, "data": {    — per-cycle record
#     "opportunities": [{...}, ...],
#     "contexts":      [{...}, ...],
#     "trades":        [{...}, ...],           — optional: actual fills
#     "market_state":  {...}                   — optional: MarketState
# }}

_OPP_TYPE_MAP = {t.value: t for t in OpportunityType}
_SIDE_MAP = {s.value: s for s in Side}
SUPPORTED_SCHEMA_VERSIONS = {1, 2}


@dataclass(frozen=True)
class WeightVector:
    """Scorer weight vector. Fields must sum to ~1.0."""
    profit: float
    fill: float
    efficiency: float
    urgency: float
    competition: float
    persistence: float
    realized_ev: float

    def label(self) -> str:
        return (
            f"P={self.profit:.2f} F={self.fill:.2f} E={self.efficiency:.2f} "
            f"U={self.urgency:.2f} C={self.competition:.2f} "
            f"R={self.persistence:.2f} V={self.realized_ev:.2f}"
        )


# Default weights from scanner/scorer.py
DEFAULT_WEIGHTS = WeightVector(
    profit=0.20, fill=0.20, efficiency=0.15,
    urgency=0.15, competition=0.05, persistence=0.15, realized_ev=0.10,
)


@dataclass(frozen=True)
class CycleRecord:
    """Parsed cycle from NDJSON recording."""
    cycle: int
    opportunities: tuple[Opportunity, ...]
    contexts: tuple[ScoringContext, ...]
    trade_pnls: tuple[float, ...]  # net_pnl per trade (empty if no trades)


@dataclass(frozen=True)
class WeightReport:
    """Results of replaying all cycles with a single weight vector."""
    weights: WeightVector
    total_pnl: float
    win_count: int
    loss_count: int
    win_rate: float
    sharpe: float
    capital_efficiency: float  # total_pnl / total_capital_deployed
    top_1_pnl: float  # P&L if we only took the top-ranked opp each cycle
    cycles_replayed: int


# ── Parsing ──────────────────────────────────────────────────────────────

def _parse_opportunity(d: dict) -> Opportunity:
    """Deserialize an Opportunity from a JSON dict."""
    opp_type = _OPP_TYPE_MAP.get(d.get("type", ""), OpportunityType.BINARY_REBALANCE)
    legs = tuple(
        LegOrder(
            token_id=leg.get("token_id", ""),
            side=_SIDE_MAP.get(leg.get("side", "BUY"), Side.BUY),
            price=float(leg.get("price", 0.0)),
            size=float(leg.get("size", 0.0)),
            platform=leg.get("platform", ""),
            tick_size=leg.get("tick_size", "0.01"),
        )
        for leg in d.get("legs", [])
    )
    return Opportunity(
        type=opp_type,
        event_id=d.get("event_id", ""),
        legs=legs,
        expected_profit_per_set=float(d.get("expected_profit_per_set", 0.0)),
        net_profit_per_set=float(d.get("net_profit_per_set", 0.0)),
        max_sets=float(d.get("max_sets", 0.0)),
        gross_profit=float(d.get("gross_profit", 0.0)),
        estimated_gas_cost=float(d.get("estimated_gas_cost", 0.0)),
        net_profit=float(d.get("net_profit", 0.0)),
        roi_pct=float(d.get("roi_pct", 0.0)),
        required_capital=float(d.get("required_capital", 0.0)),
        pair_fill_prob=float(d.get("pair_fill_prob", 1.0)),
        toxicity_score=float(d.get("toxicity_score", 0.0)),
        expected_realized_net=float(d.get("expected_realized_net", 0.0)),
        quote_theoretical_net=float(d.get("quote_theoretical_net", 0.0)),
        reason_code=str(d.get("reason_code", "")),
        risk_flags=tuple(str(x) for x in d.get("risk_flags", []) or ()),
    )


def _parse_context(d: dict) -> ScoringContext:
    """Deserialize a ScoringContext from a JSON dict."""
    return ScoringContext(
        market_volume=float(d.get("market_volume", 0.0)),
        recent_trade_count=int(d.get("recent_trade_count", 0)),
        time_to_resolution_hours=float(d.get("time_to_resolution_hours", 720.0)),
        is_spike=bool(d.get("is_spike", False)),
        book_depth_ratio=float(d.get("book_depth_ratio", 1.0)),
        confidence=float(d.get("confidence", 0.5)),
        realized_ev_score=float(d.get("realized_ev_score", 0.5)),
        ofi_divergence=float(d.get("ofi_divergence", 0.0)),
    )


def parse_recording(path: Path) -> list[CycleRecord]:
    """
    Parse an NDJSON recording file into CycleRecords.
    Skips malformed lines and non-cycle records.
    """
    cycles: list[CycleRecord] = []
    with open(path) as f:
        for raw_line in f:
            raw_line = raw_line.strip()
            if not raw_line:
                continue
            try:
                record = json.loads(raw_line)
            except json.JSONDecodeError:
                continue

            cycle_num: int | None = None
            data: dict = {}
            rec_type = record.get("type")
            if rec_type == "cycle":
                schema_version = int(record.get("schema_version", 1))
                if schema_version not in SUPPORTED_SCHEMA_VERSIONS:
                    raise ValueError(
                        f"Unsupported recording schema_version={schema_version} in {path}"
                    )
                cycle_num = int(record.get("cycle", 0))
                data = record.get("data", {}) or {}
            elif "cycle" in record and (
                "opportunities" in record or "scoring_contexts" in record or "contexts" in record
            ):
                # Backward-compatible flat schema (pre-v2 recorder)
                cycle_num = int(record.get("cycle", 0))
                data = record
            else:
                continue

            opps = tuple(
                _parse_opportunity(o) for o in data.get("opportunities", [])
            )
            ctxs = tuple(
                _parse_context(c)
                for c in (data.get("contexts") or data.get("scoring_contexts") or [])
            )
            # Pad contexts to match opportunities if short
            if len(ctxs) < len(opps):
                ctxs = ctxs + tuple(
                    ScoringContext() for _ in range(len(opps) - len(ctxs))
                )
            trade_pnls = tuple(
                float(t.get("net_pnl", 0.0)) for t in data.get("trades", [])
            )
            cycles.append(CycleRecord(
                cycle=cycle_num if cycle_num is not None else 0,
                opportunities=opps,
                contexts=ctxs,
                trade_pnls=trade_pnls,
            ))
    return cycles


# ── Scoring with custom weights ──────────────────────────────────────────

def _rescore(
    opp: Opportunity,
    ctx: ScoringContext,
    weights: WeightVector,
) -> ScoredOpportunity:
    """Score an opportunity, then reweight with custom weights."""
    scored = score_opportunity(opp, ctx)
    total = (
        weights.profit * scored.profit_score
        + weights.fill * scored.fill_score
        + weights.efficiency * scored.efficiency_score
        + weights.urgency * scored.urgency_score
        + weights.competition * scored.competition_score
        + weights.persistence * scored.persistence_score
        + weights.realized_ev * scored.realized_ev_score
    )
    return ScoredOpportunity(
        opportunity=scored.opportunity,
        total_score=total,
        profit_score=scored.profit_score,
        fill_score=scored.fill_score,
        efficiency_score=scored.efficiency_score,
        urgency_score=scored.urgency_score,
        competition_score=scored.competition_score,
        persistence_score=scored.persistence_score,
        realized_ev_score=scored.realized_ev_score,
    )


def replay_cycle(
    cycle: CycleRecord,
    weights: WeightVector,
) -> list[ScoredOpportunity]:
    """Re-rank a cycle's opportunities using the given weight vector."""
    scored = [
        _rescore(opp, ctx, weights)
        for opp, ctx in zip(cycle.opportunities, cycle.contexts)
    ]
    scored.sort(key=lambda s: s.total_score, reverse=True)
    return scored


# ── Replay engine ────────────────────────────────────────────────────────

def replay_with_weights(
    cycles: list[CycleRecord],
    weights: WeightVector,
) -> WeightReport:
    """
    Replay all recorded cycles with a given weight vector.
    Returns aggregate metrics.
    """
    if not cycles:
        return WeightReport(
            weights=weights, total_pnl=0.0, win_count=0, loss_count=0,
            win_rate=0.0, sharpe=0.0, capital_efficiency=0.0,
            top_1_pnl=0.0, cycles_replayed=0,
        )

    # For each cycle, re-rank and compute hypothetical P&L based on
    # the top-ranked opportunity's net_profit.
    per_cycle_pnl: list[float] = []
    top_1_pnl_total = 0.0
    total_capital = 0.0

    for cycle in cycles:
        if not cycle.opportunities:
            per_cycle_pnl.append(0.0)
            continue

        ranked = replay_cycle(cycle, weights)

        # If actual trades exist, use real P&L
        if cycle.trade_pnls:
            cycle_pnl = sum(cycle.trade_pnls)
        else:
            # Hypothetical: sum net_profit of all profitable ranked opps
            cycle_pnl = sum(
                s.opportunity.net_profit
                for s in ranked
                if s.opportunity.net_profit > 0
            )

        per_cycle_pnl.append(cycle_pnl)

        # Top-1 tracking
        if ranked:
            top_1 = ranked[0].opportunity
            top_1_pnl_total += top_1.net_profit
            total_capital += top_1.required_capital

    total_pnl = sum(per_cycle_pnl)
    wins = sum(1 for p in per_cycle_pnl if p > 0)
    losses = sum(1 for p in per_cycle_pnl if p <= 0)
    n = len(per_cycle_pnl)
    win_rate = wins / n if n > 0 else 0.0

    # Sharpe ratio (annualized, assuming ~1 cycle per minute)
    if n > 1:
        mean = total_pnl / n
        variance = sum((p - mean) ** 2 for p in per_cycle_pnl) / (n - 1)
        std = math.sqrt(variance) if variance > 0 else 0.0
        sharpe = (mean / std * math.sqrt(525600.0)) if std > 0 else 0.0  # minutes/year
    else:
        sharpe = 0.0

    cap_eff = total_pnl / total_capital if total_capital > 0 else 0.0

    return WeightReport(
        weights=weights,
        total_pnl=total_pnl,
        win_count=wins,
        loss_count=losses,
        win_rate=win_rate,
        sharpe=sharpe,
        capital_efficiency=cap_eff,
        top_1_pnl=top_1_pnl_total,
        cycles_replayed=n,
    )


# ── Weight sweep ─────────────────────────────────────────────────────────

def generate_weight_sweep(step: float = 0.05) -> list[WeightVector]:
    """
    Generate weight vectors by varying the 7 scorer weights in `step` increments.
    Uses integer arithmetic to avoid float drift. The 7th weight is derived
    from the constraint (sum = 1.0).
    """
    inv = round(1.0 / step)
    total = inv
    min_s = 1  # minimum 1 step (= step value)
    max_s = round(0.40 / step)  # max 40% per weight

    vectors: list[WeightVector] = []

    for w1 in range(min_s, max_s + 1):
        for w2 in range(min_s, max_s + 1):
            for w3 in range(min_s, max_s + 1):
                rem1 = total - w1 - w2 - w3
                if rem1 < 4 * min_s or rem1 > 4 * max_s:
                    continue
                for w4 in range(min_s, max_s + 1):
                    rem2 = rem1 - w4
                    if rem2 < 3 * min_s or rem2 > 3 * max_s:
                        continue
                    for w5 in range(min_s, max_s + 1):
                        rem3 = rem2 - w5
                        if rem3 < 2 * min_s or rem3 > 2 * max_s:
                            continue
                        for w6 in range(min_s, max_s + 1):
                            w7 = rem3 - w6
                            if min_s <= w7 <= max_s:
                                vectors.append(WeightVector(
                                    profit=w1 * step,
                                    fill=w2 * step,
                                    efficiency=w3 * step,
                                    urgency=w4 * step,
                                    competition=w5 * step,
                                    persistence=w6 * step,
                                    realized_ev=w7 * step,
                                ))

    return vectors


# ── Report generation ────────────────────────────────────────────────────

def run_sweep(
    cycles: list[CycleRecord],
    step: float = 0.05,
    top_n: int = 10,
) -> list[WeightReport]:
    """
    Run a full weight sweep and return top N results sorted by total_pnl.
    """
    vectors = generate_weight_sweep(step=step)
    results = [replay_with_weights(cycles, w) for w in vectors]
    results.sort(key=lambda r: r.total_pnl, reverse=True)
    return results[:top_n]


def report_to_json(reports: list[WeightReport], meta: dict | None = None) -> dict:
    """Serialize a list of WeightReport into a JSON-serializable dict."""
    return {
        "meta": meta or {},
        "results": [
            {
                "rank": i + 1,
                "weights": {
                    "profit": r.weights.profit,
                    "fill": r.weights.fill,
                    "efficiency": r.weights.efficiency,
                    "urgency": r.weights.urgency,
                    "competition": r.weights.competition,
                    "persistence": r.weights.persistence,
                    "realized_ev": r.weights.realized_ev,
                },
                "total_pnl": r.total_pnl,
                "win_count": r.win_count,
                "loss_count": r.loss_count,
                "win_rate": r.win_rate,
                "sharpe": r.sharpe,
                "capital_efficiency": r.capital_efficiency,
                "top_1_pnl": r.top_1_pnl,
                "cycles_replayed": r.cycles_replayed,
            }
            for i, r in enumerate(reports)
        ],
    }


# ── CLI ──────────────────────────────────────────────────────────────────

def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="benchmark.replay",
        description="Replay recorded cycles through the scorer with configurable weights.",
    )
    parser.add_argument("--input", required=True, help="Path to NDJSON recording file")
    parser.add_argument("--output", default=None, help="Path to write JSON report (default: stdout)")
    parser.add_argument(
        "--sweep", default=None, metavar="scorer_weights",
        help="Run weight sweep (pass 'scorer_weights' to activate)",
    )
    parser.add_argument(
        "--weights", default=None,
        help="Comma-separated weights: profit,fill,efficiency,urgency,competition,persistence,realized_ev",
    )
    parser.add_argument("--step", type=float, default=0.05, help="Sweep step size (default: 0.05)")
    parser.add_argument("--top", type=int, default=10, help="Number of top results to show (default: 10)")
    parser.add_argument(
        "--validate-only",
        action="store_true",
        help="Only validate and parse recording schema; do not replay/scored sweep.",
    )
    return parser


def _parse_weights_str(s: str) -> WeightVector:
    """Parse '0.20,0.20,0.15,0.15,0.05,0.15,0.10' into a WeightVector."""
    parts = [float(x.strip()) for x in s.split(",")]
    if len(parts) != 7:
        raise ValueError(f"Expected 7 comma-separated weights, got {len(parts)}")
    return WeightVector(*parts)


def main(argv: list[str] | None = None) -> None:
    """CLI entrypoint."""
    parser = _build_parser()
    args = parser.parse_args(argv)

    input_path = Path(args.input)
    cycles = parse_recording(input_path)

    if not cycles:
        print("No cycle records found in input file.", file=sys.stderr)
        sys.exit(1)

    total_opps = sum(len(c.opportunities) for c in cycles)
    print(f"Loaded {len(cycles)} cycles, {total_opps} total opportunities", file=sys.stderr)

    if args.validate_only:
        output = {
            "meta": {
                "mode": "validate_only",
                "cycles": len(cycles),
                "opportunities": total_opps,
                "input": str(input_path),
            },
            "results": [],
        }
        if args.output:
            output_path = Path(args.output)
            output_path.parent.mkdir(parents=True, exist_ok=True)
            with open(output_path, "w") as f:
                json.dump(output, f, indent=2)
            print(f"Validation report written to {output_path}", file=sys.stderr)
        else:
            json.dump(output, sys.stdout, indent=2)
            print(file=sys.stdout)
        return

    if args.sweep == "scorer_weights":
        # Weight sweep mode
        reports = run_sweep(cycles, step=args.step, top_n=args.top)
        meta = {
            "mode": "sweep",
            "step": args.step,
            "total_vectors": len(generate_weight_sweep(step=args.step)),
            "cycles": len(cycles),
            "opportunities": total_opps,
        }
        output = report_to_json(reports, meta=meta)

        print(f"\nTop {len(reports)} weight configurations:", file=sys.stderr)
        for i, r in enumerate(reports):
            print(
                f"  #{i + 1}: P&L=${r.total_pnl:.2f}  WR={r.win_rate:.1%}  "
                f"Sharpe={r.sharpe:.2f}  {r.weights.label()}",
                file=sys.stderr,
            )

    elif args.weights:
        # Single weight vector mode
        weights = _parse_weights_str(args.weights)
        report = replay_with_weights(cycles, weights)
        output = report_to_json([report], meta={
            "mode": "single",
            "cycles": len(cycles),
            "opportunities": total_opps,
        })

        print(f"\nReplay results ({weights.label()}):", file=sys.stderr)
        print(f"  Total P&L: ${report.total_pnl:.2f}", file=sys.stderr)
        print(f"  Win rate:  {report.win_rate:.1%} ({report.win_count}W / {report.loss_count}L)", file=sys.stderr)
        print(f"  Sharpe:    {report.sharpe:.2f}", file=sys.stderr)
        print(f"  Top-1 P&L: ${report.top_1_pnl:.2f}", file=sys.stderr)

    else:
        # Default: replay with current scorer weights
        report = replay_with_weights(cycles, DEFAULT_WEIGHTS)
        output = report_to_json([report], meta={
            "mode": "default",
            "cycles": len(cycles),
            "opportunities": total_opps,
        })

        print(f"\nReplay with default weights:", file=sys.stderr)
        print(f"  Total P&L: ${report.total_pnl:.2f}", file=sys.stderr)
        print(f"  Win rate:  {report.win_rate:.1%}", file=sys.stderr)
        print(f"  Sharpe:    {report.sharpe:.2f}", file=sys.stderr)

    # Write output
    if args.output:
        output_path = Path(args.output)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, "w") as f:
            json.dump(output, f, indent=2)
        print(f"\nReport written to {output_path}", file=sys.stderr)
    else:
        json.dump(output, sys.stdout, indent=2)
        print(file=sys.stdout)


if __name__ == "__main__":
    main()
