"""
Scorer weight grid search. Finds optimal weight combinations by maximizing
EVS (Expected Value Score) across a set of scored opportunities.

For each weight configuration, computes a reweighted composite score using
the individual factor scores from score_opportunity(), then measures how
well those scores align with actual opportunity quality (ROI, depth, confidence).

Usage:
    python -m benchmark.weight_search --input PATH --output PATH [--step 0.05] [--top 10]
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass
from pathlib import Path

from benchmark.evs import (
    ParsedOpportunity,
    classify_confidence,
    parse_log_lines,
)
from scanner.models import (
    LegOrder,
    Opportunity,
    OpportunityType,
    Side,
)
from scanner.scorer import ScoringContext, score_opportunity


# ── Data models ──────────────────────────────────────────────────────────

@dataclass(frozen=True)
class WeightConfig:
    """A set of scorer weights. All fields must sum to 1.0."""
    profit: float
    fill: float
    efficiency: float
    urgency: float
    competition: float
    persistence: float

    def as_tuple(self) -> tuple[float, ...]:
        return (
            self.profit, self.fill, self.efficiency,
            self.urgency, self.competition, self.persistence,
        )


@dataclass(frozen=True)
class WeightResult:
    """Result of evaluating a single weight configuration."""
    config: WeightConfig
    evs: float
    mean_score: float


# ── Grid generation ──────────────────────────────────────────────────────

def generate_weight_grid(
    step: float = 0.05,
    min_w: float = 0.05,
    max_w: float = 0.40,
) -> list[WeightConfig]:
    """
    Generate all valid weight combinations summing to 1.0 within bounds.

    Uses integer arithmetic internally to avoid floating-point drift.
    The 6th weight is derived from the constraint (sum = 1.0) rather than
    looped over, reducing search space from O(n^6) to O(n^5).
    """
    inv = round(1.0 / step)
    total_steps = inv
    min_s = round(min_w / step)
    max_s = round(max_w / step)

    configs: list[WeightConfig] = []

    for w1 in range(min_s, max_s + 1):
        for w2 in range(min_s, max_s + 1):
            for w3 in range(min_s, max_s + 1):
                for w4 in range(min_s, max_s + 1):
                    remainder = total_steps - w1 - w2 - w3 - w4
                    # Early pruning: remaining two weights each in [min_s, max_s]
                    if remainder < 2 * min_s or remainder > 2 * max_s:
                        continue
                    for w5 in range(min_s, max_s + 1):
                        w6 = remainder - w5
                        if min_s <= w6 <= max_s:
                            configs.append(WeightConfig(
                                profit=w1 * step,
                                fill=w2 * step,
                                efficiency=w3 * step,
                                urgency=w4 * step,
                                competition=w5 * step,
                                persistence=w6 * step,
                            ))

    return configs


# ── Evaluation ───────────────────────────────────────────────────────────

def evaluate_weights(
    opportunities: list[Opportunity],
    contexts: list[ScoringContext],
    configs: list[WeightConfig],
    target_size: float = 100.0,
) -> list[WeightResult]:
    """
    Evaluate each weight config against opportunities.
    Returns results sorted by EVS descending.

    EVS = sum(reweighted_score_i * confidence_i * roi_factor_i * depth_factor_i).
    This rewards configs that assign high scores to high-quality opportunities.
    """
    if not opportunities:
        return [
            WeightResult(config=cfg, evs=0.0, mean_score=0.0)
            for cfg in configs
        ]

    # Pre-compute factor scores once (independent of weights)
    scored = [
        score_opportunity(opp, ctx)
        for opp, ctx in zip(opportunities, contexts)
    ]

    # Pre-compute quality factors for EVS
    quality_factors: list[tuple[float, float, float]] = []
    for s in scored:
        opp = s.opportunity
        legs = len(opp.legs)
        capital = opp.required_capital
        ci = classify_confidence(legs, capital, target_size)
        ri = min(max(opp.roi_pct / 100.0, 0.0), 1.0)
        di = (
            min(max(capital / target_size, 0.0), 1.0)
            if target_size > 0
            else 0.0
        )
        quality_factors.append((ci, ri, di))

    n = len(scored)
    results: list[WeightResult] = []

    for cfg in configs:
        # Reweight each opportunity's composite score
        reweighted_scores: list[float] = []
        for s in scored:
            total = (
                cfg.profit * s.profit_score
                + cfg.fill * s.fill_score
                + cfg.efficiency * s.efficiency_score
                + cfg.urgency * s.urgency_score
                + cfg.competition * s.competition_score
                + cfg.persistence * s.persistence_score
            )
            reweighted_scores.append(total)

        mean_score = sum(reweighted_scores) / n

        # EVS: score-weighted expected value
        evs = 0.0
        for i, (ci, ri, di) in enumerate(quality_factors):
            evs += reweighted_scores[i] * ci * ri * di

        results.append(WeightResult(config=cfg, evs=evs, mean_score=mean_score))

    results.sort(key=lambda r: r.evs, reverse=True)
    return results


# ── CLI helpers ──────────────────────────────────────────────────────────

_OPP_TYPE_MAP = {
    "binary_rebalance": OpportunityType.BINARY_REBALANCE,
    "negrisk_rebalance": OpportunityType.NEGRISK_REBALANCE,
    "latency_arb": OpportunityType.LATENCY_ARB,
    "spike_lag": OpportunityType.SPIKE_LAG,
    "cross_platform_arb": OpportunityType.CROSS_PLATFORM_ARB,
}


def _parsed_to_scored(
    parsed: ParsedOpportunity,
    target_size: float,
) -> tuple[Opportunity, ScoringContext]:
    """Convert a ParsedOpportunity from log data into Opportunity + ScoringContext."""
    opp_type = _OPP_TYPE_MAP.get(parsed.opp_type, OpportunityType.BINARY_REBALANCE)
    num_legs = max(parsed.legs, 1)
    legs = tuple(
        LegOrder(
            token_id=f"tok_{i}",
            side=Side.BUY,
            price=0.50,
            size=parsed.capital / num_legs,
        )
        for i in range(parsed.legs)
    )
    opp = Opportunity(
        type=opp_type,
        event_id=parsed.event_id,
        legs=legs,
        expected_profit_per_set=parsed.profit,
        net_profit_per_set=parsed.profit / num_legs,
        max_sets=float(parsed.legs),
        gross_profit=parsed.profit * 1.1,
        estimated_gas_cost=0.50,
        net_profit=parsed.profit,
        roi_pct=parsed.roi_pct,
        required_capital=parsed.capital,
    )
    ctx = ScoringContext(
        book_depth_ratio=min(parsed.capital / target_size, 2.0) if target_size > 0 else 0.0,
        confidence=classify_confidence(parsed.legs, parsed.capital, target_size),
        is_spike=(opp_type == OpportunityType.SPIKE_LAG),
    )
    return opp, ctx


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="benchmark.weight_search",
        description="Grid search for optimal scorer weights maximizing EVS.",
    )
    parser.add_argument("--input", required=True, help="Path to dry-run NDJSON log")
    parser.add_argument("--output", required=True, help="Path to write JSON results")
    parser.add_argument("--step", type=float, default=0.05, help="Grid step size (default: 0.05)")
    parser.add_argument("--top", type=int, default=10, help="Number of top results (default: 10)")
    parser.add_argument(
        "--target-size", type=float, default=100.0,
        help="Target position size in USD (default: 100)",
    )
    return parser


def main(argv: list[str] | None = None) -> None:
    """CLI: python -m benchmark.weight_search --input PATH --output PATH"""
    parser = _build_parser()
    args = parser.parse_args(argv)

    input_path = Path(args.input)
    output_path = Path(args.output)

    # Parse NDJSON log
    lines: list[dict] = []
    with open(input_path) as f:
        for raw_line in f:
            raw_line = raw_line.strip()
            if not raw_line:
                continue
            try:
                lines.append(json.loads(raw_line))
            except json.JSONDecodeError:
                continue

    cycles = parse_log_lines(lines)

    # Collect all opportunities across cycles
    all_parsed: list[ParsedOpportunity] = []
    for cycle in cycles:
        all_parsed.extend(cycle.opportunities)

    if not all_parsed:
        print("No opportunities found in input log.")
        sys.exit(1)

    # Convert to Opportunity + ScoringContext
    opps: list[Opportunity] = []
    ctxs: list[ScoringContext] = []
    for p in all_parsed:
        opp, ctx = _parsed_to_scored(p, args.target_size)
        opps.append(opp)
        ctxs.append(ctx)

    # Generate grid and evaluate
    print(f"Parsed {len(all_parsed)} opportunities from {len(cycles)} cycles")
    configs = generate_weight_grid(step=args.step)
    print(f"Generated {len(configs)} weight combinations (step={args.step})")

    results = evaluate_weights(opps, ctxs, configs, target_size=args.target_size)
    top_results = results[: args.top]

    # Write output
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_data = {
        "meta": {
            "total_configs": len(configs),
            "total_opportunities": len(all_parsed),
            "step": args.step,
            "target_size": args.target_size,
        },
        "results": [
            {
                "rank": i + 1,
                "evs": r.evs,
                "mean_score": r.mean_score,
                "weights": {
                    "profit": r.config.profit,
                    "fill": r.config.fill,
                    "efficiency": r.config.efficiency,
                    "urgency": r.config.urgency,
                    "competition": r.config.competition,
                    "persistence": r.config.persistence,
                },
            }
            for i, r in enumerate(top_results)
        ],
    }

    with open(output_path, "w") as f:
        json.dump(output_data, f, indent=2)

    print(f"\nTop {len(top_results)} weight configurations:")
    for i, r in enumerate(top_results):
        cfg = r.config
        print(
            f"  #{i + 1}: EVS={r.evs:.6f}  mean={r.mean_score:.4f}  "
            f"P={cfg.profit:.2f} F={cfg.fill:.2f} E={cfg.efficiency:.2f} "
            f"U={cfg.urgency:.2f} C={cfg.competition:.2f} R={cfg.persistence:.2f}"
        )

    print(f"\nResults written to {output_path}")


if __name__ == "__main__":
    main()
