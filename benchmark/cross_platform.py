"""
Cross-platform EVS breakdown analyzer.

Parses dry-run NDJSON logs (reusing parse_log_lines from benchmark/evs),
splits opportunities by platform (polymarket vs kalshi), computes EVS
separately, and evaluates fuzzy matching thresholds.

Usage:
    python -m benchmark.cross_platform --input PATH --output PATH [--target-size 100]
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass
from pathlib import Path

from rapidfuzz import fuzz

from benchmark.evs import (
    CycleParsed,
    ParsedOpportunity,
    compute_evs,
    parse_log_lines,
)


# ── Data models (all frozen) ────────────────────────────────────────────

@dataclass(frozen=True)
class PlatformBreakdown:
    """EVS breakdown by platform."""
    total_evs: float
    pm_only_evs: float
    cross_platform_evs: float
    delta: float                  # cross_platform_evs (contribution added by cross-platform)
    total_arb_count: int
    pm_arb_count: int
    cross_platform_arb_count: int


@dataclass(frozen=True)
class LabeledPair:
    """A labeled pair for threshold evaluation."""
    pm_title: str
    kalshi_title: str
    is_match: bool


@dataclass(frozen=True)
class ThresholdResult:
    """Evaluation result for a single fuzzy threshold."""
    threshold: float
    matches_found: int
    true_positives: int
    false_positives: int
    precision: float
    recall: float
    f1_score: float


# ── Platform classification ─────────────────────────────────────────────

def _is_cross_platform(opp: ParsedOpportunity) -> bool:
    """Determine if an opportunity is cross-platform (kalshi)."""
    return opp.platform == "kalshi"


# ── Core breakdown computation ──────────────────────────────────────────

def compute_platform_breakdown(
    cycles: list[CycleParsed],
    target_size: float,
) -> PlatformBreakdown:
    """
    Split opportunities by platform and compute EVS for each.

    PM-only: platform == "polymarket"
    Cross-platform: platform == "kalshi" (inferred from opp_type containing "cross_platform")
    """
    pm_opps: list[ParsedOpportunity] = []
    xp_opps: list[ParsedOpportunity] = []

    for cycle in cycles:
        for opp in cycle.opportunities:
            if _is_cross_platform(opp):
                xp_opps.append(opp)
            else:
                pm_opps.append(opp)

    total_evs = compute_evs(tuple(pm_opps + xp_opps), target_size)
    pm_only_evs = compute_evs(tuple(pm_opps), target_size)
    cross_platform_evs = compute_evs(tuple(xp_opps), target_size)

    return PlatformBreakdown(
        total_evs=total_evs,
        pm_only_evs=pm_only_evs,
        cross_platform_evs=cross_platform_evs,
        delta=cross_platform_evs,
        total_arb_count=len(pm_opps) + len(xp_opps),
        pm_arb_count=len(pm_opps),
        cross_platform_arb_count=len(xp_opps),
    )


# ── Threshold evaluation ────────────────────────────────────────────────

def evaluate_thresholds(
    labeled_pairs: list[LabeledPair],
    thresholds: list[float],
) -> list[ThresholdResult]:
    """
    Evaluate fuzzy matching quality at different thresholds.

    For each threshold, compute how many labeled pairs would be matched
    by rapidfuzz.fuzz.token_set_ratio, then calculate precision/recall/F1.
    """
    results: list[ThresholdResult] = []

    for threshold in thresholds:
        true_positives = 0
        false_positives = 0
        false_negatives = 0
        matches_found = 0

        for pair in labeled_pairs:
            score = fuzz.token_set_ratio(pair.pm_title, pair.kalshi_title)
            predicted_match = score >= threshold

            if predicted_match:
                matches_found += 1
                if pair.is_match:
                    true_positives += 1
                else:
                    false_positives += 1
            else:
                if pair.is_match:
                    false_negatives += 1

        if true_positives + false_positives > 0:
            precision = true_positives / (true_positives + false_positives)
        else:
            precision = 0.0

        if true_positives + false_negatives > 0:
            recall = true_positives / (true_positives + false_negatives)
        else:
            recall = 0.0

        if precision + recall > 0:
            f1_score = 2 * precision * recall / (precision + recall)
        else:
            f1_score = 0.0

        results.append(ThresholdResult(
            threshold=threshold,
            matches_found=matches_found,
            true_positives=true_positives,
            false_positives=false_positives,
            precision=precision,
            recall=recall,
            f1_score=f1_score,
        ))

    return results


# ── CLI entrypoint ──────────────────────────────────────────────────────

def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="benchmark.cross_platform",
        description="Cross-platform EVS breakdown from dry-run NDJSON logs.",
    )
    parser.add_argument("--input", required=True, help="Path to NDJSON log file")
    parser.add_argument("--output", required=True, help="Path to write JSON report")
    parser.add_argument(
        "--target-size", type=float, default=100.0,
        help="Target position size in USD (default: 100)",
    )
    return parser


def main(argv: list[str] | None = None) -> None:
    """CLI entrypoint. Accepts argv for testability."""
    parser = _build_parser()
    args = parser.parse_args(argv)

    input_path = Path(args.input)
    output_path = Path(args.output)

    # Read and parse NDJSON
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
    breakdown = compute_platform_breakdown(cycles, target_size=args.target_size)

    report = {
        "platform_breakdown": {
            "total_evs": breakdown.total_evs,
            "pm_only_evs": breakdown.pm_only_evs,
            "cross_platform_evs": breakdown.cross_platform_evs,
            "delta": breakdown.delta,
            "total_arb_count": breakdown.total_arb_count,
            "pm_arb_count": breakdown.pm_arb_count,
            "cross_platform_arb_count": breakdown.cross_platform_arb_count,
        },
    }

    with open(output_path, "w") as f:
        json.dump(report, f, indent=2)

    print(f"Cross-platform report written to {output_path}")
    print(f"  Total EVS:          {breakdown.total_evs:.6f}")
    print(f"  PM-only EVS:        {breakdown.pm_only_evs:.6f}")
    print(f"  Cross-platform EVS: {breakdown.cross_platform_evs:.6f}")
    print(f"  Delta:              {breakdown.delta:.6f}")
    print(f"  Arbs: {breakdown.total_arb_count} total, {breakdown.pm_arb_count} PM, {breakdown.cross_platform_arb_count} cross-platform")


if __name__ == "__main__":
    main()
