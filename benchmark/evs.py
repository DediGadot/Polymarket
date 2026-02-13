"""
EVS (Expected Value Score) metric computation from dry-run NDJSON logs.

Parses structured log output from run.py dry-run mode, computes per-cycle
and session-level EVS metrics, and writes a JSON dashboard.

EVS_cycle = sum(Ci * Ri * Di) for all opportunities i in that cycle
  Ci = confidence (1.0 known, 0.7 first-seen deep, 0.3 thin)
  Ri = min(net_roi_pct / 100.0, 1.0)  -- clamped to [0, 1]
  Di = min(available_depth / target_size, 1.0) -- depth ratio

Usage:
    python -m benchmark.evs --input dry_run_output.jsonl --output evs_report.json [--target-size 100]
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from dataclasses import dataclass
from pathlib import Path


# ── Data models (all frozen) ────────────────────────────────────────────

@dataclass(frozen=True)
class ParsedOpportunity:
    """Single opportunity extracted from a log line."""
    opp_type: str
    event_id: str
    profit: float
    roi_pct: float
    legs: int
    capital: float
    platform: str


@dataclass(frozen=True)
class CycleParsed:
    """Raw parsed cycle data before metric computation."""
    cycle: int
    opportunities: tuple[ParsedOpportunity, ...]
    scan_latency_sec: float


@dataclass(frozen=True)
class CycleMetrics:
    """Dashboard metrics for a single scan cycle."""
    cycle: int
    evs: float
    arb_count: int
    mean_confidence: float
    mean_roi: float
    depth_weighted_profit: float
    scan_latency_sec: float
    hit_rate: float       # 1.0 if this cycle found opps, 0.0 otherwise
    platform_count: int   # distinct platforms in this cycle's opps


@dataclass(frozen=True)
class SessionReport:
    """Aggregate report for an entire dry-run session."""
    total_cycles: int
    overall_evs: float
    cycle_metrics: tuple[CycleMetrics, ...]

    def to_json(self, path: Path) -> None:
        """Write report as JSON to the given path."""
        data = {
            "total_cycles": self.total_cycles,
            "overall_evs": self.overall_evs,
            "cycles": [
                {
                    "cycle": m.cycle,
                    "evs": m.evs,
                    "arb_count": m.arb_count,
                    "mean_confidence": m.mean_confidence,
                    "mean_roi": m.mean_roi,
                    "depth_weighted_profit": m.depth_weighted_profit,
                    "scan_latency_sec": m.scan_latency_sec,
                    "hit_rate": m.hit_rate,
                    "platform_count": m.platform_count,
                }
                for m in self.cycle_metrics
            ],
        }
        with open(path, "w") as f:
            json.dump(data, f, indent=2)


# ── Regex patterns for log parsing ──────────────────────────────────────

_CYCLE_MARKER_RE = re.compile(r"\u2500\u2500 Cycle (\d+) ")
_CYCLE_COMPLETE_RE = re.compile(r"Cycle (\d+) complete in ([\d.]+)s")
_OPP_LINE_RE = re.compile(
    r"#(\d+)\s+(\S+)\s+event=(\S+)\s+profit=\$([\d.]+)\s+roi=([\d.]+)%\s+legs=(\d+)\s+capital=\$([\d.]+)"
)


# ── Confidence classification ───────────────────────────────────────────

def classify_confidence(legs: int, capital: float, target_size: float) -> float:
    """
    Classify opportunity confidence tier:
      1.0 = known: multi-leg (>=3) with deep capital (>= target_size)
      0.7 = first-seen deep: capital >= target_size
      0.3 = thin: capital < target_size
    """
    if legs >= 3 and capital >= target_size:
        return 1.0
    if capital >= target_size:
        return 0.7
    return 0.3


# ── Core EVS computation ────────────────────────────────────────────────

def compute_evs(
    opportunities: tuple[ParsedOpportunity, ...],
    target_size: float,
) -> float:
    """
    Compute EVS for a set of opportunities.
    EVS = sum(Ci * Ri * Di) where:
      Ci = confidence tier
      Ri = min(roi_pct / 100, 1.0), clamped to >= 0
      Di = min(capital / target_size, 1.0), clamped to >= 0
    """
    total = 0.0
    for opp in opportunities:
        ci = classify_confidence(opp.legs, opp.capital, target_size)
        ri = min(max(opp.roi_pct / 100.0, 0.0), 1.0)
        di = min(max(opp.capital / target_size, 0.0), 1.0) if target_size > 0 else 0.0
        total += ci * ri * di
    return total


# ── Log parsing ─────────────────────────────────────────────────────────

def _infer_platform(opp_type: str) -> str:
    """Infer platform from opportunity type."""
    if "cross_platform" in opp_type:
        return "kalshi"
    return "polymarket"


def parse_log_lines(lines: list[dict]) -> list[CycleParsed]:
    """
    Parse NDJSON log dicts into a list of CycleParsed.
    Each cycle boundary is detected by the cycle marker line.
    """
    cycles: list[CycleParsed] = []
    current_cycle: int | None = None
    current_opps: list[ParsedOpportunity] = []
    current_latency: float = 0.0

    def _flush() -> None:
        nonlocal current_cycle, current_opps, current_latency
        if current_cycle is not None:
            cycles.append(CycleParsed(
                cycle=current_cycle,
                opportunities=tuple(current_opps),
                scan_latency_sec=current_latency,
            ))
        current_opps = []
        current_latency = 0.0

    for line in lines:
        msg = line.get("msg", "")

        # Cycle boundary
        m = _CYCLE_MARKER_RE.search(msg)
        if m:
            _flush()
            current_cycle = int(m.group(1))
            continue

        # Cycle complete (extract latency)
        m = _CYCLE_COMPLETE_RE.search(msg)
        if m:
            current_latency = float(m.group(2))
            continue

        # Opportunity line
        m = _OPP_LINE_RE.search(msg)
        if m:
            opp_type = m.group(2)
            current_opps.append(ParsedOpportunity(
                opp_type=opp_type,
                event_id=m.group(3),
                profit=float(m.group(4)),
                roi_pct=float(m.group(5)),
                legs=int(m.group(6)),
                capital=float(m.group(7)),
                platform=_infer_platform(opp_type),
            ))

    # Flush last cycle
    _flush()
    return cycles


# ── Session report builder ──────────────────────────────────────────────

def build_session_report(
    cycles: tuple[CycleParsed, ...],
    target_size: float,
) -> SessionReport:
    """Build a full session report with per-cycle metrics and overall EVS."""
    if not cycles:
        return SessionReport(
            total_cycles=0,
            overall_evs=0.0,
            cycle_metrics=(),
        )

    metrics: list[CycleMetrics] = []
    evs_sum = 0.0

    for cycle_data in cycles:
        evs = compute_evs(cycle_data.opportunities, target_size)
        evs_sum += evs
        n = len(cycle_data.opportunities)

        if n > 0:
            confidences = tuple(
                classify_confidence(o.legs, o.capital, target_size)
                for o in cycle_data.opportunities
            )
            mean_confidence = sum(confidences) / n
            mean_roi = sum(o.roi_pct for o in cycle_data.opportunities) / n
            depth_weighted_profit = sum(
                o.profit * min(o.capital / target_size, 1.0) if target_size > 0 else 0.0
                for o in cycle_data.opportunities
            )
            platforms = frozenset(o.platform for o in cycle_data.opportunities)
            platform_count = len(platforms)
            hit_rate = 1.0
        else:
            mean_confidence = 0.0
            mean_roi = 0.0
            depth_weighted_profit = 0.0
            platform_count = 0
            hit_rate = 0.0

        metrics.append(CycleMetrics(
            cycle=cycle_data.cycle,
            evs=evs,
            arb_count=n,
            mean_confidence=mean_confidence,
            mean_roi=mean_roi,
            depth_weighted_profit=depth_weighted_profit,
            scan_latency_sec=cycle_data.scan_latency_sec,
            hit_rate=hit_rate,
            platform_count=platform_count,
        ))

    total_cycles = len(cycles)
    overall_evs = evs_sum / total_cycles if total_cycles > 0 else 0.0

    return SessionReport(
        total_cycles=total_cycles,
        overall_evs=overall_evs,
        cycle_metrics=tuple(metrics),
    )


# ── CLI entrypoint ──────────────────────────────────────────────────────

def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="benchmark.evs",
        description="Compute EVS metrics from dry-run NDJSON logs.",
    )
    parser.add_argument("--input", required=True, help="Path to NDJSON log file")
    parser.add_argument("--output", required=True, help="Path to write JSON report")
    parser.add_argument("--target-size", type=float, default=100.0, help="Target position size in USD (default: 100)")
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
    report = build_session_report(tuple(cycles), target_size=args.target_size)
    report.to_json(output_path)

    print(f"EVS report written to {output_path}")
    print(f"  Total cycles: {report.total_cycles}")
    print(f"  Overall EVS:  {report.overall_evs:.6f}")
    cycles_with_opps = sum(1 for m in report.cycle_metrics if m.arb_count > 0)
    print(f"  Cycles with opportunities: {cycles_with_opps}/{report.total_cycles}")


if __name__ == "__main__":
    main()
