"""
Architecture latency comparison simulator.

Generates synthetic arb windows, simulates scanning at different intervals,
and computes EVS metrics to compare how scan frequency affects detection rate.

Usage:
    python -m benchmark.latency_sim --output PATH [--duration 300] [--arb-rate 0.5] \
        [--mean-window 200] [--intervals 100,250,500,1000] [--seed 42]
"""

from __future__ import annotations

import argparse
import json
import math
import random
import sys
from dataclasses import dataclass
from pathlib import Path

from benchmark.evs import ParsedOpportunity, classify_confidence, compute_evs


# ── Data models (all frozen) ────────────────────────────────────────────


@dataclass(frozen=True)
class ArbWindow:
    """A transient arbitrage opportunity with a time window."""
    event_id: str
    open_time: float   # ms
    close_time: float  # ms
    profit: float
    roi_pct: float
    capital: float


@dataclass(frozen=True)
class IntervalResult:
    """Simulation result for a single scan interval."""
    interval_ms: int
    total_scans: int
    arbs_detected: int
    arbs_missed: int
    detection_rate: float
    mean_detection_delay_ms: float
    evs: float
    evs_multiplier: float


@dataclass(frozen=True)
class ArchitectureReport:
    """Full comparison report across multiple scan intervals."""
    baseline_interval_ms: int
    total_arb_windows: int
    duration_sec: float
    results: tuple[IntervalResult, ...]

    def to_json(self, path: Path) -> None:
        """Write report as JSON to the given path."""
        data = {
            "baseline_interval_ms": self.baseline_interval_ms,
            "total_arb_windows": self.total_arb_windows,
            "duration_sec": self.duration_sec,
            "results": [
                {
                    "interval_ms": r.interval_ms,
                    "total_scans": r.total_scans,
                    "arbs_detected": r.arbs_detected,
                    "arbs_missed": r.arbs_missed,
                    "detection_rate": r.detection_rate,
                    "mean_detection_delay_ms": r.mean_detection_delay_ms,
                    "evs": r.evs,
                    "evs_multiplier": r.evs_multiplier,
                }
                for r in self.results
            ],
        }
        with open(path, "w") as f:
            json.dump(data, f, indent=2)


# ── Arb window generation ──────────────────────────────────────────────


def generate_arb_windows(
    duration_sec: float = 300.0,
    arb_rate: float = 0.5,
    mean_window_ms: float = 200.0,
    seed: int = 42,
) -> list[ArbWindow]:
    """
    Generate random arb windows using a Poisson process.

    Args:
        duration_sec: Total simulation duration in seconds.
        arb_rate: Average arb arrivals per second.
        mean_window_ms: Mean arb window duration (exponential distribution).
        seed: Random seed for reproducibility.

    Returns:
        List of ArbWindow objects sorted by open_time.
    """
    if arb_rate <= 0.0:
        return []

    rng = random.Random(seed)
    duration_ms = duration_sec * 1000.0
    windows: list[ArbWindow] = []

    # Poisson process: inter-arrival times are exponential
    current_time = 0.0
    idx = 0

    while current_time < duration_ms:
        # Exponential inter-arrival time (rate = arb_rate per second = arb_rate/1000 per ms)
        inter_arrival_ms = rng.expovariate(arb_rate / 1000.0)
        current_time += inter_arrival_ms

        if current_time >= duration_ms:
            break

        # Window duration: exponential with given mean
        window_duration = rng.expovariate(1.0 / mean_window_ms)

        # Random profit, ROI, capital within specified ranges
        profit = rng.uniform(1.0, 20.0)
        roi_pct = rng.uniform(2.0, 15.0)
        capital = rng.uniform(20.0, 500.0)

        windows.append(ArbWindow(
            event_id=f"arb_{idx}",
            open_time=current_time,
            close_time=current_time + window_duration,
            profit=profit,
            roi_pct=roi_pct,
            capital=capital,
        ))
        idx += 1

    return windows


# ── Interval simulation ────────────────────────────────────────────────


def simulate_interval(
    arb_windows: list[ArbWindow],
    interval_ms: int,
    target_size: float = 100.0,
) -> IntervalResult:
    """
    Simulate scanning at a fixed interval and measure detection.

    Scans happen at t=0, t=interval_ms, t=2*interval_ms, ...
    An arb is "detected" if any scan falls within [open_time, close_time).
    """
    if not arb_windows:
        return IntervalResult(
            interval_ms=interval_ms,
            total_scans=0,
            arbs_detected=0,
            arbs_missed=0,
            detection_rate=0.0,
            mean_detection_delay_ms=0.0,
            evs=0.0,
            evs_multiplier=0.0,
        )

    # Determine scan range: cover all arb windows
    max_close = max(w.close_time for w in arb_windows)
    scan_times = []
    t = 0.0
    while t <= max_close:
        scan_times.append(t)
        t += interval_ms

    total_scans = len(scan_times)

    # For each arb window, check if any scan falls within it
    detected_opps: list[ParsedOpportunity] = []
    detection_delays: list[float] = []
    arbs_detected = 0

    for window in arb_windows:
        # Find the first scan at or after open_time
        # scan_index = ceil(open_time / interval_ms)
        if interval_ms > 0:
            first_scan_idx = math.ceil(window.open_time / interval_ms)
            first_scan_time = first_scan_idx * interval_ms
        else:
            first_scan_time = 0.0

        if first_scan_time < window.close_time:
            # Detected
            arbs_detected += 1
            delay = first_scan_time - window.open_time
            detection_delays.append(delay)
            # Build a ParsedOpportunity for EVS computation
            detected_opps.append(ParsedOpportunity(
                opp_type="latency_sim",
                event_id=window.event_id,
                profit=window.profit,
                roi_pct=window.roi_pct,
                legs=2,
                capital=window.capital,
                platform="simulated",
            ))

    arbs_missed = len(arb_windows) - arbs_detected
    detection_rate = arbs_detected / len(arb_windows) if arb_windows else 0.0
    mean_delay = sum(detection_delays) / len(detection_delays) if detection_delays else 0.0
    evs = compute_evs(tuple(detected_opps), target_size)

    return IntervalResult(
        interval_ms=interval_ms,
        total_scans=total_scans,
        arbs_detected=arbs_detected,
        arbs_missed=arbs_missed,
        detection_rate=detection_rate,
        mean_detection_delay_ms=mean_delay,
        evs=evs,
        evs_multiplier=0.0,  # filled in by compare_architectures
    )


# ── Architecture comparison ────────────────────────────────────────────


def compare_architectures(
    arb_windows: list[ArbWindow],
    intervals: list[int] | None = None,
    target_size: float = 100.0,
) -> ArchitectureReport:
    """
    Compare detection across multiple scan intervals.

    Results are sorted by interval ascending. The 1000ms interval
    is used as the baseline for EVS multiplier computation.
    """
    if intervals is None:
        intervals = [100, 250, 500, 1000]

    baseline_ms = 1000

    # Simulate each interval
    raw_results: list[IntervalResult] = []
    for iv in sorted(intervals):
        raw_results.append(simulate_interval(arb_windows, iv, target_size))

    # Find baseline EVS
    baseline_evs = 0.0
    for r in raw_results:
        if r.interval_ms == baseline_ms:
            baseline_evs = r.evs
            break

    # Compute multipliers
    final_results: list[IntervalResult] = []
    for r in raw_results:
        if baseline_evs > 0:
            multiplier = r.evs / baseline_evs
        else:
            multiplier = 1.0 if r.interval_ms == baseline_ms else 0.0
        final_results.append(IntervalResult(
            interval_ms=r.interval_ms,
            total_scans=r.total_scans,
            arbs_detected=r.arbs_detected,
            arbs_missed=r.arbs_missed,
            detection_rate=r.detection_rate,
            mean_detection_delay_ms=r.mean_detection_delay_ms,
            evs=r.evs,
            evs_multiplier=multiplier,
        ))

    # Compute duration from arb windows
    duration_sec = 0.0
    if arb_windows:
        max_close = max(w.close_time for w in arb_windows)
        duration_sec = max_close / 1000.0

    return ArchitectureReport(
        baseline_interval_ms=baseline_ms,
        total_arb_windows=len(arb_windows),
        duration_sec=duration_sec,
        results=tuple(final_results),
    )


# ── CLI entrypoint ──────────────────────────────────────────────────────


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="benchmark.latency_sim",
        description="Simulate architecture latency impact on arb detection.",
    )
    parser.add_argument("--output", required=True, help="Path to write JSON report")
    parser.add_argument("--duration", type=float, default=300.0, help="Simulation duration in seconds (default: 300)")
    parser.add_argument("--arb-rate", type=float, default=0.5, help="Arb arrivals per second (default: 0.5)")
    parser.add_argument("--mean-window", type=float, default=200.0, help="Mean arb window in ms (default: 200)")
    parser.add_argument("--intervals", type=str, default="100,250,500,1000", help="Comma-separated scan intervals in ms")
    parser.add_argument("--seed", type=int, default=42, help="Random seed (default: 42)")
    parser.add_argument("--target-size", type=float, default=100.0, help="Target position size in USD (default: 100)")
    return parser


def main(argv: list[str] | None = None) -> None:
    """CLI entrypoint. Accepts argv for testability."""
    parser = _build_parser()
    args = parser.parse_args(argv)

    output_path = Path(args.output)
    intervals = [int(x.strip()) for x in args.intervals.split(",")]

    windows = generate_arb_windows(
        duration_sec=args.duration,
        arb_rate=args.arb_rate,
        mean_window_ms=args.mean_window,
        seed=args.seed,
    )

    report = compare_architectures(windows, intervals, target_size=args.target_size)
    report.to_json(output_path)

    print(f"Latency sim report written to {output_path}")
    print(f"  Arb windows generated: {report.total_arb_windows}")
    print(f"  Intervals tested: {len(report.results)}")
    for r in report.results:
        print(
            f"    {r.interval_ms:>5d}ms: detected {r.arbs_detected}/{report.total_arb_windows} "
            f"({r.detection_rate:.1%}), EVS={r.evs:.4f}, multiplier={r.evs_multiplier:.2f}x"
        )


if __name__ == "__main__":
    main()
