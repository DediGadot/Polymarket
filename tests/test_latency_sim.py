"""
Unit tests for benchmark/latency_sim.py -- architecture latency comparison simulator.
"""

from __future__ import annotations

import json
import tempfile
from pathlib import Path

from benchmark.latency_sim import (
    ArbWindow,
    IntervalResult,
    ArchitectureReport,
    generate_arb_windows,
    simulate_interval,
    compare_architectures,
    main as latency_main,
)


# ── Tests: generate_arb_windows ───────────────────────────────────────


class TestGenerateArbWindows:
    def test_deterministic(self):
        """Same seed produces identical windows."""
        w1 = generate_arb_windows(duration_sec=60.0, arb_rate=1.0, seed=99)
        w2 = generate_arb_windows(duration_sec=60.0, arb_rate=1.0, seed=99)
        assert w1 == w2

    def test_different_seeds_differ(self):
        """Different seeds produce different windows."""
        w1 = generate_arb_windows(duration_sec=60.0, arb_rate=1.0, seed=1)
        w2 = generate_arb_windows(duration_sec=60.0, arb_rate=1.0, seed=2)
        assert w1 != w2

    def test_count_reasonable(self):
        """Number of windows is roughly arb_rate * duration."""
        windows = generate_arb_windows(duration_sec=300.0, arb_rate=0.5, seed=42)
        expected = 300.0 * 0.5  # 150
        # Poisson: allow wide margin (within 50% of expected)
        assert len(windows) > expected * 0.5
        assert len(windows) < expected * 1.5

    def test_windows_within_duration(self):
        """All arb windows open within the simulation duration."""
        windows = generate_arb_windows(duration_sec=100.0, arb_rate=1.0, seed=42)
        duration_ms = 100.0 * 1000.0
        for w in windows:
            assert w.open_time >= 0.0
            assert w.open_time < duration_ms
            assert w.close_time > w.open_time

    def test_window_fields_valid(self):
        """Each window has valid profit, roi, capital."""
        windows = generate_arb_windows(duration_sec=60.0, arb_rate=2.0, seed=42)
        assert len(windows) > 0
        for w in windows:
            assert 1.0 <= w.profit <= 20.0
            assert 2.0 <= w.roi_pct <= 15.0
            assert 20.0 <= w.capital <= 500.0
            assert w.event_id.startswith("arb_")

    def test_zero_rate_empty(self):
        """Zero arb rate produces no windows."""
        windows = generate_arb_windows(duration_sec=60.0, arb_rate=0.0, seed=42)
        assert windows == []


# ── Tests: simulate_interval ──────────────────────────────────────────


class TestSimulateInterval:
    def _make_windows(self) -> list[ArbWindow]:
        """Create a handful of known arb windows for testing."""
        return [
            ArbWindow(event_id="arb_0", open_time=100.0, close_time=300.0,
                       profit=5.0, roi_pct=10.0, capital=100.0),
            ArbWindow(event_id="arb_1", open_time=500.0, close_time=600.0,
                       profit=10.0, roi_pct=8.0, capital=200.0),
            ArbWindow(event_id="arb_2", open_time=1500.0, close_time=1700.0,
                       profit=3.0, roi_pct=5.0, capital=50.0),
        ]

    def test_all_detected_short_interval(self):
        """Very short interval (1ms) should detect all arb windows."""
        windows = self._make_windows()
        result = simulate_interval(windows, interval_ms=1)
        assert result.arbs_detected == len(windows)
        assert result.arbs_missed == 0
        assert result.detection_rate == 1.0

    def test_none_detected_huge_interval(self):
        """Very long interval (10s) may miss most short arb windows."""
        # All windows are < 200ms, interval is 10000ms
        windows = [
            ArbWindow(event_id="arb_0", open_time=50.0, close_time=100.0,
                       profit=5.0, roi_pct=10.0, capital=100.0),
            ArbWindow(event_id="arb_1", open_time=5050.0, close_time=5100.0,
                       profit=10.0, roi_pct=8.0, capital=200.0),
        ]
        result = simulate_interval(windows, interval_ms=10000)
        # With scan at t=0 and t=10000, likely misses both 50ms windows
        assert result.arbs_missed >= result.arbs_detected

    def test_faster_interval_higher_detection(self):
        """100ms interval detects more (or same) arbs than 1000ms."""
        windows = generate_arb_windows(duration_sec=60.0, arb_rate=2.0,
                                        mean_window_ms=150.0, seed=42)
        fast = simulate_interval(windows, interval_ms=100)
        slow = simulate_interval(windows, interval_ms=1000)
        assert fast.arbs_detected >= slow.arbs_detected

    def test_detection_delay_positive(self):
        """Mean detection delay is non-negative."""
        windows = self._make_windows()
        result = simulate_interval(windows, interval_ms=50)
        assert result.mean_detection_delay_ms >= 0.0

    def test_interval_preserved(self):
        """IntervalResult records the correct interval_ms."""
        windows = self._make_windows()
        result = simulate_interval(windows, interval_ms=250)
        assert result.interval_ms == 250

    def test_total_scans_correct(self):
        """Total scans = ceil(max_time / interval) roughly."""
        windows = [
            ArbWindow(event_id="arb_0", open_time=0.0, close_time=1000.0,
                       profit=5.0, roi_pct=10.0, capital=100.0),
        ]
        result = simulate_interval(windows, interval_ms=100)
        # Scans from 0 to close_time at 100ms intervals: at least 10
        assert result.total_scans >= 10


# ── Tests: EVS multiplier ─────────────────────────────────────────────


class TestEvsMultiplier:
    def test_baseline_multiplier_is_one(self):
        """1000ms interval has evs_multiplier = 1.0."""
        windows = generate_arb_windows(duration_sec=60.0, arb_rate=1.0, seed=42)
        report = compare_architectures(windows, intervals=[100, 500, 1000])
        baseline_result = [r for r in report.results if r.interval_ms == 1000][0]
        assert baseline_result.evs_multiplier == 1.0

    def test_faster_multiplier_higher(self):
        """Faster intervals have evs_multiplier >= 1.0."""
        windows = generate_arb_windows(duration_sec=120.0, arb_rate=1.0,
                                        mean_window_ms=200.0, seed=42)
        report = compare_architectures(windows, intervals=[100, 250, 500, 1000])
        multipliers = {r.interval_ms: r.evs_multiplier for r in report.results}
        # 100ms should have higher multiplier than 1000ms
        assert multipliers[100] >= multipliers[1000]
        assert multipliers[250] >= multipliers[1000]


# ── Tests: compare_architectures ──────────────────────────────────────


class TestCompareArchitectures:
    def test_sorted_by_interval(self):
        """Results are sorted by interval ascending."""
        windows = generate_arb_windows(duration_sec=30.0, arb_rate=1.0, seed=42)
        report = compare_architectures(windows, intervals=[1000, 100, 500, 250])
        intervals = [r.interval_ms for r in report.results]
        assert intervals == sorted(intervals)

    def test_report_metadata(self):
        """Report includes correct metadata."""
        windows = generate_arb_windows(duration_sec=30.0, arb_rate=1.0, seed=42)
        report = compare_architectures(windows, intervals=[100, 500, 1000])
        assert report.baseline_interval_ms == 1000
        assert report.total_arb_windows == len(windows)
        assert len(report.results) == 3


# ── Tests: empty windows ─────────────────────────────────────────────


class TestEmptyWindows:
    def test_simulate_empty(self):
        """Zero arbs produces zero EVS and zero detections."""
        result = simulate_interval([], interval_ms=100)
        assert result.arbs_detected == 0
        assert result.arbs_missed == 0
        assert result.evs == 0.0

    def test_compare_empty(self):
        """All intervals produce zero EVS for empty windows."""
        report = compare_architectures([], intervals=[100, 500, 1000])
        for r in report.results:
            assert r.evs == 0.0
            assert r.arbs_detected == 0


# ── Tests: CLI ────────────────────────────────────────────────────────


class TestCLI:
    def test_end_to_end(self, tmp_path):
        """Run main(), check JSON output exists and is valid."""
        output_path = tmp_path / "latency_report.json"
        latency_main([
            "--output", str(output_path),
            "--duration", "30",
            "--arb-rate", "1.0",
            "--mean-window", "200",
            "--intervals", "100,500,1000",
            "--seed", "42",
        ])

        assert output_path.exists()
        with open(output_path) as f:
            data = json.load(f)

        assert "baseline_interval_ms" in data
        assert "total_arb_windows" in data
        assert "duration_sec" in data
        assert "results" in data
        assert len(data["results"]) == 3
        # Results should be sorted by interval
        intervals = [r["interval_ms"] for r in data["results"]]
        assert intervals == sorted(intervals)
        # Each result has expected fields
        for r in data["results"]:
            assert "interval_ms" in r
            assert "total_scans" in r
            assert "arbs_detected" in r
            assert "arbs_missed" in r
            assert "detection_rate" in r
            assert "mean_detection_delay_ms" in r
            assert "evs" in r
            assert "evs_multiplier" in r

    def test_default_seed_deterministic(self, tmp_path):
        """Default seed produces deterministic output."""
        out1 = tmp_path / "report1.json"
        out2 = tmp_path / "report2.json"
        args = ["--output", "", "--duration", "10", "--arb-rate", "1.0", "--seed", "42"]
        args_1 = args.copy()
        args_1[1] = str(out1)
        args_2 = args.copy()
        args_2[1] = str(out2)

        latency_main(args_1)
        latency_main(args_2)

        with open(out1) as f:
            data1 = json.load(f)
        with open(out2) as f:
            data2 = json.load(f)

        assert data1 == data2
