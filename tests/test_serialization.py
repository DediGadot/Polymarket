"""Round-trip serialization tests for all tracker classes."""

from __future__ import annotations

import json
import time

from scanner.confidence import ArbTracker
from scanner.realized_ev import EVStats, RealizedEVTracker
from scanner.spike import PriceHistory, SpikeDetector
from scanner.maker import MakerPersistenceGate


# ---------------------------------------------------------------------------
# ArbTracker
# ---------------------------------------------------------------------------

class TestArbTrackerSerialization:
    def test_round_trip(self):
        tracker = ArbTracker()
        tracker._history = {
            "evt-1": [1, 2, 3],
            "evt-2": [5, 7],
        }
        tracker._failures = {"evt-1": 2, "evt-3": 1}
        tracker._max_history = 15

        d = tracker.to_dict()
        assert json.dumps(d)  # JSON-safe

        restored = ArbTracker.from_dict(d)
        assert restored._history == tracker._history
        assert restored._failures == tracker._failures
        assert restored._max_history == tracker._max_history

    def test_behavioral_equivalence(self):
        tracker = ArbTracker()
        tracker._history = {"evt-1": [4, 5]}
        tracker._failures = {"evt-1": 1}

        restored = ArbTracker.from_dict(tracker.to_dict())
        # Both should return the same confidence score
        assert tracker.confidence("evt-1") == restored.confidence("evt-1")
        assert tracker.confidence("unknown") == restored.confidence("unknown")

    def test_from_empty_dict(self):
        tracker = ArbTracker.from_dict({})
        assert tracker._history == {}
        assert tracker._failures == {}
        assert tracker._max_history == 10

    def test_from_partial_dict(self):
        tracker = ArbTracker.from_dict({"history": {"e1": [1]}})
        assert tracker._history == {"e1": [1]}
        assert tracker._failures == {}
        assert tracker._max_history == 10

    def test_immutability_of_source(self):
        """Serialized dict mutations must not affect original tracker."""
        tracker = ArbTracker()
        tracker._history = {"evt-1": [1, 2]}
        d = tracker.to_dict()
        d["history"]["evt-1"].append(99)
        assert tracker._history["evt-1"] == [1, 2]


# ---------------------------------------------------------------------------
# EVStats
# ---------------------------------------------------------------------------

class TestEVStatsSerialization:
    def test_round_trip(self):
        stats = EVStats(observations=10, full_fills=3, orphan_hedges=1, realized_pnl=0.45)
        d = stats.to_dict()
        assert json.dumps(d)

        restored = EVStats.from_dict(d)
        assert restored.observations == stats.observations
        assert restored.full_fills == stats.full_fills
        assert restored.orphan_hedges == stats.orphan_hedges
        assert restored.realized_pnl == stats.realized_pnl

    def test_from_empty_dict(self):
        stats = EVStats.from_dict({})
        assert stats.observations == 0
        assert stats.realized_pnl == 0.0


# ---------------------------------------------------------------------------
# RealizedEVTracker
# ---------------------------------------------------------------------------

class TestRealizedEVTrackerSerialization:
    def test_round_trip(self):
        tracker = RealizedEVTracker(alpha_full=3.0, beta_full=7.0, orphan_loss_ratio=0.15)
        tracker._stats = {
            "MAKER_REBALANCE:evt-1:tok-a,tok-b": EVStats(
                observations=20, full_fills=5, orphan_hedges=2, realized_pnl=1.23
            ),
        }

        d = tracker.to_dict()
        assert json.dumps(d)

        restored = RealizedEVTracker.from_dict(d)
        assert restored.alpha_full == 3.0
        assert restored.beta_full == 7.0
        assert restored.orphan_loss_ratio == 0.15
        key = "MAKER_REBALANCE:evt-1:tok-a,tok-b"
        assert restored._stats[key].observations == 20
        assert restored._stats[key].full_fills == 5
        assert restored._stats[key].realized_pnl == 1.23

    def test_from_empty_dict(self):
        tracker = RealizedEVTracker.from_dict({})
        assert tracker.alpha_full == 2.0
        assert tracker.beta_full == 6.0
        assert tracker._stats == {}

    def test_from_partial_dict(self):
        tracker = RealizedEVTracker.from_dict({"alpha_full": 5.0})
        assert tracker.alpha_full == 5.0
        assert tracker.beta_full == 6.0  # default


# ---------------------------------------------------------------------------
# PriceHistory
# ---------------------------------------------------------------------------

class TestPriceHistorySerialization:
    def test_round_trip(self):
        h = PriceHistory(max_window_sec=120.0)
        now = time.time()
        h.record(0.50, now - 10)
        h.record(0.55, now - 5)
        h.record(0.60, now)

        d = h.to_dict()
        assert json.dumps(d)

        restored = PriceHistory.from_dict(d)
        assert restored.max_window_sec == 120.0
        assert len(restored) == len(h)
        assert restored.latest == h.latest

    def test_velocity_preserved(self):
        h = PriceHistory(max_window_sec=300.0)
        now = time.time()
        h.record(0.50, now - 10)
        h.record(0.60, now)

        restored = PriceHistory.from_dict(h.to_dict())
        assert h.velocity(30.0) == restored.velocity(30.0)

    def test_from_empty_dict(self):
        h = PriceHistory.from_dict({})
        assert h.max_window_sec == 300.0
        assert len(h) == 0

    def test_from_empty_points(self):
        h = PriceHistory.from_dict({"max_window_sec": 60.0, "points": []})
        assert h.max_window_sec == 60.0
        assert len(h) == 0


# ---------------------------------------------------------------------------
# SpikeDetector
# ---------------------------------------------------------------------------

class TestSpikeDetectorSerialization:
    def test_round_trip(self):
        detector = SpikeDetector(threshold_pct=3.0, window_sec=15.0, cooldown_sec=45.0)
        now = time.time()
        detector.register_token("tok-1", "evt-1")
        detector.register_token("tok-2", "evt-1")
        detector.update("tok-1", 0.50, now - 5)
        detector.update("tok-1", 0.55, now)
        detector.update("tok-2", 0.30, now)
        detector._cooldowns["tok-1"] = now - 10

        d = detector.to_dict()
        assert json.dumps(d)

        restored = SpikeDetector.from_dict(d)
        assert restored.threshold_pct == 3.0
        assert restored.window_sec == 15.0
        assert restored.cooldown_sec == 45.0
        assert restored._token_events == detector._token_events
        assert restored._cooldowns == detector._cooldowns
        assert len(restored._histories) == len(detector._histories)
        assert len(restored._histories["tok-1"]) == len(detector._histories["tok-1"])

    def test_velocity_preserved(self):
        detector = SpikeDetector()
        now = time.time()
        detector.update("tok-1", 0.50, now - 10)
        detector.update("tok-1", 0.60, now)

        restored = SpikeDetector.from_dict(detector.to_dict())
        assert detector.get_velocity("tok-1") == restored.get_velocity("tok-1")

    def test_from_empty_dict(self):
        detector = SpikeDetector.from_dict({})
        assert detector.threshold_pct == 5.0
        assert detector.window_sec == 30.0
        assert detector.cooldown_sec == 60.0
        assert detector._histories == {}
        assert detector._cooldowns == {}
        assert detector._token_events == {}

    def test_from_partial_dict(self):
        detector = SpikeDetector.from_dict({"threshold_pct": 10.0})
        assert detector.threshold_pct == 10.0
        assert detector.window_sec == 30.0  # default


# ---------------------------------------------------------------------------
# MakerPersistenceGate
# ---------------------------------------------------------------------------

class TestMakerPersistenceGateSerialization:
    def test_round_trip(self):
        gate = MakerPersistenceGate(min_consecutive_cycles=5)
        gate._streaks = {"mkt-1": 4, "mkt-2": 1, "mkt-3": 7}

        d = gate.to_dict()
        assert json.dumps(d)

        restored = MakerPersistenceGate.from_dict(d)
        assert restored.min_consecutive_cycles == 5
        assert restored._streaks == gate._streaks

    def test_behavioral_equivalence(self):
        gate = MakerPersistenceGate(min_consecutive_cycles=3)
        gate._streaks = {"mkt-1": 2}

        restored = MakerPersistenceGate.from_dict(gate.to_dict())
        # Both should agree on next mark_viable result
        gate.begin_cycle()
        restored.begin_cycle()
        assert gate.mark_viable("mkt-1") == restored.mark_viable("mkt-1")

    def test_from_empty_dict(self):
        gate = MakerPersistenceGate.from_dict({})
        assert gate.min_consecutive_cycles == 3
        assert gate._streaks == {}

    def test_from_partial_dict(self):
        gate = MakerPersistenceGate.from_dict({"min_consecutive_cycles": 7})
        assert gate.min_consecutive_cycles == 7
        assert gate._streaks == {}

    def test_viable_set_not_serialized(self):
        """_viable_this_cycle is transient and should not be in serialized output."""
        gate = MakerPersistenceGate()
        gate._viable_this_cycle = {"mkt-1", "mkt-2"}
        d = gate.to_dict()
        assert "viable_this_cycle" not in d
        # Restored gate starts with empty viable set
        restored = MakerPersistenceGate.from_dict(d)
        assert restored._viable_this_cycle == set()

    def test_immutability_of_source(self):
        """Serialized dict mutations must not affect original gate."""
        gate = MakerPersistenceGate()
        gate._streaks = {"mkt-1": 5}
        d = gate.to_dict()
        d["streaks"]["mkt-1"] = 0
        assert gate._streaks["mkt-1"] == 5
