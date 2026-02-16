"""
Unit tests for state/checkpoint.py -- tracker state persistence.
"""

from __future__ import annotations

import json
import sqlite3
import threading
import time
from pathlib import Path
from unittest.mock import MagicMock

import pytest

from state.checkpoint import CheckpointManager, Serializable


# ---------------------------------------------------------------------------
# Fixtures / helpers
# ---------------------------------------------------------------------------

class FakeTracker:
    """Minimal Serializable tracker for testing."""

    def __init__(self, value: int = 0, name: str = "default"):
        self.value = value
        self.name = name

    def to_dict(self) -> dict:
        return {"value": self.value, "name": self.name}

    @classmethod
    def from_dict(cls, data: dict) -> FakeTracker:
        return cls(value=data.get("value", 0), name=data.get("name", "default"))


class BrokenFromDict:
    """Tracker whose from_dict always raises."""

    def to_dict(self) -> dict:
        return {"ok": True}

    @classmethod
    def from_dict(cls, data: dict) -> BrokenFromDict:
        raise RuntimeError("from_dict exploded")


class BrokenSerializer:
    """Tracker whose to_dict raises."""

    def to_dict(self) -> dict:
        raise RuntimeError("serialize boom")

    @classmethod
    def from_dict(cls, data: dict) -> BrokenSerializer:
        return cls()


@pytest.fixture
def db_path(tmp_path: Path) -> Path:
    return tmp_path / "test_state.db"


@pytest.fixture
def mgr(db_path: Path) -> CheckpointManager:
    m = CheckpointManager(db_path=db_path, auto_save_interval=3)
    yield m
    m.close()


# ---------------------------------------------------------------------------
# Round-trip save/load
# ---------------------------------------------------------------------------

class TestSaveLoad:
    def test_round_trip(self, mgr: CheckpointManager):
        tracker = FakeTracker(value=42, name="arb")
        mgr.save("arb_tracker", tracker, cycle_num=5)

        restored = mgr.load("arb_tracker", FakeTracker)
        assert restored is not None
        assert restored.value == 42
        assert restored.name == "arb"

    def test_load_nonexistent_returns_none(self, mgr: CheckpointManager):
        assert mgr.load("missing", FakeTracker) is None

    def test_overwrite_updates_value(self, mgr: CheckpointManager):
        mgr.save("t", FakeTracker(value=1))
        mgr.save("t", FakeTracker(value=2))

        restored = mgr.load("t", FakeTracker)
        assert restored is not None
        assert restored.value == 2

    def test_multiple_trackers_independent(self, mgr: CheckpointManager):
        mgr.save("a", FakeTracker(value=10, name="alpha"))
        mgr.save("b", FakeTracker(value=20, name="beta"))

        a = mgr.load("a", FakeTracker)
        b = mgr.load("b", FakeTracker)
        assert a.value == 10
        assert b.value == 20
        assert a.name == "alpha"
        assert b.name == "beta"

    def test_cycle_num_persisted(self, mgr: CheckpointManager):
        """cycle_num is stored and retrievable via load_metadata."""
        mgr.save("t1", FakeTracker(value=1), cycle_num=99)
        meta = mgr.load_metadata("t1")
        assert meta is not None
        assert meta["cycle_num"] == 99


# ---------------------------------------------------------------------------
# Corrupt JSON handling
# ---------------------------------------------------------------------------

class TestCorruptJSON:
    def test_corrupt_json_returns_none(self, mgr: CheckpointManager, db_path: Path):
        # Manually insert corrupt JSON
        conn = sqlite3.connect(str(db_path))
        conn.execute(
            "INSERT OR REPLACE INTO tracker_state (tracker_name, data_json, cycle_num, updated_at) "
            "VALUES (?, ?, ?, ?)",
            ("broken", "{invalid json!!!", 0, time.time()),
        )
        conn.commit()
        conn.close()

        result = mgr.load("broken", FakeTracker)
        assert result is None

    def test_from_dict_exception_returns_none(self, mgr: CheckpointManager):
        # Save valid JSON that BrokenFromDict.from_dict can't handle
        tracker = FakeTracker(value=1)
        mgr.save("will_break", tracker)

        result = mgr.load("will_break", BrokenFromDict)
        assert result is None

    def test_null_json_returns_none(self, mgr: CheckpointManager, db_path: Path):
        conn = sqlite3.connect(str(db_path))
        conn.execute(
            "INSERT OR REPLACE INTO tracker_state (tracker_name, data_json, cycle_num, updated_at) "
            "VALUES (?, ?, ?, ?)",
            ("null_data", "null", 0, time.time()),
        )
        conn.commit()
        conn.close()

        # json.loads("null") returns None, then from_dict gets None
        # which should be caught gracefully
        result = mgr.load("null_data", FakeTracker)
        # from_dict({}) or from_dict(None) — either way, should not crash
        assert result is not None or result is None  # just shouldn't raise


# ---------------------------------------------------------------------------
# Auto-save via tick()
# ---------------------------------------------------------------------------

class TestAutoSave:
    def test_tick_saves_at_interval(self, mgr: CheckpointManager):
        tracker = FakeTracker(value=100)
        mgr.register("t", tracker)

        # Ticks 1, 2 should not save
        assert mgr.tick() == 0
        assert mgr.tick() == 0

        # Tick 3 (interval=3) should save
        assert mgr.tick() == 1

        # Verify it was persisted
        restored = mgr.load("t", FakeTracker)
        assert restored is not None
        assert restored.value == 100

    def test_tick_saves_multiple_trackers(self, mgr: CheckpointManager):
        mgr.register("a", FakeTracker(value=1))
        mgr.register("b", FakeTracker(value=2))

        # Advance to save interval
        mgr.tick()
        mgr.tick()
        saved = mgr.tick()

        assert saved == 2

    def test_mutated_tracker_saved_on_tick(self, mgr: CheckpointManager):
        tracker = FakeTracker(value=0)
        mgr.register("mut", tracker)

        # Mutate between ticks
        tracker.value = 999

        mgr.tick()
        mgr.tick()
        mgr.tick()  # saves

        restored = mgr.load("mut", FakeTracker)
        assert restored.value == 999

    def test_unregister_stops_auto_save(self, mgr: CheckpointManager):
        tracker = FakeTracker(value=50)
        mgr.register("t", tracker)
        mgr.unregister("t")

        mgr.tick()
        mgr.tick()
        saved = mgr.tick()

        assert saved == 0


# ---------------------------------------------------------------------------
# save_all
# ---------------------------------------------------------------------------

class TestSaveAll:
    def test_save_all_persists_all(self, mgr: CheckpointManager):
        mgr.register("x", FakeTracker(value=10))
        mgr.register("y", FakeTracker(value=20))

        count = mgr.save_all()
        assert count == 2

        assert mgr.load("x", FakeTracker).value == 10
        assert mgr.load("y", FakeTracker).value == 20

    def test_save_all_explicit_dict(self, mgr: CheckpointManager):
        """save_all with explicit trackers dict saves all provided trackers."""
        trackers = {
            "a": FakeTracker(value=1, name="alpha"),
            "b": FakeTracker(value=2, name="beta"),
            "c": FakeTracker(value=3, name="gamma"),
        }
        count = mgr.save_all(trackers, cycle_num=10)
        assert count == 3

        for name, tracker in trackers.items():
            restored = mgr.load(name, FakeTracker)
            assert restored is not None
            assert restored.value == tracker.value
            assert restored.name == tracker.name

    def test_save_all_with_failing_tracker(self, mgr: CheckpointManager):
        good = FakeTracker(value=1)
        bad = MagicMock()
        bad.to_dict.side_effect = RuntimeError("serialize fail")

        mgr.register("good", good)
        mgr.register("bad", bad)

        # Should save good, skip bad, not crash
        count = mgr.save_all()
        assert count == 1
        assert mgr.load("good", FakeTracker).value == 1

    def test_save_all_explicit_with_broken_serializer(self, mgr: CheckpointManager):
        """If one tracker in explicit dict fails to_dict, others still get saved."""
        trackers = {
            "good1": FakeTracker(value=10),
            "bad": BrokenSerializer(),
            "good2": FakeTracker(value=20),
        }
        count = mgr.save_all(trackers, cycle_num=1)
        assert count == 2

        assert mgr.load("good1", FakeTracker) is not None
        assert mgr.load("bad", FakeTracker) is None
        assert mgr.load("good2", FakeTracker) is not None

    def test_save_all_empty(self, mgr: CheckpointManager):
        """save_all with no registered trackers returns 0."""
        assert mgr.save_all() == 0


# ---------------------------------------------------------------------------
# Concurrent access
# ---------------------------------------------------------------------------

class TestConcurrency:
    def test_concurrent_save_load(self, db_path: Path):
        mgr = CheckpointManager(db_path=db_path)
        errors: list[Exception] = []

        def writer(tid: int):
            try:
                for i in range(20):
                    mgr.save(f"tracker_{tid}", FakeTracker(value=i))
            except Exception as e:
                errors.append(e)

        def reader(tid: int):
            try:
                for _ in range(20):
                    mgr.load(f"tracker_{tid}", FakeTracker)
            except Exception as e:
                errors.append(e)

        threads = []
        for i in range(5):
            threads.append(threading.Thread(target=writer, args=(i,)))
            threads.append(threading.Thread(target=reader, args=(i,)))

        for t in threads:
            t.start()
        for t in threads:
            t.join()

        mgr.close()
        assert errors == []

    def test_concurrent_saves_no_corruption(self, db_path: Path):
        """Multiple threads saving different keys — all entries persisted."""
        mgr = CheckpointManager(db_path=db_path)
        errors: list[Exception] = []

        def writer(thread_id: int) -> None:
            try:
                for i in range(20):
                    name = f"thread_{thread_id}_iter_{i}"
                    mgr.save(name, FakeTracker(value=thread_id * 100 + i))
            except Exception as e:
                errors.append(e)

        threads = [threading.Thread(target=writer, args=(tid,)) for tid in range(5)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        assert errors == [], f"Thread errors: {errors}"
        entries = mgr.list_checkpoints()
        assert len(entries) == 100  # 5 threads × 20 iterations
        mgr.close()


# ---------------------------------------------------------------------------
# list_checkpoints / delete
# ---------------------------------------------------------------------------

class TestListAndDelete:
    def test_list_checkpoints(self, mgr: CheckpointManager):
        mgr.save("alpha", FakeTracker(value=1), cycle_num=5)
        mgr.save("beta", FakeTracker(value=2), cycle_num=10)

        items = mgr.list_checkpoints()
        names = [i["name"] for i in items]
        assert "alpha" in names
        assert "beta" in names
        assert all("updated_at" in i for i in items)
        assert all("data_bytes" in i for i in items)
        assert all("cycle_num" in i for i in items)
        assert all("age_sec" in i for i in items)
        # Sorted by name
        assert names == ["alpha", "beta"]

    def test_delete_existing(self, mgr: CheckpointManager):
        mgr.save("doomed", FakeTracker(value=0))
        assert mgr.delete("doomed") is True
        assert mgr.load("doomed", FakeTracker) is None

    def test_delete_nonexistent(self, mgr: CheckpointManager):
        assert mgr.delete("ghost") is False

    def test_list_empty(self, mgr: CheckpointManager):
        assert mgr.list_checkpoints() == []


# ---------------------------------------------------------------------------
# load_metadata
# ---------------------------------------------------------------------------

class TestLoadMetadata:
    def test_metadata_values(self, mgr: CheckpointManager):
        """load_metadata returns cycle_num, age_sec, data_bytes."""
        mgr.save("t1", FakeTracker(value=42, name="big"), cycle_num=42)
        meta = mgr.load_metadata("t1")

        assert meta is not None
        assert meta["cycle_num"] == 42
        assert meta["age_sec"] >= 0
        assert meta["data_bytes"] > 0

    def test_metadata_missing(self, mgr: CheckpointManager):
        """load_metadata on missing tracker returns None."""
        assert mgr.load_metadata("nope") is None


# ---------------------------------------------------------------------------
# stats property
# ---------------------------------------------------------------------------

class TestStats:
    def test_stats_tracking(self, mgr: CheckpointManager):
        """stats property tracks save/load counts."""
        assert mgr.stats["save_count"] == 0
        assert mgr.stats["load_count"] == 0

        mgr.save("s1", FakeTracker(value=1))
        mgr.save("s2", FakeTracker(value=2))
        assert mgr.stats["save_count"] == 2

        mgr.load("s1", FakeTracker)
        assert mgr.stats["load_count"] == 1
        assert mgr.stats["checkpoints"] == 2
        assert mgr.stats["db_path"] == str(mgr._db_path)


# ---------------------------------------------------------------------------
# close and reopen
# ---------------------------------------------------------------------------

class TestCloseAndReopen:
    def test_persistence_across_connections(self, db_path: Path):
        """Save, close, open new manager → data is still there."""
        mgr1 = CheckpointManager(db_path=db_path)
        mgr1.save("persist_test", FakeTracker(value=777), cycle_num=7)
        mgr1.close()

        mgr2 = CheckpointManager(db_path=db_path)
        restored = mgr2.load("persist_test", FakeTracker)
        assert restored is not None
        assert restored.value == 777

        meta = mgr2.load_metadata("persist_test")
        assert meta["cycle_num"] == 7
        mgr2.close()


# ---------------------------------------------------------------------------
# Serializable protocol check
# ---------------------------------------------------------------------------

class TestSerializableProtocol:
    def test_fake_tracker_is_serializable(self):
        assert isinstance(FakeTracker(), Serializable)

    def test_non_serializable_rejected(self):
        assert not isinstance("not a tracker", Serializable)


# ---------------------------------------------------------------------------
# Real tracker integration (ArbTracker from scanner/confidence.py)
# ---------------------------------------------------------------------------

class TestRealTrackerIntegration:
    def test_arb_tracker_round_trip(self, mgr: CheckpointManager):
        from scanner.confidence import ArbTracker

        tracker = ArbTracker()
        tracker._history = {"evt-1": [1, 2, 3], "evt-2": [5]}
        tracker._failures = {"evt-1": 2}

        mgr.save("arb", tracker)
        restored = mgr.load("arb", ArbTracker)

        assert restored is not None
        assert restored._history == tracker._history
        assert restored._failures == tracker._failures
        assert restored.confidence("evt-1") == tracker.confidence("evt-1")

    def test_spike_detector_round_trip(self, mgr: CheckpointManager):
        from scanner.spike import SpikeDetector

        detector = SpikeDetector(threshold_pct=3.0, window_sec=15.0)
        now = time.time()
        detector.register_token("tok-1", "evt-1")
        detector.update("tok-1", 0.50, now - 5)
        detector.update("tok-1", 0.55, now)

        mgr.save("spikes", detector)
        restored = mgr.load("spikes", SpikeDetector)

        assert restored is not None
        assert restored.threshold_pct == 3.0
        assert restored.window_sec == 15.0
        assert len(restored._histories) == 1

    def test_maker_gate_round_trip(self, mgr: CheckpointManager):
        from scanner.maker import MakerPersistenceGate

        gate = MakerPersistenceGate(min_consecutive_cycles=5)
        gate._streaks = {"mkt-1": 4, "mkt-2": 1}

        mgr.save("maker", gate)
        restored = mgr.load("maker", MakerPersistenceGate)

        assert restored is not None
        assert restored.min_consecutive_cycles == 5
        assert restored._streaks == gate._streaks
