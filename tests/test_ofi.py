"""Tests for scanner.ofi — Order Flow Imbalance tracker."""

from __future__ import annotations

import threading
import time

import pytest

from scanner.ofi import OFISnapshot, OFITracker


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _feed(
    tracker: OFITracker,
    token_id: str,
    events: list[tuple[str, float]],
    base_ts: float = 1000.0,
    dt: float = 0.1,
) -> None:
    """Feed a sequence of (side, size) events into the tracker."""
    for i, (side, size) in enumerate(events):
        tracker.record(token_id, side, size, timestamp=base_ts + i * dt)


# ---------------------------------------------------------------------------
# 1. record + get_ofi
# ---------------------------------------------------------------------------


class TestOFIBasic:
    def test_record_buy_sell_net(self) -> None:
        tracker = OFITracker(window_sec=30.0)
        tracker.record("tok_a", "BUY", 100.0, timestamp=1000.0)
        tracker.record("tok_a", "SELL", 30.0, timestamp=1000.1)
        assert tracker.get_ofi("tok_a") == pytest.approx(70.0)

    def test_record_all_buys(self) -> None:
        tracker = OFITracker(window_sec=30.0)
        tracker.record("tok_a", "BUY", 50.0, timestamp=1000.0)
        tracker.record("tok_a", "BUY", 25.0, timestamp=1000.1)
        assert tracker.get_ofi("tok_a") == pytest.approx(75.0)

    def test_record_all_sells(self) -> None:
        tracker = OFITracker(window_sec=30.0)
        tracker.record("tok_a", "SELL", 40.0, timestamp=1000.0)
        tracker.record("tok_a", "SELL", 60.0, timestamp=1000.1)
        assert tracker.get_ofi("tok_a") == pytest.approx(-100.0)

    def test_case_insensitive_side(self) -> None:
        tracker = OFITracker(window_sec=30.0)
        tracker.record("tok_a", "buy", 10.0, timestamp=1000.0)
        tracker.record("tok_a", "Sell", 3.0, timestamp=1000.1)
        assert tracker.get_ofi("tok_a") == pytest.approx(7.0)


# ---------------------------------------------------------------------------
# 2. normalized_ofi bounds
# ---------------------------------------------------------------------------


class TestNormalizedOFI:
    def test_all_buys_normalized_is_one(self) -> None:
        tracker = OFITracker(window_sec=30.0)
        tracker.record("tok_a", "BUY", 500.0, timestamp=1000.0)
        tracker.record("tok_a", "BUY", 300.0, timestamp=1000.1)
        assert tracker.get_normalized_ofi("tok_a") == pytest.approx(1.0)

    def test_all_sells_normalized_is_neg_one(self) -> None:
        tracker = OFITracker(window_sec=30.0)
        tracker.record("tok_a", "SELL", 200.0, timestamp=1000.0)
        tracker.record("tok_a", "SELL", 100.0, timestamp=1000.1)
        assert tracker.get_normalized_ofi("tok_a") == pytest.approx(-1.0)

    def test_mixed_normalized_bounds(self) -> None:
        tracker = OFITracker(window_sec=30.0)
        tracker.record("tok_a", "BUY", 100.0, timestamp=1000.0)
        tracker.record("tok_a", "SELL", 50.0, timestamp=1000.1)
        # ofi=50, total=150 → 50/150 ≈ 0.333
        norm = tracker.get_normalized_ofi("tok_a")
        assert -1.0 <= norm <= 1.0
        assert norm == pytest.approx(50.0 / 150.0)

    def test_balanced_normalized_is_zero(self) -> None:
        tracker = OFITracker(window_sec=30.0)
        tracker.record("tok_a", "BUY", 100.0, timestamp=1000.0)
        tracker.record("tok_a", "SELL", 100.0, timestamp=1000.1)
        assert tracker.get_normalized_ofi("tok_a") == pytest.approx(0.0)


# ---------------------------------------------------------------------------
# 3. window pruning
# ---------------------------------------------------------------------------


class TestWindowPruning:
    def test_events_expire_after_window(self) -> None:
        tracker = OFITracker(window_sec=5.0)
        tracker.record("tok_a", "BUY", 100.0, timestamp=1000.0)
        tracker.record("tok_a", "BUY", 50.0, timestamp=1003.0)
        # At t=1000, both are within 5s window
        assert tracker.get_ofi("tok_a") == pytest.approx(150.0)
        # Record new event past the window for first event
        tracker.record("tok_a", "BUY", 10.0, timestamp=1006.0)
        # First event (t=1000) should be pruned (cutoff = 1006 - 5 = 1001)
        assert tracker.get_ofi("tok_a") == pytest.approx(60.0)  # 50 + 10

    def test_all_events_expire_removes_token(self) -> None:
        tracker = OFITracker(window_sec=2.0)
        tracker.record("tok_a", "BUY", 100.0, timestamp=1000.0)
        assert tracker.tracked_tokens == 1
        # Recording on tok_a far in future prunes the old event
        tracker.record("tok_a", "BUY", 0.0, timestamp=1010.0)
        # The t=1000 event is pruned (cutoff=1008), leaving only the 0-size event
        # which itself is in window — but OFI is 0 since size was 0
        assert tracker.get_ofi("tok_a") == pytest.approx(0.0)
        # cleanup_stale can then remove it entirely
        tracker.cleanup_stale(active_tokens=set())
        assert tracker.tracked_tokens == 0


# ---------------------------------------------------------------------------
# 4. divergence
# ---------------------------------------------------------------------------


class TestDivergence:
    def test_divergence_basic(self) -> None:
        tracker = OFITracker(window_sec=30.0)
        tracker.record("tok_a", "BUY", 100.0, timestamp=1000.0)
        tracker.record("tok_b", "SELL", 50.0, timestamp=1000.0)
        # OFI_a = 100, OFI_b = -50, divergence = |100 - (-50)| = 150
        assert tracker.get_divergence("tok_a", "tok_b") == pytest.approx(150.0)

    def test_divergence_same_direction(self) -> None:
        tracker = OFITracker(window_sec=30.0)
        tracker.record("tok_a", "BUY", 100.0, timestamp=1000.0)
        tracker.record("tok_b", "BUY", 40.0, timestamp=1000.0)
        assert tracker.get_divergence("tok_a", "tok_b") == pytest.approx(60.0)

    def test_divergence_with_untracked_token(self) -> None:
        tracker = OFITracker(window_sec=30.0)
        tracker.record("tok_a", "BUY", 100.0, timestamp=1000.0)
        # tok_b never recorded → OFI = 0
        assert tracker.get_divergence("tok_a", "tok_b") == pytest.approx(100.0)

    def test_divergence_zero_when_equal(self) -> None:
        tracker = OFITracker(window_sec=30.0)
        tracker.record("tok_a", "BUY", 80.0, timestamp=1000.0)
        tracker.record("tok_b", "BUY", 80.0, timestamp=1000.0)
        assert tracker.get_divergence("tok_a", "tok_b") == pytest.approx(0.0)


# ---------------------------------------------------------------------------
# 5. snapshot
# ---------------------------------------------------------------------------


class TestSnapshot:
    def test_snapshot_returns_frozen_dataclass(self) -> None:
        tracker = OFITracker(window_sec=30.0)
        tracker.record("tok_a", "BUY", 100.0, timestamp=1000.0)
        tracker.record("tok_a", "SELL", 30.0, timestamp=1000.1)
        snap = tracker.get_snapshot("tok_a")
        assert isinstance(snap, OFISnapshot)
        assert snap.token_id == "tok_a"
        assert snap.ofi == pytest.approx(70.0)
        assert snap.total_volume == pytest.approx(130.0)
        assert snap.normalized_ofi == pytest.approx(70.0 / 130.0)
        assert snap.event_count == 2

    def test_snapshot_empty_token(self) -> None:
        tracker = OFITracker(window_sec=30.0)
        snap = tracker.get_snapshot("nonexistent")
        assert snap.ofi == 0.0
        assert snap.normalized_ofi == 0.0
        assert snap.total_volume == 0.0
        assert snap.event_count == 0

    def test_snapshot_is_immutable(self) -> None:
        tracker = OFITracker(window_sec=30.0)
        tracker.record("tok_a", "BUY", 100.0, timestamp=1000.0)
        snap = tracker.get_snapshot("tok_a")
        with pytest.raises(AttributeError):
            snap.ofi = 999.0  # type: ignore[misc]


# ---------------------------------------------------------------------------
# 6. cleanup_stale
# ---------------------------------------------------------------------------


class TestCleanupStale:
    def test_cleanup_removes_inactive_tokens(self) -> None:
        tracker = OFITracker(window_sec=30.0)
        tracker.record("tok_a", "BUY", 10.0, timestamp=1000.0)
        tracker.record("tok_b", "BUY", 20.0, timestamp=1000.0)
        tracker.record("tok_c", "BUY", 30.0, timestamp=1000.0)
        assert tracker.tracked_tokens == 3

        removed = tracker.cleanup_stale(active_tokens={"tok_a"})
        assert removed == 2
        assert tracker.tracked_tokens == 1
        assert tracker.get_ofi("tok_a") == pytest.approx(10.0)

    def test_cleanup_noop_when_all_active(self) -> None:
        tracker = OFITracker(window_sec=30.0)
        tracker.record("tok_a", "BUY", 10.0, timestamp=1000.0)
        removed = tracker.cleanup_stale(active_tokens={"tok_a"})
        assert removed == 0
        assert tracker.tracked_tokens == 1


# ---------------------------------------------------------------------------
# 7. serialization round-trip
# ---------------------------------------------------------------------------


class TestSerialization:
    def test_to_dict_from_dict_roundtrip(self) -> None:
        tracker = OFITracker(window_sec=15.0)
        tracker.record("tok_a", "BUY", 100.0, timestamp=1000.0)
        tracker.record("tok_a", "SELL", 30.0, timestamp=1000.1)
        tracker.record("tok_b", "SELL", 50.0, timestamp=1000.2)

        data = tracker.to_dict()
        restored = OFITracker.from_dict(data)

        assert restored._window_sec == 15.0
        assert restored.get_ofi("tok_a") == pytest.approx(70.0)
        assert restored.get_ofi("tok_b") == pytest.approx(-50.0)

    def test_from_dict_empty(self) -> None:
        restored = OFITracker.from_dict({})
        assert restored._window_sec == 30.0
        assert restored.tracked_tokens == 0


# ---------------------------------------------------------------------------
# 8. thread safety
# ---------------------------------------------------------------------------


class TestThreadSafety:
    def test_concurrent_record_and_read(self) -> None:
        """Hammer record() and get_ofi() from multiple threads."""
        tracker = OFITracker(window_sec=60.0)
        errors: list[Exception] = []
        n_writes = 500
        n_readers = 4

        def writer(token: str, side: str) -> None:
            try:
                for i in range(n_writes):
                    tracker.record(token, side, 1.0, timestamp=1000.0 + i * 0.001)
            except Exception as exc:
                errors.append(exc)

        def reader(token: str) -> None:
            try:
                for _ in range(n_writes):
                    tracker.get_ofi(token)
                    tracker.get_normalized_ofi(token)
                    tracker.get_snapshot(token)
            except Exception as exc:
                errors.append(exc)

        threads = [
            threading.Thread(target=writer, args=("tok_a", "BUY")),
            threading.Thread(target=writer, args=("tok_a", "SELL")),
            threading.Thread(target=writer, args=("tok_b", "BUY")),
        ]
        threads += [
            threading.Thread(target=reader, args=("tok_a",))
            for _ in range(n_readers)
        ]

        for t in threads:
            t.start()
        for t in threads:
            t.join(timeout=10.0)

        assert errors == [], f"Thread errors: {errors}"


# ---------------------------------------------------------------------------
# 9. empty / untracked token
# ---------------------------------------------------------------------------


class TestEmptyToken:
    def test_get_ofi_untracked(self) -> None:
        tracker = OFITracker()
        assert tracker.get_ofi("nonexistent") == 0.0

    def test_get_normalized_ofi_untracked(self) -> None:
        tracker = OFITracker()
        assert tracker.get_normalized_ofi("nonexistent") == 0.0

    def test_get_divergence_both_untracked(self) -> None:
        tracker = OFITracker()
        assert tracker.get_divergence("a", "b") == 0.0

    def test_tracked_tokens_starts_at_zero(self) -> None:
        tracker = OFITracker()
        assert tracker.tracked_tokens == 0
