"""Tests for benchmark/recorder.py — cycle recording for offline replay."""

from __future__ import annotations

import json
import time

import pytest

from benchmark.recorder import (
    BaseRecorder,
    CycleRecorder,
    NullRecorder,
    create_recorder,
)
from scanner.models import (
    LegOrder,
    Opportunity,
    OpportunityType,
    OrderBook,
    PriceLevel,
    Side,
)
from scanner.scorer import ScoringContext


# ── factories ──


def _make_levels(n: int, start: float = 0.40, step: float = 0.01) -> tuple[PriceLevel, ...]:
    return tuple(PriceLevel(price=round(start + i * step, 4), size=10.0) for i in range(n))


def _make_book(token_id: str = "tok_yes", n_levels: int = 10) -> OrderBook:
    return OrderBook(
        token_id=token_id,
        bids=_make_levels(n_levels, start=0.45, step=-0.01),
        asks=_make_levels(n_levels, start=0.55, step=0.01),
    )


def _make_opp(event_id: str = "evt_1", profit: float = 1.50) -> Opportunity:
    return Opportunity(
        type=OpportunityType.BINARY_REBALANCE,
        event_id=event_id,
        legs=(
            LegOrder(token_id="tok_yes", side=Side.BUY, price=0.45, size=100.0,
                     platform="polymarket", tick_size="0.01"),
            LegOrder(token_id="tok_no", side=Side.BUY, price=0.53, size=100.0,
                     platform="polymarket", tick_size="0.01"),
        ),
        expected_profit_per_set=0.02,
        net_profit_per_set=0.015,
        max_sets=100.0,
        gross_profit=2.0,
        estimated_gas_cost=0.10,
        net_profit=profit,
        roi_pct=3.5,
        required_capital=98.0,
        pair_fill_prob=0.90,
        toxicity_score=0.15,
        timestamp=time.time(),
    )


def _make_ctx() -> ScoringContext:
    return ScoringContext(
        market_volume=50000.0,
        recent_trade_count=5,
        time_to_resolution_hours=48.0,
        is_spike=False,
        book_depth_ratio=1.5,
        confidence=0.8,
        realized_ev_score=0.6,
    )


# ── NullRecorder ──


class TestNullRecorder:
    def test_record_cycle_is_noop(self):
        rec = NullRecorder()
        # Should not raise
        rec.record_cycle(
            cycle=1,
            books={"tok": _make_book()},
            opportunities=[_make_opp()],
            scoring_contexts=[_make_ctx()],
            strategy_mode="aggressive",
            config={"key": "val"},
        )

    def test_stats_shows_disabled(self):
        rec = NullRecorder()
        assert rec.stats == {"enabled": False, "cycles_recorded": 0}

    def test_close_is_noop(self):
        rec = NullRecorder()
        rec.close()  # Should not raise

    def test_zero_overhead(self):
        """NullRecorder methods are trivially fast (no IO, no serialization)."""
        rec = NullRecorder()
        start = time.monotonic()
        for _ in range(1000):
            rec.record_cycle(0, {}, [], [])
        elapsed = time.monotonic() - start
        # 1000 no-op calls should take < 10ms
        assert elapsed < 0.1


# ── CycleRecorder NDJSON output ──


class TestCycleRecorderNDJSON:
    def test_writes_three_cycles(self, tmp_path):
        rec = CycleRecorder(output_dir=str(tmp_path), max_mb=100)
        books = {"tok_yes": _make_book("tok_yes"), "tok_no": _make_book("tok_no")}
        opps = [_make_opp()]
        ctxs = [_make_ctx()]

        for i in range(3):
            rec.record_cycle(i, books, opps, ctxs, strategy_mode="aggressive")

        rec.close()

        with open(rec.stats["current_file"]) as f:
            lines = f.readlines()

        assert len(lines) == 3

        for line in lines:
            record = json.loads(line)
            assert record["type"] == "cycle"
            assert "schema_version" in record
            assert "cycle" in record
            assert "timestamp" in record
            assert "data" in record
            assert "books" in record["data"]
            assert "opportunities" in record["data"]
            assert "contexts" in record["data"]
            assert "scoring_contexts" in record["data"]
            assert "strategy_mode" in record["data"]

    def test_cycle_numbers_recorded(self, tmp_path):
        rec = CycleRecorder(output_dir=str(tmp_path))
        for i in [0, 1, 2]:
            rec.record_cycle(i, {}, [], [])
        rec.close()

        with open(rec.stats["current_file"]) as f:
            records = [json.loads(line) for line in f]

        assert [r["cycle"] for r in records] == [0, 1, 2]


# ── Book compression ──


class TestBookCompression:
    def test_compresses_to_max_levels(self, tmp_path):
        rec = CycleRecorder(output_dir=str(tmp_path), max_book_levels=5)
        book = _make_book("tok_a", n_levels=10)

        rec.record_cycle(0, {"tok_a": book}, [], [])
        rec.close()

        with open(rec.stats["current_file"]) as f:
            record = json.loads(f.readline())

        compressed = record["data"]["books"]["tok_a"]
        assert len(compressed["bids"]) == 5
        assert len(compressed["asks"]) == 5

    def test_preserves_levels_when_fewer_than_max(self, tmp_path):
        rec = CycleRecorder(output_dir=str(tmp_path), max_book_levels=5)
        book = _make_book("tok_a", n_levels=3)

        rec.record_cycle(0, {"tok_a": book}, [], [])
        rec.close()

        with open(rec.stats["current_file"]) as f:
            record = json.loads(f.readline())

        compressed = record["data"]["books"]["tok_a"]
        assert len(compressed["bids"]) == 3
        assert len(compressed["asks"]) == 3

    def test_price_and_size_preserved(self, tmp_path):
        rec = CycleRecorder(output_dir=str(tmp_path), max_book_levels=2)
        book = OrderBook(
            token_id="tok",
            bids=(PriceLevel(price=0.50, size=100.0), PriceLevel(price=0.49, size=50.0)),
            asks=(PriceLevel(price=0.55, size=80.0), PriceLevel(price=0.56, size=60.0)),
        )

        rec.record_cycle(0, {"tok": book}, [], [])
        rec.close()

        with open(rec.stats["current_file"]) as f:
            record = json.loads(f.readline())

        bids = record["data"]["books"]["tok"]["bids"]
        assert bids[0] == {"price": 0.50, "size": 100.0}
        assert bids[1] == {"price": 0.49, "size": 50.0}


# ── Opportunity serialization ──


class TestOpportunitySerialization:
    def test_all_fields_present(self, tmp_path):
        rec = CycleRecorder(output_dir=str(tmp_path))
        opp = _make_opp()
        rec.record_cycle(0, {}, [opp], [])
        rec.close()

        with open(rec.stats["current_file"]) as f:
            record = json.loads(f.readline())

        serialized = record["data"]["opportunities"][0]
        assert serialized["type"] == "binary_rebalance"
        assert serialized["event_id"] == "evt_1"
        assert len(serialized["legs"]) == 2
        assert serialized["legs"][0]["token_id"] == "tok_yes"
        assert serialized["legs"][0]["side"] == "BUY"
        assert serialized["net_profit"] == opp.net_profit
        assert serialized["roi_pct"] == opp.roi_pct
        assert serialized["pair_fill_prob"] == opp.pair_fill_prob
        assert serialized["toxicity_score"] == opp.toxicity_score

    def test_leg_fields(self, tmp_path):
        rec = CycleRecorder(output_dir=str(tmp_path))
        opp = _make_opp()
        rec.record_cycle(0, {}, [opp], [])
        rec.close()

        with open(rec.stats["current_file"]) as f:
            record = json.loads(f.readline())

        leg = record["data"]["opportunities"][0]["legs"][0]
        assert set(leg.keys()) == {"token_id", "side", "price", "size", "platform", "tick_size"}


# ── ScoringContext serialization ──


class TestScoringContextSerialization:
    def test_all_fields_present(self, tmp_path):
        rec = CycleRecorder(output_dir=str(tmp_path))
        ctx = _make_ctx()
        rec.record_cycle(0, {}, [], [ctx])
        rec.close()

        with open(rec.stats["current_file"]) as f:
            record = json.loads(f.readline())

        serialized = record["data"]["scoring_contexts"][0]
        assert serialized["market_volume"] == 50000.0
        assert serialized["recent_trade_count"] == 5
        assert serialized["time_to_resolution_hours"] == 48.0
        assert serialized["is_spike"] is False
        assert serialized["book_depth_ratio"] == 1.5
        assert serialized["confidence"] == 0.8
        assert serialized["realized_ev_score"] == 0.6
        assert serialized["ofi_divergence"] == 0.0


# ── Config recording ──


class TestConfigRecording:
    def test_config_only_on_first_cycle(self, tmp_path):
        rec = CycleRecorder(output_dir=str(tmp_path))
        cfg = {"min_profit_usd": 0.50, "scan_interval_sec": 1.0}

        rec.record_cycle(0, {}, [], [], config=cfg)
        rec.record_cycle(1, {}, [], [], config=cfg)
        rec.record_cycle(2, {}, [], [], config=cfg)
        rec.close()

        with open(rec.stats["current_file"]) as f:
            records = [json.loads(line) for line in f]

        assert records[0]["type"] == "config"
        assert records[0]["data"] == cfg
        assert records[1]["type"] == "cycle"
        assert records[2]["type"] == "cycle"
        assert records[3]["type"] == "cycle"

    def test_no_config_when_none(self, tmp_path):
        rec = CycleRecorder(output_dir=str(tmp_path))
        rec.record_cycle(0, {}, [], [])
        rec.close()

        with open(rec.stats["current_file"]) as f:
            record = json.loads(f.readline())

        assert record["type"] == "cycle"


# ── File rotation ──


class TestFileRotation:
    def test_rotation_at_max_size(self, tmp_path):
        # max_mb=0.001 → ~1KB limit → forces rotation quickly
        rec = CycleRecorder(output_dir=str(tmp_path), max_mb=0.001)
        books = {"tok": _make_book("tok", n_levels=5)}
        opps = [_make_opp()]
        ctxs = [_make_ctx()]

        # Record enough cycles to trigger rotation
        for i in range(10):
            rec.record_cycle(i, books, opps, ctxs, strategy_mode="aggressive")

        rec.close()

        assert rec.stats["files_rotated"] >= 1
        # Multiple JSONL files should exist
        jsonl_files = list(tmp_path.glob("*.jsonl"))
        assert len(jsonl_files) >= 2

    def test_all_rotated_files_are_valid_ndjson(self, tmp_path):
        rec = CycleRecorder(output_dir=str(tmp_path), max_mb=0.001)
        books = {"tok": _make_book("tok", n_levels=3)}

        for i in range(10):
            rec.record_cycle(i, books, [_make_opp()], [_make_ctx()])

        rec.close()

        for jsonl_file in tmp_path.glob("*.jsonl"):
            with open(jsonl_file) as f:
                for line in f:
                    if line.strip():
                        json.loads(line)  # Must not raise


# ── Stats tracking ──


class TestStatsTracking:
    def test_cycles_recorded(self, tmp_path):
        rec = CycleRecorder(output_dir=str(tmp_path))
        assert rec.stats["cycles_recorded"] == 0

        rec.record_cycle(0, {}, [], [])
        assert rec.stats["cycles_recorded"] == 1

        rec.record_cycle(1, {}, [], [])
        assert rec.stats["cycles_recorded"] == 2
        rec.close()

    def test_bytes_written_increases(self, tmp_path):
        rec = CycleRecorder(output_dir=str(tmp_path))
        rec.record_cycle(0, {}, [], [])
        bytes_after_one = rec.stats["bytes_written"]
        assert bytes_after_one > 0

        rec.record_cycle(1, {}, [], [])
        assert rec.stats["bytes_written"] > bytes_after_one
        rec.close()

    def test_enabled_flag(self, tmp_path):
        rec = CycleRecorder(output_dir=str(tmp_path))
        assert rec.stats["enabled"] is True
        rec.close()

    def test_current_file_set(self, tmp_path):
        rec = CycleRecorder(output_dir=str(tmp_path))
        assert rec.stats["current_file"].endswith(".jsonl")
        rec.close()


# ── Factory ──


class TestCreateRecorder:
    def test_enabled_returns_cycle_recorder(self, tmp_path):
        rec = create_recorder(enabled=True, output_dir=str(tmp_path))
        assert isinstance(rec, CycleRecorder)
        rec.close()

    def test_disabled_returns_null_recorder(self):
        rec = create_recorder(enabled=False)
        assert isinstance(rec, NullRecorder)

    def test_factory_passes_kwargs(self, tmp_path):
        rec = create_recorder(enabled=True, output_dir=str(tmp_path), max_mb=100, max_book_levels=3)
        assert isinstance(rec, CycleRecorder)
        assert rec._max_levels == 3
        rec.close()


# ── Close ──


class TestClose:
    def test_close_sets_handle_to_none(self, tmp_path):
        rec = CycleRecorder(output_dir=str(tmp_path))
        assert rec._file_handle is not None
        rec.close()
        assert rec._file_handle is None

    def test_double_close_is_safe(self, tmp_path):
        rec = CycleRecorder(output_dir=str(tmp_path))
        rec.close()
        rec.close()  # Should not raise

    def test_recording_after_close_is_noop(self, tmp_path):
        rec = CycleRecorder(output_dir=str(tmp_path))
        rec.close()
        # Should not raise, but won't write (file_handle is None)
        rec.record_cycle(0, {}, [], [])
        assert rec.stats["cycles_recorded"] == 0
