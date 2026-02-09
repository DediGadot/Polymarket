"""Tests for monitor/status.py -- rolling markdown status file."""

from __future__ import annotations

import os


from monitor.status import StatusWriter, _format_duration, _padded_table, _truncate
from scanner.models import (
    Opportunity,
    OpportunityType,
    LegOrder,
    Side,
)
from scanner.scorer import ScoredOpportunity


# ── Factories ──


def _make_opp(
    event_id: str = "evt_001",
    opp_type: OpportunityType = OpportunityType.BINARY_REBALANCE,
    net_profit: float = 1.50,
    roi_pct: float = 3.0,
    required_capital: float = 50.0,
) -> Opportunity:
    return Opportunity(
        type=opp_type,
        event_id=event_id,
        legs=(
            LegOrder(token_id="tok_yes", side=Side.BUY, price=0.48, size=100),
            LegOrder(token_id="tok_no", side=Side.BUY, price=0.49, size=100),
        ),
        expected_profit_per_set=0.03,
        net_profit_per_set=0.03,
        max_sets=50.0,
        gross_profit=1.60,
        estimated_gas_cost=0.10,
        net_profit=net_profit,
        roi_pct=roi_pct,
        required_capital=required_capital,
    )


def _make_scored(
    event_id: str = "evt_001",
    net_profit: float = 1.50,
    roi_pct: float = 3.0,
    total_score: float = 0.65,
) -> ScoredOpportunity:
    opp = _make_opp(event_id=event_id, net_profit=net_profit, roi_pct=roi_pct)
    return ScoredOpportunity(
        opportunity=opp,
        total_score=total_score,
        profit_score=0.30,
        fill_score=0.50,
        efficiency_score=0.20,
        urgency_score=0.50,
        competition_score=1.0,
    )


# ── Unit tests: helpers ──


class TestFormatDuration:
    def test_seconds(self):
        assert _format_duration(45.3) == "45s"

    def test_minutes(self):
        assert _format_duration(125.0) == "2m 5s"

    def test_hours(self):
        assert _format_duration(3700.0) == "1h 1m"


class TestPaddedTable:
    def test_columns_aligned(self):
        result = _padded_table(["A", "BB"], [["x", "yyyy"], ["zzz", "w"]])
        # All rows should have the same length
        assert len(set(len(line) for line in result)) == 1

    def test_content_present(self):
        result = _padded_table(["Name", "Val"], [["foo", "123"]])
        joined = "\n".join(result)
        assert "foo" in joined
        assert "123" in joined

    def test_separator_matches_width(self):
        result = _padded_table(["Col"], [["data"]])
        # Header, separator, one data row
        assert len(result) == 3
        assert set(result[1].replace("|", "")) == {"-"}


class TestTruncate:
    def test_short_text(self):
        assert _truncate("hello", 10) == "hello"

    def test_exact_length(self):
        assert _truncate("hello", 5) == "hello"

    def test_long_text(self):
        assert _truncate("hello world", 8) == "hello..."

    def test_truncate_boundary(self):
        result = _truncate("abcdefghij", 6)
        assert result == "abc..."
        assert len(result) == 6


# ── Integration tests: StatusWriter ──


class TestStatusWriter:
    def test_writes_file(self, tmp_path):
        path = str(tmp_path / "status.md")
        writer = StatusWriter(file_path=path)
        writer.write_cycle(
            cycle=1,
            mode="DRY-RUN",
            markets_scanned=500,
            scored_opps=[],
            event_questions={},
            total_opps_found=0,
            total_trades_executed=0,
            total_pnl=0.0,
            current_exposure=0.0,
            scan_only=True,
        )
        assert os.path.exists(path)
        content = open(path).read()
        assert "# Polymarket Arbitrage Bot -- Status" in content
        assert "DRY-RUN" in content
        assert "500" in content

    def test_no_opportunities_message(self, tmp_path):
        path = str(tmp_path / "status.md")
        writer = StatusWriter(file_path=path)
        writer.write_cycle(
            cycle=1,
            mode="SCAN-ONLY",
            markets_scanned=100,
            scored_opps=[],
            event_questions={},
            total_opps_found=0,
            total_trades_executed=0,
            total_pnl=0.0,
            current_exposure=0.0,
            scan_only=True,
        )
        content = open(path).read()
        assert "*No opportunities found.*" in content

    def test_opportunity_table(self, tmp_path):
        path = str(tmp_path / "status.md")
        writer = StatusWriter(file_path=path)
        scored = [_make_scored(event_id="evt_abc", net_profit=2.50, roi_pct=5.0)]
        questions = {"evt_abc": "Will BTC hit 100k?"}
        writer.write_cycle(
            cycle=1,
            mode="DRY-RUN",
            markets_scanned=1000,
            scored_opps=scored,
            event_questions=questions,
            total_opps_found=1,
            total_trades_executed=0,
            total_pnl=0.0,
            current_exposure=0.0,
            scan_only=True,
        )
        content = open(path).read()
        assert "Will BTC hit 100k?" in content
        assert "$2.50" in content
        assert "5.00%" in content
        assert "binary_rebalance" in content

    def test_event_id_fallback_when_no_question(self, tmp_path):
        path = str(tmp_path / "status.md")
        writer = StatusWriter(file_path=path)
        scored = [_make_scored(event_id="evt_unknown_1234567890")]
        writer.write_cycle(
            cycle=1,
            mode="DRY-RUN",
            markets_scanned=100,
            scored_opps=scored,
            event_questions={},
            total_opps_found=1,
            total_trades_executed=0,
            total_pnl=0.0,
            current_exposure=0.0,
            scan_only=True,
        )
        content = open(path).read()
        # Should fall back to truncated event_id
        assert "evt_unknown_12" in content

    def test_trading_mode_shows_pnl(self, tmp_path):
        path = str(tmp_path / "status.md")
        writer = StatusWriter(file_path=path)
        writer.write_cycle(
            cycle=3,
            mode="PAPER TRADING",
            markets_scanned=2000,
            scored_opps=[],
            event_questions={},
            total_opps_found=5,
            total_trades_executed=2,
            total_pnl=12.50,
            current_exposure=150.0,
            scan_only=False,
        )
        content = open(path).read()
        assert "Trades executed" in content
        assert "$12.50" in content
        assert "$150.00" in content

    def test_scan_only_hides_pnl(self, tmp_path):
        path = str(tmp_path / "status.md")
        writer = StatusWriter(file_path=path)
        writer.write_cycle(
            cycle=1,
            mode="SCAN-ONLY",
            markets_scanned=500,
            scored_opps=[],
            event_questions={},
            total_opps_found=0,
            total_trades_executed=0,
            total_pnl=0.0,
            current_exposure=0.0,
            scan_only=True,
        )
        content = open(path).read()
        # The live Current State table must not contain P&L rows in scan-only mode.
        # Match the live section (## heading, not ### from the guide).
        sections = content.split("\n## ")
        state_section = next(s for s in sections if s.startswith("Current State\n"))
        assert "Trades executed" not in state_section
        assert "Net P&L" not in state_section

    def test_overwrites_file_each_cycle(self, tmp_path):
        path = str(tmp_path / "status.md")
        writer = StatusWriter(file_path=path)
        for c in range(1, 4):
            writer.write_cycle(
                cycle=c,
                mode="DRY-RUN",
                markets_scanned=100 * c,
                scored_opps=[],
                event_questions={},
                total_opps_found=0,
                total_trades_executed=0,
                total_pnl=0.0,
                current_exposure=0.0,
                scan_only=True,
            )
        content = open(path).read()
        # Current state should show cycle 3 (use \n## to skip guide's ### heading)
        state_section = content.split("\n## Current State\n")[1].split("\n##")[0]
        assert "Cycle" in state_section
        assert "| 3" in state_section
        # Should NOT have duplicate headers
        assert content.count("# Polymarket Arbitrage Bot -- Status") == 1

    def test_rolling_history_keeps_last_n(self, tmp_path):
        path = str(tmp_path / "status.md")
        writer = StatusWriter(file_path=path, max_history=5)
        for c in range(1, 12):
            writer.write_cycle(
                cycle=c,
                mode="DRY-RUN",
                markets_scanned=100,
                scored_opps=[],
                event_questions={},
                total_opps_found=0,
                total_trades_executed=0,
                total_pnl=0.0,
                current_exposure=0.0,
                scan_only=True,
            )
        content = open(path).read()
        # Should only have cycles 7-11 in history (last 5)
        history_section = content.split("\n## Recent Cycles\n")[1]
        assert "| 7 " in history_section
        assert "| 11 " in history_section
        # Cycle 1 should be gone from history
        assert "| 1 " not in history_section

    def test_history_shows_best_opportunity(self, tmp_path):
        path = str(tmp_path / "status.md")
        writer = StatusWriter(file_path=path)
        scored = [
            _make_scored(event_id="evt_best", net_profit=10.0, roi_pct=8.0, total_score=0.9),
            _make_scored(event_id="evt_ok", net_profit=2.0, roi_pct=3.0, total_score=0.5),
        ]
        questions = {"evt_best": "Will ETH flip BTC?", "evt_ok": "Some other event"}
        writer.write_cycle(
            cycle=1,
            mode="DRY-RUN",
            markets_scanned=500,
            scored_opps=scored,
            event_questions=questions,
            total_opps_found=2,
            total_trades_executed=0,
            total_pnl=0.0,
            current_exposure=0.0,
            scan_only=True,
        )
        content = open(path).read()
        # History row should reference the best opportunity
        assert "8.00%" in content
        assert "$10.00" in content
        assert "Will ETH flip BTC?" in content

    def test_history_newest_first(self, tmp_path):
        path = str(tmp_path / "status.md")
        writer = StatusWriter(file_path=path)
        for c in range(1, 4):
            writer.write_cycle(
                cycle=c,
                mode="DRY-RUN",
                markets_scanned=100,
                scored_opps=[],
                event_questions={},
                total_opps_found=0,
                total_trades_executed=0,
                total_pnl=0.0,
                current_exposure=0.0,
                scan_only=True,
            )
        content = open(path).read()
        history_section = content.split("\n## Recent Cycles\n")[1]
        lines = history_section.split("\n")
        # Cycle 3 should appear before cycle 1 (newest first)
        idx_3 = next(i for i, line in enumerate(lines) if "| 3 " in line)
        idx_1 = next(i for i, line in enumerate(lines) if "| 1 " in line)
        assert idx_3 < idx_1

    def test_multiple_opportunity_types(self, tmp_path):
        path = str(tmp_path / "status.md")
        writer = StatusWriter(file_path=path)
        scored = [
            _make_scored(event_id="evt_1", net_profit=5.0, roi_pct=4.0),
            ScoredOpportunity(
                opportunity=_make_opp(
                    event_id="evt_2",
                    opp_type=OpportunityType.NEGRISK_REBALANCE,
                    net_profit=3.0,
                    roi_pct=2.5,
                ),
                total_score=0.55,
                profit_score=0.25,
                fill_score=0.50,
                efficiency_score=0.20,
                urgency_score=0.50,
                competition_score=1.0,
            ),
        ]
        questions = {"evt_1": "Binary event", "evt_2": "NegRisk event"}
        writer.write_cycle(
            cycle=1,
            mode="DRY-RUN",
            markets_scanned=1000,
            scored_opps=scored,
            event_questions=questions,
            total_opps_found=2,
            total_trades_executed=0,
            total_pnl=0.0,
            current_exposure=0.0,
            scan_only=True,
        )
        content = open(path).read()
        assert "binary_rebalance" in content
        assert "negrisk_rebalance" in content
        assert "Binary event" in content
        assert "NegRisk event" in content
