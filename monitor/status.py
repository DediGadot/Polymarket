"""
Rolling markdown status file. Overwrites status.md each cycle with:
  - Current state: mode, uptime, cycle, session totals, opportunity table
  - History: last N cycles with best finds
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field

from scanner.scorer import ScoredOpportunity


_GUIDE: list[str] = [
    "## How the Pipeline Works",
    "",
    "The bot runs in a continuous loop. Each iteration is called a **cycle**:",
    "",
    "1. **Fetch markets** -- pulls all active markets from the Gamma REST API.",
    "   Binary markets (YES/NO) and negRisk events (multi-outcome) are separated.",
    "2. **Scan for arbitrage** -- four independent scanners run on the fetched markets:",
    "   - **binary_rebalance**: buy arb when YES ask + NO ask < $1.00, or sell arb when YES bid + NO bid > $1.00.",
    "   - **negrisk_rebalance**: buy arb when sum of all YES asks < $1.00, or sell arb when sum of YES bids > $1.00.",
    "   - **latency_arb**: 15-minute crypto markets (BTC/ETH/SOL up/down) reprice slower than spot exchanges.",
    "     The bot compares Polymarket odds to live spot momentum and buys or sells when the market lags.",
    "   - **spike_lag**: during breaking news one market reprices instantly while sibling markets in the same",
    "     event lag by 5-60 seconds. The bot builds a multi-leg negRisk basket on the lagging outcomes.",
    "3. **Score and rank** -- every opportunity gets a composite score (0-1) from five weighted factors:",
    "   profit magnitude, fill probability, capital efficiency, urgency, and competition.",
    "   Opportunities are sorted best-first.",
    "4. **Size** -- half-Kelly criterion determines how many sets to trade given current bankroll and edge.",
    "   *(Skipped in DRY-RUN and SCAN-ONLY modes.)*",
    "5. **Safety checks** -- price freshness, orderbook depth, and gas cost are verified. If any check fails",
    "   the opportunity is skipped. A circuit breaker halts the bot on excessive losses.",
    "   *(Skipped in DRY-RUN and SCAN-ONLY modes.)*",
    "6. **Execute** -- FAK (fill-and-kill) orders are sent for each leg. Partial fills are unwound.",
    "   *(Skipped in DRY-RUN and SCAN-ONLY modes.)*",
    "7. **Record** -- P&L is updated, the trade is appended to the NDJSON ledger, and this status file is rewritten.",
    "   *(Skipped in DRY-RUN and SCAN-ONLY modes.)*",
    "8. **Sleep** -- the bot waits for the remaining scan interval before starting the next cycle.",
    "",
    "## Field Reference",
    "",
    "### Current State",
    "",
    "| Field | Meaning |",
    "|-------|---------|",
    "| Mode | DRY-RUN = public APIs only, no wallet. SCAN-ONLY = detect but don't trade. PAPER = simulated fills. LIVE = real orders. |",
    "| Uptime | Wall-clock time since the bot process started. |",
    "| Cycle | How many full fetch-scan-execute loops have completed. |",
    "| Markets scanned | Number of individual binary markets fetched this cycle (negRisk markets counted individually). |",
    "| Opportunities (this cycle) | Arbitrage opportunities that passed minimum profit and ROI filters this cycle. |",
    "| Opportunities (session) | Cumulative count across all cycles since startup. |",
    "| Trades executed | (Trading modes only) How many opportunities were actually sent to the exchange. |",
    "| Net P&L | (Trading modes only) Realized profit/loss across all trades this session, after fees and gas. |",
    "| Current exposure | (Trading modes only) Total capital currently locked in open positions awaiting resolution. |",
    "",
    "### Opportunities Table",
    "",
    "| Column | Meaning |",
    "|--------|---------|",
    "| Type | Which scanner found it: binary_rebalance, negrisk_rebalance, latency_arb, or spike_lag. |",
    "| Event | Human-readable event title from Polymarket (e.g. \"Will BTC be above 100k?\"). |",
    "| Profit | Net expected profit in USD after subtracting gas cost and the 2% resolution fee. |",
    "| ROI | Return on invested capital as a percentage (net_profit / required_capital * 100). |",
    "| Score | Composite score (0-1). Weights: 25% profit, 25% fill probability, 20% capital efficiency, 20% urgency, 10% competition. |",
    "| Legs | Number of separate orders required (2 for binary, N for negRisk, 1 for latency, N for spike). |",
    "| Capital | USDC needed to execute all legs at the quoted prices and sizes. |",
    "",
    "### Recent Cycles Table",
    "",
    "| Column | Meaning |",
    "|--------|---------|",
    "| Cycle | Cycle number. |",
    "| Time | Wall-clock time when the cycle completed. |",
    "| Markets | Markets scanned that cycle. |",
    "| Opps | Opportunities found that cycle. |",
    "| Best Type | Scanner type of the highest-profit opportunity. |",
    "| Best ROI | ROI of the highest-profit opportunity. |",
    "| Best Profit | Dollar profit of the highest-profit opportunity. |",
    "| Best Event | Event title of the highest-profit opportunity. |",
    "",
    "---",
]


@dataclass
class CycleSnapshot:
    """One cycle's summary for the history table."""

    cycle: int
    timestamp: float
    markets_scanned: int
    n_opportunities: int
    best_roi_pct: float
    best_profit_usd: float
    best_event_question: str
    best_type: str


@dataclass
class StatusWriter:
    """Writes a rolling status.md file each cycle."""

    file_path: str = "status.md"
    max_history: int = 20

    _session_start: float = field(default_factory=time.time)
    _history: list[CycleSnapshot] = field(default_factory=list)

    def write_cycle(
        self,
        *,
        cycle: int,
        mode: str,
        markets_scanned: int,
        scored_opps: list[ScoredOpportunity],
        event_questions: dict[str, str],
        total_opps_found: int,
        total_trades_executed: int,
        total_pnl: float,
        current_exposure: float,
        scan_only: bool,
    ) -> None:
        """Overwrite the status file with current state + rolling history."""
        # Build history entry for this cycle
        raw_opps = [s.opportunity for s in scored_opps]
        best_roi = max((o.roi_pct for o in raw_opps), default=0.0)
        best_profit = max((o.net_profit for o in raw_opps), default=0.0)
        best_event_q = ""
        best_type = ""
        if raw_opps:
            best = max(raw_opps, key=lambda o: o.net_profit)
            best_event_q = event_questions.get(best.event_id, best.event_id[:14])
            best_type = best.type.value

        snap = CycleSnapshot(
            cycle=cycle,
            timestamp=time.time(),
            markets_scanned=markets_scanned,
            n_opportunities=len(scored_opps),
            best_roi_pct=round(best_roi, 2),
            best_profit_usd=round(best_profit, 2),
            best_event_question=best_event_q,
            best_type=best_type,
        )
        self._history.append(snap)
        if len(self._history) > self.max_history:
            self._history = self._history[-self.max_history :]

        # Render
        lines = self._render(
            cycle=cycle,
            mode=mode,
            markets_scanned=markets_scanned,
            scored_opps=scored_opps,
            event_questions=event_questions,
            total_opps_found=total_opps_found,
            total_trades_executed=total_trades_executed,
            total_pnl=total_pnl,
            current_exposure=current_exposure,
            scan_only=scan_only,
        )

        with open(self.file_path, "w") as f:
            f.write("\n".join(lines) + "\n")

    def _render(
        self,
        *,
        cycle: int,
        mode: str,
        markets_scanned: int,
        scored_opps: list[ScoredOpportunity],
        event_questions: dict[str, str],
        total_opps_found: int,
        total_trades_executed: int,
        total_pnl: float,
        current_exposure: float,
        scan_only: bool,
    ) -> list[str]:
        uptime = _format_duration(time.time() - self._session_start)
        ts = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())

        lines: list[str] = []
        lines.append("# Polymarket Arbitrage Bot -- Status")
        lines.append("")
        lines.append(f"*Updated {ts}*")
        lines.append("")
        lines.extend(_GUIDE)
        lines.append("")

        # ── Current State ──
        lines.append("## Current State")
        lines.append("")
        state_rows: list[list[str]] = [
            ["Mode", mode],
            ["Uptime", uptime],
            ["Cycle", str(cycle)],
            ["Markets scanned", f"{markets_scanned:,}"],
            ["Opportunities (this cycle)", str(len(scored_opps))],
            ["Opportunities (session)", str(total_opps_found)],
        ]
        if not scan_only:
            state_rows.append(["Trades executed", str(total_trades_executed)])
            state_rows.append(["Net P&L", f"${total_pnl:.2f}"])
            state_rows.append(["Current exposure", f"${current_exposure:.2f}"])
        lines.extend(_padded_table(["Field", "Value"], state_rows))
        lines.append("")

        # ── Opportunities this cycle ──
        lines.append("## Opportunities This Cycle")
        lines.append("")
        if scored_opps:
            opp_rows: list[list[str]] = []
            for i, scored in enumerate(scored_opps, 1):
                opp = scored.opportunity
                question = event_questions.get(opp.event_id, opp.event_id[:14])
                q_display = _truncate(question, 50)
                type_label = opp.type.value
                if opp.is_sell_arb:
                    type_label = f"[SELL] {type_label}"
                opp_rows.append([
                    str(i),
                    type_label,
                    q_display,
                    f"${opp.net_profit:.2f}",
                    f"{opp.roi_pct:.2f}%",
                    f"{scored.total_score:.2f}",
                    str(len(opp.legs)),
                    f"${opp.required_capital:.2f}",
                ])
            lines.extend(_padded_table(
                ["#", "Type", "Event", "Profit", "ROI", "Score", "Legs", "Capital"],
                opp_rows,
            ))
        else:
            lines.append("*No opportunities found.*")
        lines.append("")

        # ── Rolling History ──
        lines.append("## Recent Cycles")
        lines.append("")
        if self._history:
            history_rows: list[list[str]] = []
            for snap in reversed(self._history):
                t = time.strftime("%H:%M:%S", time.localtime(snap.timestamp))
                q_display = _truncate(snap.best_event_question, 40) if snap.best_event_question else "--"
                best_type = snap.best_type if snap.best_type else "--"
                roi_str = f"{snap.best_roi_pct:.2f}%" if snap.n_opportunities > 0 else "--"
                profit_str = f"${snap.best_profit_usd:.2f}" if snap.n_opportunities > 0 else "--"
                history_rows.append([
                    str(snap.cycle),
                    t,
                    f"{snap.markets_scanned:,}",
                    str(snap.n_opportunities),
                    best_type,
                    roi_str,
                    profit_str,
                    q_display,
                ])
            lines.extend(_padded_table(
                ["Cycle", "Time", "Markets", "Opps", "Best Type", "Best ROI", "Best Profit", "Best Event"],
                history_rows,
            ))
            lines.append("")
        else:
            lines.append("*No history yet.*")
            lines.append("")

        return lines


def _padded_table(headers: list[str], rows: list[list[str]]) -> list[str]:
    """Build a Markdown table with evenly padded columns."""
    widths = [len(h) for h in headers]
    for row in rows:
        for i, cell in enumerate(row):
            widths[i] = max(widths[i], len(cell))

    def _fmt(cells: list[str]) -> str:
        parts = [f" {c:<{widths[i]}} " for i, c in enumerate(cells)]
        return "|" + "|".join(parts) + "|"

    lines = [_fmt(headers)]
    lines.append("|" + "|".join("-" * (w + 2) for w in widths) + "|")
    for row in rows:
        lines.append(_fmt(row))
    return lines


def _format_duration(seconds: float) -> str:
    """Format seconds into a human-readable duration string."""
    if seconds < 60:
        return f"{seconds:.0f}s"
    minutes = int(seconds) // 60
    secs = seconds - minutes * 60
    if minutes < 60:
        return f"{minutes}m {secs:.0f}s"
    hours = minutes // 60
    mins = minutes % 60
    return f"{hours}h {mins}m"


def _truncate(text: str, max_len: int) -> str:
    """Truncate text with ellipsis if too long."""
    if len(text) <= max_len:
        return text
    return text[: max_len - 3] + "..."
