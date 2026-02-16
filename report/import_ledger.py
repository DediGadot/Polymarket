"""
One-time import of existing pnl_ledger.json into the report database.

Usage:
    PYTHONPATH=. uv run python -m report.import_ledger
    PYTHONPATH=. uv run python -m report.import_ledger --ledger path/to/pnl_ledger.json
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from report.store import ReportStore


def import_ledger(
    store: ReportStore,
    ledger_path: str = "pnl_ledger.json",
) -> int:
    """Import pnl_ledger.json (NDJSON) into the trades table.

    Creates a synthetic "imported" session. Returns number of trades imported.
    """
    path = Path(ledger_path)
    if not path.exists():
        print(f"Ledger file not found: {path}")
        return 0

    session_id = store.start_session(
        mode="IMPORTED",
        config_json="{}",
        cli_args_json=json.dumps({"source": str(path)}),
    )

    # Create a single synthetic cycle for imported data
    cycle_id = store.insert_cycle(
        session_id=session_id,
        cycle_num=0,
        elapsed_sec=0.0,
        strategy_mode="imported",
        gas_price_gwei=0.0,
        spike_count=0,
        has_momentum=False,
        win_rate=0.0,
        markets_scanned=0,
        binary_count=0,
        negrisk_events=0,
        negrisk_markets=0,
        opps_found=0,
        opps_executed=0,
    )

    count = 0
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                entry = json.loads(line)
            except json.JSONDecodeError:
                continue

            store.insert_trade(
                session_id=session_id,
                cycle_id=cycle_id,
                event_id=entry.get("event_id", ""),
                opp_type=entry.get("opportunity_type", "unknown"),
                n_legs=entry.get("n_legs", 0),
                fully_filled=entry.get("fully_filled", False),
                fill_prices_json=json.dumps(entry.get("fill_prices", [])),
                fill_sizes_json=json.dumps(entry.get("fill_sizes", [])),
                fees=entry.get("fees", 0.0),
                gas_cost=entry.get("gas_cost", 0.0),
                net_pnl=entry.get("net_pnl", 0.0),
                execution_time_ms=entry.get("execution_time_ms", 0.0),
                total_score=0.0,
            )
            count += 1

    store.end_session(session_id)
    return count


def main() -> None:
    parser = argparse.ArgumentParser(description="Import pnl_ledger.json into report database")
    parser.add_argument("--ledger", default="pnl_ledger.json", help="Path to NDJSON ledger file")
    parser.add_argument("--db", default=None, help="Path to SQLite database (default: report/data/report.db)")
    args = parser.parse_args()

    store = ReportStore(db_path=args.db)
    count = import_ledger(store, ledger_path=args.ledger)
    print(f"Imported {count} trades from {args.ledger}")
    store.close()


if __name__ == "__main__":
    main()
