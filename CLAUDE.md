# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Commands

```bash
# Install dependencies
uv sync --all-extras

# Run all tests
PYTHONPATH=. uv run python -m pytest tests/ -v

# Run a single test file
PYTHONPATH=. uv run python -m pytest tests/test_binary_scanner.py -v

# Run with coverage
PYTHONPATH=. uv run python -m pytest tests/ --cov=. --cov-report=term-missing

# Run the bot
uv run python run.py --dry-run        # No wallet, public APIs only
uv run python run.py --scan-only      # Detect only, needs wallet
uv run python run.py                  # Paper trading (default)
uv run python run.py --live           # Real orders
```

## Architecture

Single sequential pipeline in `run.py`: **config → auth → fetch markets → scan → size → safety check → execute → track P&L → repeat**.

Four modules, each with a single responsibility:

- **`client/`** — Thin wrappers around Polymarket APIs (Gamma for market discovery, CLOB for orderbooks/orders, WebSocket for real-time feeds). `auth.py` derives L2 API credentials from a Polygon wallet private key via py-clob-client SDK.
- **`scanner/`** — Opportunity detection. `models.py` holds all domain models as frozen dataclasses (immutable). `binary.py` detects YES+NO ask < $1.00. `negrisk.py` detects sum(all YES asks) < $1.00 across multi-outcome events.
- **`executor/`** — Trade execution. `sizing.py` uses half-Kelly criterion. `safety.py` has stale-quote checks, depth verification, and circuit breakers. `engine.py` posts batch orders and unwinds partial fills.
- **`monitor/`** — `pnl.py` writes append-only newline-delimited JSON ledger. `logger.py` emits structured single-line JSON logs.

## Key Data Flow

`Market` (from Gamma API) → grouped into `Event` → scanners produce `Opportunity` (with `LegOrder` legs) → sized via Kelly → safety-checked → `execute_opportunity()` returns `TradeResult` → recorded in `PnLTracker`.

All domain models in `scanner/models.py` are `@dataclass(frozen=True)`. `bids`/`asks` are tuples sorted best-first (index 0 = best level).

## Error Handling

Two exception types drive control flow:
- `SafetyCheckFailed` — skip this opportunity, continue scanning
- `CircuitBreakerTripped` — halt the entire bot, log final P&L

General exceptions in the main loop are logged with full traceback; bot continues to next cycle. Partial fill unwind failures are logged but don't halt.

## Non-Obvious Behaviors

- `--dry-run` creates an unauthenticated ClobClient (public endpoints only). `--scan-only` still requires wallet auth.
- Binary markets with `neg_risk=True` are excluded from binary scanning (they're handled as NegRisk events).
- All legs in an opportunity execute the same size. `max_sets` is the min available depth across all legs.
- Gas cost is estimated (`gas_per_order * gas_price_gwei`), not actual on-chain cost.
- Paper trading fills at opportunity prices without slippage simulation.
- Orders are posted as GTC. The `ORDER_TIMEOUT_SEC` config exists but is not currently enforced in live execution.

## Testing Patterns

Tests use `respx` to mock httpx calls and `unittest.mock.MagicMock` for the CLOB client. Helper factories like `_make_book()`, `_make_market()`, `_make_opp()` construct test data inline. PYTHONPATH must be set since the project uses flat module imports (no package install).
