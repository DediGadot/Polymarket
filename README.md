# Polymarket Arbitrage Bot

Automated scanner and execution engine for Polymarket arbitrage, with optional
cross-platform opportunities against Kalshi.

The bot runs a continuous pipeline:

1. Fetch markets from Gamma/CLOB.
2. Scan for opportunities across multiple strategies.
3. Rank opportunities with a composite score.
4. Apply sizing and safety checks.
5. Execute (paper or live) and track session metrics.

## What It Scans

- `binary_rebalance`: YES ask + NO ask < $1.00 (and inventory-based sell variants).
- `negrisk_rebalance`: multi-outcome baskets where combined pricing is misaligned.
- `latency_arb`: short-horizon crypto-related markets lagging spot moves.
- `spike_lag`: sibling markets lagging during rapid repricing events.
- `cross_platform_arb`: optional Polymarket vs Kalshi pricing dislocations.

## Repository Layout

```text
polymarket/
├── run.py                  # Main loop and CLI entrypoint
├── config.py               # Environment-driven runtime config
├── client/                 # Polymarket + Kalshi API clients and auth
├── scanner/                # Opportunity detection, depth math, scoring, strategy
├── executor/               # Sizing, safety checks, and execution logic
├── monitor/                # PnL tracking, status writer, logging
└── tests/                  # Pytest suite
```

## Requirements

- Python 3.11+
- `uv` package manager
- For `--scan-only`, paper, or live modes:
  - Polygon private key
  - Polymarket profile/proxy address
- For cross-platform mode (`CROSS_PLATFORM_ENABLED=true`):
  - Kalshi API key ID and RSA private key

## Quick Start

```bash
uv sync --all-extras
cp .env.example .env
```

For a no-wallet smoke test:

```bash
uv run python run.py --dry-run --limit 500
```

## Run Modes

```bash
# Public APIs only (no wallet, no execution)
uv run python run.py --dry-run

# Wallet auth, scanning only (no execution)
uv run python run.py --scan-only

# Paper trading (default when no --live flag)
uv run python run.py

# Live trading (real orders)
uv run python run.py --live
```

Additional CLI flags:

- `--limit N`: cap binary markets scanned (useful for fast iteration).
- `--json-log PATH`: append NDJSON logs for machine parsing.

## Configuration

`config.py` is the source of truth for all environment variables. Key settings:

- Credentials: `PRIVATE_KEY`, `POLYMARKET_PROFILE_ADDRESS`, `SIGNATURE_TYPE`
- Opportunity filters: `MIN_PROFIT_USD`, `MIN_ROI_PCT`, `MIN_VOLUME_FILTER`
- Risk limits: `MAX_EXPOSURE_PER_TRADE`, `MAX_TOTAL_EXPOSURE`
- Circuit breakers: `MAX_LOSS_PER_HOUR`, `MAX_LOSS_PER_DAY`, `MAX_CONSECUTIVE_FAILURES`
- Runtime: `SCAN_INTERVAL_SEC`, `ORDER_TIMEOUT_SEC`, `LOG_LEVEL`
- Execution controls: `USE_FAK_ORDERS`, `MAX_LEGS_PER_OPPORTUNITY`
- Data/perf: `WS_ENABLED`, `BOOK_CACHE_MAX_AGE_SEC`, `BOOK_FETCH_WORKERS`
- Optional cross-platform: `CROSS_PLATFORM_ENABLED`, `KALSHI_*`,
  `CROSS_PLATFORM_MIN_CONFIDENCE`, `CROSS_PLATFORM_MANUAL_MAP`

`.env.example` contains a starter subset; advanced options are documented by field
names and defaults in `config.py`.

## Output Files

- `status.md`: rolling markdown dashboard (current cycle + recent history).
- `pnl_ledger.json`: append-only NDJSON trade ledger (paper/live modes).
- `--json-log <path>`: optional structured runtime log.

## Testing

Use `PYTHONPATH=.` because imports are flat modules:

```bash
PYTHONPATH=. uv run python -m pytest tests/ -v
PYTHONPATH=. uv run python -m pytest tests/ --cov=. --cov-report=term-missing
```

## Safety Notes

- Execution path revalidates freshness, depth, edge, and gas before placing orders.
- Circuit breaker halts the bot on configured loss/failure thresholds.
- Partial fill/unwind failure paths are explicit and surfaced in logs.

This project trades real money in live mode. Validate behavior in `--dry-run` and
paper modes first, and review risk limits before enabling `--live`.
