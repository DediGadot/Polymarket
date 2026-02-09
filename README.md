# Polymarket Arbitrage Bot

Automated scanner and execution engine for arbitrage opportunities on
Polymarket, with optional cross-platform opportunities against Kalshi.

This bot runs a continuous pipeline:

1. Fetch active markets from Polymarket APIs.
2. Build events and fetch relevant orderbooks.
3. Scan multiple arbitrage strategies.
4. Rank opportunities with a composite score.
5. Apply sizing + safety checks.
6. Execute (paper/live) or report (scan-only).
7. Repeat.

## Default Behavior (Important)

The current default is **all supported data integrations allowed**:

- `ALLOW_NON_POLYMARKET_APIS=true` by default.
- This means non-Polymarket external data calls are permitted by runtime policy.
- Latency strategy can run (it uses Binance spot data).
- Gas oracle can query Polygon RPC and CoinGecko.
- Cross-platform strategy is still controlled separately by
  `CROSS_PLATFORM_ENABLED` (default: `false`) and Kalshi credentials.

If you want strict Polymarket-only outbound behavior, set:

```bash
ALLOW_NON_POLYMARKET_APIS=false
```

With that setting:

- `latency_arb` is disabled.
- `cross_platform_arb` is disabled.
- Gas oracle uses configured fallback defaults (no RPC/CoinGecko calls).

## Strategy Coverage

The scanner supports the following opportunity types:

- `binary_rebalance`
  - Core condition: `YES ask + NO ask < $1.00` (buy arb) or
    inventory-backed sell variants.
- `negrisk_rebalance`
  - Multi-outcome event baskets where total YES pricing is misaligned.
- `latency_arb`
  - Short-horizon crypto prediction markets lagging spot momentum.
- `spike_lag`
  - Event siblings that lag after abrupt repricing.
- `cross_platform_arb`
  - Optional Polymarket vs Kalshi divergence opportunities.

## Data Sources and Dependencies

By strategy/data path:

- Polymarket core:
  - Gamma API (market discovery/event metadata)
  - CLOB API + WS (orderbooks, execution, real-time updates)
  - Data API (positions/inventory checks)
- External feeds (enabled by `ALLOW_NON_POLYMARKET_APIS=true`):
  - Binance spot API (latency strategy signal)
  - Polygon RPC + CoinGecko (gas/POL-USD estimation)
  - Kalshi API (cross-platform mode only)

## Repository Layout

```text
polymarket/
├── run.py                  # Main loop and CLI entrypoint
├── config.py               # Environment-driven runtime config
├── client/                 # Polymarket + external API clients/auth
├── scanner/                # Opportunity detection, scoring, strategy selection
├── executor/               # Sizing, safety checks, execution logic
├── monitor/                # PnL tracking, status writer, structured logging
└── tests/                  # Pytest suite
```

## Requirements

- Python 3.11+
- `uv` package manager
- For `--scan-only`, paper, or live:
  - Polygon private key
  - Polymarket profile/proxy address
- For cross-platform
  (`ALLOW_NON_POLYMARKET_APIS=true` and `CROSS_PLATFORM_ENABLED=true`):
  - Kalshi API key ID
  - Kalshi RSA private key file path

## Installation

```bash
uv sync --all-extras
cp .env.example .env
```

## Run Modes

```bash
# Public APIs only (no wallet, no execution)
uv run python run.py --dry-run

# Wallet auth, scan only (no execution)
uv run python run.py --scan-only

# Paper trading (default when no --live flag)
uv run python run.py

# Live trading (real orders)
uv run python run.py --live
```

Useful flags:

- `--limit N`: caps binary markets for faster iteration.
- `--json-log PATH`: writes machine-readable NDJSON logs.

Quick smoke test:

```bash
uv run python run.py --dry-run --limit 500
```

## Configuration Guide

`config.py` is the source of truth. `.env.example` includes common options.

Core credentials:

- `PRIVATE_KEY`
- `POLYMARKET_PROFILE_ADDRESS`
- `SIGNATURE_TYPE`

Trading/risk:

- `MIN_PROFIT_USD`, `MIN_ROI_PCT`, `MIN_VOLUME_FILTER`
- `MAX_EXPOSURE_PER_TRADE`, `MAX_TOTAL_EXPOSURE`
- `MAX_LOSS_PER_HOUR`, `MAX_LOSS_PER_DAY`, `MAX_CONSECUTIVE_FAILURES`

Runtime and execution:

- `SCAN_INTERVAL_SEC`, `ORDER_TIMEOUT_SEC`, `LOG_LEVEL`
- `USE_FAK_ORDERS`, `MAX_LEGS_PER_OPPORTUNITY`
- `WS_ENABLED`, `BOOK_CACHE_MAX_AGE_SEC`, `BOOK_FETCH_WORKERS`

External integration policy:

- `ALLOW_NON_POLYMARKET_APIS` (default `true`)

Cross-platform options:

- `CROSS_PLATFORM_ENABLED` (default `false`)
- `KALSHI_API_KEY_ID`, `KALSHI_PRIVATE_KEY_PATH`, `KALSHI_HOST`, `KALSHI_DEMO`
- `CROSS_PLATFORM_MIN_CONFIDENCE`, `CROSS_PLATFORM_MANUAL_MAP`

## Practical Config Profiles

### 1. Full Default (recommended starting point)

Keep defaults as-is in `.env` and run `--dry-run` first.

### 2. Strict Polymarket-only Outbound Traffic

```bash
ALLOW_NON_POLYMARKET_APIS=false
CROSS_PLATFORM_ENABLED=false
```

### 3. Cross-Platform Enabled

```bash
ALLOW_NON_POLYMARKET_APIS=true
CROSS_PLATFORM_ENABLED=true
KALSHI_API_KEY_ID=...
KALSHI_PRIVATE_KEY_PATH=...
```

## Output Files

- `status.md`
  - Rolling markdown status dashboard with recent cycles and key metrics.
- `pnl_ledger.json`
  - Append-only NDJSON ledger of trade results in paper/live modes.
- `--json-log <path>`
  - Optional structured runtime log stream.

## Testing

Use `PYTHONPATH=.` (flat imports):

```bash
PYTHONPATH=. uv run python -m pytest tests/ -v
PYTHONPATH=. uv run python -m pytest tests/ --cov=. --cov-report=term-missing
```

## Safety Model

Before execution, the engine re-checks:

- opportunity freshness/TTL
- max leg count
- price freshness
- edge integrity
- depth sufficiency
- inventory for sell legs
- gas reasonableness

Circuit breaker halts trading when configured failure/loss thresholds are hit.

## Troubleshooting

Common startup/behavior issues:

- Missing wallet vars in non-dry-run:
  - Set `PRIVATE_KEY` and `POLYMARKET_PROFILE_ADDRESS`, or use `--dry-run`.
- Cross-platform enabled without Kalshi creds:
  - Provide `KALSHI_*` vars or set `CROSS_PLATFORM_ENABLED=false`.
- Unexpected external API calls:
  - Set `ALLOW_NON_POLYMARKET_APIS=false`.
- High scan latency:
  - Use `--limit` for iteration and tune `BOOK_FETCH_WORKERS`.

## Risk Notice

This software can place real orders when run with `--live`.
Validate behavior in `--dry-run` and paper modes first, then review sizing,
exposure limits, and breaker thresholds before live deployment.
