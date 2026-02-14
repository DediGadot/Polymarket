# Polymarket Arbitrage Bot

Automated scanner and execution engine for arbitrage opportunities on
Polymarket, with optional cross-platform opportunities against Kalshi.

This bot runs a continuous pipeline:

1. Fetch active markets from Polymarket APIs (cached).
2. Build events and fetch relevant orderbooks (REST or WebSocket).
3. Scan 9 arbitrage strategies in parallel.
4. Rank opportunities with a composite score.
5. Apply sizing + safety checks.
6. Execute (paper/live) or report (scan-only).
7. Track maker orders across cycles.
8. Repeat.

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
- `stale_quote` and `resolution_snipe` are disabled.
- Gas oracle uses configured fallback defaults (no RPC/CoinGecko calls).

## Strategy Coverage

The scanner supports 9 opportunity types across taker and maker strategies:

**Taker strategies** (immediate fill-or-kill):

- `binary_rebalance` — `YES ask + NO ask < $1.00` (buy arb) or
  inventory-backed sell variants.
- `negrisk_rebalance` — Multi-outcome event baskets where total YES pricing
  is misaligned.
- `latency_arb` — Short-horizon crypto prediction markets lagging spot
  momentum.
- `spike_lag` — Event siblings that lag after abrupt repricing.
- `cross_platform_arb` — Optional Polymarket vs Kalshi divergence.
- `stale_quote` — WebSocket-driven detection of stale complementary
  orderbooks after a significant price move (real-time latency arb).
- `resolution_snipe` — Nearly-resolved markets where the outcome is publicly
  knowable but not yet settled; buys the winning side at a discount.

**Maker strategies** (limit orders, spread capture):

- `maker_spread` — Posts GTC limit orders at bid+1tick on both sides when
  the combined cost is < $1.00. Tracked across cycles by `MakerLifecycle`.

**Disabled by default:**

- `value` — Partial negrisk value scanner (assumes uniform 1/N probability;
  high false-positive rate on markets with known favorites).
  Enable with `VALUE_SCANNER_ENABLED=true`.

## Data Sources and Dependencies

By strategy/data path:

- Polymarket core:
  - Gamma API (market discovery/event metadata) — cached via `GammaCache`
    (60s markets, 300s event counts)
  - CLOB API + WS (orderbooks, execution, real-time updates)
  - Data API (positions/inventory checks)
- External feeds (enabled by `ALLOW_NON_POLYMARKET_APIS=true`):
  - Binance spot API (latency strategy signal + outcome oracle for resolution
    sniping)
  - Polygon RPC + CoinGecko (gas/POL-USD estimation)
  - Kalshi API (cross-platform mode only)

## Repository Layout

```text
polymarket/
├── run.py                  # Main loop and CLI entrypoint
├── config.py               # Environment-driven runtime config
├── client/                 # API clients, auth, caching
│   ├── cache.py            #   Thread-safe TTL cache for Gamma API
│   ├── clob.py             #   CLOB orderbook + order placement
│   ├── gamma.py            #   Market discovery REST
│   ├── gas.py              #   Polygon gas + POL/USD oracle
│   ├── kalshi.py           #   Kalshi REST v2 client
│   ├── kalshi_auth.py      #   RSA request signing
│   ├── ws.py               #   Async WebSocket manager
│   └── ws_bridge.py        #   Async WS → sync bridge
├── scanner/                # Opportunity detection + scoring
│   ├── models.py           #   Frozen dataclasses + BookFetcher alias
│   ├── binary.py           #   Binary rebalance scanner
│   ├── negrisk.py          #   NegRisk rebalance scanner
│   ├── latency.py          #   Crypto latency arb
│   ├── spike.py            #   News-driven spike lag
│   ├── cross_platform.py   #   Cross-platform divergence
│   ├── stale_quote.py      #   WS-driven stale quote detection
│   ├── resolution.py       #   Resolution sniping scanner
│   ├── maker.py            #   Maker spread capture scanner
│   ├── value.py            #   Partial negrisk value (off by default)
│   ├── outcome_oracle.py   #   Public outcome determination (Binance)
│   ├── depth.py            #   VWAP sweep + worst fill price
│   ├── fees.py             #   PM taker + resolution fee model
│   ├── fees_v2.py          #   Enhanced fee model (DCM, 15-min crypto)
│   ├── validation.py       #   Boundary validators (NaN, Inf, range)
│   ├── matching.py         #   Fuzzy cross-platform event mapping
│   ├── strategy.py         #   Adaptive strategy selector (4 modes)
│   ├── book_cache.py       #   WS-fed orderbook cache
│   └── confidence.py       #   Arb persistence tracking
├── executor/               # Sizing, safety, execution
│   ├── engine.py           #   FAK batch orders + partial unwind
│   ├── sizing.py           #   Half-Kelly position sizing
│   ├── safety.py           #   Pre-execution safety checks
│   ├── cross_platform.py   #   Dual-platform execution + unwind
│   ├── cross_platform_v2.py#   State machine cross-platform execution
│   ├── fill_state.py       #   Order lifecycle state machine
│   ├── maker_lifecycle.py  #   GTC maker order tracking across cycles
│   └── tick_size.py        #   Price quantization to tick grids
├── pipeline/               # Pipeline utilities
│   └── gas_utils.py        #   Gas cost estimation helpers
├── monitor/                # PnL tracking, status, logging
│   ├── pnl.py              #   NDJSON ledger
│   ├── display.py          #   Console pretty-printing
│   ├── scan_tracker.py     #   Scan-only session aggregator
│   └── logger.py           #   Structured JSON logs
├── benchmark/              # Offline analysis tools
│   ├── evs.py              #   Expected value of scanning metric
│   ├── weight_search.py    #   Scorer weight grid search
│   ├── latency_sim.py      #   Architecture latency simulation
│   └── cross_platform.py   #   Platform breakdown analysis
└── tests/                  # Pytest suite (~80+ test files)
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

- `--limit N`: caps binary markets for faster iteration (never truncates
  negRisk events).
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

Scanner toggles:

- `STALE_QUOTE_ENABLED` (default `true`) — WebSocket stale quote detection
- `STALE_QUOTE_MIN_MOVE_PCT`, `STALE_QUOTE_MAX_STALENESS_MS`,
  `STALE_QUOTE_COOLDOWN_SEC`
- `RESOLUTION_SNIPING_ENABLED` (default `true`) — resolution sniping
- `RESOLUTION_MAX_MINUTES`, `RESOLUTION_MIN_EDGE_PCT`
- `VALUE_SCANNER_ENABLED` (default `false`) — partial negrisk value scanner
- `VALUE_MIN_EDGE_PCT`, `VALUE_MAX_EXPOSURE`

Maker filters:

- `MAKER_MIN_LEG_PRICE` (default `0.05`) — excludes near-certain outcomes
- `MAKER_MIN_DEPTH_SETS` (default `5.0`) — filters micro-depth phantom arbs

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

### 4. Conservative (taker-only, high thresholds)

```bash
VALUE_SCANNER_ENABLED=false
RESOLUTION_SNIPING_ENABLED=false
STALE_QUOTE_ENABLED=false
```

## Output Files

- `status.md`
  - Rolling markdown status dashboard with recent cycles and key metrics.
- `pnl_ledger.json`
  - Append-only NDJSON ledger of trade results in paper/live modes.
- `stuck_positions.json`
  - Cross-platform positions that failed to unwind (requires manual review).
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
- input validation (NaN, Inf, range checks on all external data)
- tick size quantization (prices snapped to 0.01 or 0.001 grids)

Cross-platform execution follows a strict order: external platform first
(~50ms REST), then Polymarket (~2s on-chain). If Polymarket fails after an
external fill, automatic unwind sells the external position at market.
`FillState` tracks order lifecycle: PENDING → FILLED → UNWINDING → UNWOUND/STUCK.

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
- Stuck cross-platform positions:
  - Check `stuck_positions.json` for positions that failed to unwind.
- Value scanner false positives:
  - Leave `VALUE_SCANNER_ENABLED=false` (default). The uniform 1/N
    assumption fails on markets with known favorites.
- Maker phantom arbs:
  - Tune `MAKER_MIN_DEPTH_SETS` (filters micro-depth books) and
    `MAKER_MIN_LEG_PRICE` (filters near-certain outcomes).

## Risk Notice

This software can place real orders when run with `--live`.
Validate behavior in `--dry-run` and paper modes first, then review sizing,
exposure limits, and breaker thresholds before live deployment.
