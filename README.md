# Polymarket Arbitrage Bot

Automated detection and execution of risk-free mispricings on [Polymarket](https://polymarket.com). Targets two arbitrage strategies ranked by historical profitability:

1. **NegRisk Multi-Outcome Rebalancing** -- When the sum of all YES prices in a multi-outcome event falls below $1.00, buy all outcomes for a guaranteed profit at resolution. This strategy generated 73% of the $40M+ in arbitrage profits extracted from Polymarket between Apr 2024 and Apr 2025 ([IMDEA research](https://arxiv.org/abs/2508.03474)), with 29x capital efficiency over binary arbitrage.

2. **Binary Market Rebalancing** -- When YES ask + NO ask < $1.00 in a binary market, buy both sides. One must resolve to $1.00, locking in the spread as profit.

## Architecture

```
polymarket/
├── run.py                    # Single pipeline: scan -> detect -> size -> execute -> track
├── config.py                 # Pydantic config from env vars
├── client/                   # Polymarket API client
│   ├── auth.py               # Wallet auth + L2 API credential derivation
│   ├── clob.py               # CLOB REST (orderbooks, orders, cancellation)
│   ├── gamma.py              # Market discovery with auto-pagination
│   └── ws.py                 # WebSocket real-time price feeds
├── scanner/                  # Arbitrage opportunity detection
│   ├── models.py             # Data models (Market, OrderBook, Opportunity, etc.)
│   ├── binary.py             # Binary rebalancing scanner
│   └── negrisk.py            # NegRisk multi-outcome scanner
├── executor/                 # Trade execution
│   ├── engine.py             # Order placement, batch posting, partial fill unwinding
│   ├── sizing.py             # Half-Kelly criterion position sizing
│   └── safety.py             # Stale quote checks, depth verification, circuit breakers
├── monitor/                  # Observability
│   ├── pnl.py                # P&L tracking with append-only JSON ledger
│   └── logger.py             # Structured JSON logging
└── tests/                    # 100 tests across 11 test files
```

## Prerequisites

- Python 3.11+
- [uv](https://docs.astral.sh/uv/) package manager
- A Polygon wallet with USDC
- A Polymarket account (for proxy address and token approvals)

## Setup

```bash
# Clone and install
git clone <repo-url> && cd polymarket
uv sync --all-extras

# Configure credentials
cp .env.example .env
# Edit .env with your PRIVATE_KEY and POLYMARKET_PROFILE_ADDRESS
```

### Required environment variables

| Variable | Description |
|----------|-------------|
| `PRIVATE_KEY` | Polygon wallet private key (hex, no 0x prefix) |
| `POLYMARKET_PROFILE_ADDRESS` | Your Polymarket profile/proxy address |

See `.env.example` for all configurable parameters.

## Usage

```bash
# Dry-run: no wallet needed, scan real markets using public APIs
uv run python run.py --dry-run

# Scan-only: detect opportunities without executing trades (needs wallet)
uv run python run.py --scan-only

# Paper trading: simulated execution with virtual P&L tracking
uv run python run.py

# Live trading: real orders on Polymarket
uv run python run.py --live
```

### Dry-run mode

No wallet or credentials required. Connects to Polymarket's public APIs (Gamma for market discovery, CLOB for orderbooks) and scans for arbitrage opportunities. Equivalent to `--scan-only` but skips authentication entirely. Use this to verify the bot detects real opportunities before configuring a wallet.

### Scan-only mode

Logs every detected opportunity with type, profit, ROI, and number of legs. No orders are placed. Requires wallet credentials for authenticated API access. Use this to validate detection against real market conditions before risking capital.

### Paper trading mode (default)

Simulates full execution: sizing via Kelly criterion, safety checks, and virtual P&L tracking written to `pnl_ledger.json`. No real orders are placed.

### Live trading mode

Places real GTC limit orders on Polymarket's CLOB. Binary arbs submit both legs as a batch. NegRisk arbs batch in groups of 15 (CLOB limit). Partial fills are automatically unwound via FOK market orders.

## How it works

### Detection

Each scan cycle:

1. Fetches all active markets from the Gamma API (auto-paginates)
2. Separates binary markets from NegRisk multi-outcome events
3. Batch-fetches orderbooks for all relevant tokens
4. **Binary scanner**: checks if `YES_best_ask + NO_best_ask < 1.0`
5. **NegRisk scanner**: checks if `sum(all YES_best_ask) < 1.0` across an event's outcomes
6. Filters by minimum profit ($0.50 default) and minimum ROI (2% default)
7. Ranks opportunities by ROI descending

### Execution

For each opportunity that passes safety checks:

1. **Stale quote check** -- re-fetches prices, rejects if moved beyond slippage tolerance
2. **Depth check** -- verifies orderbook can fill the intended size
3. **Kelly sizing** -- computes optimal position size (half-Kelly for conservatism), capped by max exposure
4. **Batch order submission** -- posts all legs via `POST /orders` (up to 15 per batch)
5. **Fill tracking** -- monitors order status
6. **Partial fill unwinding** -- if not all legs fill, cancels unfilled orders and market-sells filled positions

### Safety

- **Circuit breakers**: halt on max hourly loss ($50), max daily loss ($200), or 5 consecutive failures
- **Graceful shutdown**: SIGINT/SIGTERM cancels all open orders and logs final P&L
- **Fail-fast**: no silent fallbacks -- crashes are surfaced immediately for supervisor restart

## Configuration

All parameters are configurable via environment variables or `.env`:

| Parameter | Default | Description |
|-----------|---------|-------------|
| `SIGNATURE_TYPE` | `1` | 0=EOA, 1=POLY_PROXY, 2=GNOSIS_SAFE |
| `MIN_PROFIT_USD` | `0.50` | Minimum absolute profit to execute |
| `MIN_ROI_PCT` | `2.0` | Minimum ROI % after gas costs |
| `MAX_EXPOSURE_PER_TRADE` | `500` | Max USDC risked per single arb |
| `MAX_TOTAL_EXPOSURE` | `5000` | Max total USDC deployed |
| `MAX_LOSS_PER_HOUR` | `50` | Circuit breaker: hourly loss limit |
| `MAX_LOSS_PER_DAY` | `200` | Circuit breaker: daily loss limit |
| `MAX_CONSECUTIVE_FAILURES` | `5` | Circuit breaker: consecutive failure limit |
| `SCAN_INTERVAL_SEC` | `1.0` | Seconds between scan cycles |
| `ORDER_TIMEOUT_SEC` | `5.0` | Cancel unfilled orders after this |
| `GAS_PER_ORDER` | `150000` | Estimated gas units per order |
| `GAS_PRICE_GWEI` | `30.0` | Default gas price (overridden at runtime) |
| `PAPER_TRADING` | `true` | Simulate execution (overridden by `--live`) |
| `LOG_LEVEL` | `INFO` | Logging verbosity |

## Tests

```bash
# Run all tests
PYTHONPATH=. uv run python -m pytest tests/ -v

# Run with coverage
PYTHONPATH=. uv run python -m pytest tests/ --cov=. --cov-report=term-missing
```

100 tests across 11 test files covering:

- **Unit tests**: data models, config validation, scanner math, Kelly sizing, circuit breakers, safety checks, execution engine, P&L tracking
- **Integration tests**: Gamma API with mocked HTTP (respx), full scan-detect-size-execute pipeline, circuit breaker halt behavior, fair-market no-opportunity scenarios, structured JSON logging

## Risks

| Risk | Mitigation |
|------|------------|
| **Leg risk** (partial fill) | Automatic unwind via FOK market orders, tight timeout |
| **Stale quotes** | Re-check prices before every execution |
| **Gas spikes** | Gas cost included in profitability calculation |
| **Oracle disputes** | Monitor UMA Optimistic Oracle, avoid markets near resolution |
| **Rate limiting** | Respects Polymarket CLOB rate limits (3,500 orders/10s burst) |
| **Smart contract risk** | Uses official py-clob-client SDK, conservative position sizes |

## License

MIT
