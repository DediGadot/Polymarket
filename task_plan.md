# Task Plan: Polymarket Arbitrage Bot

## Objective
Build a modular Python arbitrage bot that detects and executes risk-free (or near-risk-free) mispricings on Polymarket, prioritizing NegRisk multi-outcome rebalancing and binary market rebalancing.

## Architecture Overview

```
polymarket/
├── task_plan.md
├── findings.md
├── progress.md
├── pyproject.toml              # uv project config
├── run.py                      # Single pipeline script (all modules combined)
├── config.py                   # Configuration (env vars, thresholds, constants)
├── client/                     # Polymarket API client module
│   ├── __init__.py
│   ├── clob.py                 # CLOB REST client (orders, prices, books)
│   ├── gamma.py                # Gamma API client (market discovery)
│   ├── ws.py                   # WebSocket real-time feed
│   └── auth.py                 # Authentication (L1 wallet, L2 HMAC)
├── scanner/                    # Arbitrage opportunity detection module
│   ├── __init__.py
│   ├── binary.py               # Binary market rebalancing (YES+NO != 1.0)
│   ├── negrisk.py              # NegRisk multi-outcome rebalancing
│   └── models.py               # Data models (Opportunity, Market, OrderBook)
├── executor/                   # Trade execution module
│   ├── __init__.py
│   ├── engine.py               # Order placement, cancellation, fill tracking
│   ├── sizing.py               # Position sizing (Kelly criterion, max exposure)
│   └── safety.py               # Pre-trade checks, circuit breakers
├── monitor/                    # Monitoring and observability module
│   ├── __init__.py
│   ├── pnl.py                  # P&L tracking, position accounting
│   └── logger.py               # Structured logging
└── tests/                      # Test suite
    ├── test_scanner.py
    ├── test_executor.py
    └── test_sizing.py
```

## Design Principles
- **Modular**: Each concern in its own directory (client, scanner, executor, monitor)
- **Single pipeline**: `run.py` combines all modules into one working pipeline
- **Fail-fast**: No fallbacks, crash on errors, let supervisor restart
- **No stubs**: Every function fully implemented
- **No ground truth in production**: Only use known answers for validation/testing
- **uv for package management**: pyproject.toml, uv.lock

---

## Phase 1: Foundation (client + config)

### 1.1 Project Setup
- Initialize uv project with pyproject.toml
- Dependencies: py-clob-client, httpx, websockets, python-dotenv, pydantic
- Create .env.example with required vars (PRIVATE_KEY, POLYMARKET_PROFILE_ADDRESS, etc.)
- Create config.py loading from env with pydantic-settings

### 1.2 Client Module -- `client/`
- **auth.py**: Wallet setup, API credential derivation (create_or_derive_api_creds)
- **clob.py**: Thin wrapper around py-clob-client for:
  - get_orderbook(token_id) -> OrderBook
  - get_price(token_id) -> BidAsk
  - get_midpoint(token_id) -> float
  - post_order(order) -> OrderResponse
  - post_orders(orders) -> list[OrderResponse]  (batch, up to 15)
  - cancel_order(order_id)
  - cancel_all()
- **gamma.py**: Market discovery via Gamma API:
  - get_markets(active=True, closed=False) -> list[Market]
  - get_events() -> list[Event]
  - get_market(condition_id) -> Market
  - Filter: negRisk markets, binary markets, by volume/liquidity
- **ws.py**: WebSocket manager:
  - Subscribe to market price changes for monitored token_ids
  - Emit events on price_change, book update
  - Automatic reconnection with exponential backoff (no silent fallback -- raise after max retries)

### 1.3 Data Models -- `scanner/models.py`
- `Market`: condition_id, token_ids (YES/NO), neg_risk, event_id, question, min_tick_size
- `OrderBook`: bids[], asks[], best_bid, best_ask, spread, midpoint
- `Opportunity`: type (BINARY_REBALANCE | NEGRISK_REBALANCE), markets, expected_profit, required_capital, roi, confidence, timestamp
- `TradeResult`: order_ids, fill_prices, fill_sizes, fees, net_pnl, execution_time_ms

---

## Phase 2: Arbitrage Scanner

### 2.1 Binary Rebalancing Scanner -- `scanner/binary.py`
```
Algorithm:
1. For each active binary market:
   a. Fetch best_ask for YES token and best_ask for NO token
   b. cost = YES_ask + NO_ask
   c. If cost < 1.0:
      profit_per_set = 1.0 - cost
      Check depth: min(YES_ask_size, NO_ask_size) = max_sets
      gross_profit = profit_per_set * max_sets
      Estimate gas cost (2 orders * ~150k gas * gas_price)
      net_profit = gross_profit - gas_cost
      If net_profit > MIN_PROFIT_THRESHOLD and ROI > MIN_ROI:
        Emit Opportunity(BINARY_REBALANCE, ...)
   d. If cost > 1.0 (and we hold positions to sell):
      Similar logic for sell-both arbitrage
```

### 2.2 NegRisk Rebalancing Scanner -- `scanner/negrisk.py`
```
Algorithm:
1. For each active negRisk event (multi-outcome):
   a. Fetch all markets in the event
   b. For each market, get best_ask for YES token
   c. total_cost = sum(all YES_ask prices)
   d. If total_cost < 1.0:
      profit_per_set = 1.0 - total_cost
      max_sets = min(ask_size across all outcomes)
      gross_profit = profit_per_set * max_sets
      gas_cost = N_outcomes * 150k gas * gas_price (one order per outcome)
      net_profit = gross_profit - gas_cost
      If net_profit > MIN_PROFIT_THRESHOLD and ROI > MIN_ROI:
        Emit Opportunity(NEGRISK_REBALANCE, ...)
   e. Also check: sum(all YES_bid prices) > 1.0 for sell-all arbitrage
```

### 2.3 Scanner Orchestrator
- Run both scanners in parallel (asyncio)
- Deduplicate opportunities
- Rank by ROI, then by absolute profit
- Feed top opportunities to executor

---

## Phase 3: Trade Execution

### 3.1 Position Sizing -- `executor/sizing.py`
- Kelly criterion for optimal bet sizing: f* = (bp - q) / b
  - b = odds (profit/risk), p = probability of fill, q = 1-p
- Max exposure per opportunity: configurable (e.g. $500)
- Max total exposure: configurable (e.g. $5000)
- Never risk more than X% of bankroll on single arb

### 3.2 Safety Checks -- `executor/safety.py`
- Pre-trade validation:
  - Verify prices haven't moved since scan (stale quote check)
  - Verify sufficient USDC balance
  - Verify token allowances on exchange contracts
  - Verify order book depth supports intended size
  - Check if we're within rate limits
- Circuit breakers:
  - Max loss per hour / per day
  - Max consecutive failed executions
  - Kill switch: cancel all open orders and halt

### 3.3 Execution Engine -- `executor/engine.py`
```
For BINARY_REBALANCE:
  1. Build two GTC limit orders: BUY YES @ ask, BUY NO @ ask
  2. Submit both via batch endpoint (POST /orders)
  3. Wait for fills (poll or WebSocket user channel)
  4. If both fill: log success, compute P&L
  5. If partial fill (one leg only):
     - Immediately cancel unfilled leg
     - Market-sell the filled leg (FOK at aggressive bid)
     - Log the loss from partial fill
  6. If neither fills within timeout: cancel both

For NEGRISK_REBALANCE:
  1. Build N GTC limit orders: BUY YES for each outcome @ ask
  2. Submit via batch (may need multiple batches if N > 15)
  3. Wait for fills on all legs
  4. If all fill: guaranteed profit at resolution
  5. If partial fill:
     - Cancel all unfilled legs
     - Evaluate: can we convert filled YES positions using NegRisk adapter?
     - If not profitable to hold, market-sell filled positions
  6. Timeout: cancel all unfilled
```

---

## Phase 4: Monitoring & Pipeline

### 4.1 P&L Tracking -- `monitor/pnl.py`
- Track per-trade: entry cost, expected payout, fees, gas, net P&L
- Track aggregate: daily P&L, total exposure, win rate, avg ROI
- Persist to local JSON file (append-only ledger)

### 4.2 Structured Logging -- `monitor/logger.py`
- JSON-formatted logs with: timestamp, level, module, event, data
- Log every: scan result, opportunity found, order placed, fill received, error
- Console output for real-time monitoring

### 4.3 Pipeline Script -- `run.py`
```python
Main loop:
  1. Initialize client (auth, connect WS)
  2. Fetch all active markets from Gamma API
  3. Categorize: binary vs negRisk
  4. Start WebSocket feeds for all monitored tokens
  5. Loop forever:
     a. Run binary scanner on latest prices
     b. Run negrisk scanner on latest prices
     c. Rank opportunities
     d. For each opportunity above threshold:
        - Run safety checks
        - Size the position
        - Execute via engine
        - Log result
     e. Update P&L
     f. Sleep(scan_interval) -- configurable, default 1s
  6. On SIGINT/SIGTERM: cancel all open orders, log final P&L, exit
```

---

## Phase 5: Hardening & Validation

### 5.1 Testing
- Unit tests for scanner math (known price inputs -> expected opportunities)
- Unit tests for sizing (Kelly criterion edge cases)
- Integration test against Polymarket testnet or paper trading mode
- Validate: P&L calculations match expected outcomes

### 5.2 Paper Trading Mode
- Add `--paper` flag to run.py
- All execution is simulated (log what would be traded, track virtual P&L)
- Validates detection + sizing without risking capital

### 5.3 Deployment
- Run via `uv run python run.py`
- Supervisor/systemd for auto-restart on crash (fail-fast design)
- .env for secrets management

---

## Configuration Parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| PRIVATE_KEY | (required) | Polygon wallet private key |
| POLYMARKET_PROFILE_ADDRESS | (required) | Polymarket profile/proxy address |
| SIGNATURE_TYPE | 1 | 0=EOA, 1=POLY_PROXY, 2=GNOSIS_SAFE |
| MIN_PROFIT_USD | 0.50 | Minimum absolute profit to execute |
| MIN_ROI_PCT | 2.0 | Minimum ROI % after all costs |
| MAX_EXPOSURE_PER_TRADE | 500 | Max USDC per single arb |
| MAX_TOTAL_EXPOSURE | 5000 | Max total USDC deployed |
| MAX_LOSS_PER_HOUR | 50 | Circuit breaker: max hourly loss |
| MAX_LOSS_PER_DAY | 200 | Circuit breaker: max daily loss |
| SCAN_INTERVAL_SEC | 1.0 | Seconds between scan cycles |
| ORDER_TIMEOUT_SEC | 5.0 | Cancel unfilled orders after this |
| PAPER_TRADING | false | Simulate execution without real orders |
| LOG_LEVEL | INFO | Logging verbosity |

---

## Risk Assessment

| Risk | Likelihood | Impact | Mitigation |
|------|-----------|--------|------------|
| Leg risk (partial fill) | HIGH | MEDIUM | Immediate unwind, batch orders, tight timeout |
| Stale quotes | MEDIUM | LOW | Re-check prices before execution |
| Gas spikes | LOW | LOW | Gas cost included in profitability calc |
| Oracle dispute | LOW | HIGH | Avoid markets near resolution, monitor UMA |
| Rate limiting | MEDIUM | LOW | Respect limits, exponential backoff |
| Smart contract bug | VERY LOW | CRITICAL | Use official SDK, small position sizes |
| Regulatory change | LOW | HIGH | Monitor news, kill switch ready |

---

## Status: AWAITING APPROVAL

**Estimated files to create**: ~18 Python files + config
**Key dependency**: py-clob-client SDK
**First milestone**: Scanner detecting real opportunities in paper mode
