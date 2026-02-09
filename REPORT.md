# Polymarket Arbitrage Bot -- Full Report

```
Author:  automated analysis
Date:    2026-02-06
Version: 1.0
```

---

## Table of Contents

1. [What This Bot Does (The 30-Second Version)](#1-what-this-bot-does)
2. [How Polymarket Works](#2-how-polymarket-works)
3. [The Four Money-Making Strategies](#3-the-four-strategies)
4. [Pipeline Step by Step](#4-pipeline-step-by-step)
5. [Module Map](#5-module-map)
6. [Configuration and Knobs](#6-configuration-and-knobs)
7. [Safety and Risk Controls](#7-safety-and-risk-controls)
8. [Fees and Costs](#8-fees-and-costs)
9. [Data Flow Diagram](#9-data-flow-diagram)
10. [Running Modes](#10-running-modes)
11. [Monitoring and Observability](#11-monitoring-and-observability)
12. [Known Limitations and Future Work](#12-known-limitations)

---

## 1. What This Bot Does

This is a trading bot that finds and exploits pricing mistakes on Polymarket,
a prediction market platform. It runs in a continuous loop:

```
fetch 25,000+ markets  -->  scan for mispriced odds  -->  rank the best ones
      -->  check safety  -->  place orders  -->  record profit/loss  -->  repeat
```

**Plain-language example:**

Imagine a market asking "Will it rain tomorrow?" with two outcomes: YES and NO.
If you can buy YES for $0.45 and NO for $0.52, you spend $0.97 total.
One of them MUST pay out $1.00. You just locked in $0.03 profit with zero risk.
This bot finds thousands of markets like this every second and trades them
automatically.

---

## 2. How Polymarket Works

Polymarket is a prediction market on the Polygon blockchain. Here is what
matters for understanding this bot:

### Markets and Outcomes

- A **market** is a yes/no question: "Will BTC hit $100k by Friday?"
- Each market has two tokens: **YES** and **NO**.
- Tokens trade between $0.00 and $1.00.
- When the market resolves, the winning token pays $1.00 and the losing
  token pays $0.00.
- Prices reflect probabilities. YES at $0.70 means 70% chance.

### Events (Multi-Outcome)

- An **event** groups related markets: "Which party wins the election?"
- Each market in the event covers one outcome (Democrat, Republican, etc.).
- Exactly one outcome wins. The winning YES token pays $1.00, all others
  pay $0.00.
- These are called **negRisk** markets on Polymarket.

### The Orderbook

- Each token has an orderbook with **bids** (buy orders) and **asks**
  (sell orders).
- The **best ask** is the cheapest price someone will sell at.
- The **best bid** is the highest price someone will buy at.
- The **spread** is the gap between them.

### Why Mispricings Happen

- Market makers update slowly after news breaks.
- Thin orderbooks mean prices overshoot.
- Multi-outcome events have many tokens to keep in sync.
- 15-minute crypto markets lag behind spot exchanges.

---

## 3. The Four Strategies

### Strategy 1: Binary Rebalance

**What it is:** The simplest arbitrage. Buy YES + NO in a two-outcome market
when their combined ask price is less than $1.00.

**Example:**

```
Market: "Will ETH be above $4,000 on Feb 10?"

  YES best ask: $0.47  (size: 200 shares)
  NO  best ask: $0.51  (size: 150 shares)

  Combined cost:  $0.47 + $0.51 = $0.98
  Guaranteed payout:                $1.00
  Gross profit per set:             $0.02
  Max sets (limited by thinner side): 150

  Gross profit:  150 x $0.02 = $3.00
  Resolution fee (2%):              -$3.00 x 0.02 = not quite, see below
  Gas cost (2 orders):              -$0.01
  Net profit:                        ~$2.69
```

The bot buys 150 shares of YES at $0.47 and 150 shares of NO at $0.51.
No matter what happens, one of them pays $1.00. Guaranteed profit.

**Where in code:** `scanner/binary.py`

---

### Strategy 2: NegRisk Rebalance

**What it is:** The same idea but for multi-outcome events. Buy one YES share
in every outcome. Exactly one wins $1.00. If the total cost is under $1.00,
it is free money.

**Example:**

```
Event: "Who will win the Oscar for Best Picture?"

  Outcome A (Film X) YES ask: $0.25
  Outcome B (Film Y) YES ask: $0.30
  Outcome C (Film Z) YES ask: $0.20
  Outcome D (Film W) YES ask: $0.15

  Total cost: $0.25 + $0.30 + $0.20 + $0.15 = $0.90
  Guaranteed payout (one winner):               $1.00
  Gross profit per set:                          $0.10

  With 4 legs, gas cost is higher (~$0.02 for 4 orders).
  Net profit per set: ~$0.08
  If 100 sets available: ~$8.00 profit.
```

Historically this strategy accounts for about 73% of profits because
multi-outcome events are harder for market makers to keep balanced.

**Where in code:** `scanner/negrisk.py`

---

### Strategy 3: Latency Arbitrage

**What it is:** Exploit the delay between crypto spot price moves on
Binance and the odds updating on Polymarket 15-minute crypto markets.

**Example:**

```
Market: "Will BTC be up in the next 15 minutes?"
Current market YES ask: $0.50 (50% odds)

Meanwhile on Binance, BTC just jumped 1% in 30 seconds.
The implied probability of "BTC up" is now ~85%.
But the Polymarket odds have not moved yet.

  Buy YES at: $0.50
  True probability: ~85%
  Expected value: 0.85 x $1.00 = $0.85
  Edge: $0.85 - $0.50 = $0.35 per share (70% ROI before fees)

  Taker fee at 50/50 odds: 3.15% = $0.016 per share
  Net expected edge: ~$0.33 per share
```

This is NOT riskless like strategies 1 and 2. The probability estimate
could be wrong. The bot uses a minimum 5% edge threshold after fees
to only take high-confidence trades.

**Where in code:** `scanner/latency.py`

---

### Strategy 4: Spike Detection

**What it is:** When breaking news causes one market in a multi-outcome
event to spike, sibling markets often lag behind. The bot detects the
spike and trades the lagging siblings.

**Example:**

```
Event: "Will BTC close above $100k, $101k, or $102k?"

Breaking: BTC surges past $102k.

  "$102k" market spikes from $0.30 to $0.85 instantly.
  "$101k" market still sits at $0.40 (should be ~$0.90).
  "$100k" market still sits at $0.50 (should be ~$0.95).

  The bot detects the spike in "$102k" (>5% move in 30 seconds).
  It checks siblings and finds "$101k" and "$100k" are mispriced.
  It buys the underpriced siblings before they catch up.
```

**Where in code:** `scanner/spike.py`

---

## 4. Pipeline Step by Step

Each cycle of the bot follows these steps:

### Step 1: Fetch Markets

```
Source:  Gamma REST API (gamma-api.polymarket.com)
Output:  25,000+ active markets
Code:    client/gamma.py -> get_all_markets()
```

The bot downloads metadata for every active market: question text, token
IDs, whether it is a negRisk event, tick size, volume. Markets are split
into binary (2 outcomes) and negRisk (3+ outcomes grouped by event).

### Step 2: Fetch Orderbooks

```
Source:  CLOB API (clob.polymarket.com)
Output:  Bid/ask levels for each token
Code:    client/clob.py -> get_orderbooks()
```

For each market the scanners care about, the bot fetches the live orderbook.
Books are fetched in batches of 50 to avoid API limits.

**Important trap:** The SDK does NOT sort book levels. The bot sorts asks
ascending and bids descending. Without this fix, the "best ask" would
actually be the worst ask, and zero opportunities would ever be found.

### Step 3: Scan for Opportunities

```
Input:   Markets + orderbooks
Output:  List of Opportunity objects
Code:    scanner/binary.py, scanner/negrisk.py, scanner/latency.py,
         scanner/spike.py
```

All four scanners run on their respective market sets. Each scanner:

1. Checks the math (can we lock in profit?).
2. Deducts fees (taker fee + 2% resolution fee).
3. Deducts estimated gas cost (via `client/gas.py`).
4. Filters out opportunities below minimum profit ($0.50) and minimum
   ROI (2%).
5. Emits `Opportunity` objects with all details.

### Step 4: Score and Rank

```
Input:   Raw opportunities from all scanners
Output:  Sorted list, best first
Code:    scanner/scorer.py -> rank_opportunities()
```

Each opportunity gets a composite score from 0 to 1 based on five factors:

| Factor          | Weight | What it measures                       |
|-----------------|--------|----------------------------------------|
| Profit          | 25%    | Net dollar profit (log-scaled)         |
| Fill probability| 25%    | Orderbook depth vs order size          |
| Efficiency      | 20%    | ROI relative to capital locked up      |
| Urgency         | 20%    | How fast the window closes             |
| Competition     | 10%    | How many other bots are likely trading |

Spike and latency arbs score higher on urgency because the window is
seconds, not minutes.

### Step 5: Position Sizing (Half-Kelly)

```
Input:   Top opportunity + bankroll
Output:  Number of sets to trade
Code:    executor/sizing.py -> compute_position_size()
```

The bot uses the Kelly Criterion at half strength to decide how much
capital to risk:

```
edge       = profit_per_set / cost_per_set
kelly_frac = edge * 0.50           (half-Kelly for safety)
capital    = kelly_frac * bankroll
sets       = capital / cost_per_set
```

**Example:**

```
Bankroll:        $10,000
Cost per set:    $0.97
Profit per set:  $0.03
Edge:            0.03 / 0.97 = 3.09%
Kelly fraction:  3.09% * 0.50 = 1.545%
Capital:         $10,000 * 1.545% = $154.50
Sets:            $154.50 / $0.97 = 159 sets
```

The bot also respects hard caps: max $500 per trade, max $5,000 total
exposure.

### Step 6: Safety Checks

```
Input:   Sized opportunity
Output:  Pass or reject
Code:    executor/safety.py
```

Three checks must pass before any order is placed:

1. **Price freshness** -- Re-fetch the orderbook. If prices moved more
   than 0.5% since the scan, reject. The opportunity may be gone.

2. **Depth check** -- Verify the orderbook still has enough shares at
   the expected price. If someone else took the liquidity, reject.

3. **Gas reasonableness** -- Estimate real-time gas cost. If gas would
   eat more than 50% of the net profit, reject. Not worth it.

If any check fails, the opportunity is skipped (not the whole bot).

### Step 7: Execute Orders

```
Input:   Validated, sized opportunity
Output:  TradeResult with fill details
Code:    executor/engine.py -> execute_opportunity()
```

The execution engine places orders through the CLOB API:

- **Binary arbs:** 2 orders (YES + NO), batch-posted together.
- **NegRisk arbs:** N orders (one per outcome), batched in groups of 15.
- **Latency arbs:** 1 order (single leg).
- **Spike arbs:** N orders (lagging siblings).

Orders use Fill-and-Kill (FAK) by default: they either fill immediately
or cancel. No open orders left dangling.

**Partial fill handling:** If only some legs fill (e.g., YES fills but NO
does not), the bot immediately unwinds the filled positions by
market-selling them. This locks in a small loss rather than holding
unhedged risk.

### Step 8: Record and Loop

```
Input:   TradeResult
Output:  Updated P&L ledger, status file
Code:    monitor/pnl.py, monitor/status.py
```

Every trade is appended to `pnl_ledger.json` (one JSON object per line).
The status file `status.md` is overwritten each cycle with current state.
The circuit breaker checks cumulative losses. Then the bot sleeps for
1 second and starts the next cycle.

---

## 5. Module Map

```
polymarket/
|
|-- run.py                    Main pipeline orchestrator
|-- config.py                 All settings, loaded from .env
|
|-- client/                   External API communication
|   |-- auth.py               Wallet authentication (L2 HMAC keys)
|   |-- gamma.py              Market discovery (REST API)
|   |-- clob.py               Orderbook fetching + order placement
|   |-- gas.py                Real-time gas price + POL/USD oracle
|   |-- ws.py                 WebSocket manager (prepared, not wired)
|
|-- scanner/                  Opportunity detection
|   |-- models.py             All data types (frozen dataclasses)
|   |-- binary.py             YES+NO < $1 arbitrage
|   |-- negrisk.py            Multi-outcome sum(asks) < $1 arbitrage
|   |-- latency.py            Crypto spot vs prediction market lag
|   |-- spike.py              Breaking news price spike detection
|   |-- fees.py               Taker fees + resolution fee model
|   |-- depth.py              Multi-level orderbook analysis (VWAP)
|   |-- scorer.py             5-factor opportunity ranking
|   |-- strategy.py           Adaptive parameter tuning
|   |-- book_cache.py         In-memory orderbook cache
|
|-- executor/                 Trade execution
|   |-- engine.py             Order placement + partial unwind
|   |-- sizing.py             Half-Kelly position sizing
|   |-- safety.py             Pre-trade checks + circuit breaker
|
|-- monitor/                  Tracking and observability
|   |-- pnl.py                P&L aggregation + NDJSON ledger
|   |-- logger.py             Structured logging (console + JSON)
|   |-- status.py             Rolling markdown status file
|   |-- scan_tracker.py       Scan-only mode statistics
|
|-- tests/                    242 tests, 85% coverage
```

Each module is self-contained. Scanners only depend on `models.py`,
`fees.py`, and the client layer. The executor only depends on models
and the client layer. The monitor depends on nothing but models.
`run.py` is the only file that wires everything together.

---

## 6. Configuration and Knobs

All settings live in `config.py` and are loaded from a `.env` file or
environment variables using pydantic-settings. No hardcoded secrets.

### Trading Thresholds

| Setting                  | Default  | What it controls                     |
|--------------------------|----------|--------------------------------------|
| `min_profit_usd`         | $0.50    | Ignore opportunities below this      |
| `min_roi_pct`            | 2.0%     | Ignore opportunities below this ROI  |
| `max_exposure_per_trade` | $500     | Hard cap per single trade            |
| `max_total_exposure`     | $5,000   | Hard cap across all open positions   |
| `scan_interval_sec`      | 1.0s     | Pause between scan cycles            |

### Risk Limits

| Setting                    | Default | What happens when hit              |
|----------------------------|---------|------------------------------------|
| `max_loss_per_hour`        | $50     | Circuit breaker halts the bot      |
| `max_loss_per_day`         | $200    | Circuit breaker halts the bot      |
| `max_consecutive_failures` | 5       | Circuit breaker halts the bot      |
| `max_gas_profit_ratio`     | 50%     | Reject trade if gas eats >50%      |

### Gas and Fees

| Setting              | Default    | Purpose                           |
|----------------------|------------|-----------------------------------|
| `gas_per_order`      | 150,000    | Estimated gas units per order     |
| `gas_price_gwei`     | 30 gwei    | Fallback if RPC call fails        |
| `gas_cache_sec`      | 10s        | How long to cache gas prices      |
| `fee_model_enabled`  | true       | Enable dynamic fee deductions     |

### Latency and Spike

| Setting                 | Default | Purpose                            |
|-------------------------|---------|------------------------------------|
| `latency_min_edge_pct`  | 5.0%    | Minimum edge after fees            |
| `spot_price_cache_sec`  | 2.0s    | Cache Binance spot prices          |
| `spike_threshold_pct`   | 5.0%    | Detect moves larger than this      |
| `spike_window_sec`      | 30s     | Time window for spike detection    |
| `spike_cooldown_sec`    | 60s     | Cooldown after a spike triggers    |

---

## 7. Safety and Risk Controls

The bot has a layered safety system that gets progressively more
aggressive:

### Layer 1: Per-Opportunity Filters (scanner level)

- Skip if net profit < $0.50.
- Skip if ROI < 2%.
- Skip if zero depth on either side.

These are cheap checks that discard noise early.

### Layer 2: Pre-Trade Safety Checks (executor level)

Before placing any order:

1. **Price freshness:** Re-fetch the book. If prices moved >0.5%, abort.
   Someone else may have taken the opportunity.

2. **Depth verification:** Confirm enough shares exist at the quoted
   price. Thin books can vanish between scan and execution.

3. **Gas check:** Estimate real-time gas. If gas >50% of profit, abort.
   Not worth burning money on gas.

Failure at this layer skips the single opportunity. The bot continues
to the next one.

### Layer 3: Circuit Breaker (session level)

After every trade, the circuit breaker updates its counters:

```
If hourly losses    >= $50   -->  HALT bot
If daily losses     >= $200  -->  HALT bot
If consecutive losses >= 5   -->  HALT bot
```

When the circuit breaker trips, the bot prints a P&L summary and exits.
It does not retry or reset automatically. A human must investigate.

### Layer 4: Partial Fill Handling (execution level)

If a multi-leg trade only partially fills (e.g., YES fills but NO does
not), the bot:

1. Cancels all unfilled orders.
2. Market-sells the filled positions to close them.
3. Records the loss.

This prevents holding unhedged risk. The loss is small (spread cost)
but guaranteed to be bounded.

---

## 8. Fees and Costs

Understanding fees is critical because they can turn a profitable-looking
trade into a losing one.

### Resolution Fee

- **Always applies:** 2% of the $1.00 winning payout = $0.02 per set.
- This is deducted when the market resolves, not when you trade.
- For arbitrage, the bot always has a winning side, so this always costs
  $0.02 per set.

### Taker Fee (Standard Markets)

- **Most markets:** 0% taker fee.
- You just pay the spread (difference between ask and fair value).

### Taker Fee (15-Minute Crypto Markets)

- **Applies to:** BTC/ETH/SOL 15-minute up/down markets.
- **Dynamic formula:** `fee = 3.15% * 4 * price * (1 - price)`
- **At 50/50 odds (price = $0.50):** Maximum fee of 3.15%.
- **At 80/20 odds (price = $0.80):** Fee drops to 2.02%.
- **At 95/5 odds (price = $0.95):** Fee drops to 0.60%.

This fee was introduced specifically to counter latency bots. It makes
latency arbitrage unprofitable at 50/50 odds unless the edge is very
large (>5%).

### Gas Cost (Polygon Network)

- Each order costs roughly 150,000 gas units on Polygon.
- At 30 gwei and $0.50/POL, one order costs about $0.002.
- A binary arb (2 orders) costs about $0.004 in gas.
- A 5-outcome negRisk arb costs about $0.010 in gas.
- Gas is cheap on Polygon, but the bot still checks that it stays below
  50% of net profit.

### Fee Example: Binary Arb

```
YES ask: $0.47,  NO ask: $0.51
Cost per set:  $0.98
Gross profit:  $0.02

Taker fee (standard market):  $0.00
Resolution fee:               $0.02  (2% of $1.00 payout)
Gas (2 orders):               $0.004

Net profit per set: $0.02 - $0.02 - $0.004 = -$0.004

This trade is NOT profitable after fees. The bot skips it.
```

```
YES ask: $0.43,  NO ask: $0.51
Cost per set:  $0.94
Gross profit:  $0.06

Taker fee:       $0.00
Resolution fee:  $0.02
Gas:             $0.004

Net profit per set: $0.06 - $0.02 - $0.004 = $0.036
For 150 sets: 150 * $0.036 = $5.40 net profit.

This trade IS profitable. The bot takes it.
```

---

## 9. Data Flow Diagram

```
                    +---------------------+
                    |    Gamma REST API    |
                    |  (market metadata)  |
                    +---------+-----------+
                              |
                              v
                    +---------------------+
                    |   Fetch Markets      |
                    |   client/gamma.py    |
                    |   25,000+ markets    |
                    +---------+-----------+
                              |
                +-------------+-------------+
                |                           |
                v                           v
     +------------------+        +------------------+
     | Binary Markets   |        | NegRisk Events   |
     | (2 outcomes)     |        | (3+ outcomes)    |
     +--------+---------+        +--------+---------+
              |                           |
              v                           v
     +------------------+        +------------------+
     | CLOB API         |        | CLOB API         |
     | Fetch orderbooks |        | Fetch orderbooks |
     | client/clob.py   |        | client/clob.py   |
     +--------+---------+        +--------+---------+
              |                           |
              v                           v
     +------------------+        +------------------+     +------------------+
     | Binary Scanner   |        | NegRisk Scanner  |     | Latency Scanner  |
     | scanner/binary   |        | scanner/negrisk  |     | scanner/latency  |
     | YES+NO < $1?     |        | sum(asks) < $1?  |     | spot vs market?  |
     +--------+---------+        +--------+---------+     +--------+---------+
              |                           |                         |
              +-------------+-------------+------------+------------+
                            |                          |
                            v                          v
                   +------------------+       +------------------+
                   | Fee Adjustment   |       | Spike Detector   |
                   | scanner/fees.py  |       | scanner/spike.py |
                   +--------+---------+       +--------+---------+
                            |                          |
                            +------------+-------------+
                                         |
                                         v
                              +---------------------+
                              |   Score & Rank      |
                              |   scanner/scorer.py |
                              |   5-factor composite|
                              +---------+-----------+
                                        |
                                        v
                              +---------------------+
                              |   Position Sizing   |
                              |   executor/sizing   |
                              |   Half-Kelly        |
                              +---------+-----------+
                                        |
                                        v
                              +---------------------+
                              |   Safety Checks     |
                              |   executor/safety   |
                              |   fresh? deep? gas? |
                              +---------+-----------+
                                        |
                               pass?    |    fail?
                              +---------+---------+
                              |                   |
                              v                   v
                   +------------------+    (skip, try next)
                   |   Execute Orders |
                   |   executor/engine|
                   |   FAK batch post |
                   +--------+---------+
                            |
                            v
                   +------------------+        +------------------+
                   |   Record P&L     |------->|  pnl_ledger.json |
                   |   monitor/pnl    |        |  (append-only)   |
                   +--------+---------+        +------------------+
                            |
                            v
                   +------------------+        +------------------+
                   | Circuit Breaker  |------->|  HALT if limits  |
                   | executor/safety  |        |  exceeded        |
                   +--------+---------+        +------------------+
                            |
                            v
                   +------------------+        +------------------+
                   |  Write Status    |------->|  status.md       |
                   |  monitor/status  |        |  (overwritten)   |
                   +--------+---------+        +------------------+
                            |
                            v
                      Sleep 1 second
                            |
                            v
                      Next cycle...
```

---

## 10. Running Modes

The bot has four running modes, from safest to most dangerous:

### Dry Run (no wallet needed)

```bash
uv run python run.py --dry-run --limit 500
```

- Uses only public APIs (Gamma + CLOB orderbooks).
- No wallet, no authentication, no orders.
- Good for testing scanner logic and seeing what opportunities exist.
- The `--limit 500` flag caps the number of markets fetched (faster).

### Scan Only (needs wallet)

```bash
uv run python run.py --scan-only
```

- Authenticates with wallet (needed for some API endpoints).
- Scans all markets and reports opportunities.
- Does NOT place any orders.
- Useful for monitoring market conditions.

### Paper Trading (default)

```bash
uv run python run.py
```

- Simulates order execution without hitting the network.
- Assumes all orders fill at quoted prices.
- Records simulated P&L to the ledger.
- Good for validating the full pipeline before going live.

### Live Trading

```bash
uv run python run.py --live
```

- Real orders with real money on real markets.
- Requires explicit `--live` flag (cannot happen by accident).
- All safety checks, circuit breakers, and partial unwind active.
- Writes real trades to the P&L ledger.

---

## 11. Monitoring and Observability

### Console Output

Human-readable, color-coded logs on stderr:

```
14:32:01 INF  Cycle 42: scanned 24,891 markets, found 3 opportunities
14:32:01 INF  Best: BINARY $2.14 profit, 4.3% ROI (Will ETH hit $5k?)
14:32:01 INF  Executed 2 orders, filled 150 sets, net P&L: +$1.87
14:32:02 WRN  Gas spike: 85 gwei, skipping low-profit opportunities
```

### Status File (status.md)

Overwritten every cycle. Contains:

- Current mode (dry-run / scan-only / paper / live).
- Uptime, cycle count, markets scanned.
- Table of opportunities found this cycle.
- Rolling history of the last 20 cycles.

### P&L Ledger (pnl_ledger.json)

Append-only NDJSON file. One JSON object per line per trade:

```json
{"timestamp":1738850521.3,"type":"BINARY_REBALANCE","event_id":"abc123","n_legs":2,"filled":true,"prices":[0.47,0.51],"sizes":[150,150],"fees":3.0,"gas":0.004,"net_pnl":1.87,"ms":234,"orders":["ord1","ord2"]}
```

### JSON Log File (optional)

Machine-readable structured logs for external monitoring:

```bash
uv run python run.py --json-log bot.log
```

---

## 12. Known Limitations

### Not Yet Wired

- **WebSocket feed:** The `client/ws.py` WebSocket manager and
  `scanner/book_cache.py` cache are built but not connected to the main
  loop. The bot currently fetches orderbooks via REST each cycle, which
  is slower than streaming.

- **Depth-based arbitrage:** `scanner/depth.py` has functions for
  multi-level VWAP sweep analysis, but these are not called by any
  scanner yet. Currently only best-bid/best-ask is used.

- **Adaptive strategy:** `scanner/strategy.py` can tune parameters
  based on gas prices and market conditions, but the selected strategy
  is not yet fed back into scanner thresholds in the main loop.

### Design Constraints

- **No position tracking across cycles:** The bot assumes every
  arbitrage resolves within one cycle (buy both sides, wait for
  resolution). It does not track open positions from previous cycles.

- **REST polling:** At 1-second intervals, the bot may miss
  opportunities that appear and disappear within a second. WebSocket
  integration would fix this.

- **CoinGecko rate limits:** The free-tier CoinGecko API used for
  POL/USD pricing returns 429 errors frequently. The bot falls back to
  a default of $0.50/POL, which may be inaccurate.

- **Single-threaded execution:** Orders are placed sequentially within
  a cycle. A faster bot could execute multiple opportunities in parallel.

### Fee Headwinds

- The 3.15% dynamic taker fee on 15-minute crypto markets severely
  limits latency arbitrage profitability. This fee was introduced
  specifically to counter bots like this one.

- The 2% resolution fee ($0.02 per set) eats into thin binary arb
  margins. Only arbs with >$0.02 spread per set are profitable.

---

*End of report.*
