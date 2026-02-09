# Polymarket Arbitrage Bot -- Status

*Updated 2026-02-09 11:25:10*

## How the Pipeline Works

The bot runs in a continuous loop. Each iteration is called a **cycle**:

1. **Fetch markets** -- pulls all active markets from the Gamma REST API.
   Binary markets (YES/NO) and negRisk events (multi-outcome) are separated.
2. **Scan for arbitrage** -- four independent scanners run on the fetched markets:
   - **binary_rebalance**: buy arb when YES ask + NO ask < $1.00, or sell arb when YES bid + NO bid > $1.00.
   - **negrisk_rebalance**: buy arb when sum of all YES asks < $1.00, or sell arb when sum of YES bids > $1.00.
   - **latency_arb**: 15-minute crypto markets (BTC/ETH/SOL up/down) reprice slower than spot exchanges.
     The bot compares Polymarket odds to live spot momentum and buys or sells when the market lags.
   - **spike_lag**: during breaking news one market reprices instantly while sibling markets in the same
     event lag by 5-60 seconds. The bot builds a multi-leg negRisk basket on the lagging outcomes.
3. **Score and rank** -- every opportunity gets a composite score (0-1) from five weighted factors:
   profit magnitude, fill probability, capital efficiency, urgency, and competition.
   Opportunities are sorted best-first.
4. **Size** -- half-Kelly criterion determines how many sets to trade given current bankroll and edge.
   *(Skipped in DRY-RUN and SCAN-ONLY modes.)*
5. **Safety checks** -- price freshness, orderbook depth, and gas cost are verified. If any check fails
   the opportunity is skipped. A circuit breaker halts the bot on excessive losses.
   *(Skipped in DRY-RUN and SCAN-ONLY modes.)*
6. **Execute** -- FAK (fill-and-kill) orders are sent for each leg. Partial fills are unwound.
   *(Skipped in DRY-RUN and SCAN-ONLY modes.)*
7. **Record** -- P&L is updated, the trade is appended to the NDJSON ledger, and this status file is rewritten.
   *(Skipped in DRY-RUN and SCAN-ONLY modes.)*
8. **Sleep** -- the bot waits for the remaining scan interval before starting the next cycle.

## Field Reference

### Current State

| Field | Meaning |
|-------|---------|
| Mode | DRY-RUN = public APIs only, no wallet. SCAN-ONLY = detect but don't trade. PAPER = simulated fills. LIVE = real orders. |
| Uptime | Wall-clock time since the bot process started. |
| Cycle | How many full fetch-scan-execute loops have completed. |
| Markets scanned | Number of individual binary markets fetched this cycle (negRisk markets counted individually). |
| Opportunities (this cycle) | Arbitrage opportunities that passed minimum profit and ROI filters this cycle. |
| Opportunities (session) | Cumulative count across all cycles since startup. |
| Trades executed | (Trading modes only) How many opportunities were actually sent to the exchange. |
| Net P&L | (Trading modes only) Realized profit/loss across all trades this session, after fees and gas. |
| Current exposure | (Trading modes only) Total capital currently locked in open positions awaiting resolution. |

### Opportunities Table

| Column | Meaning |
|--------|---------|
| Type | Which scanner found it: binary_rebalance, negrisk_rebalance, latency_arb, or spike_lag. |
| Event | Human-readable event title from Polymarket (e.g. "Will BTC be above 100k?"). |
| Profit | Net expected profit in USD after subtracting gas cost and the 2% resolution fee. |
| ROI | Return on invested capital as a percentage (net_profit / required_capital * 100). |
| Score | Composite score (0-1). Weights: 25% profit, 25% fill probability, 20% capital efficiency, 20% urgency, 10% competition. |
| Legs | Number of separate orders required (2 for binary, N for negRisk, 1 for latency, N for spike). |
| Capital | USDC needed to execute all legs at the quoted prices and sizes. |

### Recent Cycles Table

| Column | Meaning |
|--------|---------|
| Cycle | Cycle number. |
| Time | Wall-clock time when the cycle completed. |
| Markets | Markets scanned that cycle. |
| Opps | Opportunities found that cycle. |
| Best Type | Scanner type of the highest-profit opportunity. |
| Best ROI | ROI of the highest-profit opportunity. |
| Best Profit | Dollar profit of the highest-profit opportunity. |
| Best Event | Event title of the highest-profit opportunity. |

---

## Current State

| Field                      | Value                                    |
|----------------------------|------------------------------------------|
| Mode                       | DRY-RUN (public APIs only, no execution) |
| Uptime                     | 1h 21m                                   |
| Cycle                      | 446                                      |
| Markets scanned            | 14,223                                   |
| Opportunities (this cycle) | 2                                        |
| Opportunities (session)    | 314                                      |

## Opportunities This Cycle

| # | Type                     | Event                                              | Profit | ROI    | Score | Legs | Capital |
|---|--------------------------|----------------------------------------------------|--------|--------|-------|------|---------|
| 1 | [SELL] negrisk_rebalance | Will Galatasaray SK win on 2026-02-17?             | $3.99  | 68.40% | 0.40  | 5    | $5.84   |
| 2 | [SELL] negrisk_rebalance | Will the price of Bitcoin be less than $60,000 ... | $1.33  | 26.53% | 0.35  | 14   | $5.00   |

## Recent Cycles

| Cycle | Time     | Markets | Opps | Best Type         | Best ROI | Best Profit | Best Event                               |
|-------|----------|---------|------|-------------------|----------|-------------|------------------------------------------|
| 446   | 11:25:10 | 14,223  | 2    | negrisk_rebalance | 68.40%   | $3.99       | Will Galatasaray SK win on 2026-02-17?   |
| 445   | 11:25:00 | 14,223  | 2    | negrisk_rebalance | 68.38%   | $3.99       | Will Galatasaray SK win on 2026-02-17?   |
| 444   | 11:24:51 | 14,223  | 3    | negrisk_rebalance | 68.38%   | $3.99       | Will Galatasaray SK win on 2026-02-17?   |
| 442   | 11:24:33 | 14,223  | 2    | negrisk_rebalance | 68.36%   | $3.99       | Will Galatasaray SK win on 2026-02-17?   |
| 441   | 11:24:24 | 14,223  | 3    | negrisk_rebalance | 68.36%   | $3.99       | Will Galatasaray SK win on 2026-02-17?   |
| 440   | 11:24:12 | 14,223  | 3    | negrisk_rebalance | 68.34%   | $3.99       | Will Galatasaray SK win on 2026-02-17?   |
| 438   | 11:23:59 | 14,223  | 3    | negrisk_rebalance | 68.33%   | $3.99       | Will Galatasaray SK win on 2026-02-17?   |
| 437   | 11:23:43 | 14,223  | 2    | negrisk_rebalance | 68.30%   | $3.99       | Will Galatasaray SK win on 2026-02-17?   |
| 436   | 11:23:34 | 14,223  | 2    | negrisk_rebalance | 68.30%   | $3.99       | Will Galatasaray SK win on 2026-02-17?   |
| 435   | 11:23:13 | 14,210  | 1    | negrisk_rebalance | 3.31%    | $0.30       | Will SpaceX's market cap be less than... |
| 434   | 11:23:04 | 14,210  | 0    | --                | --       | --          | --                                       |
| 433   | 11:22:55 | 14,210  | 0    | --                | --       | --          | --                                       |
| 432   | 11:22:45 | 14,210  | 1    | negrisk_rebalance | 3.30%    | $0.29       | Will SpaceX's market cap be less than... |
| 430   | 11:22:27 | 14,210  | 0    | --                | --       | --          | --                                       |
| 429   | 11:22:18 | 14,210  | 1    | negrisk_rebalance | 3.28%    | $0.29       | Will SpaceX's market cap be less than... |
| 428   | 11:22:06 | 14,210  | 0    | --                | --       | --          | --                                       |
| 426   | 11:21:52 | 14,210  | 0    | --                | --       | --          | --                                       |
| 425   | 11:21:36 | 14,210  | 0    | --                | --       | --          | --                                       |
| 424   | 11:21:26 | 14,210  | 1    | negrisk_rebalance | 3.19%    | $0.29       | Will SpaceX's market cap be less than... |
| 423   | 11:21:06 | 14,214  | 1    | negrisk_rebalance | 6.34%    | $1.59       | Will the price of XRP be less than $1... |

