# Polymarket Arbitrage Bot -- Status

*Updated 2026-02-16 15:52:21*

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
| Event | Market question for single-market arbs; event title for multi-market baskets. |
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

| Field                        | Value                                    |
|------------------------------|------------------------------------------|
| Mode                         | DRY-RUN (public APIs only, no execution) |
| Uptime                       | 0s                                       |
| Cycle                        | 0                                        |
| Markets scanned              | 0                                        |
| Opportunities (this cycle)   | 0                                        |
| Opportunities (session)      | 0                                        |
| Executable lane (this cycle) | 0                                        |
| Research lane (this cycle)   | 0                                        |
| Executable lane profit       | $0.00                                    |
| Research lane profit         | $0.00                                    |

## Opportunities This Cycle

*No opportunities found.*

## Recent Cycles

| Cycle | Time     | Markets | Opps | Best Type | Best ROI | Best Profit | Best Event |
|-------|----------|---------|------|-----------|----------|-------------|------------|
| 0     | 15:52:21 | 0       | 0    | --        | --       | --          | --         |

