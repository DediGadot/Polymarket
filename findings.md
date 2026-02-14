# Findings

Research notes and technical details for the profit-maximization implementation.

**Last updated:** 2026-02-14

---

## Dry-Run Baseline (2026-02-14)

### Pipeline Output
- **14,118 total markets**: 449 binary, 13,669 negrisk (2,345 events)
- **1 opportunity per cycle**: SpaceX market cap negrisk arb
  - $2.10 net profit, 4.67% ROI, 8 legs, $44.95 capital
  - Repeating every cycle (persistent arb)
- **Strategy mode**: CONSERVATIVE (locked due to gas threshold)
- **Zero** binary arbs, latency arbs, spike arbs, or cross-platform arbs found
- **Cycle time**: ~8-10 seconds

### Root Cause Analysis: Why Only 1 Opportunity

| Bottleneck | Impact | Evidence |
|-----------|--------|----------|
| CONSERVATIVE mode raises thresholds 50% | Kills opps between $0.50-$0.75 profit, 2-3% ROI | `strategy.py:142` — `HIGH_GAS_GWEI=100` triggers on normal Polygon gas |
| Kelly deploys 23% of bankroll | Even when arbs exist, position sizes are tiny | `sizing.py:66` — `kelly_odds_confirmed=0.10` |
| 0.5% slippage ceiling | Misses 80%+ of available depth | `binary.py:120-121` — `* 1.005` hardcoded |
| $500 per-trade cap | Limits even perfectly-sized arbs | `config.py:28` |
| min_hours_to_resolution=1.0 | Blocks near-resolution sniping | `config.py:73` |
| No maker strategy | 100% taker, paying spread instead of earning it | All scanners use FAK orders |
| No resolution sniping | Discards markets approaching resolution | `filters.py:23-44` |
| No partial negrisk | Requires complete set (sum < $1) | `negrisk.py` — must buy ALL outcomes |
| No stale-quote detection | Reacts to 30-second windows, not 100ms ticks | `spike.py:113` — threshold_pct=5.0, window=30s |

---

## Gas Price Analysis: Polygon vs Ethereum

Polygon gas is fundamentally different from Ethereum:
- **Polygon 100 gwei**: ~$0.003 per 150K gas order (at POL=$0.50)
- **Ethereum 100 gwei**: ~$30 per 150K gas order (at ETH=$2000)
- **Factor**: ~10,000x difference in dollar cost

The `HIGH_GAS_GWEI = 100.0` threshold was designed for Ethereum. On Polygon, even 1000 gwei is cheap ($0.03/order).

**New thresholds should be dollar-denominated:**
- AGGRESSIVE: gas_cost_usd < $0.01/order
- CONSERVATIVE: gas_cost_usd > $0.10/order

---

## Kelly Sizing Calibration

### Current (Broken)
```
kelly_odds_confirmed = 0.10 (10:1 implied)
For 5% edge: f = 0.05/0.10 = 0.50 → half-Kelly = 0.25
Capital deployed: 25% of bankroll
```

### Correct Model
For confirmed arbs (YES + NO < $1):
- **Success = both orders fill completely** (~85% probability)
- **Failure = partial fill requiring unwind** (~15% probability)
- **Loss on failure** = slippage on unwind (~1-3% of notional)

True Kelly: `f = (p*b - q) / b`
- p = 0.85, q = 0.15
- b = edge / risk = 0.05 / 0.02 = 2.5
- f = (0.85 * 2.5 - 0.15) / 2.5 = 0.79

**Half-Kelly = 0.40** → deploy 40% of available capital. Current: 12.5%.

Setting `kelly_odds_confirmed = 0.65` yields comparable aggressiveness.

---

## Slippage Analysis

### Current: 0.5% fixed ceiling
For best_ask = $0.10:
- Ceiling = $0.1005 (half a cent above best)
- Only sees top-of-book depth

### Proposed: Edge-proportional
For 5% edge and slippage_fraction = 0.4:
- Allowable slippage = 5% × 0.4 = 2%
- Ceiling = $0.102 (2 cents above best)
- Sees 4x more depth levels
- Net edge after slippage = 5% - 2% = 3% (still profitable)

For 1% edge:
- Allowable slippage = 1% × 0.4 = 0.4%
- Nearly same as current (conservative for thin edges)

---

## New Scanner Revenue Estimates

| Scanner | Edge Range | Frequency | Est. Daily Revenue (at $50K capital) |
|---------|-----------|-----------|--------------------------------------|
| Maker rebalance | 0.2-2% per set | 50-200/day | $500-2,000 |
| Resolution sniping | 3-15% per snipe | 5-20/day | $200-1,000 |
| NegRisk value bets | 10-30% edge | 10-30/day | $300-1,500 |
| Stale-quote sniping | 2-5% per snipe | 20-50/day | $200-800 |

These are rough estimates based on Polymarket market structure and volume. Actual results depend on competition from other bots.

---

## Technical Constraints

### Maker Strategy
- Polymarket has **0% taker fee** on standard markets but charges resolution fee
- GTC orders require separate lifecycle management
- Cancel all maker orders if book moves >1 tick (prevent adverse selection)
- Need separate capital bucket — maker ties up capital until fill

### Resolution Sniping
- Must know outcome BEFORE market moves to $0.99+
- External data sources have their own latency
- Sports: ESPN/live scores update within 5-30 seconds
- Crypto: Binance price is real-time (already have integration via LatencyScanner)
- Elections: minutes-to-hours latency (low frequency but high edge)

### Stale-Quote Sniping
- WS feed granularity: per-trade (not per-tick)
- Last_trade_price updates are NOT guaranteed to precede book updates
- Need to handle false positives: price reverted before we can execute
- Rate limit REST fetches: max 10/second to avoid 429s

---

## Cross-References

| Finding | Plan Item |
|---------|-----------|
| Gas threshold wrong for Polygon | 1.1 |
| Kelly 5x underdeployed | 1.2 |
| Slippage ceiling too tight | 1.3 |
| Exposure caps too low | 1.4 |
| ArbTracker needs verification | 1.5 |
| Maker strategy missing | 2.1-2.3 |
| Resolution sniping missing | 3.1-3.3 |
| Partial negrisk missing | 4.1-4.2 |
| Stale-quote sniping missing | 5.1-5.2 |
