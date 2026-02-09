# Findings: Polymarket Arbitrage Bot

## Research Date: 2026-02-05 (Updated with web research + codex analysis)

---

## 1. Polymarket Architecture

### Hybrid-Decentralized CLOB
- **Off-chain**: Operator handles order matching, sequencing, execution
- **On-chain**: Settlement on Polygon via EIP-712 signed orders
- **Atomic swaps**: CTF Exchange contract swaps ERC-1155 outcome tokens for USDC
- **Mint/merge**: YES order + NO order can be matched via minting/merging token pairs

### APIs
| API | Base URL | Purpose |
|-----|----------|---------|
| CLOB | `https://clob.polymarket.com` | Orders, pricing, orderbooks |
| Gamma | `https://gamma-api.polymarket.com` | Market discovery, metadata |
| Data | `https://data-api.polymarket.com` | Positions, activity, history |
| WS Market | `wss://ws-subscriptions-clob.polymarket.com/ws/market` | Real-time orderbook |
| WS User | `wss://ws-subscriptions-clob.polymarket.com/ws/user` | Authenticated updates |

### SDKs
- **Python**: `py-clob-client` v0.34.5 (latest Jan 2026)
- **TypeScript**: `@polymarket/clob-client`
- **Rust**: `polymarket-client-sdk`

### Authentication
- **L1**: Wallet-based (EIP-712 signatures) for credential creation
- **L2**: HMAC-SHA256 API keys (key, secret, passphrase) for trading
- Signature types: EOA (0), POLY_PROXY (1), GNOSIS_SAFE (2)

### Rate Limits
| Endpoint | Burst/10s | Sustained/10min |
|----------|-----------|-----------------|
| POST /order | 3,500 | 36,000 |
| DELETE /order | 3,000 | 30,000 |
| POST /orders (batch) | 1,000 | 15,000 |
| /book, /price | 1,500 | -- |

### Fee Structure (UPDATED 2026)
- **Most markets**: Zero fees
- **15-min crypto markets**: Dynamic taker fee, highest at 50/50 odds (~3.15%), drops toward zero near 0% or 100%. Fees redistributed daily to market makers as rebates.
- **Polymarket US (DCM)**: 10 bps taker fee
- **Winning position**: 2% fee on winning positions at resolution

### Order Types
- GTC (Good-Til-Cancelled) -- resting limit order
- GTD (Good-Til-Date) -- limit with expiration
- FOK (Fill-Or-Kill) -- immediate full fill or cancel
- FAK (Fill-and-Kill) -- immediate fill available quantity, cancel remainder (IOC semantics)

### WebSocket Market Channel Messages
| Message Type | Trigger | Data |
|-------------|---------|------|
| `book` | On subscription + after trades | Full bid/ask orderbook snapshot |
| `price_change` | Order placed/cancelled | Individual price level update with best bid/ask |
| `last_trade_price` | Maker/taker match | Trade price, size, side, fee rate, timestamp |
| `tick_size_change` | Price >0.96 or <0.04 | New tick size for affected side |
| `best_bid_ask` | Best price shifts | Spread + timestamp (feature-flagged) |

### Key Contracts (Polygon)
| Contract | Address |
|----------|---------|
| CTF Exchange | `0x4bFb41d5B3570DeFd03C39a9A4D8dE6Bd8B8982E` |
| Neg Risk CTF Exchange | `0xC5d563A36AE78145C45a50134d48A1215220f80a` |
| Neg Risk Adapter | `0xd91E80cF2E7be2e162c6513ceD06f1dD0dA35296` |
| USDC (Polygon) | `0x2791Bca1f2de4661ED88A30C99A7a9449Aa84174` |
| Conditional Tokens | `0x4D97DCd97eC945f40cF65F87097ACe5EA0476045` |

---

## 2. Arbitrage Types -- Research-Backed Ranking

### Type A: NegRisk Multi-Outcome Rebalancing -- $28.68M (72.4%)
- **Principle**: In multi-outcome markets, sum of all YES prices must = $1.00
- **Mechanism**: NegRisk adapter converts NO share → YES shares in all other markets
- **Capital efficiency**: 29x advantage over binary arbitrage
- **Historical**: $28.68M extracted (73% of all arbitrage profits despite 8.6% of opportunities)
- **Key insight**: Liquidity fragments across outcomes; retail flow concentrates on 1-2 favorites while complementary probability space trades thin
- **Status**: Already implemented

### Type B: Binary Single-Condition Rebalancing -- $10.58M (26.7%)
- **Principle**: YES_price + NO_price must = $1.00
- **Opportunity**: When sum < $1.00, buy both; when sum > $1.00, sell both
- **Edge**: Typically 2-5 cents, requires speed and volume
- **Status**: Already implemented

### Type C: Latency Arbitrage (15-min Crypto) -- ~$5M+ estimated
- **Principle**: Polymarket 15-min crypto markets (BTC/ETH/SOL up/down) reprice slower than spot exchanges (Binance, Coinbase)
- **Best known result**: $313 → $414,000 in one month, 98% win rate
- **Mechanism**: When spot momentum confirms direction (e.g., BTC up 0.5% in 2 minutes), real probability is ~85% but market still shows ~50/50. Bot buys at 50c, expected value is 85c.
- **Fee challenge**: Polymarket introduced dynamic taker fees (up to 3.15% at 50/50) specifically to kill this strategy
- **Key insight**: Profitable when odds are NOT near 50/50 (fees drop to near zero at extreme odds). Bot needs to wait for confirmation before entering.
- **Status**: NOT IMPLEMENTED -- highest-alpha new opportunity

### Type D: Spike Lag Arbitrage -- estimated high EV
- **Principle**: During breaking news, one market reprices instantly while related markets lag 5-60s
- **Edge**: 10-50x wider than steady-state rebalancing
- **Example**: Candidate drops out → their market crashes → "Who will be nominee?" hasn't adjusted yet
- **Status**: NOT IMPLEMENTED

### Type E: Deep-Book Arbitrage
- **Principle**: Arb may not exist at top-of-book but exists 2-3 levels deeper
- **Example**: best YES ask=$0.52, NO ask=$0.49 (sum=$1.01, no arb). Level 2: YES 200@$0.50, NO 150@$0.48 (sum=$0.98, 2% arb on 150 shares)
- **Why bots miss it**: Most bots only check best bid/ask
- **Status**: NOT IMPLEMENTED

### Type F: Combinatorial Cross-Market -- $95K (0.24%) -- LOW VALUE
- **Principle**: Logically related markets priced inconsistently
- **IMDEA findings**: Only 13 valid dependent pairs among 46,360 combinations. Only 5 pairs produced profit totaling $95K.
- **62% of LLM-detected dependencies fail** due to liquidity asymmetry + non-atomic execution
- **Key insight**: Sounds sophisticated but generated only 0.24% of total profits. NOT worth engineering effort.
- **Status**: DEPRIORITIZED

### Type G: Cross-Platform (Polymarket vs Kalshi)
- **Principle**: Same event priced differently across platforms
- **Challenge**: Non-atomic, different settlement definitions (2024 govt shutdown: different resolutions on same event)
- **Threshold**: Spreads must exceed 6% after combined 5%+ fees
- **Status**: Out of scope (requires Kalshi integration)

---

## 3. Profitability Analysis

### Historical Data (Apr 2024 - Apr 2025, IMDEA Research)
- Total arbitrage profits: **$39.6M** (verified)
- 86 million bets analyzed across thousands of markets
- Top arbitrageur: $2.01M across 4,049 transactions (~$496 avg per trade)
- **Only 40% realization rate** -- $100M+ theoretical, $39.6M actually captured
- **Execution frequency matters more than position size**

### Current Competitive Landscape (2026)
- Bots dominate; arb windows close in ~200ms
- Institutional capital entering (ICE $2B investment Oct 2025)
- Spreads compressing: "Opportunities that paid 3-5% in 2024 now pay 1-2%"
- VPS infrastructure critical: 1-30ms latency advantage via proximity to Polygon nodes
- 78% of arb opportunities in low-volume markets fail due to execution inefficiency
- Manual identification essentially impossible in 2026

### Fee Impact on Profitability
- Most markets: zero fees → structural arb still viable
- 15-min crypto: dynamic fee up to 3.15% at 50/50 odds → latency arb needs >6.3% edge at midpoint
- Polymarket US DCM: 10bps → negligible impact
- Winning position: 2% at resolution → reduces long-hold arb returns
- **Combined fees 5%+ make sub-5% spreads unprofitable on fee-bearing markets**

### Realistic 2026 Expectations
- Retail ($5K capital): 5-10 opportunities/month, $250-$1,000 gross
- Bot ($50K+ capital): $5-10K daily possible with speed advantage
- Top bots: $200K+ with 85%+ win rate

---

## 4. Critical Technical Insights

### Execution > Detection (IMDEA finding)
The 40% realization rate means **60% of detected arbs are lost to execution**:
- Non-atomic multi-leg execution (legs can be sniped between fills)
- Stale orderbook data from REST polling
- GTC orders sitting unfilled while opportunity evaporates
- Competing bots with lower latency getting there first

### WebSocket is Mandatory
- REST polling at 1s cycle: opportunity already captured by WS-connected bots
- WS provides sub-100ms price updates
- `book` message gives full snapshot on subscription
- `price_change` gives incremental updates (trigger scan immediately)
- `last_trade_price` enables spike detection in real-time

### Order Type Selection Matters
- **GTC** (current): Order rests on book indefinitely. Bad for arb -- opportunity may disappear but order stays.
- **FAK/FOK**: Immediate fill or cancel. Correct for arb execution.
- **GTD**: Time-limited. Good for arbs with known expiry windows.
- py-clob-client v0.34+ supports all types

### IMDEA Combinatorial Detection Algorithm (for reference)
Search space reduction heuristics:
1. Temporal alignment: only compare markets with identical end dates
2. Topic clustering: text embeddings (Linq-Embed-Mistral) into 7 categories
3. Liquidity filtering: reduce >4 condition markets to top 4 by volume
4. LLM validation: DeepSeek-R1-Distill-Qwen-32B for logical dependency detection
Result: 46,360 combinations → 13 valid pairs → 5 profitable ($95K total)

---

## 5. Key Open Source References

| Repository | Strategy | Notes |
|-----------|----------|-------|
| [Polymarket/agents](https://github.com/Polymarket/agents) | AI agent trading | Official |
| [runesatsdev/polymarket-arbitrage-bot](https://github.com/runesatsdev/polymarket-arbitrage-bot) | Single + NegRisk arb | Community, good reference |
| [Trust412/Polymarket-spike-bot-v1](https://github.com/Trust412/Polymarket-spike-bot-v1) | Spike detection HFT | Community |
| [franko74/polymarket-arbitrage-bot-btc-eth-15m](https://github.com/franko74/polymarket-arbitrage-bot-btc-eth-15m) | 15-min crypto latency arb (Rust) | Community, latency reference |

---

## 6. Risk Factors

| Risk | Likelihood | Impact | Mitigation |
|------|-----------|--------|------------|
| Leg risk (partial fill) | HIGH | MEDIUM | FAK/FOK orders, timeout enforcement |
| Stale quotes | MEDIUM | LOW | WS feed, re-check before execution |
| Dynamic fee erosion | HIGH (crypto markets) | HIGH | Fee model, avoid 50/50 odds on fee markets |
| Gas spikes | LOW | LOW | Gas oracle, reject if gas > 50% of profit |
| Oracle dispute | LOW | HIGH | Avoid markets near resolution |
| Rate limiting | MEDIUM | LOW | Respect limits, exponential backoff |
| Competition (speed) | HIGH | HIGH | WS + VPS + FAK orders |
| Settlement risk (cross-platform) | MEDIUM | CRITICAL | Avoid cross-platform for now |

---

## 7. CRITICAL BUG: Orderbook Sort Order (2026-02-06)

### Root Cause
The `client/clob.py` functions `get_orderbook()` and `get_orderbooks()` do NOT sort orderbook levels after converting from the py-clob-client SDK. The SDK returns:
- **Asks in DESCENDING order** (highest/worst price first, index 0 = $0.999)
- **Bids in ASCENDING order** (lowest/worst price first, index 0 = $0.001)

But `OrderBook.best_ask` (asks[0]) and `OrderBook.best_bid` (bids[0]) assume:
- Asks sorted ASCENDING (best/lowest first)
- Bids sorted DESCENDING (best/highest first)

### Impact
- `best_ask` returns the **WORST** ask ($0.999) instead of the **BEST** ask
- `best_bid` returns the **WORST** bid ($0.001) instead of the **BEST** bid
- Binary scanner: `yes_ask + no_ask ≈ $2.00` (sum of worst asks) → never < $1.00 → zero buy arbs
- NegRisk scanner: `sum(yes_asks) ≈ N × $0.999` → never < $1.00 → zero buy arbs
- Sell arb: `yes_bid + no_bid ≈ $0.002` (sum of worst bids) → never > $1.00 → zero sell arbs
- **100% of opportunities are missed** due to this single bug

### Evidence
```
YES orderbook for top liquid market (vol=$55M):
  Asks (first 5 levels):
    [0] price=0.9990  ← WORST ask used as "best_ask"
    [1] price=0.9980
    [2] price=0.9970
    [3] price=0.9960
    [4] price=0.9950
  Asks sorted ascending (best first): False

  Bids (first 5 levels):
    [0] price=0.0010  ← WORST bid used as "best_bid"
    [1] price=0.0020
    [2] price=0.0030
  Bids sorted descending (best first): False
```

### Fix
Sort asks ascending and bids descending in `get_orderbook()` and `get_orderbooks()` in `client/clob.py`.

### Secondary Issues (lower priority)
1. **CoinGecko API failure**: `matic-network` ID may be stale (Polygon rebranded to POL). Returns wrong key then 429 rate limit. Falls back to $0.50 default.
2. **Resolution fee**: $0.02/set (2% on winning payout) is correctly applied per Polymarket docs. Reduces profit but not a bug.
3. **Gas cost**: $0.09 per 2-leg at 598 gwei is small but real.

---

## Sources
- [IMDEA "Probabilistic Forest" paper](https://arxiv.org/html/2508.03474v1)
- [62% of LLM-detected dependencies fail](https://medium.com/@navnoorbawa/combinatorial-arbitrage-in-prediction-markets-why-62-of-llm-detected-dependencies-fail-to-26f614804e8d)
- [Polymarket dynamic fees](https://www.financemagnates.com/cryptocurrency/polymarket-introduces-dynamic-fees-to-curb-latency-arbitrage-in-short-term-crypto-markets/)
- [Bot $313 → $414K](https://finance.yahoo.com/news/arbitrage-bots-dominate-polymarket-millions-100000888.html)
- [Polymarket WS docs](https://docs.polymarket.com/developers/CLOB/websocket/market-channel)
- [Polymarket fee docs](https://docs.polymarket.com/polymarket-learn/trading/fees)
- [2026 arb guide](https://newyorkcityservers.com/blog/prediction-market-arbitrage-guide)
- [Polymarket HFT](https://www.quantvps.com/blog/polymarket-hft-traders-use-ai-arbitrage-mispricing)
- [Polymarket CLOB docs](https://docs.polymarket.com/developers/CLOB/introduction)
- [NegRisk $29M extraction](https://medium.com/@navnoorbawa/negrisk-market-rebalancing-how-29m-was-extracted-from-multi-condition-prediction-markets-2f1f91644c5b)
- [Polymarket API Reference](https://docs.polymarket.com/quickstart/reference/endpoints)

---

## 8. Validation Session Findings (2026-02-07)

### Pre-Validation Baseline
- **270 tests**, all passing, **88% coverage** (up from 242 tests / 85%)
- Test suite runs in 1.28s
- 17 source modules, 26 test files

### Coverage Analysis
| Module | Coverage | Assessment |
|--------|----------|------------|
| `client/ws.py` | 0% | Not wired in yet -- acceptable |
| `run.py` | 21% | Main loop hard to unit test -- utility functions testable |
| `client/clob.py` | 50% | SDK wrappers -- tested via integration mocks |
| `client/auth.py` | 50% | Needs real wallet -- tested via integration |
| `executor/engine.py` | 78% | Missing: single-leg, unwind, wait_for_fill terminal states |
| `executor/safety.py` | 85% | Missing: gas check with non-positive profit |
| `scanner/latency.py` | 80% | Missing: sell-side arb path, error handling |
| Everything else | 94-100% | Good coverage |

### Key Observations
1. Tests grew from 242 → 270 since last session (new test files for run.py, status.py added)
2. `_sort_book_levels()` fix from Session 4 is properly tested and working
3. Gas oracle uses `polygon-ecosystem-token` (fixed from `matic-network`)
4. Resolution fee correctly applies $0.02/set on ALL markets
5. `strategy.py` StrategySelector not actually wired into run.py's scan params -- run.py initializes it but never calls `strategy.select()`. The scored params from StrategySelector are computed but not used to tune the scan. **FIXED**: Now wired in, calls `strategy.select(MarketState)` per cycle.

### Bugs Found and Fixed
1. **StrategySelector not wired** (severity: MEDIUM) -- adaptive scan parameters were never applied. Fixed by calling `strategy.select()` per cycle and using returned `ScanParams` for min_profit, min_roi, and per-scanner enable/disable.
2. **Reentrant logging in signal handler** (severity: LOW) -- `logger.info()` in `handle_signal()` could race with ongoing log writes, causing `RuntimeError: reentrant call`. Fixed by using `print(file=sys.stderr)` instead.

### Arithmetic Validation Results (35 tests)
All profit calculations confirmed correct:
- Binary buy/sell arb: cost/proceeds per set, max_sets, gas, net_profit, ROI all match manual calculation
- NegRisk: gas cost scales correctly with n_legs
- Fee model: dynamic fee formula matches specification (parabolic, symmetric, 3.15% peak at 50/50)
- Resolution fee: correctly $0.02/set on all markets
- Gas oracle: `estimate_cost_usd()` formula verified (n * gas_per_order * gwei * 1e9 / 1e18 * matic_usd)
- Depth sweep: VWAP multi-level fill pricing correct
- Kelly sizing: half-Kelly correctly caps at 50% of full Kelly

### Live Market Validation
Full dry-run against 27,460 markets confirmed:
- 9 NegRisk arbs detected (Nebraska Senate: $48.75 profit, 30.31% ROI)
- 2028 US Presidential Election: consistent $0.91/3.12% arb across cycles
- No binary arbs found (expected -- spreads are tight in current market)
- Spike detector functional: 1-3 spikes detected per cycle
- CoinGecko rate limits confirmed: 429s after a few calls, fallback works

---

## 9. Pipeline Reliability Audit (2026-02-07)

### "Too Good to Be True" Analysis

Reviewed all 9 opportunities from the live dry-run scan. Found that the results are misleading:

#### Phantom Opportunities (#1, #2)
- **Elche CF** ($4,116.38, 48,900% ROI): Match date was 2026-01-19 -- three weeks ago. Market already resolved. Abandoned liquidity on the book creates a false arb signal.
- **Taylor Swift** ($3,628.01, 24,400% ROI): Same class -- stale/resolved market with dead liquidity.
- **Root cause**: Market dataclass has only `active: bool`. No `end_date`, `closed`, or `resolved` field. Gamma API filters `active=true, closed=false` but some resolved markets leak through with abandoned liquidity.

#### Non-Capturable Opportunities (#4-#9)
- Multi-leg arbs (6-31 legs) persist across cycles because no one can execute them atomically
- Polymarket batch endpoint processes orders sequentially -- no all-or-nothing guarantee
- 31-leg JD Vance arb requires 2 batch API calls with ~100ms gap between them
- Partial fill unwind is fire-and-forget: exceptions silently swallowed, orphaned positions possible

#### Genuine Opportunity (#3 Nebraska Senate)
- $48.75 profit, 30.31% ROI on 2-leg arb appears real
- BUT: verify_depth() only checks best-level size, not multi-level depth
- max_sets calculation may be overstated

### Depth Validation Gaps Found
1. **Scanners use best-level only**: `max_sets = min(yes_ask.size, no_ask.size)` -- does NOT walk multiple book levels
2. **verify_depth() checks best-level only**: `best_ask.size >= leg.size` -- not multi-level sweep
3. **depth.py functions exist but unused**: `sweep_cost()`, `effective_price()`, `sweep_depth()` are all tested but never called by any scanner
4. **ScoringContext never populated**: `book_depth_ratio` defaults to 1.0, `fill_score` is always 0.50 for all opportunities. The scorer literally cannot distinguish thin vs deep books.

### Execution Atomicity Analysis
- **Zero atomicity guarantee** on multi-leg orders
- Batch endpoint sends all orders in one POST but Polymarket processes them sequentially
- Safety checks happen ONCE before execution, not at execution time -- prices can move between check and batch POST
- NegRisk >15 outcomes requires multiple batch calls (~100ms between each)
- Partial fill unwind: cancel unfilled legs + market-sell filled legs. Each step can fail independently with exceptions silently caught.
- Failed unwind leaves orphaned positions with no alerting mechanism

### Key Conclusion
Detection is correct (the math works). Capturable profit in production would be a small fraction of what the scan shows. The main issues are: phantom markets (fixable), depth overestimation (fixable), non-atomic execution (partially mitigable), and REST latency (fixable with WebSocket).

---

## 10. Battle-Test Analysis (2026-02-07, Session 8)

### Codex + Deep Code Review

Used `codex exec -m gpt-5.3-codex` to analyze the pipeline for real-world profitability gaps. Combined with line-by-line code review of all 17 source modules.

### Critical Bug Found: FAK/FOK Order Type Mismatch

**File**: `executor/engine.py:56`
**Bug**: `OrderType.FOK if use_fak else OrderType.GTC`
**Should be**: `OrderType.FAK if use_fak else OrderType.GTC`

The py-clob-client SDK has BOTH `OrderType.FAK` and `OrderType.FOK`:
- **FAK** (Fill-and-Kill): Fill whatever's available, cancel remainder. Correct for arb.
- **FOK** (Fill-or-Kill): Fill ALL or fill NOTHING. Too strict -- rejects thin-book opportunities entirely.

Verified via SDK inspection:
```python
from py_clob_client.clob_types import OrderType
# OrderType.FAK = "FAK", OrderType.FOK = "FOK", OrderType.GTC = "GTC", OrderType.GTD = "GTD"
```

Config says "FAK (Fill-and-Kill)" in UI, but engine sends FOK. This means every order that can't fill 100% of requested size gets zero fills instead of partial fills. Silent profit killer on thin-book opportunities.

### Analysis: Why Dry-Run Opportunities Are Mostly Phantoms

| Factor | Impact | Fix Complexity |
|--------|--------|----------------|
| REST polling 25K markets = 5-30s stale | 60-90% of arbs already gone | HIGH (WebSocket) |
| FOK instead of FAK order type | Kills all partial-fill profit | TRIVIAL (1 line) |
| VWAP used as limit price (should be worst-fill) | Orders can't fill upper levels | LOW (new helper) |
| No opportunity TTL | Stale opps executed 1-5s after detection | LOW |
| No edge revalidation at execution time | Prices moved since scan | LOW |
| No inventory check for sell legs | Sell orders fail silently | MEDIUM |
| Single-thread REST fetch blocks scan | Can't scan while fetching | HIGH (async) |

### Realistic Arb Window Analysis

Based on IMDEA research + Polymarket competitive landscape:
- **Binary rebalance**: Window ~200-500ms. REST polling catches 10-20% of real opportunities.
- **NegRisk rebalance**: Window ~500ms-5s. More legs = slower competition. REST catches 20-40%.
- **Latency arb (15-min crypto)**: Window ~1-5s. Dynamic fees (3.15% at 50/50) kill naive approach. Viable only when odds away from 50/50 (fee < 1%).
- **Spike lag**: Window ~5-60s. Highest capture rate from REST (50-80%). Requires fast news detection.

### Strategy Viability Assessment (2026)

| Strategy | Viable? | Why | Monthly $$ (est, $10K bankroll) |
|----------|---------|-----|-------------------------------|
| Binary rebalance | MARGINAL | Spreads tight, 200ms windows, need WS | $200-$800 |
| NegRisk rebalance | YES | Wider windows, less competition on high-leg | $500-$2,000 |
| Latency arb (crypto) | YES WITH CAVEATS | Only when odds away from 50/50, fee < 1% | $300-$1,500 |
| Spike lag | YES | Widest windows, highest edge, needs news feed | $1,000-$5,000 |

### Missing Alpha Sources (Not Yet Implemented)

1. **Market making (passive limits)**: Instead of taking liquidity, provide it. Earn the spread. Requires inventory management and continuous quoting. Expected: $500-$2000/day on $25K bankroll.

2. **Cross-event correlation**: When one event in a category resolves, sibling events reprice. E.g., if candidate X drops out of a race, all related markets adjust. First mover captures 5-20% edge on siblings.

3. **Resolution front-running**: Markets near resolution with mispriced remaining outcomes. When outcome is "obvious" but market still shows 92/8 instead of 99/1, buy the certainty at 92c for 8% guaranteed return.

4. **Aggregated spot feed for latency arb**: Use Binance + Coinbase + Bybit + OKX aggregated VWAP instead of single Binance ticker. Reduces false signals from exchange-specific noise.

5. **News-driven spike prediction**: Monitor Twitter/X firehose + RSS feeds for breaking news. Pre-position before spike hits Polymarket. Requires NLP + low-latency news ingestion.

---

## 11. Cross-Platform + BookCache Reliability Audit (2026-02-08)

### Issue 1: Kalshi Price Conversion — CRITICAL BUG in Execution Path

**Where**: `executor/cross_platform.py:74-75`

The Kalshi side mapping is wrong. When the scanner creates a SELL leg (selling Kalshi YES), the executor maps it to:
```python
kalshi_side = "yes" if leg.side == Side.BUY else "no"  # SELL → "no"
```
This sends `side="no", action="sell"` — selling NO tokens instead of selling YES tokens. The correct mapping should always be `side="yes"` since our Kalshi OrderBook model represents YES tokens, and the `action` field determines buy/sell.

Additionally, `dollars_to_cents` conversion via `int(round(leg.price * 100))` has no validation that the result falls in Kalshi's 1-99 cent range. Prices at extremes ($0.001, $0.999) produce 0 or 100 cents, which Kalshi rejects.

The scanner correctly converts Kalshi cents→dollars in `client/kalshi.py:get_orderbook()` (divides by 100). The gap is only in the execution path going dollars→cents.

### Issue 2: Kalshi Fee Model — Correct but Missing Edge Guard

**Where**: `scanner/kalshi_fees.py:43`, `scanner/cross_platform.py:269`

The formula `ceil(7 * p * (1-p))` cents is correct per Kalshi docs. At extreme prices:
- Price $0.02: `ceil(7 * 0.02 * 0.98)` = `ceil(0.1372)` = 1 cent = $0.01 → **50% effective fee rate**
- Price $0.50: `ceil(7 * 0.50 * 0.50)` = `ceil(1.75)` = 2 cents = $0.02 → **4% effective fee rate**
- Price $0.98: `ceil(7 * 0.98 * 0.02)` = `ceil(0.1372)` = 1 cent = $0.01 → **1% effective fee rate**

The ceil rounding creates a fee floor of $0.01 per contract regardless of price. At extreme prices this is disproportionate. The cross-platform scanner doesn't guard against this — it would emit arbs where Kalshi fees consume the entire profit.

### Issue 3: Cross-Platform Execution Order — Multiple Gaps

**Where**: `executor/cross_platform.py:34-260`

Five specific gaps found:

1. **"resting" treated as filled** (line 90): Kalshi "resting" means the order is on the book, NOT filled. Treating it as filled proceeds to PM leg based on an unfilled Kalshi position.

2. **No fill polling**: Unlike PM where we have `_wait_for_fill()`, Kalshi has no fill wait. The `get_order()` endpoint exists but isn't used for polling.

3. **Unwind loss not tracked**: When PM fails and we unwind Kalshi via market sell, `TradeResult.net_pnl=0.0`. Actual loss = buy cost + sell slippage + fees. This understates real losses and breaks circuit breaker accuracy.

4. **No execution deadline**: If PM leg takes 10 seconds (network issue), the Kalshi book has moved. No abort mechanism.

5. **Market sell for unwind**: `_unwind_kalshi()` uses `type="market"` which may fill at any price. On illiquid Kalshi markets, this could mean selling at $0.01 what we bought at $0.50.

### Issue 4: Cross-Platform Fuzzy Matching — Catastrophic Risk

**Where**: `scanner/matching.py:113-138`

`token_set_ratio` at 85% threshold is dangerously permissive:
- "Will Biden win the 2024 election?" vs "Will Biden win the popular vote in 2024?" → score ~92% but completely different settlements
- "Super Bowl LVIII winner" vs "Super Bowl LIX winner" → high score, different events entirely
- "Will X resign by March 2026?" vs "Will X resign by June 2026?" → same event, different resolution dates

Settlement mismatch = 100% capital loss on both legs. This is the highest-impact risk in cross-platform arb.

Current mitigations (confidence > 90% filter in config) are insufficient because:
- The filter only blocks <90% matches. A 92% match with different settlement terms passes.
- No semantic analysis of resolution criteria, dates, or qualifying conditions.
- No human review gate — fuzzy matches auto-trade.

### Issue 5: BookCache Threading — Low Risk in Practice, Easy Fix

**Where**: `scanner/book_cache.py`

CPython's GIL makes single-dict-assignment atomic, so `apply_snapshot()` (which does `self._books[token_id] = new_book`) is safe. The frozen OrderBook dataclass means readers get either the old or new object, never a torn one.

The real risk is **stale-read ordering**: when the main thread reads `get_book(A)` then `get_book(B)`, book A may be from cycle N and book B from cycle N+5. For cross-platform arb, this means PM and Kalshi books from different moments, leading to phantom edge calculations.

`apply_delta()` is the riskiest operation: it reads the current book, computes new levels, then writes back. If the main thread reads between the read and write, it gets the old book — which is fine (consistent, just stale). No corruption possible.

Conclusion: technically safe under CPython GIL for current access pattern, but adding a `threading.Lock` + `get_books_snapshot()` is cheap insurance and enables future multi-reader patterns.
