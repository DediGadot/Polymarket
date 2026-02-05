# Findings: Polymarket Arbitrage Bot

## Research Date: 2026-02-05

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
- **Python**: `py-clob-client` (pip install py-clob-client)
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

### Fee Structure
- **Most markets**: Zero fees
- **15-min crypto markets**: Dynamic taker fee up to 3.15% at 50/50
- **Polymarket US (DCM)**: 10 bps taker fee

### Order Types
- GTC (Good-Til-Cancelled) -- resting limit order
- GTD (Good-Til-Date) -- limit with expiration
- FOK (Fill-Or-Kill) -- immediate full fill or cancel

### Key Contracts (Polygon)
| Contract | Address |
|----------|---------|
| CTF Exchange | `0x4bFb41d5B3570DeFd03C39a9A4D8dE6Bd8B8982E` |
| Neg Risk CTF Exchange | `0xC5d563A36AE78145C45a50134d48A1215220f80a` |
| Neg Risk Adapter | `0xd91E80cF2E7be2e162c6513ceD06f1dD0dA35296` |
| USDC (Polygon) | `0x2791Bca1f2de4661ED88A30C99A7a9449Aa84174` |
| Conditional Tokens | `0x4D97DCd97eC945f40cF65F87097ACe5EA0476045` |

---

## 2. Arbitrage Types Identified

### Type A: Binary Market Rebalancing (Single-Condition)
- **Principle**: In any binary market, YES_price + NO_price must = $1.00
- **Opportunity**: When sum < $1.00, buy both sides. When sum > $1.00, sell both sides.
- **Profit**: Guaranteed $1.00 payout at resolution minus cost
- **Historical**: $10.58M extracted Apr 2024-Apr 2025
- **Edge**: Typically thin (2-5 cents), requires speed and volume

### Type B: NegRisk Multi-Outcome Rebalancing
- **Principle**: In multi-outcome markets (e.g. "Who will win?"), sum of all YES prices must = $1.00
- **Mechanism**: NegRisk adapter allows converting NO share in any market to YES shares in all other markets
- **Capital efficiency**: 29x advantage over binary arbitrage
- **Historical**: $28.99M extracted (73% of all arbitrage profits despite 8.6% of opportunities)
- **Key insight**: This is the highest-value arbitrage type on Polymarket
- **Example**: 5-candidate race, if sum of all YES prices = $0.92, buy all YES for $0.92, guaranteed $1.00 payout = $0.08 profit per set

### Type C: Combinatorial/Cross-Market Arbitrage
- **Principle**: Logically related markets should have consistent pricing
- **Example**: "Will X happen?" at 60% while "Will X AND Y happen?" at 70% (impossible)
- **Detection**: Requires NLP/LLM analysis to find semantic relationships between markets
- **Research found**: 7,000+ markets with measurable combinatorial mispricings
- **Complexity**: O(2^(n+m)) naive, requires heuristic reduction

### Type D: Cross-Platform Arbitrage (Polymarket vs Kalshi)
- **Principle**: Same event priced differently on different platforms
- **Challenge**: Non-atomic execution, different settlement, oracle risk
- **Threshold**: Spreads must exceed 15 cents to be profitable after fees/risk
- **Infrastructure**: Requires accounts and capital on multiple platforms

---

## 3. Profitability Analysis

### Historical Data (Apr 2024 - Apr 2025, IMDEA Research)
- Total arbitrage profits: **$40M+**
- 86 million bets analyzed across thousands of markets
- Top arbitrageur: $2.01M across 4,049 transactions (~$496 avg per trade)
- **Execution frequency matters more than position size**

### Current Competitive Landscape (2026)
- Bots dominate, humans falling behind
- Institutional capital entering (ICE $2B investment Oct 2025)
- Spreads compressing but still exploitable
- 78% of arb opportunities in low-volume markets fail due to execution inefficiency
- Combined fees of 5%+ make spreads under 5% unprofitable on fee-bearing markets

### Most Promising Strategies (Ranked by Expected Value)
1. **NegRisk Rebalancing** -- highest capital efficiency, most volume
2. **Binary Rebalancing** -- simpler, lower edge, high frequency
3. **Event-driven spikes** -- mispricings cluster during news events
4. **Cross-platform** -- highest edge but highest complexity/risk

---

## 4. Technical Implementation Insights

### Mispricing Detection Algorithm
```
For each market/event:
  1. Fetch all outcome token prices (best bid/ask)
  2. For binary: check if YES_best_ask + NO_best_ask < 1.0 (buy arb)
  3. For binary: check if YES_best_bid + NO_best_bid > 1.0 (sell arb)
  4. For negRisk multi-outcome: check if sum(all YES_best_ask) < 1.0
  5. Apply fee deduction and gas cost estimation
  6. Filter by minimum ROI threshold (suggest 2% after all costs)
  7. Verify liquidity depth at target prices
```

### Execution Risk Mitigation
- Simultaneous order placement via async
- 5-second timeout, cancel unfilled legs
- Partial fill handling: exit both sides immediately
- Gas estimation: ~150k gas per CLOB order on Polygon
- Batch orders: up to 15 per request

### Infrastructure Requirements
- Low-latency connection to Polymarket CLOB
- WebSocket for real-time price feeds
- Polygon RPC node (for on-chain operations)
- Python + py-clob-client SDK

---

## 5. Key Open Source References

| Repository | Strategy | Stars |
|-----------|----------|-------|
| [Polymarket/agents](https://github.com/Polymarket/agents) | AI agent trading | Official |
| [runesatsdev/polymarket-arbitrage-bot](https://github.com/runesatsdev/polymarket-arbitrage-bot) | Single + NegRisk arb | Community |
| [Trust412/Polymarket-spike-bot-v1](https://github.com/Trust412/Polymarket-spike-bot-v1) | Spike detection HFT | Community |
| [warproxxx/poly-maker](https://github.com/warproxxx/poly-maker) | Market making | Community |
| [discountry/polymarket-trading-bot](https://github.com/discountry/polymarket-trading-bot) | Beginner-friendly | Community |
| [djienne/Polymarket-bot](https://github.com/djienne/Polymarket-bot) | HFT + Kelly Criterion | Community |

---

## 6. Risk Factors

- **Oracle risk**: UMA Optimistic Oracle resolution disputes
- **Smart contract risk**: CTF Exchange bugs
- **Execution risk**: Leg risk in non-atomic arb
- **Regulatory risk**: CFTC/SEC evolving stance on prediction markets
- **MEV risk**: Front-running by Polygon validators
- **Liquidity risk**: Thin books can't absorb large arb orders
- **Gas spikes**: Polygon congestion during high-activity events

---

## Sources
- [Polymarket CLOB Docs](https://docs.polymarket.com/developers/CLOB/introduction)
- [Polymarket API Reference](https://docs.polymarket.com/quickstart/reference/endpoints)
- [Polymarket NegRisk](https://docs.polymarket.com/developers/neg-risk/overview)
- [IMDEA Arbitrage Paper](https://arxiv.org/abs/2508.03474)
- [Arbitrage Bot Implementation Guide](https://navnoorbawa.substack.com/p/building-a-prediction-market-arbitrage)
- [Polymarket HFT Analysis](https://www.quantvps.com/blog/polymarket-hft-traders-use-ai-arbitrage-mispricing)
- [Institutional Strategies 2026](https://www.ainvest.com/news/capitalizing-prediction-markets-2026-institutional-grade-strategies-market-making-arbitrage-2601/)
