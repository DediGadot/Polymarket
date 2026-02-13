# Findings

Research notes and technical details from the 5-agent deep inspection.

**Last updated:** 2026-02-13

---

## Agent Reports Summary

### Security Audit
- **2 CRITICAL**: Path traversal in RSA key loading (`kalshi_auth.py:28`), no orderbook data validation at any ingestion point
- **5 HIGH**: Float arithmetic for financial calcs, non-atomic cross-platform execution, config mutability, BookCache thread safety gaps, no TLS cert pinning
- **7 MEDIUM**: Private key in env var, debug logs expose architecture, ticker injection risk, unbounded SpikeDetector memory, NDJSON ledger no integrity protection, gas oracle manipulation via RPC, no rate limiting on outbound calls

### Architecture & Performance
- **Critical bottleneck**: Gamma API pagination = 30s+ per cycle. Event counts = 10-20s additional. Combined = 40-50s cycle time vs 1s configured interval.
- **asyncio.Queue not thread-safe** for cross-thread WS bridge usage
- **All 5 scanners run sequentially** on disjoint data (trivially parallelizable)
- **3 serial book fetch paths**: latency scanner, Kalshi orderbooks, safety re-fetches
- **No unwind retry** on cross-platform execution failure
- **ArbTracker** infrastructure complete but unused (W_PERSISTENCE=0.15 is dead weight)

### Python Code Quality
- **5 CRITICAL**: Config mutation, `ext_client` typed as `object`, `_dollars_to_cents` applied to all platforms, `TradeResult` not frozen, fanatics_auth hmac concern
- **7 HIGH**: run.py 954 lines, gas cost duplication in 4 scanners, broad exception handling in retry, BookCache lock gaps, ScanTracker unbounded memory, sell-side edge verification assumes same-side legs, exposure tracking monotonic

### Trading Risk & Edge Analysis
- **C1**: NegRisk completeness bypass (event_market_counts=0) — up to $500 per occurrence
- **C2**: Safety check top-of-book vs VWAP — $0.50-5.00/trade
- **H1**: Kelly sizing too conservative — $4K/day missed revenue estimate
- **H2**: Non-atomic multi-batch NegRisk — $150/1000 trades
- **H3**: Cross-platform unwind loss underestimated at $0.02/contract
- **What the code gets right**: Orderbook sort enforcement, NegRisk grouping, fuzzy match blocking, resolution fee handling, Kalshi cent conversion, FAK default, VWAP in scanners, graceful shutdown

### Test Coverage Gaps
- **696 tests, 91% coverage** — strong numerically but critical paths untested
- **P0 untested**: `_filled_size_from_response()`, cascading cross-platform failure, `_dollars_to_cents` out-of-range, WSManager message handling
- **P1 untested**: NegRisk partial fill + unwind, `verify_edge_intact` latency branch, BookCache concurrent access, empty legs through safety
- **run.py at 27% coverage**, **ws.py at 39% coverage**

---

## Key Technical Decisions

| Decision | Rationale |
|----------|-----------|
| Validation in `scanner/validation.py` | Domain-specific rules (price 0-1), not transport-layer |
| Generic `TTLCache[T]` in `client/cache.py` | Reusable across 3+ call sites; `lru_cache` lacks TTL |
| State machine in `executor/fill_state.py` | Isolates complexity; cross_platform.py delegates to it |
| Kelly odds configurable in config.py | Manual tuning safer than computed (Kelly sensitive to probability input) |
| run.py breakup deferred to Phase 5 | Functional changes first, structural refactor once stable |

---

## Cross-References: Finding to Plan Item

| Finding | Source Agent(s) | Plan Item |
|---------|----------------|-----------|
| Orderbook price validation | Security, Trading Risk | 1.1 |
| NegRisk completeness bypass | Trading Risk | 1.2 |
| asyncio.Queue thread safety | Architecture, Security | 1.3 |
| BookCache lock gaps | Architecture, Security, Code Quality | 1.4 |
| Exposure monotonic increase | Trading Risk, Code Quality | 1.5 |
| Config mutation | Security, Code Quality | 1.6 |
| Cross-platform state machine | Architecture, Trading Risk | 2.1 |
| Runtime risk controls unwired | IDEAS #1 | 2.2 |
| Tick-size precision | IDEAS #3 | 2.3 |
| PlatformClient typing | Code Quality | 2.4 |
| TradeResult mutable | Code Quality | 2.5 |
| API pagination bottleneck | Architecture, IDEAS #2 | 3.1 |
| Serial scanners/fetches | Architecture | 3.2 |
| Kelly sizing too conservative | Trading Risk | 3.3 |
| ArbTracker unused | Architecture, IDEAS #8 | 3.4 |
| DCM fee realism | IDEAS #4 | 4.1 |
| VWAP safety check | Trading Risk | 4.2 |
| Contract-level matching | IDEAS #6 | 4.3 |
| Float epsilon | Security, Code Quality | 4.4 |
| Strategy win-rate-only gate | Trading Risk | 4.5 |
| WS health/failover | IDEAS #7 | 5.1 |
| run.py 954 lines + gas duplication | Code Quality | 5.2 |
| CI gates + test gaps | Testing, IDEAS #9 | 5.3 |
| Memory leaks | Architecture, Code Quality | 5.4 |
| Docs drift | IDEAS #10 | 5.5 |
