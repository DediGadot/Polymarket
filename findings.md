# Findings

Research notes and technical details for pipeline augmentation.

**Last updated:** 2026-02-15

---

## Session 5 Research: Framework Landscape & Augmentation Gaps

### Framework Analysis (13 codebases evaluated)

| Framework | Stars | Relevance | Borrowable Pattern |
|-----------|-------|-----------|-------------------|
| NautilusTrader | 19.2K | Highest | WS connection pooling (500/conn), reconciliation, BinaryOption instrument |
| Freqtrade | 46.8K | High (ML) | FreqAI background retraining, Bayesian hyperopt, RL strategy selection |
| VectorBT | 6.7K | High (backtest) | Numba JIT depth calcs, vectorized parameter sweeps |
| Hummingbot | 16.1K | Moderate | V2 Controller/Executor pattern, InFlightOrder state machine |
| QuantConnect Lean | 16.5K | Low | Pluggable SlippageModel/FillModel (study only) |

### Key Finding: NautilusTrader Has Production Polymarket Adapter

NautilusTrader already has:
- `PolymarketWebSocketClient` — Rust-powered WS with auto-reconnect
- `PolymarketInstrumentProvider` — Loads markets from Gamma API
- `PolymarketExecutionClient` — Order placement with signature types 0/1/2
- **WS connection pooling**: 500 instruments per connection, auto-shards
- **5-second subscription buffering** before WS connection (batching)

**Decision:** Don't adopt the framework (too opinionated, paradigm shift). Borrow the connection pooling pattern for Phase 3.

### Key Finding: $40M+ Extracted via Rebalancing (Academic)

ArXiv 2508.03474 ("Unravelling the Probabilistic Forest: Arbitrage in Prediction Markets"):
- 86 million bets analyzed on Polymarket (Apr 2024 - Apr 2025)
- $40M+ in documented arbitrage extraction
- **Hierarchical decomposition** of multi-outcome events reveals logical inconsistencies
- Directly maps to proposed correlation scanner (Phase 5)

### Key Finding: FreqAI Pattern for ML Integration

Freqtrade's FreqAI architecture:
- Background thread retrains ML model while bot trades
- Feature engineering pipeline standardizes indicators into fixed-width vectors
- Adaptive retraining every N trades or M time units
- Model cached in RAM for sub-ms inference
- Supports classifiers, regressors, and reinforcement learning

**Decision:** Extract the pattern, not the library. Build `scanner/feature_engine.py` + `scanner/ml_scorer.py` following this architecture.

---

## Gap Analysis: Current Pipeline vs. Target

### What's Missing (Ranked by Impact)

| # | Gap | Current State | Impact | Phase |
|---|-----|--------------|--------|-------|
| 1 | OFI signal | Price velocity only (spike.py) | Leading indicator, 200-500ms prediction | 1 |
| 2 | State persistence | All state in-memory, lost on restart | Production reliability | 2 |
| 3 | WS sharding | Single WS connection | Limits coverage to ~500 tokens | 3 |
| 4 | Pre-signed orders | Sign at execution time (~200ms) | Latency reduction for spike/stale arbs | 4 |
| 5 | Correlation scanner | Markets treated independently | New arb type across related events | 5 |
| 6 | Backtesting replay | Log analysis only (evs.py) | Can't answer "what if different weights?" | 6 |
| 7 | ML scoring | Hand-tuned 7-factor weights | Adaptive, self-improving scorer | 7 |

### What's Already Strong (No Changes Needed)

| Capability | Implementation | Quality |
|-----------|---------------|---------|
| 9 scanner types | binary, negrisk, latency, spike, cross_platform, maker, stale_quote, resolution, value | Production |
| Cross-platform matching | 3-tier (manual, verified, fuzzy) + contract-level | Production |
| Maker microstructure | 3-layer gate (persistence + execution model + Bayesian EV) | Production |
| Safety layer | 10+ pre-trade checks + circuit breaker | Production |
| Depth analysis | VWAP sweep + slippage ceiling + worst fill price | Production |
| Fee models | PM (taker + 2% resolution) + Kalshi (parabolic + ceil) | Production |
| Pipeline dashboard | SQLite + SSE + FastAPI | Production |

---

## OFI (Order Flow Imbalance) Technical Details

### What is OFI?
Net aggressive order volume = signed sum of trades hitting the book.
- Aggressive buy = market order lifting the ask → positive OFI
- Aggressive sell = market order hitting the bid → negative OFI

### Why It's a Leading Indicator
- Price lags order flow by 200-500ms in prediction markets
- OFI divergence between YES/NO tokens in same event predicts imminent rebalancing
- Example: OFI on YES token = +500 (aggressive buying), OFI on NO token = -100 → market about to reprice YES higher

### Implementation Approach
```python
@dataclass
class OFITracker:
    window_sec: float = 30.0
    _events: dict[str, deque[tuple[float, float]]]  # token_id -> deque[(timestamp, signed_volume)]
    _lock: threading.Lock

    def record(self, token_id: str, side: str, size: float, timestamp: float):
        signed = size if side == "BUY" else -size
        self._events[token_id].append((timestamp, signed))
        # prune old entries

    def get_ofi(self, token_id: str) -> float:
        # sum(signed_volumes) in window
        return sum(v for _, v in self._events.get(token_id, []))

    def get_divergence(self, token_a: str, token_b: str) -> float:
        return abs(self.get_ofi(token_a) - self.get_ofi(token_b))
```

### Data Source
- WSBridge `price_change` events already carry `side` and `size`
- Currently fed to BookCache.apply_delta() — add OFI tracker as parallel consumer
- No new data sources needed

---

## State Persistence Technical Details

### Trackers That Need Checkpointing

| Tracker | State Size | Staleness Tolerance | Fields |
|---------|-----------|-------------------|--------|
| ArbTracker | ~500 entries | 60s | event→cycle count, confidence |
| RealizedEVTracker | ~100 entries | 300s | opp_key→(observations, fills, orphans, pnl) |
| SpikeDetector | ~1000 entries | 30s | token→price_history, active_spikes |
| MakerPersistenceGate | ~200 entries | 60s | market→consecutive_cycle_count |
| OFITracker | ~1000 entries | 10s | token→recent_events (volatile, partial restore OK) |

### SQLite Schema
```sql
CREATE TABLE tracker_state (
    tracker_name TEXT PRIMARY KEY,
    data_json TEXT NOT NULL,
    cycle_num INTEGER,
    updated_at REAL NOT NULL
);
```

### Checkpoint Strategy
- Every 10 cycles (configurable): batch-write all trackers in single transaction
- On SIGTERM/SIGINT: write immediately before exit
- On startup: read all, log staleness per tracker
- Corrupt data: log warning, start fresh (don't crash)

---

## WS Connection Sharding Technical Details

### Problem
Polymarket WS supports max 500 instruments per connection.
Current pipeline: 14K+ binary markets × 2 tokens each = 28K+ token IDs.
Single WS connection can only cover ~250 markets.

### NautilusTrader's Approach
- Auto-detect instrument count per connection
- When adding instrument N+1 that would exceed limit, create new connection
- 5-second buffering window: collect subscriptions, batch them, then connect
- Health monitoring: per-shard heartbeat, reconnect on timeout

### Proposed Sharding Strategy
```
tokens_to_subscribe = [t1, t2, ..., t28000]
shard_size = 500
n_shards = ceil(28000 / 500) = 56 connections

# But we don't need WS for ALL tokens — only "hot" ones:
# - Markets with active arb opportunities (tracked by ArbTracker)
# - Markets in MakerPersistenceGate (being monitored for maker setups)
# - Markets with recent spike activity
# Realistic hot set: 500-2000 tokens → 1-4 WS connections
```

**Decision:** Start with 2000-token cap (4 shards). Expand only if needed.

---

## Pre-Signed Order Templates Technical Details

### Current Signing Path
```
detect_opportunity() → create_limit_order() → EIP-712 sign → post_order()
                        ^^^^^^^^^^^^^^^^^^^^
                        ~200ms with py-clob-client
```

### Proposed Path
```
[background thread]
  BookCache delta → price changed → pre-sign at ±2 tick levels → cache

[execution path]
  detect_opportunity() → presigner.get_or_sign() → post_order()
                         ^^^^^^^^^^^^^^^^^^^^
                         ~0ms cache hit, ~200ms miss
```

### Cache Invalidation Rules
1. Price moves > 2 ticks from pre-signed level → invalidate that token's entries
2. Order size doesn't match → re-sign (size is part of the signature)
3. LRU eviction when cache exceeds 200 entries
4. Full flush on config change or manual trigger

### Limitations
- Size is baked into the signature → must pre-sign at expected sizes
- Pre-sign at Kelly-sized positions for the top-N recurring markets
- Cache miss rate will be high initially, improves as patterns emerge

---

## Correlation Scanner Technical Details

### Relationship Types

| Type | Example | Constraint | Violation = Arb |
|------|---------|-----------|-----------------|
| Parent-child | "X wins presidency" → "X wins Ohio" | P(parent) >= P(child) | Buy parent, sell child |
| Complement | "A wins" + "B wins" + "C wins" in same race | sum <= 1.0 | NegRisk-style multi-leg |
| Temporal | "Event by March" → "Event by June" | P(earlier) <= P(later) | Buy later, sell earlier |
| Conditional | "Party wins" ↔ "Candidate of party wins" | P(party) >= P(candidate) | Buy party if underpriced |

### Graph Construction Algorithm
1. Fetch all events from Gamma API with categories + tags
2. Group by category (e.g., "US Politics", "Sports", "Crypto")
3. Within each category, extract entities from titles (NER or regex)
4. Connect events sharing entities + overlapping time frames
5. Classify relationship type based on title analysis
6. Check probability constraints for connected events

### False Positive Prevention
- Require price data on both sides of the constraint violation
- Minimum edge: 3% (after fees) to avoid noise
- Confidence scoring: manual-mapped relationships = 1.0, rule-inferred = 0.7, entity-matched = 0.4
- Never auto-trade below 0.9 confidence (same as cross-platform)

---

## Correlation Scanner V2 — Research Notes

### Current V1 Weaknesses (Code Audit)

**Entity extraction limitations (`scanner/correlation.py:268-290`):**
- `_ENTITY_PATTERN = re.compile(r"\b([A-Z][a-z]+(?:\s+[A-Z][a-z]+)*)\b")` — only matches CamelCase words. Misses lowercase entities, acronyms in context, and multi-format entity references.
- `_ACRONYM_PATTERN = re.compile(r"\b([A-Z]{2,})\b")` — captures raw acronyms but doesn't understand they might refer to the same entity as a spelled-out name.
- Entity overlap threshold of 2 for complements is arbitrary. Two events can share 2 entities ("Super Bowl" + "MVP") and still be completely unrelated (e.g., different years, different sports).

**Constraint validation gap (`scanner/correlation.py:440-529`):**
- `_check_violation()` assumes the relationship type is correct and applies the constraint directly. No verification that the two events actually ask the same underlying question.
- Temporal example: "GDP growth by Q1 2026" and "GDP contraction by Q2 2026" would match as temporal (shared entity "GDP", different temporal markers). But the constraint P(growth by Q1) ≤ P(growth by Q2) is WRONG here — one is about growth, the other about contraction.

**No liquidity floor:**
- `_violation_to_opportunity()` checks `net_profit <= 0` and ROI cap, but never checks minimum `required_capital`. A $0.50 opportunity with 5% edge = $0.025 profit passes all filters.

**Deduplication absence:**
- `scan()` returns all violations without grouping. If Bitcoin-related events form a cluster of 5 events, that's potentially C(5,2) = 10 pairings × 3 relation types = 30 violations from one cluster.

### Similarity Approach Analysis

| Method | Deps | Quality | Speed (1K events) | Notes |
|--------|------|---------|-------------------|-------|
| Regex entity overlap (current) | None | Low | ~5ms | Misses semantic similarity, false positives on shared words |
| TF-IDF + cosine (proposed default) | scikit-learn (installed) | Medium-High | ~30ms | Good for structured titles with word overlap. N-grams capture phrases. |
| rapidfuzz token_set_ratio | rapidfuzz (installed) | Medium | ~200ms (N²) | Already used in matching.py. Good for fuzzy text, not semantic. |
| Sentence-transformers embedding | torch (~2GB) | Highest | ~500ms + model load | Best semantic understanding. "Trump wins" ↔ "Republican candidate elected". |
| fastembed (ONNX) | onnxruntime (~50MB) | High | ~200ms + model load | Lighter than torch. Good middle ground. |
| OpenAI/Claude embeddings API | Network call | Highest | ~2s (API latency) | Best quality but adds API dependency + cost + latency. |

**Decision:** TF-IDF default, sentence-transformers optional. TF-IDF with (1,2)-grams and cosine similarity covers 90%+ of Polymarket title matching without new dependencies. scikit-learn's sparse matrix operations are fast even at N²/2 comparisons.

### Stem Comparison for Constraint Validation

For temporal pairs, we need to verify that two events ask the same question with different deadlines. Approach:

1. Strip temporal markers from titles using existing `extract_temporal()` + month/year removal
2. Strip common prepositions ("by", "before", "after", "in")
3. Normalize remaining text (lowercase, strip whitespace)
4. Compare stems using TF-IDF cosine similarity or simple token overlap

Example:
```
"Bitcoin to $100K by March 2026" → strip → "bitcoin to $100k"
"Bitcoin to $100K by June 2026"  → strip → "bitcoin to $100k"
Stem similarity: 1.0 → VALID temporal constraint

"Bitcoin to $100K by 2026"         → strip → "bitcoin to $100k"
"Bitcoin mining banned by 2027"    → strip → "bitcoin mining banned"
Stem similarity: ~0.3 → INVALID, reject
```

For parent-child pairs, verify the parent is a generalization:
```
"Will Trump win?" → tokens: {trump, win}
"Will Trump win Ohio?" → tokens: {trump, win, ohio}
{trump, win} ⊂ {trump, win, ohio} → parent tokens are subset of child → VALID

"Will Trump win?" → tokens: {trump, win}
"Will Trump resign?" → tokens: {trump, resign}
{trump, win} ⊄ {trump, resign} → "win" not in child → INVALID
```

### Deduplication Strategy

Per-event capping is more effective than per-pair dedup alone. Consider:
- Event A paired with B, C, D, E, F → 5 opportunities
- All 5 are effectively "Event A is mispriced relative to its cluster"
- Trading all 5 is redundant — you'd buy/sell A five times

Cap at 3 pairings per event, keep the 3 with highest net_profit. This:
1. Reduces opp count proportional to cluster size
2. Preserves the strongest signals
3. Prevents the per-cycle cap from being entirely consumed by one cluster

## Cross-References

| Finding | Plan Item |
|---------|-----------|
| NautilusTrader WS pooling | 3.1 (WSPool) |
| FreqAI ML pattern | 7.1-7.2 (feature engine + ML scorer) |
| FreqAI RL pattern | 7.3 (RL strategy selector) |
| VectorBT Numba JIT | 6.2 (replay engine depth calcs) |
| $40M rebalancing paper | 5.1 (correlation scanner) |
| BookService shared fetch | 0.1 (prerequisite) |
| Tracker serialization | 0.2 → 2.1-2.2 (state persistence) |
| OFI as leading indicator | 1.1-1.3 (scorer integration) |
| Pre-signed templates | 4.1-4.2 (latency reduction) |
