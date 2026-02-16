# Task Plan: Pipeline Intelligence & Infrastructure Layer

**Created:** 2026-02-15
**Builds on:** Session 4 (17 profit-max items, 986 tests, 137 opps/cycle)
**Goal:** Add 7 infrastructure/intelligence augmentations that transform the pipeline from reactive scanner into an adaptive learning system with production resilience. Each augmentation is a standalone module that plugs into the existing architecture.

## Current Phase
ALL PHASES COMPLETE (Session 6)

## Architecture Overview

```
Current pipeline (Sessions 1-4):
  fetch -> pre-filter -> scan (9 scanners) -> score -> size -> safety -> execute -> P&L

Target pipeline (this plan):
  fetch -> pre-filter -> scan (10 scanners + OFI + correlation) -> score (ML-enhanced) -> size -> safety -> execute (presigned) -> P&L
       ^                    ^                                        ^                                     ^
       |                    |                                        |                                     |
  [WS sharding]      [state restore]                        [learned weights]                    [order presigner]
       |                    |                                        |
  [tick recorder]    [checkpoint every N cycles]             [background retrain thread]
```

## Phases

### Phase 0: Architectural Prerequisites
**Theme:** Refactors that unblock multiple downstream phases. No new features — structural plumbing only.

#### 0.1 — Extract BookFetcher into shared service pattern
- **Files:** `scanner/book_service.py` (new), `run.py`
- **Problem:** Binary, maker, resolution, and stale-quote scanners each call `get_orderbooks_parallel()` independently, hammering the CLOB rate limiter. Phase 1 (OFI) and Phase 2 (state persistence) both need a centralized book data flow.
- **Fix:** Create `BookService` that owns the single book fetch per cycle, stores in `BookCache`, and exposes `BookFetcher` callable. All scanners consume from service, not from raw CLOB calls.
- **Key constraint:** Must remain backward-compatible with existing `BookFetcher = Callable[[list[str]], dict[str, OrderBook]]` type alias.
- **Tests:** Integration test: BookService fetches once, 3 scanners read from cache. Verify CLOB called exactly once per token set.
- [x] Complete (Session 6)

#### 0.2 — Add serialization protocol to tracker classes
- **Files:** `scanner/confidence.py`, `scanner/realized_ev.py`, `scanner/spike.py`, `scanner/maker.py` (MakerPersistenceGate)
- **Problem:** `ArbTracker`, `RealizedEVTracker`, `SpikeDetector`, `MakerPersistenceGate` all hold learned state in memory. Phase 2 (state persistence) needs a uniform way to serialize/deserialize them.
- **Fix:** Add `Serializable` Protocol with `to_dict() -> dict` and `from_dict(cls, data: dict) -> Self` to each tracker. Use frozen dataclass-safe patterns (no pickle).
- **Tests:** Round-trip test per tracker: populate with data, serialize, deserialize, verify identical behavior.
- [x] Complete (Session 6)

- **Status:** complete

**Phase 0 exit criteria:** BookService reduces CLOB calls by 50%+. All 4 trackers can round-trip through JSON serialization.

---

### Phase 1: Order Flow Imbalance (OFI) Tracker
**Theme:** Add a leading indicator that predicts price moves 200-500ms before they happen.

#### 1.1 — Implement scanner/ofi.py
- **Files:** `scanner/ofi.py` (new)
- **Approach:**
  - `OFITracker` class accumulates aggressive buy/sell volume from WS delta events
  - Per-token rolling window (configurable, default 30s)
  - `OFI_t = sum(aggressive_buy_volume) - sum(aggressive_sell_volume)`
  - Normalized OFI: `ofi_t / total_volume_t` (bounded -1 to +1)
  - Expose `get_ofi(token_id) -> float` and `get_divergence(token_a, token_b) -> float`
  - Divergence = OFI difference between correlated tokens (same event YES vs NO)
- **Design:** Frozen `OFISnapshot` dataclass per read. Mutable accumulator internal only. Thread-safe via Lock (same pattern as BookCache).
- **Tests:** Feed synthetic WS deltas, verify OFI accumulates correctly. Test window pruning. Test normalized bounds. Test divergence between two tokens.
- [x] Complete (Session 6)

#### 1.2 — Wire OFI into WSBridge drain loop
- **Files:** `client/ws_bridge.py`
- **Problem:** WSBridge currently feeds BookCache (snapshots) and SpikeDetector (price updates). OFI needs the same delta stream with side + size info.
- **Fix:** Add `OFITracker` as third consumer in `WSBridge.__init__()`. In `drain()`, feed price_change events to OFI tracker alongside BookCache deltas.
- **Tests:** Mock WS bridge with OFI tracker. Verify deltas flow to all 3 consumers.
- [x] Complete (Session 6)

#### 1.3 — Add OFI to ScoringContext and scorer
- **Files:** `scanner/scorer.py`, `run.py`
- **Problem:** Scorer has 7 factors but no order flow signal. OFI divergence between YES/NO tokens predicts imminent price correction — high-OFI-divergence arbs are more likely to fill before the book adjusts.
- **Fix:** Add `ofi_divergence: float = 0.0` to `ScoringContext`. Add 8th scoring factor `_score_ofi()` (W=0.10, steal from competition W=0.05→0.00 since OFI subsumes it). Score: high absolute divergence = favorable (market about to correct toward our arb).
- **Tests:** Test scorer with OFI factor. Verify high-divergence opps rank higher. Verify backward compat (default 0.0 = neutral score).
- [x] Complete (Session 6)

- **Status:** complete

**Phase 1 exit criteria:** OFI tracker accumulates from WS deltas. Scorer uses 8 factors. Dry-run with WS shows OFI-aware ranking.

---

### Phase 2: State Persistence & Hot Restart
**Theme:** Survive restarts without losing learned state. Production reliability.

#### 2.1 — Implement state/checkpoint.py
- **Files:** `state/checkpoint.py` (new), `state/__init__.py` (new)
- **Approach:**
  - `CheckpointManager` writes/reads tracker state to SQLite (`state.db`)
  - Tables: `tracker_state(tracker_name TEXT PK, data_json TEXT, updated_at REAL)`
  - `save(name, tracker)` → calls `tracker.to_dict()`, upserts JSON
  - `load(name, tracker_cls)` → reads JSON, calls `tracker_cls.from_dict()`
  - Auto-save every N cycles (configurable, default 10) + on graceful shutdown (SIGTERM/SIGINT)
  - Atomic writes (SQLite transaction per save)
- **Tests:** Write state, kill process, restart, verify state restored. Test corrupt JSON handling (graceful fallback to fresh state). Test concurrent save/load safety.
- [x] Complete (Session 6)

#### 2.2 — Integrate checkpoint into run.py
- **Files:** `run.py`, `config.py`
- **Approach:**
  - Add `state_checkpoint_enabled: bool = True` and `state_checkpoint_interval: int = 10` to Config
  - On startup: load checkpoint for ArbTracker, RealizedEVTracker, SpikeDetector, MakerPersistenceGate, OFITracker
  - Every N cycles: save all tracker states
  - On SIGTERM/SIGINT: save before exit (already have signal handler)
  - Log age of restored state: "Restored ArbTracker from 47s ago (23 events tracked)"
- **Tests:** Integration test: run 5 cycles, checkpoint, create new pipeline instance, verify trackers have prior state. Test with checkpoint disabled.
- [x] Complete (Session 6)

- **Status:** complete

**Phase 2 exit criteria:** Pipeline survives restart with <5s state loss. All 5 trackers checkpoint/restore correctly.

---

### Phase 3: WebSocket Connection Sharding
**Theme:** Scale WS coverage from hundreds to thousands of instruments without hitting Polymarket's 500-instrument limit.

#### 3.1 — Implement client/ws_pool.py
- **Files:** `client/ws_pool.py` (new)
- **Approach:**
  - `WSPool` manages N `WSManager` instances, each subscribing to <= 500 tokens
  - `subscribe(token_ids: list[str])` → shards across connections, creates new connections as needed
  - `unsubscribe(token_ids: list[str])` → removes from appropriate shard, closes empty connections
  - Health monitoring: track last message time per shard, reconnect stale shards
  - Token→shard mapping stored in dict for O(1) routing
  - Merge all shard queues into single output queue consumed by WSBridge
- **Design:** Extend existing `WSManager` — pool is a higher-level orchestrator, not a replacement.
- **Tests:** Test sharding with 1200 tokens (3 shards). Test rebalancing when tokens added/removed. Test shard failure + reconnect. Test message merging from multiple shards.
- [x] Complete (Session 6)

#### 3.2 — Integrate WSPool into WSBridge
- **Files:** `client/ws_bridge.py`
- **Problem:** Current WSBridge creates a single WSManager. Need to swap for WSPool when token count > 500.
- **Fix:** WSBridge constructor accepts `max_tokens_per_conn: int = 500`. If `len(token_ids) > max_tokens_per_conn`, use WSPool instead of single WSManager. Transparent to downstream consumers (BookCache, SpikeDetector, OFITracker).
- **Tests:** Test bridge with 100 tokens (single conn). Test with 1500 tokens (3 shards). Verify drain() merges updates from all shards.
- [x] Complete (Session 6)

- **Status:** complete

**Phase 3 exit criteria:** WS covers 2000+ tokens across multiple sharded connections. No dropped updates during shard failover.

---

### Phase 4: Pre-Signed Order Templates
**Theme:** Eliminate signing latency from the critical execution path.

#### 4.1 — Implement executor/presigner.py
- **Files:** `executor/presigner.py` (new)
- **Approach:**
  - `OrderPresigner` maintains a cache of pre-signed orders for hot markets
  - Runs on background thread, signs orders at common price levels (best_ask ± 2 ticks)
  - Cache key: `(token_id, side, price, tick_size, neg_risk)` → signed order
  - Invalidation: when BookCache delta moves price > 2 ticks, invalidate stale entries
  - `get_or_sign(token_id, side, price, size, neg_risk) -> SignedOrder` — cache hit = 0ms, miss = fallback to live signing
  - Max cache size: configurable (default 200 entries = ~100 markets × 2 sides)
  - LRU eviction when cache full
- **Design:** Read from BookCache to know which price levels to pre-sign. Listen for BookCache deltas to invalidate.
- **Tests:** Test cache hit path. Test invalidation on price move. Test LRU eviction. Test fallback to live signing on miss. Benchmark: measure signing latency reduction.
- [x] Complete (Session 6)

#### 4.2 — Integrate presigner into execution engine
- **Files:** `executor/engine.py`, `run.py`, `config.py`
- **Fix:** Add `presigner_enabled: bool = True` to Config. In `execute_opportunity()`, call `presigner.get_or_sign()` instead of `create_limit_order()` for the PM leg. Presigner runs its background thread alongside WSBridge.
- **Tests:** Integration test: execute with presigner enabled, verify signed order used from cache. Test with presigner disabled (backward compat). Test cache miss fallback.
- [x] Complete (Session 6)

- **Status:** complete

**Phase 4 exit criteria:** PM leg signing latency < 5ms (cache hit) vs ~200ms+ (live signing). Cache hit rate > 70% for recurring arb markets.

---

### Phase 5: Correlation / Relationship Scanner
**Theme:** Detect logical inconsistencies across related markets that individual scanners miss.

#### 5.1 — Implement scanner/correlation.py (rule-based)
- **Files:** `scanner/correlation.py` (new)
- **Approach:** Phase 1 — rule-based correlations from Gamma API metadata:
  - **Parent-child**: P(X wins presidency) >= P(X wins state Y) for all Y
  - **Complement**: P(A wins) + P(B wins) + P(C wins) <= 1.0 across related markets
  - **Temporal**: P(event by March) <= P(event by June) <= P(event by December)
  - Build relationship graph: `{event_id: [(related_event_id, relationship_type), ...]}`
  - Graph construction: category matching, tag overlap, shared entity extraction from titles
  - Scan: for each relationship, check if implied probabilities violate the constraint
  - Emit `OpportunityType.CORRELATION_ARB` with legs spanning multiple events
- **Key data:** Use `GammaClient` event categories + market questions for graph construction. No LLM needed for phase 1.
- **Tests:** Create 3 related markets violating parent-child constraint. Verify arb detected. Test complement constraint. Test temporal constraint. Test no false positives on unrelated markets.
- [x] Complete (Session 6)

#### 5.2 — Integrate correlation scanner into run.py
- **Files:** `run.py`, `scanner/models.py` (add CORRELATION_ARB to OpportunityType), `executor/engine.py`
- **Approach:** Correlation scanner runs after individual scanners, using their market/book data. Separate execution path (multi-event legs). Add to scorer with urgency=0.50 (steady-state, not time-sensitive).
- **Tests:** Integration test: pipeline finds correlation arb across 2 related events. Verify scoring and ranking.
- [x] Complete (Session 6)

- **Status:** complete

**Phase 5 exit criteria:** Relationship graph built from Gamma metadata. Parent-child and complement violations detected. At least 1 real-world example found in dry-run.

---

### Phase 6: Vectorized Replay Backtester
**Theme:** Answer "what if?" questions about scorer weights, thresholds, and strategy parameters.

#### 6.1 — Implement benchmark/recorder.py
- **Files:** `benchmark/recorder.py` (new)
- **Approach:**
  - `CycleRecorder` captures full pipeline state per cycle as NDJSON:
    - All `OrderBook` snapshots (compressed: token_id + top 5 levels only)
    - All `Opportunity` objects emitted by scanners
    - `ScoringContext` for each opportunity
    - `MarketState` (strategy mode inputs)
    - Config snapshot (first cycle only)
  - Appends to `recordings/{timestamp}.jsonl`
  - Storage budget: ~1MB/cycle × 60 cycles/min × 60 min = ~3.6GB/hour → configurable max file size with rotation
  - Zero-overhead when disabled (NullRecorder pattern, same as report/)
- **Config:** `recording_enabled: bool = False`, `recording_max_mb: int = 500`
- **Tests:** Record 3 cycles, verify NDJSON parseable. Test NullRecorder has zero overhead. Test file rotation at max size.
- [x] Complete (Session 6)

#### 6.2 — Implement benchmark/replay.py
- **Files:** `benchmark/replay.py` (new)
- **Approach:**
  - `ReplayEngine` reads recorded NDJSON, replays through `rank_opportunities()` with configurable weights
  - Supports weight sweeps: provide list of weight vectors, get P&L for each
  - Uses Numba-accelerated `depth.py` calculations where possible
  - Output: JSON report with per-weight-vector metrics (total P&L, Sharpe, win rate, capital efficiency)
  - CLI: `python -m benchmark.replay --input recordings/2026-02-15.jsonl --sweep scorer_weights`
- **Weight sweep:** Vary W_PROFIT, W_FILL, W_EFFICIENCY, W_URGENCY, W_PERSISTENCE, W_OFI in 0.05 increments under constraint sum=1.0. Use `itertools` or random sampling for high-dimensional space.
- **Tests:** Record → replay with same weights → verify identical ranking. Replay with modified weights → verify different ranking. Test Numba path vs pure Python path produce same results.
- [x] Complete (Session 6)

- **Status:** complete

**Phase 6 exit criteria:** Can record a session and replay it with different parameters. Weight sweep finds demonstrably better weights than hand-tuned defaults.

---

### Phase 7: ML-Enhanced Scoring & Strategy Selection
**Theme:** Replace hand-tuned heuristics with learned models. Biggest long-term alpha generator.

#### 7.1 — Implement scanner/feature_engine.py
- **Files:** `scanner/feature_engine.py` (new)
- **Approach:** Standardize all opportunity features into a fixed-width vector for ML:
  - **Opportunity features:** net_profit, roi_pct, required_capital, n_legs, opp_type (one-hot)
  - **Market features:** volume, trade_count, time_to_resolution, spread_width
  - **Book features:** depth_ratio, bid_ask_imbalance, VWAP vs best_price
  - **Temporal features:** hour_of_day, day_of_week, ofi_divergence, spike_velocity
  - **Confidence features:** arb_confidence, realized_ev_score, maker_persistence_cycles
  - Output: `np.ndarray` of shape `(n_features,)` per opportunity
  - Feature normalization: z-score based on rolling statistics (updated per cycle)
- **Tests:** Test feature extraction from mock opportunity. Test one-hot encoding. Test normalization. Test shape consistency.
- [x] Complete (Session 6)

#### 7.2 — Implement scanner/ml_scorer.py
- **Files:** `scanner/ml_scorer.py` (new)
- **Approach:**
  - `MLScorer` wraps a scikit-learn classifier/regressor
  - Training: features → label (1 = trade was profitable, 0 = trade was not profitable or was rejected)
  - Inference: `predict_proba(features) -> float` (probability of profitable execution)
  - Background retraining: every M cycles or N trades (configurable), retrain on accumulated data
  - Fallback: if insufficient training data (< 100 labeled examples), fall back to hand-tuned scorer
  - Model persistence: save/load via joblib alongside state checkpoint
  - Integration: ML score becomes a factor in `rank_opportunities()` (replace or augment existing factors)
- **Design:** Start with `GradientBoostingClassifier` (fast, no GPU needed, handles mixed features well).
- **Tests:** Test with synthetic labeled data. Test fallback when insufficient data. Test background retraining thread doesn't block main loop. Test model persistence.
- [x] Complete (Session 6)

#### 7.3 — RL-based StrategySelector (experimental)
- **Files:** `scanner/rl_strategy.py` (new)
- **Approach:**
  - Replace `StrategySelector` 4-mode heuristic with tabular Q-learning agent
  - State: discretized `MarketState` (gas_level × spike_active × momentum × win_rate_bucket)
  - Actions: AGGRESSIVE, CONSERVATIVE, SPIKE_HUNT, LATENCY_FOCUS
  - Reward: cycle P&L (positive = good action, negative = bad action)
  - Epsilon-greedy exploration (start high, decay over cycles)
  - Q-table persisted via state checkpoint system
  - **Guard:** if RL recommends action that would violate safety checks, override to CONSERVATIVE
- **Scope:** Experimental. Run alongside heuristic selector in shadow mode first, compare actions.
- **Tests:** Test Q-table updates. Test epsilon decay. Test safety override. Test shadow mode comparison.
- [x] Complete (Session 6)

- **Status:** complete

**Phase 7 exit criteria:** Feature engine produces consistent vectors. ML scorer improves ranking quality (A/B test vs hand-tuned). RL selector runs in shadow mode alongside heuristic.

---

## Dependencies

```
Phase 0 (prerequisites — unblocks everything)
  |
  +-- Phase 1 (OFI) depends on 0.1 (BookService for centralized data flow)
  |
  +-- Phase 2 (State) depends on 0.2 (serialization protocol)
  |     |
  |     +-- Phase 7 (ML) depends on 2.1 (checkpoint for model persistence)
  |
  +-- Phase 3 (WS Sharding) — independent, can run in parallel with 1/2
  |
  +-- Phase 4 (Presigning) — independent, can run in parallel with 1/2/3
  |
  +-- Phase 5 (Correlation) depends on 0.1 (BookService for multi-event data)
  |
  +-- Phase 6 (Replay) depends on 0.1 (BookService for recording)
  |     |
  |     +-- Phase 7.2 (ML Scorer) can use replay data for offline training
```

**Parallelization:** Phases 1, 3, 4 can run simultaneously. Phases 5, 6 can start once Phase 0 is done. Phase 7 should wait for Phase 2 (state persistence) and benefits from Phase 6 (replay data for training).

## Effort Estimate

| Phase | Items | Est. Lines | New Files | Description |
|-------|-------|-----------|-----------|-------------|
| 0     | 2     | ~300      | 1         | BookService + serialization protocol |
| 1     | 3     | ~350      | 1         | OFI tracker + scorer integration |
| 2     | 2     | ~300      | 2         | State checkpoint + integration |
| 3     | 2     | ~400      | 1         | WS connection pool + bridge integration |
| 4     | 2     | ~350      | 1         | Order presigner + engine integration |
| 5     | 2     | ~500      | 1         | Correlation scanner + integration |
| 6     | 2     | ~450      | 2         | Cycle recorder + replay engine |
| 7     | 3     | ~600      | 3         | Feature engine + ML scorer + RL strategy |
| **Total** | **18** | **~3,250** | **12 new files** | Full intelligence layer |

## Key Questions

1. Should ML scorer replace or augment hand-tuned scorer? → **Augment first** — ML score as additional factor (W=0.15), reduce other weights proportionally. Replace only after >1000 labeled examples prove superior.
2. Should RL strategy selector be live or shadow-only? → **Shadow first** — log what RL would choose vs what heuristic chose. Compare P&L over 1000+ cycles before switching.
3. Should correlation scanner trade multi-event arbs or just flag them? → **Flag first** — multi-event execution is complex (need fills on separate events). Start with alerting, add execution in Phase 5b.
4. How much recording storage is acceptable? → **500MB default** — ~2 hours of full recording. Rotate older files. User can increase via config.
5. Should presigner pre-sign for all tracked markets or only hot ones? → **Hot only** — markets that appeared in last 3 scan cycles. Avoids wasting signing on inactive markets.

## Decisions Made

| Decision | Rationale |
|----------|-----------|
| BookService as shared data layer | Eliminates redundant CLOB calls across 9+ scanners. Single fetch, multiple consumers. |
| Serialization via JSON, not pickle | Pickle is insecure and version-fragile. JSON is human-readable, debuggable, safe. |
| OFI steals weight from competition score | Competition (W=0.05) is weak signal. OFI (W=0.10) is strictly better — measures actual order flow, not proxy trade count. |
| Numba JIT for depth calculations | Hot path in every scan. Numba gives 10-50x speedup on array operations without rewriting in C. |
| GradientBoosting over neural nets | Small dataset (<10K examples initially). GBT handles mixed features, needs no GPU, trains in <1s. |
| Tabular Q-learning over deep RL | 4 actions × ~100 discrete states = tiny Q-table. Deep RL is overkill and unstable. |
| Shadow mode before live ML | ML decisions are hard to debug. Shadow mode lets us compare without risking capital. |

## Errors Encountered

| Error | Attempt | Resolution |
|-------|---------|------------|
| (none yet) | - | - |

## Notes
- Previous session baseline: 986 tests, 137 opps/cycle, 9 scanners, 4 strategy modes
- All new modules follow existing patterns: frozen dataclasses, Protocol-based interfaces, thread-safe via Lock
- New files go in existing directories (scanner/, executor/, benchmark/, state/) — no new top-level packages
- Each phase is independently valuable — pipeline improves incrementally, not all-or-nothing
- Delta hedging (perp exchanges) and Kalshi FIX protocol are deferred — bottleneck is PM on-chain latency, not Kalshi REST

---

# Task Plan: Actionable Arbitrage Execution (Session 7+)

**Created:** 2026-02-16  
**Source plan:** `docs/ACTIONABLE_ARB_IMPLEMENTATION_PLAN.md`  
**Execution model:** Sequential delivery with measurable KPI gates.

## Current Sequential Phase
All phases complete

## Phase 0 — Actionable vs Research Lane Split
- [x] Add runtime config flags for lane controls in `config.py`
  - `research_lane_enabled`
  - `correlation_execute_enabled`
- [x] Add research/executable lane classification logic in `run.py`
- [x] Route scan-only tracking through lane-aware `ScanTracker.record_cycle(...)`
- [x] Add lane metrics to scan summaries (`executable_opp_*`, `research_opp_*`)
- [x] Add lane line to cycle display output in `monitor/display.py`
- [x] Add lane fields to `status.md` writer (`monitor/status.py`)
- [x] Add/adjust unit tests for lane split behavior (`tests/test_scan_tracker.py`, `tests/test_run.py`, `tests/test_display.py`)
- [x] Run full test suite and capture baseline lane metrics from dry-run

**Exit criteria:** scan/reporting clearly separates actionable vs research output; dry-run summary includes lane totals.

## Phase 1 — Correlation Precision Hardening
- [x] Replace first-market implied probability with robust event-level aggregation
- [x] Add strict liquidity/depth filters + reason codes
- [x] Keep correlation non-executable by default; only surface in research lane unless explicitly enabled
- [x] Add regression tests for false-positive suppression

**Exit criteria:** correlation opp count drops materially, precision improves, no executable queue pollution by default.

## Phase 2 — Presigner Hot-Path Integration
- [x] Instantiate and manage `OrderPresigner` lifecycle in `run.py`
- [x] Pass `presigner` into `execute_opportunity(...)` from run call sites
- [x] Add prewarm policy and metrics logging (hit/miss/hit-rate)
- [x] Validate latency improvement with benchmark/dry-run logs

**Exit criteria:** presigner is active in runtime and measurable in logs.

## Phase 3 — Cross-Platform Safety Semantics
- [x] Scope PM inventory checks to PM SELL legs only
- [x] Add explicit venue-aware preflight checks and reason codes
- [x] Add fill-gap guardrail metrics for legging risk
- [x] Add/adjust cross-platform execution tests

**Exit criteria:** reduced false rejections + lower orphan/unwind rate.

## Phase 4 — Recorder/Replay Schema Unification
- [x] Define schema version and align recorder/replay structures
- [x] Add compatibility path or explicit version failure mode
- [x] Wire recorder into runtime via config gates
- [x] Add parse/replay validation tests

**Exit criteria:** replay parse success is 100% for new recordings.

## Phase 5 — BookService Runtime Integration
- [x] Replace ad-hoc scanner fetch patterns with centralized `BookService` orchestration
- [x] Ensure WS event-driven scan mode still behaves correctly
- [x] Add integration coverage for fetch dedup behavior

**Exit criteria:** lower read churn and stable cycle latency under load.

## Phase 6 — OFI Signal Upgrade
- [x] Move OFI from snapshot imbalance proxy toward aggressor-flow approximation
- [x] Extend WS consumption path for richer OFI inputs
- [x] Retune OFI scoring contribution after quality check
- [x] Validate predictive lift

**Exit criteria:** OFI signal shows measurable incremental predictive value.

## Phase 7 — ML Scorer Gated Runtime Integration
- [x] Wire `MLScorer` into ranking as an augmenting factor (feature-flagged)
- [x] Add sample labeling pipeline from realized execution outcomes
- [x] Add model persistence/reload + retrain cadence
- [x] Run shadow/A-B evaluation

**Exit criteria:** statistically positive lift without harming fallback stability.

## Phase 8 — Large-Event NegRisk Actionability
- [x] Add bounded subset/basket builder for >15 outcome events
- [x] Add safety semantics for partial-event execution risk
- [x] Add tests for large-event handling

**Exit criteria:** increased executable negrisk coverage with stable reject/error rates.

---

# Task Plan: Correlation Scanner V2 — Precision & Intelligence Upgrade

**Created:** 2026-02-16
**Baseline:** 1496 tests, correlation scanner producing mostly research-lane noise
**Goal:** Transform correlation scanner from regex-heuristic prototype into a precision instrument with semantic matching, logical validation, liquidity filtering, and deduplication.

## Current Phase
Phase 0 — Planning (awaiting approval)

## Problem Statement

The correlation scanner (Phase 5 from Session 6) successfully detects cross-event probability violations but has four structural weaknesses:

1. **Regex entity extraction is brittle** — `extract_entities()` uses capitalization patterns (`[A-Z][a-z]+`) and stop-word stripping. Misses semantic relationships ("Trump wins presidency" ↔ "Republican candidate elected"), produces false positives on coincidental entity overlap ("Super Bowl" in unrelated prop bets).

2. **No constraint validation** — If the scanner classifies events as temporal ("X by 2026" / "X by 2027"), it blindly assumes P(earlier) ≤ P(later). But "Bitcoin reaches $100K by 2026" and "Bitcoin mining banned by 2027" share entities + temporal markers yet have ZERO logical relationship. The constraint is applied without verifying the core question is the same.

3. **Micro-liquidity phantoms** — Opportunities with `required_capital < $5` (tiny book depth) are emitted and inflate counts. They can't be traded, waste scorer cycles, and pollute research-lane analysis.

4. **N×N pairing inflation** — Event E appearing in pairs (E,A), (E,B), (E,C) emits 3 separate opportunities. Same underlying signal counted 3 times. The per-cycle cap (`correlation_max_opps_per_cycle=80`) is a bandaid, not a fix.

## Architecture Overview

```
Current (V1):
  events → regex entity extraction → pairwise entity overlap → type classify → check violation → emit

Target (V2):
  events → TF-IDF similarity (candidate pairs) → type classify → constraint validation → check violation
        → liquidity floor → deduplicate → emit
                                                      ↑
                                           [optional: sentence-transformer embeddings]
```

## Phases

### Phase 1 — Semantic Similarity (replace regex entity extraction)
**Theme:** Better candidate pair selection via text similarity instead of regex entity sets.

#### 1.1 — Create `scanner/similarity.py`
- **Files:** `scanner/similarity.py` (new)
- **Approach:**
  - `SimilarityBackend` Protocol with `compute_pairwise(titles: list[str]) -> list[tuple[int, int, float]]`
  - **Default backend: `TFIDFSimilarity`** — Uses `sklearn.feature_extraction.text.TfidfVectorizer` (already installed) with `(1,2)` n-grams + `sklearn.metrics.pairwise.cosine_similarity`. Zero new dependencies.
  - **Optional backend: `EmbeddingSimilarity`** — Uses `sentence-transformers` (optional dep) with `all-MiniLM-L6-v2` model. Better semantic understanding but adds ~2GB torch dependency.
  - Configurable `min_similarity: float = 0.35` threshold for candidate pairs.
  - Output: list of (event_idx_a, event_idx_b, similarity_score) above threshold.
  - **Performance guard:** For N>500 events, use `sklearn.neighbors.NearestNeighbors` with cosine metric for approximate top-K instead of O(N²) pairwise.
- **Key constraint:** Must remain fast (<200ms for 1000 events). TF-IDF vectorization + sparse cosine is typically <50ms.
- **Tests:** Test TF-IDF backend with known similar/dissimilar titles. Test threshold filtering. Test performance with 1000 synthetic titles.
- [ ] Complete

#### 1.2 — Integrate similarity into `CorrelationScanner.build_relationship_graph()`
- **Files:** `scanner/correlation.py`
- **Problem:** `build_relationship_graph()` currently calls 3 separate relation finders, each doing O(N²) pairwise comparison with regex entity overlap. Replace with single similarity-based candidate selection.
- **Fix:**
  1. Compute TF-IDF similarity for all event titles via `SimilarityBackend`
  2. For each candidate pair above threshold, classify relationship type:
     - If both have temporal markers with different dates → TEMPORAL
     - If titles suggest complementary outcomes (win pattern check) → COMPLEMENT
     - If one title's question is a subset of the other → PARENT_CHILD
  3. Use similarity score to boost/reduce relation confidence (replace fixed 0.6/0.7/0.8)
  4. Keep `extract_temporal()` — regex works fine for date extraction
  5. Deprecate `extract_entities()` — no longer primary pair-finding mechanism
- **Backward compat:** Add `similarity_backend: str = "tfidf"` to constructor. Legacy mode (`similarity_backend="regex"`) preserves old behavior.
- **Tests:** Verify same test cases pass. Test that semantically similar titles are paired. Test that unrelated titles are NOT paired despite sharing an entity.
- [ ] Complete

#### 1.3 — Add config flags
- **Files:** `config.py`
- **New fields:**
  - `correlation_similarity_backend: str = "tfidf"` (options: "tfidf", "embedding", "regex")
  - `correlation_min_similarity: float = 0.35`
- **Tests:** Config validation tests.
- [ ] Complete

- **Status:** pending

**Phase 1 exit criteria:** Candidate pair selection uses text similarity instead of regex entity overlap. Precision improves (fewer false positives). TF-IDF backend runs with zero new dependencies.

---

### Phase 2 — Constraint Validation
**Theme:** Verify that detected relationships actually hold logically before checking price violations.

#### 2.1 — Implement `_validate_constraint()` in correlation.py
- **Files:** `scanner/correlation.py`
- **Approach:**
  - For **TEMPORAL** pairs: Strip temporal markers from both titles, compute stem similarity. If stem similarity < 0.80, the two events ask different questions despite sharing entities + dates. Reject.
    - Example PASS: "Bitcoin to $100K by March 2026" → stem "Bitcoin to $100K" / "Bitcoin to $100K by June 2026" → stem "Bitcoin to $100K" → stems match → valid temporal constraint.
    - Example FAIL: "Bitcoin to $100K by 2026" / "Bitcoin mining banned by 2027" → stems "Bitcoin to $100K" / "Bitcoin mining banned" → stems differ → reject.
  - For **PARENT_CHILD** pairs: Verify the child question is a RESTRICTION of the parent question (same core predicate, additional constraint). Use token-level containment check: child tokens ⊃ parent tokens (after stop-word removal). If parent title is not a substring/subsequence of child semantics, reject.
    - Example PASS: "Trump wins" (parent) / "Trump wins Ohio" (child) → "wins Ohio" restricts "wins" → valid.
    - Example FAIL: "Trump wins" / "Trump resigns" → different predicates → reject.
  - For **COMPLEMENT** pairs: Verify both titles ask the same question with different answer entities. Strip the varying entity, compare the question frames.
    - Example PASS: "Will Mahomes win MVP?" / "Will Allen win MVP?" → frame "Will _ win MVP?" → same → valid.
    - Example FAIL: "Will Mahomes win MVP?" / "Will Allen win Super Bowl?" → frames differ → reject.
- **Confidence adjustment:** Validation failure → set confidence to 0.0 (effectively filters pair). Partial validation (stems similar but not identical) → reduce confidence by 0.2.
- **Tests:** Test each rejection case above. Test that valid pairs pass through. Test confidence adjustment.
- [ ] Complete

- **Status:** pending

**Phase 2 exit criteria:** False positives from mismatched question stems eliminated. Temporal/parent-child/complement violations are only emitted when the underlying constraint is logically valid.

---

### Phase 3 — Liquidity Floor
**Theme:** Filter out micro-liquidity phantom opportunities that can't actually be traded.

#### 3.1 — Add `min_required_capital` filter
- **Files:** `scanner/correlation.py`, `config.py`
- **Approach:**
  - Add `min_required_capital: float = 5.0` to `CorrelationScanner.__init__()` and config (`correlation_min_required_capital`).
  - In each `_*_opportunity()` function, after computing `required_capital`, check `if required_capital < self._min_required_capital: return None`.
  - Log filtered opps at DEBUG level with reason code `"corr_micro_liquidity"`.
- **Why $5:** Below $5, tx gas + slippage eats most of the edge. Even at $0.005 gas, a $2 trade with 5% edge = $0.10 profit - $0.01 gas = $0.09. Not worth the slot in the ranking queue.
- **Tests:** Test that opportunities with required_capital < threshold are filtered. Test that opportunities above threshold pass through. Test with threshold=0 (no filter).
- [ ] Complete

- **Status:** pending

**Phase 3 exit criteria:** Micro-liquidity phantoms (required_capital < $5) no longer appear in output. Opp count drops, average quality increases.

---

### Phase 4 — Deduplication
**Theme:** Prevent the same event from appearing N times with N different pairings, inflating opportunity count.

#### 4.1 — Implement `_deduplicate_opportunities()` in correlation.py
- **Files:** `scanner/correlation.py`
- **Approach:**
  - After generating all opportunities in `scan()`, deduplicate before returning.
  - **Canonical pair key:** Sort `(source_event_id, target_event_id)` alphabetically to ensure (A,B) and (B,A) are the same pair.
  - **Per-pair dedup:** If the same pair produces multiple violations (e.g., complement + parent_child), keep only the highest net_profit one.
  - **Per-event cap:** Each event_id can appear in at most `max_pairings_per_event` (default 3) opportunities. If event E appears in 10 pairs, keep the 3 highest net_profit. This prevents one event from dominating the opportunity list.
  - Sort retained opps by net_profit descending.
- **Config:** `correlation_max_pairings_per_event: int = 3` in config.py.
- **Tests:** Test pair dedup (same pair, different violation types). Test per-event cap. Test with cap=0 (no dedup). Test sort order preserved.
- [ ] Complete

- **Status:** pending

**Phase 4 exit criteria:** Same event appears at most `max_pairings_per_event` times. Opp count reflects unique trading opportunities, not combinatorial explosion.

---

## Dependencies

```
Phase 1 (Similarity) ← Phase 2 depends on this (validation uses stem similarity)
Phase 3 (Liquidity) — independent, can run in parallel with 1/2
Phase 4 (Dedup) — independent, can run in parallel with 1/2/3
```

**Parallelization:** Phases 3 and 4 are independent and can be implemented in parallel with Phases 1-2. Phase 2 benefits from the similarity infrastructure in Phase 1 (stem comparison reuses TF-IDF) but doesn't strictly require it.

## Effort Estimate

| Phase | Items | Est. Lines | New Files | Description |
|-------|-------|-----------|-----------|-------------|
| 1     | 3     | ~250      | 1         | TF-IDF similarity backend + scanner integration + config |
| 2     | 1     | ~120      | 0         | Constraint validation logic in correlation.py |
| 3     | 1     | ~40       | 0         | Liquidity floor filter + config |
| 4     | 1     | ~80       | 0         | Deduplication logic + config |
| **Total** | **6** | **~490** | **1 new file** | Correlation scanner V2 |

## Key Design Decisions

| Decision | Rationale |
|----------|-----------|
| TF-IDF over embeddings as default | Zero new dependencies. scikit-learn already installed. Polymarket titles are standardized enough for TF-IDF to work well. Embedding backend available as optional upgrade. |
| Stem comparison for constraint validation | Simple, interpretable, fast. Strip temporal/entity markers → compare remaining text. No LLM needed for this step. |
| $5 liquidity floor | Below $5, gas + slippage dominate. Even aggressive micro-arbs need minimum book depth to be actionable. |
| Per-event cap of 3 | Prevents one popular event from consuming the entire opportunity list while still allowing multi-pairing signals. |
| Keep `extract_temporal()` | Regex works perfectly for date extraction. No need to use ML for structured patterns like "by March 2026". |
| `similarity_backend="regex"` preserves V1 | Backward compatibility for anyone relying on V1 behavior. Can be removed later. |
