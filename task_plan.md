# Task Plan: Surgical & Modular Implementation of All Improvements

**Created:** 2026-02-13
**Goal:** Implement all 24 unified improvements from the 5-agent deep inspection + IDEAS.md, using surgical, modular changes that can be merged independently.

**Principles:**
- Each phase produces a working, testable codebase
- No phase depends on a later phase (strict topological order)
- Each item is a single PR-sized diff (< 400 lines changed)
- Every change includes tests proving correctness
- Zero mutations to existing passing tests (unless fixing a bug in the test itself)

---

## Phase 1: Data Integrity Foundation (CRITICAL — do before any live trading)

**Theme:** Ensure the bot cannot lose money from bad data, thread bugs, or self-disabling behavior.

### 1.1 — Validate all orderbook/price data at ingestion boundaries
- **Files:** New `scanner/validation.py`, edit `client/clob.py`, `client/kalshi.py`, `scanner/book_cache.py`, `client/ws.py`, `scanner/latency.py`, `client/gas.py`
- **Approach:** Create a single `scanner/validation.py` module with `validate_price(p, context)`, `validate_size(s, context)`, `validate_gas_gwei(g)`. Import and call at every `float()` conversion from external data. Raises `ValueError` on `NaN`, `Inf`, negative, or out-of-range values.
- **Tests:** `tests/test_validation.py` — NaN, Inf, negative, >1.0, exactly 0.0, exactly 1.0, normal values. Integration: mock API returning bad data, verify pipeline rejects it.
- **Risk:** Low. Pure additive. Worst case: rejects a valid edge case we haven't seen yet (and we log it).
- [ ] Not started

### 1.2 — Fix NegRisk completeness bypass when event_market_counts returns 0
- **Files:** Edit `scanner/negrisk.py:76-83`
- **Approach:** When `expected_total == 0` for a NegRisk event, treat it as "unknown completeness" and skip the event (conservative). Add a log warning. Alternatively, derive count from the markets list itself as a fallback.
- **Tests:** Add test case with `event_market_counts = {}` and verify the event is skipped.
- **Risk:** Low. Conservative change — may skip some valid events, but prevents catastrophic loss.
- [ ] Not started

### 1.3 — Replace asyncio.Queue with queue.Queue + bound WS queues
- **Files:** Edit `client/ws.py:46-47`, `client/ws_bridge.py` (drain logic)
- **Approach:** `import queue`; replace `asyncio.Queue` with `queue.Queue(maxsize=10000)`. In WS handler, catch `queue.Full` and log warning (drop oldest or newest). Drain loop uses `queue.Queue.get_nowait()` (same API).
- **Tests:** `tests/test_ws_bridge.py` — verify drain works with `queue.Queue`, verify `Full` is handled gracefully.
- **Risk:** Low. Drop-in replacement. Same `put_nowait()`/`get_nowait()` API.
- [ ] Not started

### 1.4 — Fix BookCache get_book()/get_books() to acquire the lock
- **Files:** Edit `scanner/book_cache.py:107-113`
- **Approach:** Add `with self._lock:` to `get_book()` and `get_books()`. Consider deprecating `get_books_snapshot()` since `get_books()` now acquires the lock.
- **Tests:** Existing `tests/test_book_cache.py` should still pass. Add a test that calls `get_books()` for multiple tokens and verifies consistency.
- **Risk:** Low. May add ~microseconds of lock contention. Single-reader model means no contention in practice.
- [ ] Not started

### 1.5 — Fix exposure tracking: decrement on resolution/unwind
- **Files:** Edit `monitor/pnl.py`
- **Approach:** Add `reduce_exposure(amount: float)` method. Call it from: (a) `record()` when trade is a sell/unwind, (b) new `record_resolution(token_id, payout)` method for resolved positions. In `record()`, if any leg is `Side.SELL`, decrement exposure by `sell_notional`.
- **Tests:** `tests/test_pnl.py` — buy $100, verify exposure=100. Sell $50, verify exposure=50. Buy again, verify exposure doesn't exceed realistic bounds.
- **Risk:** Low. Pure additive to PnLTracker. No change to existing record() signature.
- [ ] Not started

### 1.6 — Make Config frozen + use model_copy pattern
- **Files:** Edit `config.py` (add `frozen=True`), edit `run.py:121-128,194-195`
- **Approach:** Add `model_config = SettingsConfigDict(env_file=".env", frozen=True)` to Config. Change `_enforce_polymarket_only_mode(cfg) -> Config` to return `cfg.model_copy(update={...})`. Change `run.py:194` to `cfg = cfg.model_copy(update={"paper_trading": False})`.
- **Tests:** Existing tests that mutate Config will need `model_copy()` instead. Add test that Config raises on direct mutation.
- **Risk:** Medium. Any other code mutating Config will break (by design — that's the point). Grep for `cfg.` assignments.
- [ ] Not started

**Phase 1 exit criteria:** All 6 items pass tests. `--dry-run` and `--scan-only` modes work. No regressions in existing 696 tests.

---

## Phase 2: Stop Bleeding Money (HIGH — unwind, risk controls, tick precision)

**Theme:** Ensure execution failures are handled deterministically and risk controls are enforced.

### 2.1 — Cross-platform fill state machine with unwind retry
- **Files:** New `executor/fill_state.py`, edit `executor/cross_platform.py`
- **Approach:** Define `FillState` enum: `PENDING, FILLED, PARTIAL, REJECTED, RESTING, UNWINDING, UNWOUND, STUCK`. Replace the linear if/else in `execute_cross_platform()` with state transitions. Add retry (3 attempts, 0.5s backoff) to `_unwind_platform()`. Persist stuck positions to `stuck_positions.json`. On startup, check for and log stuck positions.
- **Tests:** `tests/test_fill_state.py` — state transitions, retry on unwind failure, stuck position persistence. Extend `tests/test_cross_platform_exec.py` with cascading failure scenario.
- **Risk:** Medium. Most complex change. Isolate in new module to minimize blast radius.
- [ ] Not started

### 2.2 — Wire unused runtime risk controls end-to-end (IDEAS #1)
- **Files:** Edit `run.py`, `executor/cross_platform.py`, `executor/safety.py`
- **Approach:** (a) Pass `cfg.cross_platform_deadline_sec` into `execute_cross_platform()` and enforce it. (b) Add `verify_platform_limits(platform, position, cfg)` to safety.py. (c) Respect `cfg.*_enabled` toggles in the scan phase (some are already respected; verify all).
- **Tests:** Add tests proving: deadline exceeded -> abort + unwind, platform limit hit -> skip, disabled platform -> skip.
- **Risk:** Low. Wiring existing config fields to existing code paths.
- [ ] Not started

### 2.3 — Enforce market tick-size precision at execution (IDEAS #3)
- **Files:** New `executor/tick_size.py`, edit `executor/engine.py`, `executor/cross_platform.py`
- **Approach:** Create `quantize_price(price: float, tick_size: float) -> float` that rounds to nearest valid tick. Call before every `create_limit_order()` and `place_order()`. Reject orders where quantization changes price by more than `tick_size / 2`.
- **Tests:** `tests/test_tick_size.py` — 0.01 markets, 0.001 markets, round-up, round-down, rejection on large quantization shift.
- **Risk:** Low. Pure pre-execution validation.
- [ ] Not started

### 2.4 — Type platform_clients as dict[str, PlatformClient] everywhere
- **Files:** Edit `run.py:269`, `executor/engine.py:47`, `executor/cross_platform.py:117`
- **Approach:** Replace `dict[str, object]` with `dict[str, PlatformClient]`. Import from `client/platform.py`. Move price conversion into each platform client (Kalshi converts to cents internally, not in executor).
- **Tests:** Existing tests pass. Add mypy/pyright check to CI (Phase 5).
- **Risk:** Low. Type annotation change only. Price conversion move is a small refactor.
- [ ] Not started

### 2.5 — Freeze TradeResult + use tuples for fill data
- **Files:** Edit `scanner/models.py:157`, edit callers that construct `TradeResult`
- **Approach:** `@dataclass(frozen=True)`, change `fill_prices: list[float]` to `tuple[float, ...]`, same for `fill_sizes`. Update all constructors to pass tuples.
- **Tests:** Existing tests adapted. Add test that `TradeResult` raises on mutation.
- **Risk:** Low. Grep for `TradeResult(` and update constructors.
- [ ] Not started

**Phase 2 exit criteria:** Cross-platform unwind has retry + persistence. All config risk controls are wired. Tick-size validation active. `--dry-run` shows no regressions.

---

## Phase 3: Make More Money (HIGH — latency, sizing, scoring)

**Theme:** Reduce cycle time from 40-60s to <5s and deploy capital more effectively.

### 3.1 — Cache Gamma API (60s TTL) + event market counts (5min TTL) + external get_all_markets
- **Files:** New `client/cache.py`, edit `run.py`, `client/gamma.py`
- **Approach:** Create a generic `TTLCache[T]` class in `client/cache.py` with `get(key, factory, ttl)`. Wrap `get_all_markets()`, `get_event_market_counts()`, and each platform's `get_all_markets()`. Call `cache.get("gamma_markets", lambda: get_all_markets(...), ttl=60)` in the main loop.
- **Tests:** `tests/test_cache.py` — TTL expiry, stale return, factory error handling.
- **Risk:** Low. Markets don't change second-by-second. Stale markets are filtered by `is_market_stale()`.
- [ ] Not started

### 3.2 — Parallelize scanners + all book fetches
- **Files:** Edit `run.py:452-591`, `client/kalshi.py:177-187`, `run.py:495-496`, `run.py:856-858`
- **Approach:** (a) Wrap 5 scanner calls in `ThreadPoolExecutor(max_workers=5)` + `as_completed()`. (b) Add `max_workers` param to `KalshiClient.get_orderbooks()` using `ThreadPoolExecutor`. (c) Use `poly_book_fetcher` for latency scanner instead of serial `get_orderbooks`. (d) Batch safety-check book fetches for all scored opportunities.
- **Tests:** Verify same results as serial execution. Add timing assertion that parallel is faster.
- **Risk:** Medium. Scanner parallelism is safe (disjoint data). Kalshi parallelism needs rate-limit respect (4 workers = safe under 20 req/sec). Safety batch needs freshness guard.
- [ ] Not started

### 3.3 — Recalibrate Kelly sizing
- **Files:** Edit `executor/sizing.py`
- **Approach:** Replace hardcoded `odds = 1.0` with `odds` derived from arb type. For confirmed arbs (binary/negrisk), set `odds = 0.1` (10:1 implied, half-Kelly deploys ~15% of bankroll). For cross-platform, set `odds = 0.2` (accounting for execution risk). Make these configurable in `config.py`.
- **Tests:** Update `tests/test_sizing.py` with new expected values. Verify half-Kelly at various fill probabilities.
- **Risk:** Medium. Larger position sizes = more money at risk per trade. Start conservative (odds=0.2) and tune with live data.
- [ ] Not started

### 3.4 — Integrate ArbTracker persistence confidence into scoring (IDEAS #8)
- **Files:** Edit `run.py` (instantiate `ArbTracker`), edit `run.py:_build_scoring_contexts()`
- **Approach:** Create `ArbTracker()` during initialization. After scanning, call `arb_tracker.record(cycle, opps)`. In `_build_scoring_contexts()`, call `arb_tracker.confidence(opp.event_id, ...)` and set `ScoringContext.confidence`. The scorer already has `W_PERSISTENCE = 0.15` — it just needs real data.
- **Tests:** `tests/test_confidence.py` should already exist. Add integration test verifying confidence flows through scoring.
- **Risk:** Low. Wire existing code. Scorer weights already account for this factor.
- [ ] Not started

**Phase 3 exit criteria:** Cycle time <10s with caching. Kelly sizing deploys 5-15% of bankroll per confirmed arb. ArbTracker active in scoring. `--dry-run --limit 500` completes in <5s.

---

## Phase 4: Improve Edge Accuracy (MEDIUM — fees, safety, matching, strategy)

**Theme:** Tighten the accuracy of profit estimation and risk assessment.

### 4.1 — Correct fee-model realism for DCM and fee-bearing markets (IDEAS #4)
- **Files:** Edit `scanner/fees.py`
- **Approach:** Add DCM (Dynamic Cost Model) detection via market metadata. Implement actual taker fee schedule for DCM markets (fee increases toward 50/50 odds). Add "golden case" tests against manually computed fee examples from Polymarket docs.
- **Tests:** `tests/test_fees.py` — golden cases at 10/90, 30/70, 50/50 odds for DCM markets.
- **Risk:** Low. Tightens edge estimation = fewer but more profitable trades.
- [ ] Not started

### 4.2 — Use VWAP in verify_edge_intact (not top-of-book)
- **Files:** Edit `executor/safety.py:306-339`
- **Approach:** Import `sweep_cost` from `scanner/depth.py`. In `verify_edge_intact`, compute VWAP cost for the actual execution size instead of using `best_ask.price`. Pass the position size through to the safety check.
- **Tests:** Add test with thin book where top-of-book shows edge but VWAP doesn't.
- **Risk:** Low. More conservative = fewer trades but each is more reliably profitable.
- [ ] Not started

### 4.3 — Upgrade cross-platform matching to contract-level (IDEAS #6)
- **Files:** Edit `scanner/matching.py`
- **Approach:** After event-level fuzzy match (existing), add a second pass: match individual outcomes/contracts by name. Add settlement-equivalence guards (e.g., "Over 2.5" != "Over 3.5"). Block execution when per-contract confidence < threshold.
- **Tests:** Add test cases with same-event different-settlement contracts. Verify they're blocked.
- **Risk:** Medium. More restrictive matching = fewer cross-platform opportunities, but each is safer.
- [ ] Not started

### 4.4 — Add epsilon-based comparisons for cost < 1.0 checks
- **Files:** Edit `scanner/binary.py:115`, `scanner/negrisk.py` (similar locations)
- **Approach:** Define `FLOAT_EPSILON = 1e-9` in `scanner/models.py`. Replace `cost_per_set >= 1.0` with `cost_per_set >= 1.0 - FLOAT_EPSILON`. Same for `<= 0.0` checks.
- **Tests:** Add test with `yes_ask=0.49, no_ask=0.51` verifying arb IS detected.
- **Risk:** Very low. One-line changes.
- [ ] Not started

### 4.5 — Strategy mode: gate AGGRESSIVE on avg P&L, not just win rate
- **Files:** Edit `scanner/strategy.py:148-151`
- **Approach:** Add `avg_pnl` to `MarketState`. Change condition to `state.recent_win_rate >= 0.50 and state.avg_pnl > 0`. Compute `avg_pnl` from PnLTracker's recent trades.
- **Tests:** Add test: 50% win rate but negative avg P&L -> CONSERVATIVE mode.
- **Risk:** Low. Purely additive gate condition.
- [ ] Not started

**Phase 4 exit criteria:** Fee model matches Polymarket docs for DCM markets. Safety checks use VWAP. Matching blocks different-settlement contracts. Float comparisons have epsilon. Strategy considers P&L magnitude.

---

## Phase 5: Code Health & Sustainability

**Theme:** Testing, memory, maintainability, documentation.

### 5.1 — Harden WebSocket health and failover (IDEAS #7)
- **Files:** Edit `client/ws.py`, `client/ws_bridge.py`, `run.py` (scan phase)
- **Approach:** (a) Add `last_message_time` tracking in WSManager. (b) Add `is_healthy(max_silence_sec=30)` method. (c) In scan phase, check `ws_bridge.is_connected` and `is_healthy()`. If unhealthy, skip spike scanner and log warning. (d) Add auto-restart on repeated failure.
- **Tests:** Mock WS going silent, verify health check fires. Mock repeated failures, verify restart.
- **Risk:** Low. Degradation to REST is safe (just slower).
- [ ] Not started

### 5.2 — Break up run.py + extract gas cost utility
- **Files:** New `pipeline/init_clients.py`, `pipeline/scan_cycle.py`, `pipeline/exec_cycle.py`, new `scanner/gas_utils.py`. Edit `run.py`.
- **Approach:** Extract `_init_platform_clients()`, `_run_scan_cycle()`, `_run_execution_cycle()` into separate modules under `pipeline/`. Extract gas cost fallback into `scanner/gas_utils.py` with `estimate_gas_cost(gas_oracle, n_legs, gas_per_order, gas_price_gwei) -> float`. Replace 4 duplicated blocks in scanners.
- **Tests:** Existing tests pass with imports redirected. New `tests/test_gas_utils.py`.
- **Risk:** Medium. Large refactor. Do after all functional changes are stable.
- [ ] Not started

### 5.3 — Add CI quality gates (IDEAS #9) + test critical untested paths
- **Files:** New `.github/workflows/ci.yml`, new/expanded test files
- **Approach:** (a) CI: `ruff check .`, `mypy .`, `pytest --cov=. --cov-fail-under=85`. (b) Write tests for 8 critical untested paths: `_filled_size_from_response`, negrisk partial fill, WS reconnect, cascading cross-platform failure, `_build_scoring_contexts`, `verify_edge_intact` latency, empty-legs through safety, `_dollars_to_cents` out-of-range. (c) Fix conditional assertions (`if len(...) >= 2:` -> `assert len(...) >= 2`).
- **Tests:** The tests ARE the deliverable.
- **Risk:** Low. Pure additive.
- [ ] Not started

### 5.4 — Fix memory leaks for long-running sessions
- **Files:** Edit `monitor/scan_tracker.py`, `scanner/spike.py`, `scanner/book_cache.py`
- **Approach:** (a) `ScanTracker`: cap `self.opportunities` to last N cycles (e.g., 100). (b) `SpikeDetector`: add `cleanup_stale(active_tokens: set[str])`, call per cycle. (c) `BookCache`: add `prune(max_age_sec=300)`, call per cycle.
- **Tests:** Add tests verifying bounded memory after N cycles.
- **Risk:** Low. Pruning stale data.
- [ ] Not started

### 5.5 — Fix documentation/config drift (IDEAS #10)
- **Files:** Edit `CLAUDE.md`, new `.env.example`, edit `config.py`
- **Approach:** Auto-generate `.env.example` from `Config` fields with defaults and descriptions. Sync CLAUDE.md scoring weights with actual `scorer.py` values. Verify all run modes documented match actual behavior.
- **Tests:** N/A (documentation).
- **Risk:** Very low.
- [ ] Not started

**Phase 5 exit criteria:** CI green with lint/type/test/coverage gates. Memory stable over simulated long runs. run.py under 300 lines. All docs accurate.

---

## Dependencies

```
Phase 1 (all items independent, can parallelize)
  |-- Phase 2 depends on: 1.1 (validation), 1.4 (BookCache), 1.6 (Config frozen)
       |-- Phase 3 depends on: 2.1 (state machine), 2.4 (PlatformClient types)
            |-- Phase 4 depends on: 3.1 (caching), 3.4 (confidence)
                 |-- Phase 5 depends on: all functional changes stable
```

Within each phase, items CAN be parallelized (independent PRs).

## Effort Estimate

| Phase | Items | Lines | Sessions |
|-------|-------|-------|----------|
| 1     | 6     | ~400  | 1        |
| 2     | 5     | ~600  | 1-2      |
| 3     | 4     | ~500  | 1-2      |
| 4     | 5     | ~400  | 1        |
| 5     | 5     | ~800  | 2        |
| **Total** | **25** | **~2,700** | **6-8** |

## Decision Log

| Date | Decision | Rationale |
|------|----------|-----------|
| 2026-02-13 | Phase 1 first (data integrity) | Prevents catastrophic loss; all other work pointless if bad data can bankrupt the bot |
| 2026-02-13 | State machine in `executor/fill_state.py` | Isolates complexity; cross_platform.py delegates to it |
| 2026-02-13 | TTLCache as generic `client/cache.py` | Reusable across Gamma, event counts, and external platform calls |
| 2026-02-13 | Kelly odds configurable, not hardcoded | Allows gradual tuning without code changes |
| 2026-02-13 | run.py breakup deferred to Phase 5 | Functional changes first, structural refactor once behavior is stable |
| 2026-02-13 | Validation in `scanner/validation.py` | Single import for all ingestion points; testable in isolation |
