# Actionable Arbitrage Implementation Plan (2026-02-16)

## Objective

Increase **realized, executable arbitrage capture** and reduce theoretical/phantom opportunities by restructuring the pipeline around actionable execution.

This plan is based on:
- Deep local code audit (`run.py`, `scanner/`, `executor/`, `client/`, `benchmark/`)
- Fresh runtime validation (`--dry-run --limit 400`)
- External architecture review via `gemini` CLI and `claude` CLI

## Baseline Evidence (Current State)

- Test suite is healthy: `1483 passed`.
- Runtime integration gaps:
  - `OrderPresigner` exists but is not wired from `run.py` into `execute_opportunity`.
  - `MLScorer`/`FeatureEngine` exist but are not wired into live ranking.
  - `BookService` exists but is not used in `run.py`.
- Correlation output dominates scan-only results but is non-actionable:
  - 33/33 opportunities in one dry-run cycle were `correlation_arb`.
  - Session summary reported `Actionable now (taker BUY): 0`.
- Correlation math uses first-market heuristics:
  - Event implied probability uses first active market with a best ask.
  - Opportunity construction uses first active books and mixed BUY/SELL legs.
- Replay tooling mismatch:
  - `benchmark/recorder.py` writes flat records.
  - `benchmark/replay.py` expects `{type: "cycle", data: {...}}`.
- OFI currently uses snapshot volume imbalance, not aggressor flow.

## Success Criteria

Primary success criteria after rollout:

1. `actionable_now_count / opportunities_found` materially increases from near-zero.
2. `signal_to_order_ms` p50 and p95 improve (especially for latency/spike paths).
3. Realized PnL quality improves (higher win-rate, better realized/theoretical ratio).
4. Replay/backtest loop becomes trustworthy (schema-compatible, reproducible).

## KPI Definitions (must be logged)

- `opps_found_total`: total opportunities emitted per cycle.
- `opps_actionable_now`: count passing executable gates.
- `actionable_ratio`: `opps_actionable_now / opps_found_total`.
- `signal_to_order_ms`: timestamp at ranking -> first order post.
- `execution_fill_ratio`: fully filled trades / submitted trades.
- `unwind_rate`: partial/unwind incidents / submitted trades.
- `realized_to_theoretical`: realized net PnL / expected net PnL.
- `cross_platform_orphan_rate`: one-leg fill failures / cross-platform attempts.
- `replay_parse_success`: parseable recording files / total recording files.

## Implementation Status (2026-02-16)

All phases in this plan have been implemented in runtime code and tests.

- Phase 0: lane split + lane metrics in cycle/status/summary output.
- Phase 1: correlation precision hardening (event-level aggregation, liquidity/depth filters, ROI cap, regression tests).
- Phase 2: presigner lifecycle + prewarm + execution-path wiring.
- Phase 3: cross-platform safety semantics (PM-only inventory scope, venue preflight, fill-gap guardrail).
- Phase 4: recorder/replay schema unification (schema v2 + legacy compatibility + validate-only replay mode).
- Phase 5: centralized `BookService` prefetch orchestration in runtime.
- Phase 6: OFI upgrade to aggressor-flow approximation and quality telemetry.
- Phase 7: gated ML reranking integration with sample collection and model persistence/retrain cadence.
- Phase 8: large-event negRisk bounded subset mode with conservative tail-risk payout cap and tests.

---

## Phase 0 (P0): Actionable-First Lanes + Instrumentation

### Goal
Separate research signals from executable opportunities and stop letting non-actionable signal volume obscure pipeline quality.

### Tasks

1. Split pipeline output into two lanes in `run.py`:
   - `executable_opps`
   - `research_opps`

2. Keep `correlation_arb` in research lane by default.

3. Add per-lane metrics to `monitor/scan_tracker.py`, `monitor/status.py`, and display output.

4. Add config flags in `config.py`:
   - `correlation_execute_enabled: bool = False`
   - `research_lane_enabled: bool = True`

### Files
- `run.py`
- `config.py`
- `monitor/scan_tracker.py`
- `monitor/display.py`
- `monitor/status.py`
- tests: `tests/test_run.py`, `tests/test_scan_tracker.py`, `tests/test_display.py`

### Acceptance Criteria
- Scan summaries always show both lane counts.
- Execution queue is derived strictly from `executable_opps`.
- Correlation opportunities are visible but do not dilute actionable metrics.

### KPI Target
- Baseline actionable ratio is measurable and stable across sessions.

---

## Phase 1 (P1): Correlation Scanner Hardening (Research-Grade First)

### Goal
Convert correlation from headline-ROI generator into a high-precision research signal.

### Tasks

1. Replace first-market implied probability logic with event-level aggregation:
   - Compute robust implied probability from a configurable subset (`top_liquidity`, `median`, etc.).

2. Add strict filters:
   - Minimum depth and volume per selected market.
   - Max theoretical ROI cap for research alerts.
   - Reject low-confidence relation pairs.

3. Keep SELL-dependent opportunities as non-executable unless inventory/hedge constraints are explicitly satisfied.

4. Add explicit reason codes in correlation outputs (why emitted, why non-executable).

### Files
- `scanner/correlation.py`
- `scanner/models.py` (optional metadata fields for reason codes)
- `run.py` (lane assignment)
- tests: `tests/test_correlation.py`, `tests/test_correlation_integration.py`

### Acceptance Criteria
- Correlation emissions drop substantially but precision improves.
- No correlation signal enters executable queue unless all execution constraints pass.

### KPI Target
- Correlation precision in paper mode improves by >2x vs current baseline.

---

## Phase 2 (P1): Presigner Integration in Real Execution Path

### Goal
Reduce signing overhead on latency-sensitive opportunities.

### Tasks

1. Instantiate `OrderPresigner` in `run.py` when:
   - `cfg.presigner_enabled`
   - not dry-run
   - signer/client available

2. Pass `presigner` into both execution call sites in `run.py`:
   - `_execute_single(...) -> execute_opportunity(..., presigner=...)`
   - maker paper/live paths as applicable.

3. Add warm-up policy:
   - Pre-sign near top ranked executable opps each cycle.

4. Add presigner stats logging:
   - cache hits, misses, hit-rate, stale invalidations.

### Files
- `run.py`
- `executor/engine.py` (if minor interface cleanup needed)
- `executor/presigner.py` (stats/eviction observability)
- tests: `tests/test_presigner_integration.py`, `tests/test_engine.py`

### Acceptance Criteria
- Presigner is active in live execution path.
- Hit-rate and latency improvements are observable in logs.

### KPI Target
- `signal_to_order_ms` p50 down by >=30% on binary/negrisk/scalpable paths.

---

## Phase 3 (P1): Cross-Platform Safety and Legging-Risk Semantics

### Goal
Fix safety semantics so cross-platform opportunities are validated correctly per venue and reduce orphan leg risk.

### Tasks

1. Scope Polymarket inventory checks to Polymarket SELL legs only.
   - Do not apply PM inventory checks to external tickers.

2. Add platform-specific preflight checks:
   - depth
   - limits
   - venue availability/warm cache state

3. Add execution timing guardrails:
   - reject if fill-time gap exceeds configured threshold.

4. Improve unwind telemetry with structured reason codes.

### Files
- `run.py` (inventory check scope)
- `executor/safety.py`
- `executor/cross_platform.py`
- `scanner/cross_platform.py` (optional stricter prefilters)
- tests: `tests/test_cross_platform.py`, `tests/test_cross_platform_exec.py`, `tests/test_safety.py`

### Acceptance Criteria
- Cross-platform opportunities are not incorrectly blocked by PM-only inventory checks.
- Orphan/unwind incidents are explicitly tracked with reasons.

### KPI Target
- `cross_platform_orphan_rate` reduced by >=50% from baseline.

---

## Phase 4 (P1): Recorder/Replay Schema Unification

### Goal
Make offline optimization trustworthy and compatible with runtime data.

### Tasks

1. Define schema version in recorder output (`schema_version`).

2. Align record structure between recorder and replay parser.
   - Add backward-compatible parser for prior schema if needed.

3. Add replay validation command:
   - parse-only and parse+replay sanity checks.

4. Integrate recorder into runtime with config flags:
   - `recording_enabled`
   - `recording_max_mb`

### Files
- `benchmark/recorder.py`
- `benchmark/replay.py`
- `run.py` (runtime hook)
- `config.py`
- tests: `tests/test_recorder.py`, `tests/test_replay.py`

### Acceptance Criteria
- Fresh recordings replay successfully end-to-end.
- Old recordings either parse via compatibility mode or fail with explicit version error.

### KPI Target
- `replay_parse_success = 100%` on new recordings.

---

## Phase 5 (P2): BookService Runtime Integration

### Goal
Centralize fetch orchestration and remove scanner-level redundant fetch patterns.

### Tasks

1. Replace direct `poly_book_fetcher` usage in scanner closures with `BookService`.

2. Do single prefetch for required token universe per cycle, then scanner reads from cached fetcher.

3. Ensure no behavior regressions under WS event-driven mode.

### Files
- `run.py`
- `scanner/book_service.py` (minor API refinements)
- tests: `tests/test_book_service.py`, `tests/test_pipeline_integration.py`

### Acceptance Criteria
- One coherent fetch plan per cycle.
- Reduced fetch churn and more deterministic scan times.

### KPI Target
- Cycle scan latency and read-call volume both improve under load.

---

## Phase 6 (P2): OFI Signal Upgrade (Aggressor-Flow Approximation)

### Goal
Turn OFI from passive imbalance proxy into a more predictive execution signal.

### Tasks

1. Extend WS message handling to capture richer delta/trade-like signals if available.

2. In `WSBridge`, compute OFI from directional liquidity consumption heuristics (not simple bid-ask snapshot difference).

3. Add OFI quality metric:
   - correlation of OFI score with short-horizon price move.

4. Retune scorer OFI contribution after signal quality check.

### Files
- `client/ws.py`
- `client/ws_bridge.py`
- `scanner/ofi.py`
- `scanner/scorer.py`
- tests: `tests/test_ws_bridge.py`, `tests/test_ofi.py`, `tests/test_scorer.py`

### Acceptance Criteria
- OFI has measurable predictive lift over baseline.
- OFI signal no longer dominated by static depth imbalance.

### KPI Target
- OFI-selected opportunities show improved fill/realization performance vs control.

---

## Phase 7 (P2): ML Scorer Integration (Gated Augmentation)

### Goal
Use ML as a controlled reranker on top of deterministic gates.

### Tasks

1. Add runtime wiring:
   - instantiate/load `MLScorer` when enabled.
   - inject ML probability as a bounded score component in ranking.

2. Add sample collection path from realized trade outcomes.

3. Retraining cadence:
   - cycle-based or trade-count based thresholds.
   - persisted model reload on startup.

4. Add feature sanity checks to guard drift.

### Files
- `run.py`
- `scanner/scorer.py`
- `scanner/ml_scorer.py`
- `scanner/feature_engine.py`
- tests: `tests/test_ml_scorer.py`, `tests/test_feature_engine.py`, `tests/test_pipeline_e2e.py`

### Acceptance Criteria
- ML path can be toggled safely.
- Fallback path remains deterministic and fully functional.

### KPI Target
- Positive lift in realized PnL quality metrics in shadow/A-B mode.

---

## Phase 8 (P3): Large-Event NegRisk Actionability (>15 Outcomes)

### Goal
Recover executable edge from large outcome spaces currently skipped by max-leg limits.

### Tasks

1. Add constrained basket builder:
   - select best executable subset with bounded leg count.

2. Add solver-like optimization (greedy first, exact solver optional later).

3. Add safety semantics for partial-event execution risk.

### Files
- `scanner/negrisk.py`
- `executor/safety.py`
- tests: `tests/test_negrisk.py`, `tests/test_arithmetic_validation.py`

### Acceptance Criteria
- Large events produce bounded executable candidates without violating safety rules.

### KPI Target
- Increase executable negrisk count without increasing safety reject rate.

---

## Rollout Strategy

Use staged rollout with feature flags and strict rollback criteria.

### Stage A (Days 1-3)
- P0 + P1 (lane split, correlation hardening)
- Objective: restore signal quality visibility

### Stage B (Days 3-6)
- P2 + P3 + P4 (presigner, cross-platform semantics, recorder/replay fixes)
- Objective: execution path and research loop reliability

### Stage C (Days 6-10)
- P5 + P6 (BookService runtime + OFI upgrade)
- Objective: scan efficiency and signal quality

### Stage D (Days 10-14)
- P7 shadow mode (ML integrated, not dominant)
- Objective: validated incremental lift only

### Stage E (Post-stability)
- P8 negrisk expansion

## Definition of Done (Program Level)

Program is complete when all are true:

1. Actionable ratio is consistently non-trivial and no longer drowned by research-only signals.
2. Presigner is active and measurable in production path.
3. Cross-platform safety semantics are venue-correct and orphan risk is reduced.
4. Recorder/replay loop is schema-consistent and used for tuning.
5. OFI and ML are integrated with measurable incremental value.

## Immediate Next Actions (Ready to Execute)

1. Implement P0 lane split and metric instrumentation.
2. Implement P2 presigner runtime wiring (low risk, high impact).
3. Implement P4 recorder/replay schema fix to unlock reliable optimization.
