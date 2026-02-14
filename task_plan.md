# Task Plan: 10 Profit-Maximization Fixes

**Created:** 2026-02-14
**Builds on:** Session 2 (25 foundational items, 834 tests passing)
**Goal:** Implement 10 changes that unlock maximum revenue from the existing pipeline — fix throttled parameters, widen scanning aperture, and add 4 new revenue-generating scanner modules. Success = all implemented, tested, and pipeline finds every viable opportunity.

## Current Phase
Phase 1

## Phases

### Phase 1: Unlock Existing Pipeline (Quick Wins)
**Theme:** Fix parameters and thresholds that are artificially choking the pipeline. No new modules, just tuning.

#### 1.1 — Fix gas threshold in strategy selector (Polygon != Ethereum)
- **Files:** `scanner/strategy.py`
- **Problem:** `HIGH_GAS_GWEI = 100.0` triggers CONSERVATIVE mode on Polygon, where 100 gwei costs ~$0.003 per order. The threshold was designed for Ethereum-era gas costs.
- **Fix:** Change to dollar-denominated gas thresholds. Add `gas_cost_usd` to `MarketState`. AGGRESSIVE when gas < $0.01/order, CONSERVATIVE when gas > $0.10/order. Fall back to gwei thresholds only when gas oracle is unavailable.
- **Tests:** Unit tests for each strategy mode with Polygon-realistic gas prices (30-500 gwei at $0.50 POL). Verify AGGRESSIVE is selected at 100 gwei on Polygon. Verify CONSERVATIVE only at extreme gas ($0.10+/order).
- [ ] Not started

#### 1.2 — Raise Kelly odds defaults for confirmed arbs
- **Files:** `config.py`, `executor/sizing.py`, `.env.example`, `tests/test_sizing.py`
- **Problem:** `kelly_odds_confirmed = 0.10` (10:1 implied). For confirmed arbs where YES+NO < $1, fill probability is ~85-95%. Current setting deploys only ~23% of available capital.
- **Fix:** Change default `kelly_odds_confirmed` from 0.10 to 0.65 (reflecting ~85% fill probability). Change `kelly_odds_cross_platform` from 0.20 to 0.40 (reflecting ~70% fill probability with execution risk). Update .env.example.
- **Tests:** Update existing sizing tests with new default values. Add test verifying capital deployment is 40-60% for confirmed arbs (vs previous 15-25%).
- [ ] Not started

#### 1.3 — Edge-proportional slippage ceiling
- **Files:** `scanner/binary.py`, `scanner/negrisk.py`, `scanner/cross_platform.py`, `config.py`
- **Problem:** Hardcoded `1.005` (0.5%) slippage ceiling in all scanners. For a 5% edge arb, accepting 2% slippage still leaves 3% profit. But we only see depth within 0.5%, missing 80%+ of available liquidity.
- **Fix:** Replace `best_ask.price * 1.005` with `best_ask.price * (1 + min(edge * slippage_fraction, max_slippage))`. Add `slippage_fraction: float = 0.4` and `max_slippage_pct: float = 3.0` to config. For 5% edge: accept up to 2% slippage. For 2% edge: accept up to 0.8%.
- **Tests:** Test with multi-level orderbook: verify at 5% edge, depth includes levels up to 2% above best ask. Verify at 1% edge, ceiling is tighter. Verify max_slippage caps the ceiling.
- [ ] Not started

#### 1.4 — Scale exposure limits + remove min_hours filter
- **Files:** `config.py`, `.env.example`
- **Problem:** `max_exposure_per_trade = $500`, `max_total_exposure = $5,000` cap profit at the source. `min_hours_to_resolution = 1.0` blocks near-resolution opportunities.
- **Fix:** Change defaults: `max_exposure_per_trade = $5,000`, `max_total_exposure = $50,000`, `min_hours_to_resolution = 0.0`. Update .env.example with comments explaining the reasoning.
- **Tests:** Verify sizing function deploys up to new limits. Verify filter passes markets resolving in <1 hour.
- [ ] Not started

#### 1.5 — Verify ArbTracker integration is live (integration test)
- **Files:** `tests/test_run.py` or new `tests/test_integration_pipeline.py`
- **Problem:** Previous session "wired" ArbTracker into run.py, but need to verify confidence flows through scoring end-to-end.
- **Fix:** Write integration test: create 2 cycles of opportunities for the same event. Verify cycle 1 gets confidence < 1.0, cycle 2 gets confidence = 1.0 (persistent). Verify scored rank changes.
- **Tests:** Integration test as described above.
- [ ] Not started

- **Status:** pending

**Phase 1 exit criteria:** `--dry-run` finds 2-5x more opportunities than before. Strategy shows AGGRESSIVE mode on Polygon gas. Kelly deploys 40-60% of capital for confirmed arbs.

---

### Phase 2: Maker Strategy Scanner (NEW Revenue Stream)
**Theme:** Stop paying the spread — earn it. Post passive orders on both sides of deep markets.

#### 2.1 — Design scanner/maker.py
- **Files:** New `scanner/maker.py`
- **Approach:** For each binary market with sufficient volume and spread > 1 tick:
  - Compute the "maker edge": if we post YES bid and NO bid such that combined cost < $1
  - E.g., YES bid at $0.48, NO bid at $0.50 → cost = $0.98 → $0.02 edge if both fill
  - Target combined cost of $0.995-$0.998 for deep markets, $0.990-$0.995 for thin markets
  - Size each side based on minimum depth at target price level
  - Emit Opportunity with OpportunityType.MAKER_REBALANCE
  - Need GTC order management (not FAK) — post and wait
  - Stale order cancellation: cancel unfilled orders after N seconds or if prices move
- **Key design:** This scanner produces PENDING opportunities that require lifecycle management (post → monitor → cancel/fill). Different from instant-fill arbs.
- **Tests:** Unit tests for edge calculation, maker pricing, size computation. Mock orderbook tests. Test cancellation logic.
- [ ] Not started

#### 2.2 — Maker order lifecycle manager
- **Files:** New `executor/maker_lifecycle.py`
- **Approach:** Track posted maker orders. On each cycle: check fills, cancel stale orders, repost at better prices if spread moved. Integrate with PnLTracker for fill events.
- **Tests:** Test state transitions: POSTED → FILLED, POSTED → CANCELLED → REPOSTED, POSTED → STALE → CANCELLED.
- [ ] Not started

#### 2.3 — Integrate maker into run.py
- **Files:** `run.py`, `scanner/models.py` (add MAKER_REBALANCE to OpportunityType), `executor/engine.py`
- **Approach:** Add maker scan phase after arb scanning. Maker runs independently — doesn't compete with arbs for capital. Use separate exposure bucket `max_maker_exposure`.
- **Tests:** Integration test: full pipeline with maker enabled finds maker opportunities on wide-spread markets.
- [ ] Not started

- **Status:** pending

**Phase 2 exit criteria:** Maker scanner identifies opportunities on wide-spread markets. Lifecycle manager tracks order state. GTC orders post and cancel correctly in paper mode.

---

### Phase 3: Resolution Sniping Scanner (NEW Revenue Stream)
**Theme:** Buy the winning side of nearly-resolved markets at < $1.

#### 3.1 — Design scanner/resolution.py
- **Files:** New `scanner/resolution.py`
- **Approach:** For markets within 0-60 minutes of resolution:
  - Check if outcome is publicly determinable (sports scores via free APIs, election results, etc.)
  - If YES is priced at < $0.97 and outcome is confirmed YES → buy YES (profit = $1 - price)
  - If NO is priced at < $0.97 and outcome is confirmed NO → buy NO
  - For markets where outcome is unknown: skip
  - Conservative approach: only snipe when price < $0.95 (5%+ edge) and resolution is < 30 min away
  - Use external data sources: sports scores (ESPN API), crypto prices (Binance), public data
- **Risk model:** Resolution sniping is nearly risk-free when outcome is confirmed. Kelly odds should be 0.95+.
- **Tests:** Mock market at $0.90 YES with confirmed outcome → verify opportunity emitted. Mock unresolved market → verify skip. Test edge cases: partially resolved, disputed.
- [ ] Not started

#### 3.2 — External outcome oracle
- **Files:** New `scanner/outcome_oracle.py`
- **Approach:** Modular outcome resolver. For sports: check live scores. For crypto: check current price vs threshold. For elections: check AP/Reuters feed. Returns `OutcomeStatus`: CONFIRMED_YES, CONFIRMED_NO, UNKNOWN, DISPUTED.
- **Tests:** Mock API responses for each data source. Verify correct status for each scenario.
- [ ] Not started

#### 3.3 — Integrate into run.py
- **Files:** `run.py`, `scanner/models.py` (add RESOLUTION_SNIPE to OpportunityType)
- **Approach:** Add resolution scan as a separate phase that runs on near-expiry markets (those filtered OUT by the current `min_hours` filter). Use the newly-removed filter to feed markets TO this scanner instead of throwing them away.
- **Tests:** Integration test verifying near-expiry markets flow to resolution scanner.
- [ ] Not started

- **Status:** pending

**Phase 3 exit criteria:** Resolution scanner identifies snipeable near-expiry markets. Outcome oracle resolves sports/crypto outcomes correctly. Integration flows markets to the right scanner.

---

### Phase 4: Partial NegRisk Value Scanner (NEW Revenue Stream)
**Theme:** Don't require buying ALL outcomes. Buy underpriced individual outcomes directionally.

#### 4.1 — Design scanner/value.py
- **Files:** New `scanner/value.py`
- **Approach:** For multi-outcome NegRisk events:
  - Compute implied probabilities from current ask prices
  - Identify outcomes where market probability is significantly below "fair" probability
  - "Fair" probability estimation approaches:
    a. Sum-normalization: if sum(asks) > 1.0 but one outcome is disproportionately cheap
    b. Historical anchor: if outcome was at $0.30 yesterday and is now $0.05 with no news
    c. Cross-event comparison: related events pricing the same outcome differently
  - Emit single-leg BUY opportunities with DIRECTIONAL risk (not risk-free arbs)
  - Size conservatively: use lower Kelly odds (0.30) since these aren't guaranteed arbs
  - Minimum edge requirement: 10% (implied probability vs market price)
- **Tests:** Test probability calculation from ask prices. Test edge detection. Test conservative sizing. Test that risk-free negrisk arbs (sum < 1) are NOT duplicated here (leave those to negrisk scanner).
- [ ] Not started

#### 4.2 — Integrate into run.py
- **Files:** `run.py`, `scanner/models.py` (add NEGRISK_VALUE to OpportunityType), `executor/engine.py`
- **Approach:** Value scanner runs on negrisk events where sum(asks) > 1.0 (i.e., no risk-free arb exists). Separate capital bucket. Lower priority in scorer than confirmed arbs.
- **Tests:** Integration test verifying value scanner only activates on non-arb events.
- [ ] Not started

- **Status:** pending

**Phase 4 exit criteria:** Value scanner finds underpriced outcomes in multi-outcome events. Sizing is conservative. No overlap with existing negrisk scanner.

---

### Phase 5: Stale-Quote WS Sniping (NEW Revenue Stream)
**Theme:** Exploit the 50-200ms information advantage from WebSocket price updates.

#### 5.1 — Design scanner/stale_quote.py
- **Files:** New `scanner/stale_quote.py`
- **Approach:** When WS feed shows a significant price move (>3%) in one token:
  - Immediately check the complementary token's book via REST
  - If the complementary token hasn't moved yet (stale quote), the combined cost may be < $1
  - This exploits the latency gap between WS updates and REST-polling competitors
  - Emit opportunity with short TTL (must execute within 500ms or discard)
  - Track "stale quote windows" — the average time between WS move and REST convergence
  - Backoff if stale quote window is consistently < 100ms (bots are too fast, no edge)
- **Key difference from spike scanner:** Spike scanner needs 2+ data points over 30 seconds. Stale quote sniping reacts to a single WS tick in < 100ms.
- **Tests:** Mock WS update showing 5% move in YES token. Mock REST showing stale NO price. Verify opportunity emitted. Mock both updated (no stale) → verify no opportunity.
- [ ] Not started

#### 5.2 — Fast-path integration into WS bridge
- **Files:** `client/ws_bridge.py`, `scanner/book_cache.py`
- **Approach:** Add callback in WS bridge: on significant price move, trigger stale-quote check immediately (not waiting for next scan cycle). This bypasses the 1-second scan interval for time-critical opportunities.
- **Tests:** Test callback fires on >3% move. Test callback doesn't fire on <3% move. Test rate limiting (max 10 checks/second).
- [ ] Not started

- **Status:** pending

**Phase 5 exit criteria:** Stale quote scanner detects and acts on price divergences within 200ms. WS callback bypasses scan cycle for time-critical opportunities.

---

### Phase 6: Integration Testing & Pipeline Validation
**Theme:** Verify the complete pipeline finds every viable opportunity.

#### 6.1 — End-to-end dry-run validation
- **Files:** `tests/test_pipeline_e2e.py`
- **Approach:** Run full `--dry-run` with crafted mock data containing:
  - 1 binary arb (YES+NO < $1)
  - 1 negrisk arb (sum asks < $1, 5 outcomes)
  - 1 near-resolution market (sports, confirmed outcome)
  - 1 wide-spread market (maker opportunity)
  - 1 underpriced negrisk outcome (value bet)
  - 1 WS price spike with stale complementary quote
  - Verify ALL 6 are detected and properly scored/sized
- **Tests:** The test IS the deliverable.
- [ ] Not started

#### 6.2 — Live dry-run performance benchmark
- Run `--dry-run` with real market data. Measure:
  - Opportunities found per cycle (target: 3-10x improvement over baseline)
  - Cycle time (target: < 10s)
  - Capital utilization (target: 40-60% of bankroll deployed)
  - Strategy mode distribution (target: AGGRESSIVE > 70% of cycles on Polygon)
- Document results in progress.md.
- [ ] Not started

- **Status:** pending

**Phase 6 exit criteria:** E2E test passes with all 6 opportunity types detected. Live dry-run shows 3-10x improvement in opportunity detection.

---

## Dependencies

```
Phase 1 (all items independent, can parallelize)
  |-- Phase 2 depends on: 1.4 (exposure limits), 1.1 (strategy mode)
  |-- Phase 3 depends on: 1.4 (min_hours removal feeds markets to resolution scanner)
  |-- Phase 4 depends on: 1.3 (slippage ceiling for better depth)
  |-- Phase 5 depends on: 1.1 (strategy mode for latency-sensitive ops)
       |-- Phase 6 depends on: ALL prior phases complete
```

Within phases 2-5, items CAN be parallelized (independent modules).

## Effort Estimate

| Phase | Items | Est. Lines | Description |
|-------|-------|-----------|-------------|
| 1     | 5     | ~300      | Parameter fixes + integration test |
| 2     | 3     | ~500      | Maker strategy + lifecycle |
| 3     | 3     | ~400      | Resolution sniping + outcome oracle |
| 4     | 2     | ~300      | Partial negrisk value scanner |
| 5     | 2     | ~350      | Stale-quote WS sniping |
| 6     | 2     | ~200      | Integration tests + benchmark |
| **Total** | **17** | **~2,050** | Full profit-max implementation |

## Key Questions
1. Should maker strategy use a separate capital bucket from arb trading? → **YES** — maker ties up capital for longer (GTC orders), shouldn't compete with instant arbs
2. How aggressive should resolution sniping be? → **Conservative first** — only snipe confirmed outcomes with >5% edge. Expand later.
3. Should partial negrisk bets have a separate risk limit? → **YES** — these are directional, not risk-free. Use 20% of max_total_exposure.
4. What's the minimum profitable edge for stale-quote sniping? → **2%** after fees. Below this, the timing advantage isn't reliable enough.

## Decisions Made

| Decision | Rationale |
|----------|-----------|
| Dollar-denominated gas thresholds | Polygon gwei != Ethereum gwei. $0.01/order is cheap regardless of gwei value |
| Kelly odds 0.65 for confirmed arbs | Fill probability ~85%. Previous 0.10 deployed 5x too little capital |
| Maker uses separate capital bucket | GTC orders tie up capital for seconds-minutes, shouldn't block instant arbs |
| Resolution oracle is modular | Different data sources per market type. Easy to add new sources |
| Value scanner sized at 0.30 Kelly odds | Directional risk, not guaranteed arb. Size conservatively |
| Stale-quote TTL of 500ms | WS→REST convergence window is typically 200-500ms. Staler than that = already arbitraged |
| Phase 1 all-parallel | All items independent. Maximum velocity. |

## Errors Encountered

| Error | Attempt | Resolution |
|-------|---------|------------|
| (none yet) | - | - |

## Notes
- Previous session baseline: 834 tests, 25 foundational items complete
- All new scanners follow existing patterns: emit `Opportunity`, integrate via `run.py` scan loop
- New `OpportunityType` values: MAKER_REBALANCE, RESOLUTION_SNIPE, NEGRISK_VALUE, STALE_QUOTE_ARB
- Each new scanner is its own module — modular, testable, independently deployable
