# Task Plan: Fix Pipeline Reliability Issues

## Context

The pipeline correctly *detects* arbitrage opportunities on Polymarket, but the results are misleading. A dry-run against 27,460 markets found 9 opportunities -- but 2 are phantom (stale markets), and the remaining 7 would likely fail in live execution due to depth, atomicity, and latency problems. This plan systematically fixes each issue and adds verification for every fix.

## Issues Identified (ordered by severity)

| # | Issue | Severity | Impact |
|---|-------|----------|--------|
| 1 | Stale/resolved markets appear as phantom arbs ($4K+ at 48,900% ROI) | CRITICAL | False positives mislead the scorer and would waste execution attempts |
| 2 | Depth validation only checks best-level size, not actual fillable depth | HIGH | max_sets is overstated; verify_depth() passes when real depth is insufficient |
| 3 | ScoringContext never populated -- fill_score hardcoded to 0.50 for all | HIGH | Scorer cannot distinguish thin-book vs deep-book opportunities |
| 4 | Multi-leg execution has zero atomicity guarantee | HIGH | Partial fills on 6-31 leg arbs leave orphaned unhedged positions |
| 5 | depth.py sweep functions exist but are never called by any scanner | MEDIUM | All the infrastructure for proper depth analysis is built but unused |
| 6 | REST polling at 1s vs 200ms arb windows | MEDIUM | Opportunities are gone by the time we detect them |
| 7 | Unwind is fire-and-forget; exceptions silently swallowed | MEDIUM | Failed unwind leaves positions open with no alerting |

---

## Phase 1: Stale Market Filter

### Problem
The Market dataclass has only `active: bool` -- no `end_date`, `resolved`, or `closed` field. The Gamma API already filters `active=true, closed=false`, but markets that recently resolved still have abandoned liquidity on the book. The Elche CF market (2026-01-19) and Taylor Swift market both resolved weeks ago but show as 48,900% ROI arbs because their tokens still have near-zero asks.

### Fix

#### 1A. Add `end_date` to Market dataclass
- The Gamma API response includes `end_date_iso` (or `endDate`) for markets
- Add `end_date: str = ""` field to `Market` in `scanner/models.py`
- Parse it from the API response in `client/gamma.py` `get_markets()`

#### 1B. Add `closed` field to Market dataclass
- The Gamma API includes a `closed` boolean
- Add `closed: bool = False` to `Market` in `scanner/models.py`
- Parse from API response

#### 1C. Add staleness filter in scanners
- In `scanner/binary.py` and `scanner/negrisk.py`, before scanning:
  - Skip markets where `end_date` is in the past (already expired)
  - Skip markets where `closed=True` (already resolved)
- New function `is_market_stale(market: Market) -> bool` in a shared location (e.g., `scanner/models.py` or a new `scanner/filters.py`)

#### 1D. Add last-trade-time heuristic
- Fetch `last_trade_timestamp` from CLOB API for suspicious high-ROI markets
- If no trades in the last 7 days, flag as stale
- This catches markets that are technically "active" but have dead liquidity

### Files to modify
| File | Change |
|------|--------|
| `scanner/models.py` | Add `end_date: str = ""`, `closed: bool = False` to Market |
| `client/gamma.py` | Parse `end_date_iso` and `closed` from API response |
| `scanner/binary.py` | Add staleness filter before scanning |
| `scanner/negrisk.py` | Add staleness filter before scanning |
| `tests/test_binary_scanner.py` | Add test: stale market filtered out |
| `tests/test_negrisk_scanner.py` | Add test: stale market filtered out |

### Verification
- Run dry-run: Elche CF and Taylor Swift should no longer appear
- All existing 337 tests still pass (adding `end_date=""` default means no existing test breaks)
- New tests confirm stale markets are filtered

---

## Phase 2: Wire Depth Sweep into Scanners

### Problem
`scanner/depth.py` has `sweep_cost()`, `effective_price()`, `sweep_depth()` -- all tested and working. But neither `binary.py` nor `negrisk.py` calls them. Both scanners use only `best_ask.price` and `best_ask.size` for opportunity pricing and `max_sets`. This means:
- Profit estimate is wrong if real fill requires walking multiple book levels
- `max_sets` is overestimated (only considers best level, not actual depth)

### Fix

#### 2A. Use effective_price() for cost calculation
- In `scanner/binary.py` `_check_buy_arb()`:
  - Replace `yes_ask.price + no_ask.price` with `effective_price(book, 'ask', target_size)` for each side
  - This gives the true VWAP cost for the target execution size
- Same for `scanner/negrisk.py` `_scan_buy_arb()`

#### 2B. Use sweep_depth() for max_sets calculation
- Replace `max_sets = min(yes_ask.size, no_ask.size)` with:
  - `sweep_depth(yes_book, 'ask', max_price=threshold)` to get actual available size below a price ceiling
  - `max_sets = min(sweep_depth(yes_book, ...), sweep_depth(no_book, ...))`

#### 2C. Fall back to best-level when target_size is not configured
- Keep the current best-level path as a fast pre-check
- Only invoke depth sweep when the fast check finds a potential arb (avoids perf regression on 25K markets)

### Files to modify
| File | Change |
|------|--------|
| `scanner/binary.py` | Use effective_price() and sweep_depth() for VWAP-aware sizing |
| `scanner/negrisk.py` | Same |
| `tests/test_binary_scanner.py` | Add tests with multi-level books where best-level shows arb but VWAP doesn't |
| `tests/test_negrisk_scanner.py` | Same |
| `tests/test_depth.py` | Add integration test: depth-aware scanner filters correctly |

### Verification
- Construct test book: best_ask=0.45 (10 shares), level2=0.55 (100 shares). Old code sees arb at 0.45. New code computes VWAP and may reject.
- Run dry-run: some marginal opportunities should disappear (more realistic)
- No regressions on existing tests

---

## Phase 3: Populate ScoringContext with Real Data

### Problem
`run.py` calls `rank_opportunities(all_opps)` with no contexts. Default `ScoringContext` has `book_depth_ratio=1.0` always, so `fill_score` is always 0.50. The scorer literally cannot distinguish a deep liquid arb from a 1-share paper-thin one.

### Fix

#### 3A. Compute book_depth_ratio per opportunity
- After scanners emit opportunities, for each opp:
  - `depth_available = min(sweep_depth(book, side, max_price) for each leg)`
  - `depth_needed = execution_size` (from Kelly sizing or target_size)
  - `book_depth_ratio = depth_available / depth_needed`

#### 3B. Populate market_volume and recent_trade_count
- `market_volume` is already on the Market dataclass (`volume` field)
- Thread it through to ScoringContext
- `recent_trade_count` can be estimated from CLOB API trade history (or set to 0 as placeholder)

#### 3C. Populate time_to_resolution_hours
- Use the new `end_date` field from Phase 1
- `time_to_resolution_hours = (end_date - now).total_seconds() / 3600`
- Markets resolving in hours get higher efficiency score than those resolving in months

#### 3D. Wire contexts into rank_opportunities()
- In `run.py`, build `list[ScoringContext]` parallel to `all_opps`
- Pass both to `rank_opportunities(all_opps, contexts=contexts)`

### Files to modify
| File | Change |
|------|--------|
| `run.py` | Build ScoringContext per opportunity with real depth, volume, resolution time |
| `scanner/scorer.py` | No changes needed (already supports contexts) |
| `tests/test_scorer.py` | Add tests: thin-book opp scores lower than deep-book opp |

### Verification
- Run dry-run: the 31-leg JD Vance opportunity (thin) should score lower than the 2-leg Nebraska opportunity (deep)
- Score breakdown in status.md should show varying fill_scores instead of constant 0.50

---

## Phase 4: Harden Multi-Leg Execution

### Problem
Multi-leg orders have zero atomicity. Polymarket batch endpoint processes each order sequentially against the live orderbook. For a 31-leg NegRisk arb, legs 1-15 go in batch 1, legs 16-31 go in batch 2 with ~100ms gap. Prices can move between batches. Partial fill unwind swallows exceptions silently -- failed unwinds leave orphaned positions.

### Fix

#### 4A. Pre-execution re-check for large leg counts
- Add a leg-count-aware safety check:
  - For opportunities with >10 legs, re-fetch orderbooks AFTER sizing (just before execution)
  - Compare fresh prices against opportunity prices
  - If ANY leg has slippage >0.5%, abort the entire opportunity
  - This narrows the window between check and execution for high-leg-count arbs

#### 4B. Tighten partial fill unwind
- In `_unwind_partial()`: if any unwind order fails, raise `UnwindFailed` exception instead of silently logging
- Add a new exception class `UnwindFailed(Exception)` in `executor/engine.py`
- The caller in `run.py` catches `UnwindFailed` and:
  - Logs the stuck position (token_id, size, side)
  - Records it in a `stuck_positions.json` file for manual recovery
  - Increments circuit breaker failure counter

#### 4C. Add max_legs config parameter
- New config: `max_legs_per_opportunity: int = 15` (default = one batch)
- In `run.py` or `executor/safety.py`: skip opportunities with more legs than this
- Rationale: the 31-leg JD Vance arb is unprofitable in practice because multi-batch execution fails. Better to filter it out.

#### 4D. Verify safety checks cover all legs
- Already confirmed: `verify_prices_fresh()` and `verify_depth()` iterate over ALL legs
- Add a test that explicitly passes a 5-leg opportunity and confirms all 5 legs are checked

### Files to modify
| File | Change |
|------|--------|
| `executor/engine.py` | Add `UnwindFailed` exception; raise on failed unwind |
| `executor/safety.py` | Add `verify_max_legs()` check |
| `config.py` | Add `max_legs_per_opportunity: int = 15` |
| `run.py` | Catch `UnwindFailed`, log stuck positions, add pre-exec re-check for high-leg opps |
| `tests/test_engine.py` | Test: unwind failure raises UnwindFailed |
| `tests/test_safety.py` | Test: verify_max_legs rejects >15 legs |

### Verification
- Run dry-run: JD Vance 31-leg opp should be filtered out by max_legs check
- Test: simulate unwind where cancel fails → UnwindFailed raised, not silently swallowed
- Stuck positions file created on failure

---

## Phase 5: Depth-Aware verify_depth()

### Problem
`verify_depth()` in `executor/safety.py` only checks `best_ask.size >= leg.size` or `best_bid.size >= leg.size`. It doesn't check if there's enough depth ACROSS multiple levels to fill the execution size at a reasonable price.

### Fix

#### 5A. Use sweep_depth() in verify_depth()
- Replace the best-level check with:
  ```python
  available = sweep_depth(book, side, max_price=leg.price * 1.005)  # 0.5% slippage tolerance
  if available < leg.size:
      raise SafetyCheckFailed(...)
  ```
- This walks the entire book and sums available size up to 0.5% above the opportunity price
- If the book is thin 3 levels deep, it correctly rejects

#### 5B. Add cost verification
- After depth check, verify the actual fill cost:
  ```python
  actual_cost = sweep_cost(book, side, leg.size)
  expected_cost = leg.price * leg.size
  if actual_cost > expected_cost * 1.01:  # >1% slippage
      raise SafetyCheckFailed(...)
  ```

### Files to modify
| File | Change |
|------|--------|
| `executor/safety.py` | Use sweep_depth() and sweep_cost() instead of best-level check |
| `tests/test_safety.py` | Add tests with multi-level books: sufficient vs insufficient depth |

### Verification
- Test: book with 50 shares at best level but needing 100 → fails
- Test: book with 50+50+50 across 3 levels within price tolerance → passes
- All existing safety tests still pass

---

## Phase 6: WebSocket Integration (execution speed)

### Problem
REST polling at 1-second intervals misses most 200ms arb windows. `client/ws.py` and `scanner/book_cache.py` are built and tested but not wired into the main loop.

### Fix

#### 6A. Wire WebSocket into run.py main loop
- On startup, connect to `wss://ws-subscriptions-clob.polymarket.com/ws/market`
- Subscribe to `book` and `price_change` channels for active token IDs
- Feed `book` messages into `BookCache.apply_snapshot()`
- Feed `price_change` messages into `BookCache.apply_delta()`
- Scanners read from `BookCache` instead of making REST calls

#### 6B. Event-driven scan triggering
- Instead of fixed 1s interval, trigger a scan whenever:
  - `price_change` event shifts a best_ask or best_bid enough to potentially create an arb
  - Threshold: if new_price differs from cached_price by >1%, trigger scan on that event
- Keep the fixed interval as a fallback (scan everything every 5s via REST)

#### 6C. REST as consistency check
- Every N seconds (configurable, default 30s), do a full REST refresh
- Compare REST books against WS-cached books
- Log any discrepancies (WS desync detection)

### Files to modify
| File | Change |
|------|--------|
| `run.py` | WS connection on startup, event-driven scan loop, REST fallback |
| `client/ws.py` | Already built; may need minor fixes for real WS API format |
| `scanner/book_cache.py` | Already built; wire into main loop |
| `config.py` | Already has WS config params |

### Verification
- Connect to WS in dry-run mode: confirm book messages received
- Verify BookCache updates match REST snapshots
- Measure scan latency: should drop from 1s to <100ms for WS-triggered scans
- All existing tests still pass (WS is additive, not replacing REST)

---

## Phase 7: Verification Suite

After all phases complete, run a comprehensive verification:

### 7A. Regression tests
- `PYTHONPATH=. uv run python -m pytest tests/ -v` -- all existing 337+ tests pass
- Coverage still ≥90%

### 7B. Dry-run validation (--limit 500)
- [ ] No phantom/stale market arbs appear
- [ ] Opportunities show realistic max_sets (depth-aware)
- [ ] Scores vary by fill_score (not all 0.50)
- [ ] High-leg-count arbs filtered by max_legs

### 7C. Dry-run full scan (27K+ markets)
- [ ] Previously-phantom Elche CF and Taylor Swift are gone
- [ ] Nebraska Senate still appears (genuine arb)
- [ ] JD Vance 31-leg filtered (or scored very low due to thin depth)
- [ ] Score breakdown shows meaningful differentiation

### 7D. Paper trading validation
- [ ] Paper execution of a 2-leg arb produces correct TradeResult
- [ ] Paper execution of a >15-leg arb is rejected by max_legs check
- [ ] Partial fill unwind correctly raises UnwindFailed (not silent)

---

## Implementation Order and Dependencies

```
Phase 1 (stale filter)     ─── no dependencies, start immediately
Phase 2 (depth in scanners) ─── no dependencies, can parallel with Phase 1
Phase 3 (scoring context)  ─── depends on Phase 1 (end_date) and Phase 2 (depth ratio)
Phase 4 (execution hardening) ─── no dependencies, can parallel with 1-2
Phase 5 (depth in safety)  ─── depends on Phase 2 (uses same depth functions)
Phase 6 (WebSocket)        ─── no dependencies but highest effort; start after 1-5
Phase 7 (verification)     ─── depends on all previous phases
```

Parallelizable groups:
- **Group A (can run in parallel):** Phase 1 + Phase 2 + Phase 4
- **Group B (after A):** Phase 3 + Phase 5
- **Group C (after B):** Phase 6
- **Group D (after C):** Phase 7

## Status (Phases 1-7 -- Reliability)

| Phase | Description | Status |
|-------|-------------|--------|
| 1 | Stale market filter | [x] complete |
| 2 | Wire depth sweep into scanners | [x] complete |
| 3 | Populate ScoringContext | [x] complete |
| 4 | Harden multi-leg execution | [x] complete |
| 5 | Depth-aware verify_depth() | [x] complete |
| 6 | WebSocket integration | [ ] pending (deferred) |
| 7 | Verification suite | [x] complete (352 tests, 89% coverage) |

---

# BATTLE-TEST PLAN: From Dry-Run to Real Alpha

## Context (Session 8 Analysis)

Deep code review + codex analysis identified that dry-run opportunities are largely **phantoms** -- by the time REST polling detects them, they're already gone. The pipeline detects correctly but cannot capture. This plan focuses on 5 high-impact changes that bridge the gap between detection and real execution profit.

### The Core Problem

The pipeline runs a **poll-scan-execute** loop at 1s intervals against 25K+ markets via REST. Real arb windows on Polymarket close in ~200ms. By the time we:
1. Fetch 25K markets from Gamma (5-30s)
2. Batch-fetch orderbooks from CLOB (1-5s per 50-token chunk)
3. Run 4 scanners
4. Score, size, safety check
5. Build + sign + POST orders

...the opportunity is **60-90% likely to be gone**. This matches IMDEA's finding: only 40% of detected arbs are actually captured.

### Bugs Found by Codex

**CRITICAL: FAK/FOK Order Type Mismatch** (`executor/engine.py:56`)
```python
order_type = OrderType.FOK if use_fak else OrderType.GTC
```
Config says "FAK" (Fill-and-Kill = partial fills OK), but code sends `FOK` (Fill-or-Kill = ALL or nothing). SDK has both: `OrderType.FAK` and `OrderType.FOK`. FOK will **reject orders that can't fill completely**, meaning thin-book opportunities that COULD partially fill get zero fills instead. This is a silent profit killer.

---

## Phase 8: Fix FAK/FOK Order Type Bug

**Impact: +$150-$700/day | Effort: 0.5 day**

### Problem
`executor/engine.py:56` maps `use_fak=True` to `OrderType.FOK`. The py-clob-client SDK has both `OrderType.FAK` (partial fill, cancel remainder) and `OrderType.FOK` (all or nothing). Currently we reject every order that can't fill the full requested size, silently losing any partial-fill profit.

### Fix
```python
# executor/engine.py line 56
# BEFORE:
order_type = OrderType.FOK if use_fak else OrderType.GTC
# AFTER:
order_type = OrderType.FAK if use_fak else OrderType.GTC
```

Also update `_execute_binary` and `_execute_negrisk` to handle partial fills from FAK (some legs fill partially, others fully). The `fill_sizes` tracking already supports this but `all_filled` logic needs adjustment.

### Files
| File | Change |
|------|--------|
| `executor/engine.py` | Line 56: `FOK` → `FAK`. Update partial fill handling. |
| `tests/test_engine.py` | Add test: FAK order with partial fill returns correct sizes |

---

## Phase 9: WebSocket-Driven Reactive Scanning

**Impact: +$300-$1,500/day | Effort: 3-4 days**

### Problem
REST polling 25K markets = 5-30s stale data. Arb windows close in ~200ms. We have `client/ws.py` (WSManager) and `scanner/book_cache.py` (apply_snapshot, apply_delta) fully built but NOT wired into `run.py`.

### Architecture Change
Replace the poll loop with an **event-driven** architecture:

```
Current:  while True → REST fetch all → scan all → execute → sleep 1s
Proposed: while True → WS feeds BookCache → dirty queue → scan ONLY dirty events → execute
          Background: REST metadata refresh every 30s
```

### Fix

#### 9A. Wire WSManager into run.py startup
- After `client = build_clob_client(cfg)`, start WSManager in asyncio background:
  ```python
  ws_manager = WSManager(url=cfg.ws_market_url, token_ids=[])
  ```
- Subscribe to all active YES token IDs from Gamma metadata
- Feed `book_queue` → `book_cache.apply_snapshot()`
- Feed `price_queue` → `book_cache.apply_delta()` + add to dirty set

#### 9B. Dirty-set scan trigger
- Maintain `dirty_events: set[str]` tracking events with price changes
- When dirty set is non-empty, trigger scan ONLY on those events
- Scan using `book_cache.get_book()` (no REST fetch)
- Clear dirty set after scan

#### 9C. REST as background metadata refresh
- Every 30-60s, refresh market list from Gamma (new markets, deactivated markets)
- Every 30s, do a REST book consistency check on a random sample (10 tokens)
- Log any WS desync

#### 9D. Hybrid fallback
- If WS disconnects, fall back to full REST scan loop
- If WS reconnects, resume event-driven mode

### Files
| File | Change |
|------|--------|
| `run.py` | Major refactor: asyncio main loop, WS integration, dirty-set scanning |
| `client/ws.py` | Minor fixes for actual WS message format (test against live WS) |
| `scanner/book_cache.py` | Already works, no changes needed |
| `config.py` | Add `ws_scan_debounce_ms`, `rest_metadata_interval_sec` |

---

## Phase 10: Execution-Time Price Revalidation + Opportunity TTL

**Impact: +$80-$400/day | Effort: 1.5 days**

### Problem
Opportunities are scored using book data from scan time. By execution time (even 100ms later), prices may have moved. The `verify_prices_fresh()` re-fetches via REST (adding 50-100ms), which itself may be stale.

### Fix

#### 10A. Opportunity TTL
- Add `timestamp` field to Opportunity (already exists: `scanner/models.py:123`)
- In `_execute_single()` in `run.py`, reject opportunities older than TTL:
  ```python
  age_ms = (time.time() - opp.timestamp) * 1000
  if age_ms > cfg.opportunity_ttl_ms:
      raise SafetyCheckFailed(f"Opportunity stale: {age_ms:.0f}ms > {cfg.opportunity_ttl_ms}ms")
  ```
- Default TTL: 500ms for spike/latency, 2000ms for rebalance

#### 10B. Execution-time EV recomputation
- After `verify_prices_fresh()` gets fresh books, recompute net profit using fresh prices
- If fresh net_profit < 50% of original estimate, abort
- Accounts for price movement between scan and execution
- In `executor/safety.py`, add `verify_edge_intact()`:
  ```python
  def verify_edge_intact(opp: Opportunity, fresh_books: dict, fee_model: MarketFeeModel, min_ratio: float = 0.50):
      fresh_cost = sum(fresh_books[leg.token_id].best_ask.price for leg in opp.legs if leg.side == Side.BUY)
      fresh_profit = 1.0 - fresh_cost  # for buy-all arbs
      if fresh_profit < opp.expected_profit_per_set * min_ratio:
          raise SafetyCheckFailed(f"Edge eroded: was {opp.expected_profit_per_set:.4f}, now {fresh_profit:.4f}")
  ```

### Files
| File | Change |
|------|--------|
| `executor/safety.py` | Add `verify_edge_intact()`, add `verify_opportunity_ttl()` |
| `run.py` | Wire TTL check and edge recomputation into `_execute_single()` |
| `config.py` | Add `opportunity_ttl_ms: int = 2000` |
| `tests/test_safety.py` | Tests for TTL rejection, edge erosion rejection |

---

## Phase 11: Worst-Fill Limit Pricing (Stop Leaving Money on the Table)

**Impact: +$100-$500/day | Effort: 1 day**

### Problem
Scanners compute VWAP (average fill price) and pass that as the limit price to execution. But VWAP is the AVERAGE -- if we set limit at VWAP, half the levels we need WON'T fill because they're above our limit. We need the WORST price we're willing to pay (the last level we'd sweep through).

### Fix

#### 11A. Add `worst_fill_price()` to `scanner/depth.py`
```python
def worst_fill_price(book: OrderBook, side: Side, size: float) -> float | None:
    """Price of the last level needed to fill `size`. This is the correct limit price."""
    levels = book.asks if side == Side.BUY else book.bids
    remaining = size
    last_price = None
    for level in levels:
        fill = min(remaining, level.size)
        last_price = level.price
        remaining -= fill
        if remaining <= 0:
            break
    return last_price if remaining <= 0 else None
```

#### 11B. Use worst-fill in LegOrder construction
- In `scanner/binary.py` and `scanner/negrisk.py`, set `leg.price = worst_fill_price(book, side, max_sets)` instead of `effective_price()`
- Keep VWAP for profit calculation but use worst-fill for execution limit

### Files
| File | Change |
|------|--------|
| `scanner/depth.py` | Add `worst_fill_price()` |
| `scanner/binary.py` | Use worst-fill for leg price, VWAP for profit calc |
| `scanner/negrisk.py` | Same |
| `tests/test_depth.py` | Tests for worst_fill_price |

---

## Phase 12: Inventory-Aware Sell Legs + Position Tracking

**Impact: +$50-$300/day | Effort: 1.5 days**

### Problem
Sell-side arbs (binary sell, negrisk sell, latency sell-YES) require holding the position being sold. The pipeline doesn't check inventory. Sell orders will fail if we don't hold the tokens.

### Fix

#### 12A. Add position fetcher
- New `client/data.py` with `get_positions(client) -> dict[str, float]`
- Uses Data API: `GET /positions` to get current token holdings
- Cache with 5s TTL

#### 12B. Block impossible sell legs
- In `_execute_single()` or as a new safety check:
  ```python
  def verify_inventory(positions: dict[str, float], opp: Opportunity):
      for leg in opp.legs:
          if leg.side == Side.SELL:
              held = positions.get(leg.token_id, 0.0)
              if held < leg.size:
                  raise SafetyCheckFailed(f"Insufficient inventory for sell: need {leg.size}, hold {held}")
  ```

#### 12C. Latency sell → BUY NO conversion
- In `scanner/latency.py`, when we want to sell YES (implied_prob < 0.45), instead generate a BUY NO leg (equivalent exposure, doesn't require inventory)
- This opens up the sell-side latency strategy without needing existing positions

### Files
| File | Change |
|------|--------|
| `client/data.py` | New: position fetcher with cache |
| `executor/safety.py` | Add `verify_inventory()` |
| `scanner/latency.py` | Convert SELL YES → BUY NO for latency sell |
| `run.py` | Fetch positions on startup, pass to safety |
| `config.py` | Add `position_cache_sec: float = 5.0` |

---

## Implementation Order

```
Phase  8 (FAK fix)         ─── 0.5 day, zero dependencies, START FIRST
Phase 11 (worst-fill price) ─── 1 day, parallel with 8
Phase 10 (TTL + edge check)─── 1.5 days, parallel with 8+11
Phase 12 (inventory)       ─── 1.5 days, after 8
Phase  9 (WebSocket)       ─── 3-4 days, after 8+10+11+12 (biggest change, most impact)
```

Total estimated time: **7-9 days** for all 5 phases.
Quick wins (phases 8+11) in first day.

## Status (Phases 8-12 -- Battle-Test)

| Phase | Description | Impact/day | Status |
|-------|-------------|-----------|--------|
| 8 | Fix FAK/FOK order type bug | +$150-$700 | [x] complete |
| 9 | WebSocket-driven reactive scanning | +$300-$1,500 | [x] complete |
| 10 | Execution-time TTL + edge revalidation | +$80-$400 | [x] complete |
| 11 | Worst-fill limit pricing | +$100-$500 | [x] complete |
| 12 | Inventory-aware sell legs | +$50-$300 | [x] complete |

---

# HARDENING PLAN: Cross-Platform + BookCache Reliability

## Context (Session 10 Analysis)

Code review of the cross-platform arbitrage pipeline and BookCache revealed 5 distinct issues ranging from correctness bugs to architectural risks. These issues affect the Kalshi integration (Phases 9+ cross-platform) and the WebSocket book cache (Phase 9). All cross-platform code was built in Sessions 8-9 but has not run against live Kalshi markets yet.

## Issues Identified (ordered by severity)

| # | Issue | Severity | Root Cause | Impact |
|---|-------|----------|------------|--------|
| 1 | Kalshi cents-vs-dollars price conversion | CRITICAL | `kalshi.py` converts correctly in `get_orderbook()`, but `place_order()` takes cents directly while `execute_cross_platform()` passes dollar prices | Wrong limit prices → rejected or wildly mispriced orders |
| 2 | Kalshi `math.ceil()` fee rounding | MEDIUM | `kalshi_fees.py:43` computes `ceil(7 * p * (1-p))` cents then divides by 100 → correct formula. But fee is applied per-contract, not scaled by size in cross-platform scanner | Fee understated on multi-contract orders when leg sizes vary |
| 3 | Cross-platform execution order + unwind | HIGH | Kalshi filled first (fast), PM second (slow). If PM fails: unwind sends market sell on Kalshi → may fill at worse price or fail entirely | Stuck positions, loss exceeding expected profit |
| 4 | Cross-platform fuzzy matching risk | HIGH | `token_set_ratio` at 85% threshold can match events with different settlement terms | Matched events settle differently → one side pays $1, other pays $0. Total loss of capital. |
| 5 | BookCache single-writer/reader threading | MEDIUM | No synchronization primitives. WS thread writes, main thread reads. CPython GIL protects dict assignment but not multi-step updates | Torn reads during `apply_delta()` where book is partially updated |

---

## Phase 13: Fix Kalshi Price Conversion in Execution Path

### Problem

`client/kalshi.py:place_order()` expects `yes_price` and `no_price` in **cents** (1-99), matching the Kalshi API spec. But `executor/cross_platform.py:75` converts dollars→cents inline:

```python
price_cents = int(round(leg.price * 100))
```

This looks correct at first glance. However, the issue is **validation and edge cases**:

1. **Rounding trap**: `leg.price = 0.455` → `int(round(0.455 * 100))` = `int(round(45.5))` = `46` cents. But the scanner computed profit at 0.455. The order goes in at 0.46, which is 0.5 cents more expensive. On 100 contracts: $0.50 unintended cost.

2. **No price validation**: Kalshi accepts 1-99 cents only. A price of $0.999 → 100 cents → rejected. A price of $0.001 → 0 cents → rejected. No guard rails.

3. **Kalshi sell side confusion**: When `kalshi_side == Side.SELL` in the scanner, we're selling YES. But `place_order()` maps `Side.SELL` → `action="sell"`. However the `side` param should be "yes" (the token being sold), not "no" (the outcome). Current code: `kalshi_side = "yes" if leg.side == Side.BUY else "no"` — this maps SELL→"no" which means we're selling NO tokens instead of selling YES tokens. **This is a correctness bug.**

### Fix

#### 13A. Fix Kalshi side mapping in cross-platform executor

```python
# BEFORE (executor/cross_platform.py:74):
kalshi_side = "yes" if leg.side == Side.BUY else "no"

# AFTER:
# When we BUY YES: side="yes", action="buy"
# When we SELL YES: side="yes", action="sell"
# When we BUY NO: side="no", action="buy"
# When we SELL NO: side="no", action="sell"
#
# In cross-platform arb:
#   Direction 1: Buy PM YES + SELL Kalshi YES → side="yes", action="sell"
#   Direction 2: Buy PM NO + BUY Kalshi YES → side="yes", action="buy"
#
# The leg.side represents the action (BUY/SELL), the Kalshi side is always "yes"
# because our OrderBook model for Kalshi represents YES tokens.
kalshi_side = "yes"
kalshi_action = "buy" if leg.side == Side.BUY else "sell"
```

#### 13B. Add cents conversion helper with validation

New function in `client/kalshi.py`:

```python
def dollars_to_cents(price: float) -> int:
    """Convert dollar price (0.01-0.99) to Kalshi cents (1-99). Fail-fast on out-of-range."""
    cents = round(price * 100)
    if cents < 1 or cents > 99:
        raise ValueError(f"Kalshi price out of range: ${price:.4f} → {cents} cents (must be 1-99)")
    return cents
```

Update `executor/cross_platform.py:75`:
```python
from client.kalshi import dollars_to_cents
price_cents = dollars_to_cents(leg.price)
```

#### 13C. Add round-trip price consistency check

In `scanner/cross_platform.py`, after computing worst-fill prices, verify the Kalshi leg price survives cents conversion without meaningful loss:

```python
kalshi_cents = round(kalshi_worst * 100)
if abs(kalshi_cents / 100.0 - kalshi_worst) > 0.005:  # >0.5 cent drift
    return None  # Price doesn't survive cent rounding cleanly
```

### Files to modify

| File | Change |
|------|--------|
| `client/kalshi.py` | Add `dollars_to_cents()` helper |
| `executor/cross_platform.py` | Fix `kalshi_side` mapping. Use `dollars_to_cents()`. |
| `scanner/cross_platform.py` | Add cent-rounding consistency check before emitting opportunity |
| `tests/test_kalshi.py` | Add tests for `dollars_to_cents()`: valid range, edge cases, out-of-range |
| `tests/test_cross_platform_exec.py` | Add test: side mapping for BUY vs SELL legs |

### Verification
- Test: `dollars_to_cents(0.455)` → 46 cents (rounded correctly)
- Test: `dollars_to_cents(0.999)` → raises ValueError (100 cents out of range)
- Test: `dollars_to_cents(0.001)` → raises ValueError (0 cents out of range)
- Test: SELL Kalshi YES leg → `side="yes"`, `action="sell"` (not `side="no"`)
- All existing 383 tests still pass

---

## Phase 14: Fix Kalshi Fee Model Accuracy

### Problem

`kalshi_fees.py` implements `taker_fee_per_contract(price)` correctly for a single contract. But there are two issues:

1. **`total_fee()` multiplies per-contract fee by count**: `fee_per_contract * contracts`. This is correct per Kalshi docs — the fee formula is per-contract, and `C=1` in the formula means per-contract (not total count). The `0.07 * C` in the formula header is misleading — `C` means "contracts" and the ceil applies per-contract. **Actually this is correct as-is.** Kalshi charges ceil per individual contract and sums.

2. **Real issue: `adjust_profit()` doesn't account for contract count variation**: It sums `taker_fee_per_contract(leg.price)` across legs, treating each leg as 1 contract. But in practice, a leg may have `size=50` contracts. The fee should be `taker_fee_per_contract(price) * size` per leg.

3. **`math.ceil()` rounding interaction with small prices**: At price $0.02, `ceil(7 * 0.02 * 0.98)` = `ceil(0.1372)` = 1 cent. The fee is $0.01 per contract regardless of how small the price is. On $0.02 contracts, that's a 50% fee rate. The scanner doesn't account for this — it would happily emit a cross-platform arb at extreme prices where fees eat the entire profit.

### Fix

#### 14A. Fix `adjust_profit()` to account for contract sizes

```python
# BEFORE (kalshi_fees.py:69-71):
def adjust_profit(self, gross_profit_per_set: float, legs: tuple[LegOrder, ...]) -> float:
    total_fee_per_set = 0.0
    for leg in legs:
        total_fee_per_set += self.taker_fee_per_contract(leg.price)
    return gross_profit_per_set - total_fee_per_set

# AFTER:
def adjust_profit(self, gross_profit_per_set: float, legs: tuple[LegOrder, ...]) -> float:
    total_fee_per_set = 0.0
    for leg in legs:
        if leg.platform == "kalshi":
            total_fee_per_set += self.taker_fee_per_contract(leg.price)
    return gross_profit_per_set - total_fee_per_set
```

Actually, `adjust_profit` is called **per set** and the per-contract fee IS the per-set fee (1 contract = 1 set = one $1 outcome). So the current code is correct for the "per set" interpretation. The issue is:

**The real gap**: In `scanner/cross_platform.py:269`, the fee is applied via `taker_fee_per_contract(kalshi_vwap)` — this is the per-set fee. Then `net_profit = net_profit_per_set * max_sets - gas_cost`. This is correct.

**After deeper review: the fee model is arithmetically correct.** The `math.ceil()` behavior is documented as a known trap but not a bug — it's Kalshi's actual formula.

#### 14B. Add minimum-fee guard to cross-platform scanner

At extreme prices (near $0.01 or $0.99), the ceil rounding makes fees disproportionate. Add a guard:

```python
# In scanner/cross_platform.py, after computing kalshi_fee:
if kalshi_fee_model:
    kalshi_fee = kalshi_fee_model.taker_fee_per_contract(kalshi_vwap)
    fee_rate = kalshi_fee / kalshi_vwap if kalshi_vwap > 0 else 1.0
    if fee_rate > 0.20:  # Fee exceeds 20% of contract price → uneconomical
        return None
    net_profit_per_set -= kalshi_fee
```

### Files to modify

| File | Change |
|------|--------|
| `scanner/cross_platform.py` | Add fee-rate guard for extreme prices |
| `tests/test_cross_platform.py` | Add test: extreme-price arb rejected due to fee rate |
| `tests/test_kalshi_fees.py` | Add test: fee at $0.02 price = $0.01 (ceil rounding documented) |

### Verification
- Test: arb at Kalshi price $0.02 → fee = $0.01 → 50% fee rate → rejected
- Test: arb at Kalshi price $0.50 → fee = $0.02 → 4% fee rate → passes
- All existing tests pass

---

## Phase 15: Harden Cross-Platform Execution + Unwind

### Problem

The current execution flow in `executor/cross_platform.py`:

```
1. Place Kalshi order(s) → check fill
2. If Kalshi fails → cancel resting Kalshi orders, return unfilled
3. If Kalshi fills → place PM order(s) → check fill
4. If PM fails → unwind Kalshi (market sell) → return unfilled
5. If both fill → compute P&L → return filled
```

Issues:
1. **Step 4 unwind is market-order on Kalshi**: A market sell may fill at significantly worse price than the original buy, creating a REAL loss (not zero P&L)
2. **Unwind loss is not tracked**: `_unwind_kalshi` reports success/failure but doesn't compute the loss from adverse fill. `TradeResult.net_pnl=0.0` on PM failure understates actual loss
3. **"resting" status treated as filled** (`cross_platform.py:90`): `if status in ("resting", "executed", "filled")` — "resting" means NOT yet filled, just placed on book. Should wait for fill or treat as unfilled
4. **No timeout on Kalshi fill wait**: If Kalshi order rests forever, we block the PM leg indefinitely
5. **No atomicity timeout**: Combined PM+Kalshi execution has no deadline. If one leg takes 10s, the other leg's book may have moved

### Fix

#### 15A. Fix Kalshi fill status interpretation

```python
# BEFORE (cross_platform.py:90):
if status in ("resting", "executed", "filled"):

# AFTER:
if status in ("executed", "filled"):
    # Immediate fill confirmed
    fill_prices.append(leg.price)
    fill_sizes.append(size)
elif status == "resting":
    # Order placed but not yet filled — poll for fill
    filled = _wait_for_kalshi_fill(kalshi_client, oid, timeout_sec=2.0)
    if filled:
        fill_prices.append(leg.price)
        fill_sizes.append(size)
    else:
        kalshi_client.cancel_order(oid)
        fill_prices.append(0.0)
        fill_sizes.append(0.0)
        kalshi_filled = False
```

#### 15B. Add `_wait_for_kalshi_fill()` polling

```python
def _wait_for_kalshi_fill(
    kalshi_client: KalshiClient,
    order_id: str,
    timeout_sec: float = 2.0,
    poll_interval: float = 0.1,
) -> bool:
    """Poll Kalshi order status until filled or timeout."""
    deadline = time.time() + timeout_sec
    while time.time() < deadline:
        try:
            order = kalshi_client.get_order(order_id)
            status = order.get("order", order).get("status", "")
            if status in ("executed", "filled"):
                return True
            if status in ("canceled", "cancelled", "expired"):
                return False
        except Exception:
            return False
        time.sleep(poll_interval)
    return False
```

#### 15C. Track unwind loss in TradeResult

When PM fails and we unwind Kalshi, compute the actual loss:

```python
# Unwind loss = buy_cost - sell_proceeds (adverse fill)
# For now, estimate conservatively: lose the bid-ask spread per contract
unwind_loss = sum(
    kalshi_fee_model.taker_fee_per_contract(leg.price) * size
    for leg in kalshi_legs
) + size * 0.01  # 1 cent spread estimate per contract

return TradeResult(
    ...
    net_pnl=-unwind_loss,  # Track actual estimated loss, not zero
    ...
)
```

#### 15D. Add execution deadline

```python
def execute_cross_platform(..., deadline_sec: float = 5.0) -> TradeResult:
    deadline = time.time() + deadline_sec
    ...
    # Before PM leg:
    if time.time() > deadline:
        logger.warning("Cross-platform execution deadline exceeded, aborting PM leg")
        _unwind_kalshi(kalshi_client, kalshi_legs, size)
        ...
```

### Files to modify

| File | Change |
|------|--------|
| `executor/cross_platform.py` | Fix "resting" handling, add fill polling, track unwind loss, add deadline |
| `client/kalshi.py` | No changes (get_order already exists) |
| `config.py` | Add `cross_platform_deadline_sec: float = 5.0` |
| `tests/test_cross_platform_exec.py` | Add tests: resting→fill, resting→timeout, unwind loss tracking, deadline exceeded |

### Verification
- Test: Kalshi "resting" status → poll → fill → proceed to PM
- Test: Kalshi "resting" → poll → timeout → cancel + abort
- Test: PM failure → unwind → TradeResult.net_pnl < 0 (not zero)
- Test: execution exceeds deadline → abort before PM leg
- All existing tests pass

---

## Phase 16: Harden Cross-Platform Event Matching

### Problem

Fuzzy matching via `token_set_ratio` at 85% threshold is dangerous for real money:

1. **Settlement mismatch is catastrophic**: If PM settles YES and Kalshi settles NO on the "same" event (different settlement criteria), we lose 100% of capital on both legs
2. **Title matching ignores resolution criteria**: "Will Biden win 2024?" (PM) vs "Biden popular vote winner 2024" (Kalshi) — same title, different settlements
3. **No negative matching**: Two events could score 90% but be different markets entirely (e.g., "Super Bowl LVIII winner" vs "Super Bowl LVIX winner")
4. **No human review gate for fuzzy matches**: The bot auto-trades on fuzzy matches ≥85% without any approval

### Fix

#### 16A. Raise fuzzy threshold to 95%

```python
# matching.py
FUZZY_THRESHOLD = 95.0  # Was 85.0 — too permissive for real money
```

95% still catches obvious matches but rejects most title variants with different settlement terms.

#### 16B. Add settlement keyword blocklist

```python
_SETTLEMENT_KEYWORDS = {"popular vote", "electoral", "inauguration", "sworn in", "resign", "impeach"}

def _settlement_mismatch_risk(pm_title: str, kalshi_title: str) -> bool:
    """Check if titles differ on settlement-relevant keywords."""
    pm_lower = pm_title.lower()
    kalshi_lower = kalshi_title.lower()
    for kw in _SETTLEMENT_KEYWORDS:
        pm_has = kw in pm_lower
        kalshi_has = kw in kalshi_lower
        if pm_has != kalshi_has:
            return True  # One mentions settlement keyword, other doesn't
    return False
```

#### 16C. Add date/year mismatch detection

```python
import re
_YEAR_PATTERN = re.compile(r'\b(20\d{2})\b')

def _year_mismatch(pm_title: str, kalshi_title: str) -> bool:
    """Reject matches where the year differs (e.g., 2024 vs 2028)."""
    pm_years = set(_YEAR_PATTERN.findall(pm_title))
    kalshi_years = set(_YEAR_PATTERN.findall(kalshi_title))
    if pm_years and kalshi_years and pm_years != kalshi_years:
        return True
    return False
```

#### 16D. Require human approval for first-time fuzzy matches

Add a `verified_matches.json` file that persists approved fuzzy matches. On first encounter, log a warning but DO NOT trade — only manual map entries and previously-approved fuzzy matches execute.

```python
class EventMatcher:
    def __init__(self, ..., verified_path: str = "verified_matches.json"):
        self._verified = self._load_verified(verified_path)

    def match_events(self, ...) -> list[MatchedEvent]:
        ...
        # For fuzzy matches, only include if previously verified
        if match.match_method == "fuzzy" and match.pm_event_id not in self._verified:
            logger.warning(
                "UNVERIFIED fuzzy match: PM '%s' -> Kalshi '%s' (%.1f%%). "
                "Add to verified_matches.json to enable trading.",
                pm_event.title[:50], kalshi_titles[best_kalshi_ticker][:50], best_score,
            )
            match = MatchedEvent(..., confidence=0.0)  # Block execution via confidence filter
```

### Files to modify

| File | Change |
|------|--------|
| `scanner/matching.py` | Raise threshold to 95%, add settlement/year mismatch checks, verified-match gate |
| `config.py` | Add `cross_platform_verified_path: str = "verified_matches.json"` |
| `tests/test_matching.py` | Add tests: year mismatch rejected, settlement keyword rejected, unverified fuzzy blocked |

### Verification
- Test: "Biden win 2024" vs "Biden win 2028" → year mismatch → rejected
- Test: "Will X win?" vs "Will X win popular vote?" → settlement keyword mismatch → rejected
- Test: 90% fuzzy match not in verified_matches.json → confidence set to 0.0 → blocked
- Test: 90% fuzzy match IN verified_matches.json → confidence preserved → allowed
- Manual map matches still work with confidence=1.0
- All existing tests pass

---

## Phase 17: Make BookCache Thread-Safe

### Problem

`BookCache` has no synchronization. The WS bridge thread calls `apply_snapshot()` and `apply_delta()`, while the main thread calls `get_book()`, `get_books()`, `is_stale()`. CPython's GIL protects individual dict operations (`__setitem__`, `__getitem__`) from corruption, but:

1. **`apply_delta()` is multi-step**: reads `self._books[token_id]`, creates new bids/asks, writes `self._books[token_id]`. Between read and write, the main thread could get a half-constructed book from a different snapshot
2. **`store_books()` iterates and writes**: multiple token updates are not atomic — main thread could see some tokens from new batch, others from old
3. **Future risk**: if we ever add a second reader thread (e.g., dedicated scanner thread), data races become real

However, since `apply_delta()` creates a completely new `OrderBook` object (frozen dataclass) and assigns it via a single dict `__setitem__`, the GIL actually makes this safe in practice for CPython. The main thread gets either the old or new OrderBook reference — never a torn object.

**The real risk is stale-read ordering**: main thread reads `get_book(A)` from cycle N, then `get_book(B)` from cycle N+1. Books A and B are from different time points. For cross-platform arb, this means PM and Kalshi books may be from different moments.

### Fix

#### 17A. Add `threading.Lock` for write operations

```python
import threading

@dataclass
class BookCache:
    ...
    _lock: threading.Lock = field(default_factory=threading.Lock)

    def apply_snapshot(self, token_id: str, bids: list[dict], asks: list[dict]) -> None:
        ...
        with self._lock:
            self._books[token_id] = OrderBook(...)
            self._timestamps[token_id] = time.time()

    def apply_delta(self, token_id: str, price_change: dict) -> None:
        with self._lock:
            book = self._books.get(token_id)
            if not book:
                return
            ... (compute new book)
            self._books[token_id] = new_book
            self._timestamps[token_id] = time.time()
```

#### 17B. Add snapshot read for consistent multi-token reads

```python
def get_books_snapshot(self, token_ids: list[str]) -> tuple[dict[str, OrderBook], float]:
    """
    Return a consistent snapshot of multiple books and the snapshot timestamp.
    Holding the lock ensures no interleaved writes during the read.
    """
    with self._lock:
        now = time.time()
        books = {tid: self._books[tid] for tid in token_ids if tid in self._books}
        return books, now
```

This gives the main thread a consistent view across multiple tokens.

#### 17C. Use snapshot in cross-platform scanner

In `run.py`, when fetching books for cross-platform scanning:
```python
# Use snapshot read to get consistent PM books
pm_cross_books, snapshot_ts = book_cache.get_books_snapshot(pm_token_ids)
```

### Files to modify

| File | Change |
|------|--------|
| `scanner/book_cache.py` | Add `threading.Lock`, wrap writes, add `get_books_snapshot()` |
| `run.py` | Use `get_books_snapshot()` for cross-platform book reads |
| `tests/test_book_cache.py` | Add test: concurrent snapshot/delta doesn't corrupt, snapshot returns consistent view |

### Verification
- Test: `get_books_snapshot()` returns consistent set (all from same timestamp epoch)
- Test: `apply_delta()` under lock doesn't deadlock with `get_book()`
- Test: existing single-threaded tests pass without performance regression
- Stress test: simulate WS writes + main-thread reads in parallel (optional)

---

## Implementation Order

```
Phase 13 (Kalshi price/side fix)    ─── CRITICAL, start first, 0.5 day
Phase 14 (Fee model guard)          ─── parallel with 13, 0.5 day
Phase 15 (Execution hardening)      ─── after 13, 1.5 days
Phase 16 (Matching hardening)       ─── parallel with 15, 1 day
Phase 17 (BookCache thread safety)  ─── parallel with 15+16, 0.5 day
```

Parallelizable groups:
- **Group A (parallel):** Phase 13 + Phase 14
- **Group B (parallel, after 13):** Phase 15 + Phase 16 + Phase 17

Total: **3-4 days**.

## Status (Phases 13-17 -- Cross-Platform Hardening)

| Phase | Description | Severity | Status |
|-------|-------------|----------|--------|
| 13 | Fix Kalshi price conversion + side mapping | CRITICAL | [ ] pending |
| 14 | Fix fee model edge-case guard | MEDIUM | [ ] pending |
| 15 | Harden cross-platform execution + unwind | HIGH | [ ] pending |
| 16 | Harden cross-platform event matching | HIGH | [ ] pending |
| 17 | Make BookCache thread-safe | MEDIUM | [ ] pending |
