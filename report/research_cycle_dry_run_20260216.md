# Dry-Run Research Cycle Report (2026-02-16)

## Scope
- Objective: run a full pipeline research cycle for 10 minutes in dry-run mode and assess actionability.
- Command run:
  - `timeout -k 30s 10m uv run python run.py --dry-run --limit 1200 --json-log logs/research_cycle_20260216_081606.jsonl`
- Primary logs:
  - `logs/run_20260216_081609.log`
  - `logs/research_cycle_20260216_081606.jsonl`
  - `status.md`

## Run Outcome
- Start: `2026-02-16 08:16:09`
- Last log event: `2026-02-16 08:26:11`
- Window observed: ~10m02s
- Timeout behavior: SIGTERM fired at 10m, process was later force-killed (`exit 137`) before graceful shutdown summary.

## Key Metrics (Observed)
- Completed cycles: `3` (`cycle 4` started but did not finish)
- Markets scanned per completed cycle: `2,147` (`1,200` binary + `947` negRisk in `54` events)
- Opportunities found (completed cycles): `255`
- Lane split (completed cycles):
  - Executable lane: `0`
  - Research lane: `255`
- Opportunity type mix in completed cycles: `100% correlation_arb`
- Correlation violations reported by scanner: `80`, `85`, `90` (+ `91` in partial cycle 4)
- Scan duration (completed cycles):
  - Values: `61.1s`, `61.2s`, `250.2s`
  - Avg: `124.17s`, median: `61.2s`, max: `250.2s`
- Fetch duration (completed cycles):
  - Values: `13.5s`, `9.8s`, `9.3s`
  - Avg: `10.87s`
- Book prefetch cost (`BookService`):
  - `3347` tokens each cycle
  - ~`60.3s` to `60.6s` per cycle

## Warning/Error Signals (Real)
- `UNVERIFIED fuzzy match` warnings: `15` total (`8` unique mappings, many repeated)
- `Fuzzy match REJECTED (year mismatch)`: `2`
- `Kalshi 429 rate limited`: `1`
- `CoinGecko 429` (MATIC/USD fallback to default): `1`
- `Kalshi market cache not ready`: `2` (early cycles, warm-up period)
- negRisk filtering churn:
  - `SKIP incomplete outcome group`: `188`
  - `SKIP ... exceeds max 15 legs`: `8`
- OFI tracking: `tracked_tokens=0` in completed cycles (dry-run path did not feed OFI)

## Actionability Classification

### Actionable + Real
1. **Pipeline currently yields zero executable opportunities** in dry-run (`0 executable / 255 research`).
2. **Cross-platform matching pipeline is blocked by verification gating** (repeated unverified matches + confidence skips).
3. **Cycle latency is unstable and sometimes extreme** (`250.2s`) once cross-platform matching/warnings ramp up.
4. **External API rate-limit handling needs hardening** (Kalshi/CoinGecko 429 events).
5. **Shutdown resilience is weak for long cycles** (forced kill before graceful summary after SIGTERM).
6. **negRisk scan spends significant work on non-tradable/incomplete groups** (high skip volume).

### Real but Research-Only (Not Immediately Tradable)
1. **Correlation violations are consistently detected** (80-91 per cycle), so there is research signal.
2. **High theoretical ROI/profit values** in correlation output are not currently executable in this runtime path.

### Not Actionable As-Is / Likely Misleading for Execution
1. Raw `correlation_arb` theoretical P&L in dry-run should not be treated as deployable edge until execution semantics are enabled and validated.
2. Repeated duplicate/near-duplicate correlation lines inflate perceived opportunity volume.

## Required Pipeline Improvements

### P0 (Immediate)
1. **Reduce cross-platform churn in dry-run**:
   - Skip expensive platform orderbook fetch/matching unless there are verified mappings.
   - Add negative cache for rejected/unverified mapping pairs to prevent repeated retries each cycle.
2. **Strengthen rate-limit strategy**:
   - Backoff with jitter + longer cooldown cache for 429s (Kalshi/CoinGecko).
   - Avoid immediate re-query patterns during same cycle.
3. **Improve cooperative cancellation**:
   - Add shutdown checks inside long scanner loops (cross-platform + matching) so SIGTERM exits cleanly without force kill.

### P1 (High)
1. **Make executable lane non-empty by design**:
   - Add KPI gates: fail run-quality if executable lane remains `0` for N cycles.
   - Rebalance scanner priority/quotas so research output cannot fully starve executable candidates.
2. **Cut baseline scan cost**:
   - `BookService` prefetch is ~60s every cycle for 3347 tokens; add incremental or event-driven token subset prefetching.
3. **Improve negRisk prefiltering**:
   - Pre-eliminate groups likely to be incomplete or over max-leg threshold before deep scan passes.

### P2 (Medium)
1. **Expose per-scanner runtime telemetry** in structured logs (scanner elapsed, skipped counts, match quality).
2. **Clarify OFI behavior in dry-run**:
   - Either feed OFI from available source in dry-run or explicitly disable/report it as inactive.
3. **Summary correctness for long sessions**:
   - `ScanTracker.max_opportunities=100` can under-report session totals in shutdown summary for longer runs.

## Bottom Line
- The 10-minute run produced **strong research signal but zero executable signal**.
- Main blockers are **cross-platform mapping quality, rate-limit resilience, and scan-time efficiency**.
- Until those are fixed, the pipeline will continue to look profitable in theory while being weak in immediate execution readiness.
