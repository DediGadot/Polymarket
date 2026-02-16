# Dry-Run Deep Dive Report (5 Minutes) -- 2026-02-16

## Scope
- Objective: run the full pipeline in dry-run mode for 5 minutes, then evaluate what is truly actionable vs. what is not.
- Command:
  - `timeout -k 30s 5m uv run python run.py --dry-run --limit 1200 --json-log logs/deep_dive_5m_20260216_115259.jsonl`
- Artifacts:
  - `logs/deep_dive_5m_20260216_115259.stdout.log`
  - `logs/deep_dive_5m_20260216_115259.jsonl`
  - `logs/run_20260216_115302.log`
  - `status.md`

## Agent Team Deep Dive

### 1) Runtime/Infra Agent

What worked:
- Process shut down gracefully on timeout with full summary (`no forced kill`), unlike earlier unstable runs.
- Throughput in steady-state is high once caches are warm (sub-second cycles for most loops).

What did not:
- Book prefetch has periodic long stalls:
  - Cycle 1: `60.493s` prefetch (`logs/run_20260216_115302.log:48`)
  - Cycle 63: `20.073s` prefetch (`logs/run_20260216_115302.log:12018`)
  - Cycle 134: `20.065s` prefetch (`logs/run_20260216_115302.log:25430`)
- Final cycle (134) is shutdown-biased:
  - After long prefetch, all scanners returned `0.00s (0 opps)` due shutdown flag (`logs/run_20260216_115302.log:25432`, `logs/run_20260216_115302.log:25439`).

Interpretation:
- Most cycle speed is excellent, but periodic cache-refresh cliffs create blind windows and distort per-cycle signal.

### 2) Strategy Coverage Agent

What worked:
- Correlation scanner is consistently productive:
  - 133/134 cycles with non-zero correlation output.
  - Violations per cycle stayed in a tight band (`175-181`).
- Maker scanner is also consistently non-zero (133/134 cycles).

What did not:
- Zero opportunity output from:
  - `binary`, `negrisk`, `latency`, `spike`, `resolution`, `cross-platform` (all run, all `0 opps` every cycle).
- Cross-platform path is effectively disabled in dry-run due missing tradeable mappings:
  - `Skipping cross-platform scan in dry-run (no manual/verified mappings)` (example: `logs/run_20260216_115302.log:53`).
- OFI is inactive in dry-run:
  - `WebSocket disabled in dry-run mode (REST only)` (`logs/run_20260216_115302.log:36`).
  - `OFI inactive in dry-run (WS disabled)` repeated.

Interpretation:
- Current dry-run opportunity generation is effectively a 2-strategy system: correlation + maker.

### 3) Execution-Readiness Agent

Session totals (`logs/run_20260216_115302.log:25455` onward):
- Cycles: `134`
- Markets scanned (cumulative): `287,430`
- Opportunities: `13,297`
- Executable lane: `11,361` (85.4%)
- Research lane: `1,936` (14.6%)
- Actionable now (taker BUY): `5,361` (40.3% of all, 47.2% of executable)
- Maker candidates: `606`
- Sell/inventory-dependent: `0`

Actionable gate rejection profile (aggregated from debug logs):
- `low_fill_score`: `5,382`
- `low_persistence`: `12`
- `maker`: `606`
- First-cycle example: `low_fill_score=41, low_persistence=12, maker=2` (`logs/run_20260216_115302.log:124`)

Interpretation:
- Fill quality, not persistence, is the dominant blocker between executable and actionable.

### 4) Market-Quality/Risk Agent

What worked:
- No hard runtime errors.
- Safety gating around cross-platform mappings is conservative (good for avoiding bad matches).

What did not:
- Opportunity set is highly repetitive:
  - 13,297 displayed rows but only 757 unique `(type,event,profit,roi,legs)` signatures.
  - Duplicate ratio: `94.3%`.
  - 77.3% of cycles had identical signature sets to the immediately previous cycle.
- Concentration risk is high:
  - Top 5 events account for ~58.3% of theoretical correlation profit.
  - Top 10 events account for ~82.7%.
- NegRisk churn/noise is high:
  - `5,985` repeated `SKIP incomplete outcome group` logs across `45` groups (same incompletes repeating each cycle).
  - Example repeated group: Gavin Newsom 2028 pool with `43/128` markets (`logs/run_20260216_115302.log:64`).
- External data reliability:
  - CoinGecko 429 fallback warnings seen 4x (example: `logs/run_20260216_115302.log:492`).

Interpretation:
- Reported theoretical edge is materially inflated by repeated near-identical signals and concentration.

## What Is Working and Actionable

1. Correlation scanner is reliably finding recurring BUY-side candidates at production cadence.
2. Actionable lane is non-empty and substantial (`5,361 taker BUY`) in this run.
3. Maker scanner contributes stable, low-profit resting opportunities.
4. Runtime shutdown path is now stable enough to preserve summary and checkpoints.

## What Is Not Working / Needs Improvement

1. Cycle quality is highly duplicated; unique information per cycle is low.
2. Non-correlation scanners contribute zero useful output in this profile.
3. Periodic prefetch stalls (20-60s) create blind windows and timeout artifacts.
4. Actionability bottleneck is overwhelmingly fill-score rejection.
5. NegRisk incomplete-group rescanning creates heavy repeated noise.
6. Analytics tooling drift:
   - `benchmark.evs` and `benchmark.cross_platform` currently return zeros on this log format, so automated scorecards are unreliable.

## Prioritized Improvement Plan (How)

### P0 (Immediate, high impact)

1. Make shutdown-aware cycle accounting.
- If `shutdown_requested` trips during prefetch/scan, mark cycle as `aborted` instead of `No opportunities found`.
- Prevents false negatives like cycle 134.

2. Add opportunity de-dup window to session accounting.
- Fingerprint opportunities (e.g., `type + reason_code + event_id + rounded prices`) and suppress repeats across a short TTL (e.g., 15-60s) for dashboard/session stats.
- Keep raw stream for audit, but separate `new` vs `repeat`.

3. Add negative cache for repeated NegRisk incompletes.
- Cache `event_id -> incomplete` until markets timestamp changes.
- Skip repeated deep checks/log spam for unchanged incomplete groups.

### P1 (High)

1. Flatten prefetch latency spikes.
- Split large prefetch into chunks (e.g., 300-500 tokens) and spread across cycles.
- Or use asynchronous prefetch refresh so scan loop is not blocked by 20-60s fetch cliffs.
- Make prefetch cancellable mid-flight on shutdown.

2. Fill-score calibration workflow.
- Emit fill-score histogram quantiles per cycle and per strategy.
- Run threshold sensitivity (`correlation_actionable_min_fill_score`) in replay to maximize precision@actionable, not raw count.

3. Expose scanner health KPIs.
- Alert if scanner has `0 opps` for N consecutive cycles (binary/negrisk/latency/spike/resolution currently all dead in this run).

### P2 (Medium)

1. Improve dry-run mode semantics.
- Surface `WS disabled` and `OFI inactive` clearly in the top-of-run mode banner.
- Explicitly tag cross-platform as `config enabled but runtime skipped (no verified mappings)`.

2. Harden external data backoff.
- Global cooldown + jitter for CoinGecko fallback path to reduce repeated 429 churn.

3. Fix analysis toolchain drift.
- Update `benchmark.evs` parsers to current display/log format or consume structured collector outputs directly.
- Add a CI guard so parser and log schema cannot silently diverge.

## Bottom Line

- This 5-minute dry-run demonstrates a functioning high-throughput research/executable funnel, but practical readiness is overstated by heavy duplication and concentration in correlation signals.
- The key upgrades are not new scanners first; they are signal de-duplication, prefetch latency flattening, fill-score calibration, and cleaner shutdown/metrics semantics.
