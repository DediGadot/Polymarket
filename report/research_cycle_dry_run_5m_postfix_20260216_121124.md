# Post-Fix Dry-Run Analysis (5 Minutes) -- 2026-02-16

## Scope
- Command:
  - `timeout -k 30s 5m uv run python run.py --dry-run --limit 1200 --json-log logs/deep_dive_5m_postfix_20260216_121124.jsonl`
- Logs:
  - `logs/deep_dive_5m_postfix_20260216_121124.stdout.log`
  - `logs/run_20260216_121127.log`

## What Was Implemented

1. Shutdown-aware cycle abort handling
- Added cycle-abort logic so timeout/shutdown during prefetch/scan does not get recorded as a fake "No opportunities found" cycle.
- Evidence in run log:
  - `Cycle 135 aborted: shutdown requested during prefetch` (`logs/run_20260216_121127.log:18034`)
  - Shutdown summary now reports completed cycles cleanly (`logs/run_20260216_121127.log:18036`, `logs/run_20260216_121127.log:18044`)

2. Session-level opportunity de-dup accounting
- Added short-window fingerprint tracking in `ScanTracker` and summary fields for unique vs repeated opportunities.
- Evidence:
  - `Unique opportunities: 797 (dedup window 30.0s)` (`logs/run_20260216_121127.log:18048`)
  - `Repeated opportunities: 11253` (`logs/run_20260216_121127.log:18049`)
  - Profit split:
    - unique: `$37,760.03` (`logs/run_20260216_121127.log:18054`)
    - repeated: `$657,920.46` (`logs/run_20260216_121127.log:18055`)

3. NegRisk incomplete-group negative cache
- Added run-level cache of known incomplete negRisk groups keyed to market snapshot timestamp so unchanged incomplete groups are skipped upstream.
- Practical effect: repeated `SKIP incomplete outcome group` churn eliminated in this run (0 matches in `logs/run_20260216_121127.log`).

## Post-Fix Session Summary

- Completed cycles: `134` (`logs/run_20260216_121127.log:18044`)
- Opportunities found: `12,050` (`logs/run_20260216_121127.log:18047`)
- Executable lane: `$609,797.38 (9,648 opps)` (`logs/run_20260216_121127.log:18056`)
- Research lane: `$85,883.10 (2,402 opps)` (`logs/run_20260216_121127.log:18057`)
- Actionable now (taker BUY): `$555,178.49 (4,884 opps)` (`logs/run_20260216_121127.log:18058`)
- Maker candidates: `$113.43 (268 opps)` (`logs/run_20260216_121127.log:18059`)
- Type mix:
  - `correlation_arb`: `11,782`
  - `maker_rebalance`: `268` (`logs/run_20260216_121127.log:18064`, `logs/run_20260216_121127.log:18065`)

## Pre vs Post (Key Deltas)

- Fake terminal "No opportunities found" cycle removed:
  - Pre-run had one (`logs/run_20260216_115302.log:25443`)
  - Post-run has none; cycle is explicitly aborted (`logs/run_20260216_121127.log:18034`)
- NegRisk incomplete skip noise:
  - Pre: heavy repeated skip logs (examples around `logs/run_20260216_115302.log:25247`)
  - Post: 0 skip lines in runtime log
- Actionable gate pressure improved:
  - Pre total `low_fill_score` rejects: `5,382`
  - Post total `low_fill_score` rejects: `4,484`
  - Cycle-134 pattern: `low_fill_score=33, maker=2` (`logs/run_20260216_121127.log:17924`)

Note:
- Throughput still has periodic prefetch cliffs:
  - cycle 1 prefetch: `60.668s` (`logs/run_20260216_121127.log:48`)
  - cycle 63 prefetch: `20.059s` (`logs/run_20260216_121127.log:8438`)

## Deep Dive: 20 Found Opportunities (Cycle 134)

Source rows: `logs/run_20260216_121127.log:17932` to `logs/run_20260216_121127.log:17951`

### 1) Rank 1 -- Will Trump deport less than 250,000?
- Metrics: `correlation_arb`, `$137.33`, `32.62% ROI`, `score 0.65`, `2 legs`
- Session persistence: event appeared `526` times; exact signature in `21` cycles.
- Domain interpretation:
  - This is a structural threshold-consistency mispricing candidate (likely complements or related deportation-band constraints).
  - Strong score + decent ROI implies a relatively balanced edge with good notional.
- Actionability note:
  - Candidate is in executable lane for this cycle context, but cycle-level fill gating still rejected 33 others, so depth stability remains the critical live filter.

### 2) Rank 2 -- MegaETH market cap (FDV) > $2B one day after launch?
- Metrics: `$104.21`, `23.69% ROI`, `score 0.65`
- Persistence: `402` event occurrences; exact signature in `82` cycles.
- Interpretation:
  - High persistence indicates a stable cross-market inconsistency, not a one-off print.
  - ROI is moderate but recurring, making it a repeated capture candidate if execution costs are controlled.

### 3) Rank 3 -- Will xAI release a dLLM by June 30?
- Metrics: `$180.06`, `72.40% ROI`, `score 0.65`
- Persistence: `402` occurrences; exact signature in `72` cycles.
- Interpretation:
  - Higher ROI than #2 with strong persistence suggests durable disagreement between linked xAI markets.
  - Likely more sensitive to quote freshness than lower-ROI spreads.

### 4) Rank 4 -- Will MetaMask launch a token by June 30?
- Metrics: `$121.60`, `49.24% ROI`, `score 0.65`
- Persistence: `402` occurrences; exact signature in `82` cycles.
- Interpretation:
  - Repeated, medium-high edge candidate.
  - This belongs to the same "token-launch narrative cluster" as other crypto release markets.

### 5) Rank 5 -- Will the US confirm aliens exist before 2027?
- Metrics: `$492.50`, `49.25% ROI`, `score 0.65`
- Persistence: `536` occurrences; exact signature in `62` cycles.
- Interpretation:
  - Very large gross edge candidate with high recurrence.
  - Large projected profit often indicates either broad structural mismatch or thin-side book asymmetry; execution realism checks are crucial.

### 6) Rank 6 -- Will the U.S. invade Iran before 2027?
- Metrics: `$138.44`, `40.84% ROI`, `score 0.65`
- Persistence: `402` occurrences; exact signature in `82` cycles.
- Interpretation:
  - Geopolitical consistency trade with robust recurrence.
  - This event appears again at rank 7 with identical metrics, indicating duplicate relation-path emission.

### 7) Rank 7 -- Will the U.S. invade Iran before 2027? (duplicate signature)
- Metrics: identical to rank 6
- Interpretation:
  - Same economics, separate emitted candidate line.
  - This is exactly the kind of duplicate inflation the new de-dup session metrics were designed to expose.

### 8) Rank 8 -- Will JD Vance win the 2028 US Presidential Election?
- Metrics: `$1136.72`, `113.67% ROI`, `score 0.65`
- Persistence: event seen `134` times; exact signature in `82` cycles.
- Interpretation:
  - Largest dollar edge in this cycle.
  - High ROI plus large net profit indicates highly asymmetric linked-market pricing; this is the flagship candidate for deeper fill simulation.

### 9) Rank 9 -- StandX FDV above $800M one day after launch?
- Metrics: `$129.67`, `24.53% ROI`, `score 0.65`
- Persistence: `402` occurrences; exact signature in `83` cycles.
- Interpretation:
  - Stable moderate edge in the same token-FDV cohort.
  - This is more "repeatable spread capture" than explosive one-off edge.

### 10) Rank 10 -- Will Trump resign before 2027?
- Metrics: `$219.48`, `21.95% ROI`, `score 0.65`
- Persistence: `670` event occurrences; exact signature in `82` cycles.
- Interpretation:
  - Extremely frequent event family in this run.
  - Moderate ROI but large recurrence means this market family heavily drives session totals.

### 11) Rank 11 -- Will xAI release a dLLM by June 30? (alt structure)
- Metrics: `$90.01`, `26.57% ROI`, `score 0.65`
- Persistence: exact signature in `52` cycles.
- Interpretation:
  - Secondary xAI structure with lower notional/ROI than rank 3.
  - Suggests multiple relation transforms against the same base event.

### 12) Rank 12 -- StandX FDV above $800M one day after launch? (alt structure)
- Metrics: `$83.58`, `14.54% ROI`, `score 0.64`
- Persistence: exact signature in `72` cycles.
- Interpretation:
  - Lower edge variant, still very persistent.
  - Typically where fill realism can decide whether this survives actionable filtering.

### 13) Rank 13 -- Will Trump deport less than 250,000? (alt structure)
- Metrics: `$72.97`, `18.47% ROI`, `score 0.64`
- Persistence: exact signature in `11` cycles.
- Interpretation:
  - Shorter-lived variant than rank 1, likely from a narrower relation combination.
  - Less persistent signatures are higher risk for quote drift.

### 14) Rank 14 -- OpenAI $1t+ IPO before 2027?
- Metrics: `$423.60`, `222.56% ROI`, `score 0.64`
- Persistence: `402` occurrences; exact signature in `73` cycles.
- Interpretation:
  - Extremely high ROI signal with strong recurrence.
  - This is a prime candidate for stress-testing slippage assumptions because theoretical edge is very large.

### 15) Rank 15 -- StandX FDV above $800M one day after launch? (third structure)
- Metrics: `$70.42`, `11.98% ROI`, `score 0.64`
- Persistence: exact signature in `82` cycles.
- Interpretation:
  - Third recurring variant on same event.
  - Indicates relation-family multiplicity; useful for alpha, but also a source of duplicate exposure if not grouped.

### 16) Rank 16 -- Will the U.S. invade Iran before 2027? (lower edge branch)
- Metrics: `$57.26`, `13.63% ROI`, `score 0.63`
- Persistence: exact signature in `62` cycles.
- Interpretation:
  - Same macro thesis as ranks 6/7 with lower intensity.
  - Often functions as fallback edge when top quotes compress.

### 17) Rank 17 -- Will Trump deport less than 250,000? (lower edge branch)
- Metrics: `$47.99`, `9.40% ROI`, `score 0.63`
- Persistence: exact signature in `21` cycles.
- Interpretation:
  - Lower-margin version of rank 1.
  - More vulnerable to taker fees and micro-slippage.

### 18) Rank 18 -- Will MetaMask launch a token by June 30? (lower edge branch)
- Metrics: `$48.61`, `23.44% ROI`, `score 0.63`
- Persistence: exact signature in `82` cycles.
- Interpretation:
  - Stable middle-tier member of the MetaMask relation set.
  - Good candidate for historical fill-quality calibration due repetition.

### 19) Rank 19 -- Will Trump resign before 2027? (lower edge branch)
- Metrics: `$42.08`, `9.88% ROI`, `score 0.62`
- Persistence: exact signature in `62` cycles.
- Interpretation:
  - Recurring but narrower spread branch.
  - Could be deprioritized when capital is scarce versus higher-ROI siblings.

### 20) Rank 20 -- Will the US confirm aliens exist before 2027? (lower edge branch)
- Metrics: `$41.81`, `19.03% ROI`, `score 0.62`
- Persistence: exact signature in `72` cycles.
- Interpretation:
  - Lower-dollar companion to rank 5.
  - High recurrence makes it analytically valuable for model validation and slippage realism checks.

## 20-Opportunity Takeaways

1. The top 20 are entirely `correlation_arb` and all are 2-leg structures in this cycle.
2. Many top entries are repeated event families with multiple relation transforms (or duplicate signatures), confirming why de-dup accounting is essential.
3. High-ROI outliers (e.g., OpenAI/StandX/JD Vance branches) are likely where theoretical-vs-executable drift is largest; they should be first targets for stricter fill simulation.
4. Cycle-level gating still rejects a meaningful block (`low_fill_score=33`) even in this high-signal cycle (`logs/run_20260216_121127.log:17924`), so execution realism remains the main frontier.

## Next Iteration (Recommended)

1. Add relation-family grouping in correlation output to prevent duplicate notional exposure (same event, near-identical legs).
2. Add a "top-N unique actionable" dashboard view (fingerprint-grouped), not raw row count.
3. For the top recurring signatures above, replay with stricter depth assumptions and compute realized fill-risk adjusted EV.
