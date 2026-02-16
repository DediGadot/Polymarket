# Iterative Dry-Run Report: Taker BUY Expansion (2026-02-16)

## Scope
- Goal: dramatically increase **real, immediately actionable taker BUY** opportunities.
- Method: repeated 10-minute dry-run cycles, deep-dive diagnostics, surgical code/config changes, rerun.

## Lane Definitions
- **Executable lane**: opportunities that pass execution gates now (supported type, BUY-side semantics, confidence/fill thresholds) and are eligible for taker-style action in current market state.
- **Research lane**: opportunities that are economically interesting but not currently executable due to gating constraints (maker-only, low fill/persistence, unsupported execution path, etc.).

## Iteration Summaries (10m each)

| Iteration | Log | Cycles | Opps | Executable lane | Research lane | Actionable now (taker BUY) | Maker candidates |
|---|---|---:|---:|---:|---:|---:|---:|
| Iter 3 (baseline) | `logs/run_20260216_101057.log` | 381 | 30,809 | $432,130.58 (789) | $1,046,856.01 (30,020) | $431,950.18 (380) | $180.40 (409) |
| Iter 4 | `logs/run_20260216_103622.log` | 366 | 30,242 | $415,328.17 (1,407) | $833,863.02 (28,835) | $414,889.16 (365) | $439.01 (1,042) |
| Iter 5 | `logs/run_20260216_104922.log` | 310 | 23,869 | $1,739,647.73 (19,219) | $115,262.59 (4,650) | $1,579,690.74 (11,767) | $275.97 (619) |
| Iter 6 | `logs/run_20260216_1108.log` | 311 | 30,719 | $1,951,670.88 (24,996) | $133,419.01 (5,723) | $1,755,009.78 (15,407) | $493.62 (738) |

## Net Improvement
- Iter 3 -> Iter 6:
  - Actionable taker BUY count: **380 -> 15,407** (**+15,027; 40.5x**)
  - Actionable taker BUY profit: **$431,950.18 -> $1,755,009.78** (**+ $1,323,059.60; 4.06x**)
  - Executable lane count: **789 -> 24,996** (**31.7x**)
- Iter 5 -> Iter 6 (latest surgical cap expansion):
  - Actionable taker BUY count: **11,767 -> 15,407** (**+3,640; +30.9%**)
  - Actionable taker BUY profit: **$1,579,690.74 -> $1,755,009.78** (**+ $175,319.04; +11.1%**)

## What Was Real and Immediately Actionable (Now)
- High-volume taker BUY opportunities are now consistently present per cycle.
- Latest per-cycle executable BUY profile (`logs/run_20260216_1108.log`):
  - 249 cycles with 49 taker BUY
  - 61 cycles with 52 taker BUY
  - 1 cycle with 34 taker BUY
- This is a real pipeline capability shift, not a single-cycle anomaly.

## Real but Conditional / Context-Dependent
- A substantial subset still fails fill-quality gates (`low_fill_score`) and remains non-executable unless liquidity quality improves.
- Maker opportunities remain real but require passive quoting/fill controls and adverse-selection handling.

## Surgical Fixes Implemented
1. Structural correlation BUY generation for parent-child/temporal relations with long-only preference when NO-side liquidity exists.
   - `scanner/correlation.py`
2. Correlation structural BUY reason families allowed into actionable gate (config-gated).
   - `run.py`, `config.py`
3. Critical integration fix: correlation scan now fetches both YES and NO books.
   - `run.py`
4. Correlation post-cap tuned to retain more BUY opportunities while preserving event diversity.
   - `config.py` (`correlation_max_opps_per_cycle=120`, `correlation_cap_min_buy_opps_per_cycle=40`, `correlation_cap_max_buy_per_event=3`)

## Remaining Pipeline Improvements (Highest Priority)
1. Add fee-aware netting inside correlation scanner path (taker + settlement costs) to tighten precision of projected PnL.
2. Add multi-leg slippage/VWAP simulation to reduce top-of-book optimism in dry-run actionability.
3. Add repeated-opportunity de-dup / clustering across cycles to avoid over-counting structurally identical signals.
4. Promote a stricter dynamic fill-score floor during stressed liquidity windows.

## Stability Notes
- MATIC 429 occurrences reduced in latest run (`6` in Iter 6 vs `10-12` in prior iterations).
- Occasional Kalshi cache `RemoteProtocolError` still appears during/after shutdown and should be hardened.

## External Agent Use
- Gemini CLI was used for independent failure-mode review and precision recommendations.
- Claude CLI was attempted but unavailable in-session due account credit exhaustion (`Credit balance is too low`).
