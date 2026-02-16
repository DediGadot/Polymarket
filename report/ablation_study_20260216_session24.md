# Ablation Study -- Session 24 (2026-02-16)

## Scope

- Dataset: report session `24` from fresh 5-minute dry-run (`logs/run_20260216_130921.log`).
- Sample: 12,504 opportunities across 125 cycles; actionable candidate subset: 5,482.
- Candidate filter (proxy for actionable now): `is_buy_arb=1`, `opp_type!=maker_rebalance`, `net_profit>0`, `fill_score>=0.35`, `persistence_score>=0.30`.
- Cycle constraints used in ablation simulator:
  - max cycle capital: `$10000`
  - max picks per cycle: `40`
  - family cap (when enabled): `$2000`
  - strict depth threshold (when enabled): `1.35`

## Main Results (Single-Change + Full Combo)

| Variant | Selected | Raw Profit | Robust Profit | Utility | Family Conc. | Calibration Gap |
|---|---:|---:|---:|---:|---:|---:|
| baseline | 2,249 | $526,986.97 | $317,084.15 | 0.9663 | 24.01% | 16.18% |
| family_budget_only | 2,124 | $497,647.94 | $299,481.67 | 0.9392 | 20.06% | 17.14% |
| risk_rank_only | 2,621 | $593,683.61 | $355,935.59 | 1.0324 | 24.16% | 18.47% |
| unique_actionable_only | 2,249 | $526,986.97 | $317,084.15 | 0.9663 | 24.01% | 16.18% |
| staleness_penalty_only | 2,249 | $510,182.49 | $309,708.67 | 0.9533 | 24.01% | 16.54% |
| strict_depth_only | 2,978 | $353,445.75 | $336,833.53 | 1.0168 | 15.34% | 0.84% |
| full_combo | 2,976 | $353,297.09 | $336,700.48 | 1.0166 | 15.35% | 0.84% |

## Leave-One-Out Decisions

| Change | Decision | Utility Δ (Full - LOO) | Robust Profit Δ | Calibration Gap Δ |
|---|---|---:|---:|---:|
| family_budget | DROP (neutral) | -0.0002 | $-133.04 | 0.00% |
| risk_ranked_ev | KEEP (neutral) | 0.0000 | $0.00 | 0.00% |
| unique_actionable_view | KEEP (UI-only neutral) | 0.0000 | $0.00 | 0.00% |
| staleness_penalty | DROP for now (neutral in LOO) | 0.0000 | $0.00 | 0.00% |
| strict_depth_replay | DROP | -0.0131 | $-15,309.30 | -17.62% |

## Top-20 Recurring Signature Calibration (Replay Proxy)

- Baseline MAE (projected hit-rate vs next-cycle realized persistence): `0.1552`
- Strict-depth MAE: `0.2992`
- Strict-depth calibration shift: `+0.1440` (worse)

## Targeted Incremental Checks (On Top of `risk_rank_only`)

| Variant | Selected | Raw Profit | Robust Profit | Outlier Share | Realized Hit |
|---|---:|---:|---:|---:|---:|
| risk | 2,621 | $593,683.61 | $355,935.59 | 9.46% | 99.20% |
| risk + stale | 2,621 | $593,683.61 | $355,935.59 | 9.46% | 99.20% |
| risk + family | 2,622 | $589,779.93 | $352,009.78 | 9.46% | 99.20% |
| risk + strict | 2,978 | $353,445.75 | $336,833.53 | 4.16% | 99.16% |

Interpretation:
- `stale` contributed no incremental lift once risk-ranked EV was active.
- `family` reduced robust profit by ~1.1% at tested cap.
- `strict` reduced robust profit by ~5.4% and worsened signature calibration MAE.

## Keep / Drop

1. `risk_ranked_ev`: **KEEP**
2. `strict_depth_replay` (as a hard gate/default rank modifier): **DROP**
3. `family_budget`: **DROP for now** (revisit only if concentration risk grows)
4. `staleness_penalty`: **DROP for now** (no measurable incremental lift once risk-ranked EV is on)
5. `unique_actionable_view`: **KEEP** as a UI/reporting feature only (neutral execution impact in this dataset)

## Why

- `risk_rank_only` gave the highest utility (`1.0324`) and best robust profit (`$355,935.59`).
- `strict_depth_only` improved calibration gap (0.84%) but underperformed `risk_rank_only` on robust profit by `$-19,102.06`.
- `family_budget_only` reduced concentration, but profit/utility tradeoff was negative at tested cap.
- `staleness_penalty_only` reduced outlier share but cut robust profit; no incremental gain in combined modes.
- Unique-actionable dedup did not change execution metrics here, but remains useful for operator clarity and duplicate suppression in UI surfaces.
