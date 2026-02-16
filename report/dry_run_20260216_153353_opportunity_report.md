# Dry-Run Opportunity Reality Report (20260216_153353)

## Run Setup

- Mode: `--dry-run` with `RECORDING_ENABLED=true`
- Date: 2026-02-16
- Cycles completed: **301**
- Opportunities found: **4458**
- Unique opportunity signatures: **16**
- Unique events represented: **8**
- Artifacts: `logs/dryrun10_20260216_153353.console.log`, `logs/dryrun10_20260216_153353.jsonl`, `recordings/dryrun10_20260216_153353`, `report/dry_run_20260216_153353_opportunity_deep_dive.csv`

## Headline Findings

- Composition: correlation_arb=4458.
- Top reason codes: corr_parent_child_buy_liquidity_weighted=3254, corr_parent_child_liquidity_weighted=903, corr_complement_buy_liquidity_weighted=301.
- Realness mean score: **85.05/100**.
- Realness tier distribution: High=4, Medium=1, Medium-High=5, Very High=6.
- Timing proxy: median resolution horizon **10.5mo**, mean **8.0mo**.
- Correlation graph density (first logged cycle): **16 violations / 89 relations**.
- Weak semantic GTA/Jesus pair instances detected: **0**.
- Operational warnings: dry-run auto-limit=1, CoinGecko 429=6.

## Before vs After (Semantic Fix Impact)

- Baseline run: `20260216_145154` | Post-fix run: `20260216_153353`
- Total opportunities: **22615 -> 4458** (-80.3%).
- Unique signatures: **76 -> 16** (-78.9%).
- Unique events: **33 -> 8** (-75.8%).
- First-cycle graph: **142 viol / 1050 rel** -> **16 viol / 89 rel**.
- Weak GTA/Jesus pair: **302 -> 0** (eliminated).
- Interpretation: the semantic compatibility filter removed a large set of weakly-linked correlations while preserving a smaller, more coherent set.

## Scoring Method

- Score blends persistence, confidence, fill probability, depth, toxicity, realized-EV, and profit stability.
- Penalties apply for inventory-dependent legs, extremely high ROI likely driven by tiny notional, and very thin depth.

## Per-Opportunity Deep Dive

| # | Opportunity | Type | Reason | Sides | Seen | Avg Profit | Avg ROI | Realness | Why | When You See Money |
|---|---|---|---|---|---:|---:|---:|---|---|---|
| 1 | Will JD Vance win the 2028 US Presidential Election? | correlation_arb | `corr_complement_buy_liquidity_weighted` | `BUY/BUY` | 301/301 (100.0%) | $1136.53 | 113.65% | 97.26 (Very High) | persisted 301/301 cycles; high model confidence; good depth; risks: none | Usually on resolution (~2.7y); may be monetized earlier if prices reconverge. |
| 2 | StandX FDV above $800M one day after launch? | correlation_arb | `corr_parent_child_buy_liquidity_weighted` | `BUY/BUY` | 301/301 (100.0%) | $202.90 | 36.38% | 97.26 (Very High) | persisted 301/301 cycles; high model confidence; good depth; risks: none | Usually by later parent/child resolution (~10.5mo); earlier reconvergence exit possible. |
| 3 | StandX FDV above $800M one day after launch? | correlation_arb | `corr_parent_child_buy_liquidity_weighted` | `BUY/BUY` | 301/301 (100.0%) | $149.65 | 24.50% | 97.26 (Very High) | persisted 301/301 cycles; high model confidence; good depth; risks: none | Usually by later parent/child resolution (~10.5mo); earlier reconvergence exit possible. |
| 4 | StandX FDV above $800M one day after launch? | correlation_arb | `corr_parent_child_buy_liquidity_weighted` | `BUY/BUY` | 301/301 (100.0%) | $2.48 | 12.97% | 94.02 (Very High) | persisted 301/301 cycles; high model confidence; good depth; risks: none | Usually by later parent/child resolution (~10.5mo); earlier reconvergence exit possible. |
| 5 | OpenAI $1t+ IPO before 2027? | correlation_arb | `corr_parent_child_buy_liquidity_weighted` | `BUY/BUY` | 301/301 (100.0%) | $77.47 | 49.11% | 92.53 (Very High) | persisted 301/301 cycles; high model confidence; good depth; risks: none | Usually by later parent/child resolution (~10.5mo); earlier reconvergence exit possible. |
| 6 | Will OpenAI’s market cap be less than $500B at market close on IPO day? | correlation_arb | `corr_parent_child_buy_liquidity_weighted` | `BUY/BUY` | 301/301 (100.0%) | $111.61 | 90.29% | 91.74 (Very High) | persisted 301/301 cycles; high model confidence; good depth; risks: none | Usually by later parent/child resolution (~4.4mo); earlier reconvergence exit possible. |
| 7 | Will StandX launch a token by March 31? | correlation_arb | `corr_parent_child_buy_liquidity_weighted` | `BUY/BUY` | 301/301 (100.0%) | $1.97 | 3.54% | 89.55 (High) | persisted 301/301 cycles; high model confidence; moderate depth; risks: none | Usually by later parent/child resolution (~6.3w); earlier reconvergence exit possible. |
| 8 | Will StandX launch a token by March 31? | correlation_arb | `corr_parent_child_buy_liquidity_weighted` | `BUY/BUY` | 301/301 (100.0%) | $5.09 | 15.59% | 88.76 (High) | persisted 301/301 cycles; high model confidence; moderate depth; risks: none | Usually by later parent/child resolution (~6.3w); earlier reconvergence exit possible. |
| 9 | Will Gavin Newsom announce a Presidential run before 2027? | correlation_arb | `corr_parent_child_buy_liquidity_weighted` | `BUY/BUY` | 301/301 (100.0%) | $5.99 | 18.63% | 88.75 (High) | persisted 301/301 cycles; high model confidence; moderate depth; risks: none | Usually by later parent/child resolution (~10.5mo); earlier reconvergence exit possible. |
| 10 | StandX FDV above $800M one day after launch? | correlation_arb | `corr_parent_child_buy_liquidity_weighted` | `BUY/BUY` | 301/301 (100.0%) | $4.64 | 90.52% | 87.46 (High) | persisted 301/301 cycles; high model confidence; thin depth; risks: none | Usually by later parent/child resolution (~10.5mo); earlier reconvergence exit possible. |
| 11 | MegaETH market cap (FDV) >$2B one day after launch? | correlation_arb | `corr_parent_child_buy_liquidity_weighted` | `BUY/BUY` | 122/301 (40.5%) | $98.68 | 9.87% | 79.35 (Medium-High) | persisted 122/301 cycles; high model confidence; good depth; risks: none | Usually by later parent/child resolution (~4.3w); earlier reconvergence exit possible. |
| 12 | EdgeX FDV above $1B one day after launch? | correlation_arb | `corr_parent_child_liquidity_weighted` | `BUY/SELL` | 301/301 (100.0%) | $15.70 | 40.05% | 75.71 (Medium-High) | persisted 301/301 cycles; high model confidence; moderate depth; inventory-dependent; risks: sell_inventory_required (301) | Needs inventory for SELL leg; full edge typically settles by close/resolution (~10.5mo). |
| 13 | Will StandX launch a token by March 31? | correlation_arb | `corr_parent_child_liquidity_weighted` | `BUY/SELL` | 301/301 (100.0%) | $4.49 | 86.27% | 74.34 (Medium-High) | persisted 301/301 cycles; high model confidence; thin depth; inventory-dependent; risks: sell_inventory_required (301) | Needs inventory for SELL leg; full edge typically settles by close/resolution (~6.3w). |
| 14 | EdgeX FDV above $1B one day after launch? | correlation_arb | `corr_parent_child_liquidity_weighted` | `BUY/SELL` | 301/301 (100.0%) | $3.17 | 56.55% | 74.33 (Medium-High) | persisted 301/301 cycles; high model confidence; thin depth; inventory-dependent; risks: sell_inventory_required (301) | Needs inventory for SELL leg; full edge typically settles by close/resolution (~10.5mo). |
| 15 | MegaETH market cap (FDV) >$2B one day after launch? | correlation_arb | `corr_parent_child_buy_liquidity_weighted` | `BUY/BUY` | 122/301 (40.5%) | $2.88 | 41.70% | 69.55 (Medium-High) | persisted 122/301 cycles; high model confidence; thin depth; risks: none | Usually by later parent/child resolution (~4.3w); earlier reconvergence exit possible. |
| 16 | Will StandX launch a token by March 31? | correlation_arb | `corr_parent_child_buy_liquidity_weighted` | `BUY/BUY` | 301/301 (100.0%) | $6.81 | 172.06% | 62.94 (Medium) | persisted 301/301 cycles; high model confidence; thin depth; ROI likely notional-inflated; risks: none | Usually by later parent/child resolution (~6.3w); earlier reconvergence exit possible. |

## Notes

- Dry-run PnL is theoretical and assumes quoted liquidity can be captured at observed prices.
- For real trading, execution latency, queue position, and settlement timing remain major constraints.
