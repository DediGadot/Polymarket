# IDEAS

Stack-ranked step-function improvements for the codebase (ranked by estimated success rate and impact).

1. **[92%] Wire unused runtime risk controls end-to-end**
   - Enforce `cross_platform_deadline_sec`, platform position limits, and platform enable toggles in scan/safety/execute paths.
   - Pass configured deadlines from `run.py` into cross-platform execution.
   - Add tests that prove these controls stop trades as intended.

2. **[89%] Add TTL caching for heavy per-cycle API calls**
   - Cache `get_event_market_counts` and external `get_all_markets` with short TTLs.
   - Define stale/fallback behavior when APIs fail.
   - Reduce cycle latency and rate-limit risk under load.

3. **[87%] Enforce market tick-size precision at execution**
   - Propagate `Market.min_tick_size` into order creation (including 0.001 markets).
   - Validate/quantize prices before placing orders.
   - Add regression tests for rejection and rounding edge cases.

4. **[84%] Correct fee-model realism for DCM and fee-bearing markets**
   - Implement actual DCM taker fee detection/application.
   - Add fee “golden case” tests against known examples.
   - Prevent systematic edge overestimation.

5. **[81%] Replace cross-platform fill handling with a state machine**
   - Handle full/partial/rejected/resting states explicitly on both venues.
   - Track exact fill quantity and unwind residual exposure deterministically.
   - Log state transitions for auditability.

6. **[76%] Upgrade cross-platform matching from event-level to contract-level**
   - Match outcomes/contracts, not just first market/ticker.
   - Add settlement-equivalence guards per matched leg.
   - Block execution when mapping confidence is insufficient.

7. **[74%] Harden WebSocket health and failover**
   - Add heartbeat/task-health monitoring (not just thread liveness).
   - Auto-restart on task failure; degrade to REST on repeated failure.
   - Emit explicit connectivity status and stale-book alarms.

8. **[72%] Integrate persistence confidence into live ranking**
   - Wire `ArbTracker` confidence into `ScoringContext` in runtime.
   - Use confidence as execution gating input (not just ranking).
   - Recalibrate weights with historical dry-run logs.

9. **[70%] Add CI quality gates for behavior-critical paths**
   - Add lint/type/test/coverage gates in CI.
   - Include scenario tests for partial fills, cross-platform mismatch, and tick-size errors.
   - Ensure high-risk modules keep or improve coverage.

10. **[97%] Fix documentation/config drift**
   - Sync README defaults with actual config defaults.
   - Add/maintain `.env.example` generated from `Config` fields.
   - Keep scoring-weight and mode docs aligned with implementation.

## Suggested Execution Order

1. Risk controls wiring (#1)
2. API caching (#2)
3. Tick-size correctness (#3)
4. Fee realism (#4)
5. Cross-platform execution state machine (#5)
6. Contract-level matching (#6)
7. WS hardening (#7)
8. Confidence integration (#8)
9. CI gates (#9)
10. Docs alignment (#10)
