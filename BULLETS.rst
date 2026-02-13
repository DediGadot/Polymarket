Polymarket Arbitrage Bot: 20-Point Deep Analysis
=================================================

WHAT IT DOES (10 points)
------------------------

**1. Five-scanner arbitrage engine covering every known prediction market edge type.** Binary rebalancing (YES+NO < $1), NegRisk multi-outcome rebalancing (historically 73% of arb profits), 15-min crypto latency arb (spot vs prediction lag), news-driven spike detection (sibling mispricing), and cross-platform Polymarket<->Kalshi divergence. 6,594 LOC core, 661 tests, 89-91% coverage.

**2. VWAP-aware, depth-first scanning prevents the "phantom profit" trap.** Every scanner walks the full orderbook via ``sweep_depth()`` / ``effective_price()`` / ``worst_fill_price()`` rather than trusting best-bid/ask. Worst-fill becomes the execution limit price; VWAP drives profit math. This separation is correct and critical -- most open-source PM bots miss this.

**3. Adaptive 4-mode strategy system tunes itself per cycle.** ``StrategySelector`` dynamically switches AGGRESSIVE (1.5x size, low thresholds), CONSERVATIVE (0.5x size, high thresholds), SPIKE_HUNT (disables steady-state scanners), or LATENCY_FOCUS based on real-time gas prices, spread widths, spike activity, crypto momentum, and trailing win rate.

**4. 6-factor composite scorer replaces naive ROI sorting.** Profit (22%, log-scaled), fill probability (22%, depth ratio), capital efficiency (17%, annualized ROI), urgency (17%, spike=1.0/latency=0.85/steady=0.50), competition (7%, trade count exponential decay), persistence (15%, arb stability tracking). Weight grid search benchmark tool enables data-driven tuning.

**5. 8-point safety stack + 3-condition circuit breaker.** Pre-trade: price freshness, edge revalidation (re-fetches books), depth verification, gas/profit ratio cap (50%), opportunity TTL (0.5s spikes, 2s steady), max legs (15, one batch), inventory for sell legs, cross-platform book validation. Circuit breaker: hourly loss ($50), daily loss ($200), 5 consecutive failures -> halt.

**6. NegRisk completeness validation prevents the #1 false positive.** Compares active market count against ``event_market_counts`` from the events API. Inactive "Other" outcomes create a probability gap that looks like arb but isn't. ``neg_risk_market_id`` grouping correctly handles events with multiple outcome pools (moneyline vs spread vs totals). This alone probably saved more money than any single trade made.

**7. Fee-correct dual-platform profit calculation.** ``MarketFeeModel``: PM taker (0% standard, parabolic up to 3.15% on 15-min crypto at 50/50 odds) + $0.02/set resolution fee. ``KalshiFeeModel``: ``math.ceil(0.07 * C * P * (1-P))`` per contract, no resolution fee. Fees deducted *before* min-profit filtering, not after.

**8. BookFetcher abstraction cleanly decouples scanners from data source.** ``Callable[[list[str]], dict[str, OrderBook]]`` with three implementations (parallel REST, BookCache caching layer, WSBridge). Scanners never know where data comes from. This is textbook interface segregation.

**9. Half-Kelly sizing with 4 hard caps.** ``compute_position_size()``: Kelly at 50% (halves growth rate by 25%, cuts variance by 50%), bounded by max-per-trade ($500), max-total ($5K), current exposure headroom, and orderbook depth. Rejects sizes where fixed gas overwhelms edge. Academically sound for high-frequency arb.

**10. Cross-platform Kalshi integration with execution-order optimization.** Kalshi first (~50ms REST), PM second (~2s on-chain). If PM fails after Kalshi fills, unwind logic market-sells the Kalshi position. Fuzzy matching (``token_set_ratio=95%``) with 3-tier confidence: manual JSON map (1.0), verified matches (preserved), unverified fuzzy (0.0, logged but blocked).


STEP-FUNCTION IMPROVEMENT OPPORTUNITIES (10 points)
----------------------------------------------------

**11. Replace the poll loop with event-driven execution -- this is the single highest-ROI change.** The synchronous ``while True: fetch->scan->score->execute->sleep`` loop means a 30s+ full-market scan while arbs decay in milliseconds. A 2025 study found 78% of arb opportunities in low-volume markets failed due to execution lag. **Fix:** WebSocket-triggered scan-on-price-change. When a WS book update creates a potential arb (sum of cached asks dips below $1), fire the safety/execution pipeline immediately for *that market only*, not after scanning 25K markets. Target: sub-100ms detect-to-order.

**12. Parallelize the 5 independent scanners -- 3-5x scan speed for free.** Binary, NegRisk, latency, spike, and cross-platform scanners share no mutable state. They all take a ``BookFetcher`` and return ``list[Opportunity]``. Currently sequential. **Fix:** ``concurrent.futures.ThreadPoolExecutor(max_workers=5)`` with one future per scanner. Scan time drops from ~sum(all) to ~max(slowest). The ``BookCache`` needs ``threading.RLock`` (point 17's fix) to support this.

**13. Add a maker strategy to eliminate taker fees entirely.** The bot is 100% taker. Polymarket charges 0% maker fees -- resting limit orders are free. On 15-min crypto markets, taker fees peak at 3.15% and eat most of the latency edge. **Fix:** When an arb is detected but not urgent (steady-state binary/negrisk), post resting limit orders at the required prices instead of crossing the spread. This turns a fee headwind into zero-cost execution and captures spread as bonus profit.

**14. Colocate + private RPC + HTTP/2 -- infrastructure-level latency reduction.** All PM orders go through ``py-clob-client`` over public HTTPS with HTTP/2 disabled (GOAWAY workaround). Professional arbitrageurs target sub-10ms latency via dedicated RPC nodes and colocated servers. **Fix:** Private Polygon RPC (Alchemy/QuickNode), re-enable HTTP/2 with keep-alive (py-clob-client v0.34.5 supports it natively -- test against GOAWAY fix), VPS near Polymarket's matching engine (US-East). Combined with event-driven architecture (point 11), this could achieve sub-100ms end-to-end.

**15. Persist arb history across sessions for warm-start intelligence.** Every restart is cold: no learned market profiles, no historical arb tracking, no warm BookCache. The ``ArbTracker``/``confidence`` module exists but is session-scoped only. **Fix:** SQLite or append-only NDJSON for arb history (market->win rate, avg edge, typical depth, time-of-day patterns). Feed into the scorer's persistence factor (currently 15% weight but seeded at default 0.5) and the strategy selector's mode choice. This turns the bot from stateless to learning.

**16. Address cross-platform settlement divergence -- the silent portfolio killer.** The 2024 government shutdown case showed Polymarket and Kalshi can settle identically-titled events differently, turning "risk-free" arb into total loss. Fuzzy title matching (95% threshold) catches naming differences but not settlement methodology differences. **Fix:** Parse both platforms' resolution sources and flag divergent criteria before execution. Block any match where resolution methodology, timeframe, or source differs. The $40M extracted from PM arbs historically includes losses from exactly this trap.

**17. Replace single-writer/single-reader BookCache with concurrent-safe cache.** The current "single-writer (WS thread) + single-reader (main loop) only" constraint blocks parallel scanner reads during execution and prevents the scanner parallelization from point 12. **Fix:** ``threading.RLock`` on the ``_books`` dict (multiple concurrent readers, exclusive writer), or a lock-free approach using ``dict.copy()`` snapshots. This is a prerequisite for points 11 and 12.

**18. Replace the toy latency probability model with learned calibration.** ``compute_implied_probability()`` uses ``prob = 0.50 + momentum * 0.35`` -- a hardcoded linear mapping calibrated to one historical bot's results, with no regime detection, volatility adjustment, or order flow signals. **Fix:** Logistic regression or gradient-boosted classifier trained on historical 15-min market resolutions vs (volume-weighted momentum, ATR, time-to-expiry, bid-ask spread, recent trade velocity). Even a simple logistic model with 3 features would dramatically improve edge estimation and reduce false signals.

**19. Extract run.py into a composable Pipeline -- the 898-line God object blocks everything.** The entire fetch->filter->scan->score->size->safety->execute->P&L flow lives in one function with deep nesting. Every improvement (parallel scanners, event-driven triggers, maker mode) requires touching this monolith. **Fix:** Extract a ``Pipeline`` class with ``FetchStage``, ``ScanStage``, ``ScoreStage``, ``ExecuteStage`` that can be composed, tested independently, and eventually run as event-driven coroutines instead of sequential steps.

**20. Implement graduated unwind with retry escalation.** ``_unwind_partial()`` uses single-attempt FOK market orders to exit stuck positions. In fast-moving markets, FOK fails when there's no resting liquidity at the required price -- leaving the bot with naked exposure and no recovery path. **Fix:** Tiered unwind: (1) FOK at market, (2) GTC limit +1% slippage / 10s timeout, (3) GTC +5% slippage / 30s timeout, (4) alert operator + track in circuit breaker as open exposure, not just P&L loss. The circuit breaker should also account for stuck-position risk, not only realized losses.
