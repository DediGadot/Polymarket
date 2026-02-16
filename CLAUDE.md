# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Commands

```bash
uv sync --all-extras                                      # install
PYTHONPATH=. uv run python -m pytest tests/ -v            # all tests
PYTHONPATH=. uv run python -m pytest tests/test_fees.py -v  # single file
PYTHONPATH=. uv run python -m pytest tests/ --cov=. --cov-report=term-missing
uv run python run.py --dry-run --limit 500                # fast dev loop (public APIs, no wallet)
uv run python run.py --dry-run                            # full scan, no wallet
uv run python run.py --scan-only                          # detect only (needs wallet)
uv run python run.py --live                               # real orders
uv run python run.py --report --report-port 8787          # with pipeline dashboard
PYTHONPATH=. uv run python -m report.import_ledger        # import pnl_ledger.json into report DB
PYTHONPATH=. uv run python -m benchmark.replay --input recordings/FILE.jsonl --sweep  # weight sweep replay
```

PYTHONPATH required -- flat module imports, no package install.

## Architecture

Pipeline in `run.py`: **fetch markets → pre-filter → scan (10 scanners in parallel) → score (8-factor + optional ML) → size → safety → execute (presigned) → P&L → checkpoint → repeat**.

### Modules

- **`client/`** — `gamma.py` (market discovery REST, `build_events()` groups by `nrm:`/`evt:` prefix), `clob.py` (orderbooks/orders via py-clob-client SDK, sliding-window rate limiter 25/10s, HTTP/2 disabled to avoid GOAWAY crashes), `cache.py` (`GammaClient` wrapping `GammaCache`: 60s markets, 300s event counts, exposes `markets_timestamp` for change detection), `gas.py` (cached Polygon gas + POL/USD oracle, `allow_network=False` disables all RPC/CoinGecko calls), `auth.py` (L2 HMAC from wallet key), `data.py` (position tracker via Data API for sell-side validation), `ws.py` (async WebSocket manager with retry/backoff, queue overflow drops oldest entry), `ws_bridge.py` (async WS→sync bridge, auto-selects WSManager vs WSPool by token count, feeds BookCache + SpikeDetector + OFITracker, exposes `last_changed_tokens` for event-driven scan skip), `ws_pool.py` (WS connection sharding: 500 tokens/shard, merged output queues, per-shard health via `ShardHealth`), `platform.py` (PlatformClient Protocol — see below), `kalshi.py` (Kalshi REST v2 client, cent→dollar conversion, parallel multi-ticker `get_orderbooks()`, `book_fetcher` property, rate limit 10 reads/sec with 429 retry+backoff), `kalshi_auth.py` (RSA request signing), `kalshi_cache.py` (`KalshiMarketCache` background daemon: `KalshiMarketSnapshot` refreshed every 300s with exponential backoff on failure, non-blocking warm), `fanatics.py` (Fanatics PlatformClient stub — API not yet available, all trading methods raise NotImplementedError), `fanatics_auth.py` (Fanatics auth placeholder)
- **`scanner/`** — `models.py` (all frozen dataclasses + `BookFetcher` type alias + `Platform` enum: POLYMARKET/KALSHI/FANATICS, `is_market_stale()` with module-level cache). **Scanners**: `binary.py` (YES+NO < $1, config-driven slippage ceiling), `negrisk.py` (sum YES asks < $1 across outcomes, large-event subset mode for >max_legs outcomes, cooperative `should_stop` cancellation), `latency.py` (15-min crypto lagging spot), `spike.py` (news-driven dislocations), `cross_platform.py` (N-platform price divergence with nested `platform_books`/`platform_fee_models`/`platform_markets` dicts, contract-level matching, cent-rounding validation, fee-guard rejects ext_fee/ext_vwap > 0.20), `maker.py` (GTC limit orders inside bid-ask spread on both sides, paired-fill EV model), `stale_quote.py` (fast movers with lagging complementary quotes), `resolution.py` (near-resolution outcome sniping via `outcome_oracle.py` OutcomeOracle), `correlation.py` (cross-event probability violation scanner with `RelationType`: PARENT_CHILD/COMPLEMENT/TEMPORAL, research or executable lane via `correlation_execute_enabled`, per-cycle cap with BUY-side reservation). **Support**: `depth.py` (VWAP sweep + `worst_fill_price` + `effective_price` + `slippage_ceiling` with fee-aware budget), `fees.py` (PM taker + 2% resolution fee), `fees_v2.py` (adds DCM parabolic taker: 0.10% at p=0.50, 0.01% at p=0.90), `kalshi_fees.py` (parabolic taker, no resolution fee), `fanatics_fees.py` (Fanatics fee placeholder, uses Kalshi formula), `platform_fees.py` (PlatformFeeModel Protocol), `scorer.py` (8-factor composite ranking with OFI), `strategy.py` (adaptive params: 4 modes), `book_cache.py` (WS-fed orderbook cache, `get_books_snapshot()` for consistent reads, `prune()` for long sessions), `book_service.py` (`BookService`: single prefetch per cycle, all scanners consume from cache), `ofi.py` (`OFITracker`: thread-safe, `record_aggressor()` from book transitions, `record_quality()` telemetry, serializable), `matching.py` (fuzzy event mapping across platforms, `PlatformMatch` generalized from Kalshi-only, settlement keyword + year mismatch detection, negative TTL cache), `confidence.py` (arb persistence tracking with serialization), `filters.py` (volume + time-to-resolution pre-filters), `validation.py` (boundary validators for price/size/gas — NaN, Inf, range checks), `labels.py` (human-readable opportunity labels from token→market question mapping), `realized_ev.py` (Bayesian realized-edge estimator for maker fill quality, serializable), `feature_engine.py` (fixed-width numpy feature extraction with rolling z-score normalization for ML), `ml_scorer.py` (GradientBoosting trade profitability classifier, background retraining thread, joblib persistence, blend weight 0.15), `rl_strategy.py` (tabular Q-learning strategy selector, shadow mode only — logs RL choice vs heuristic)
- **`executor/`** — `engine.py` (FAK batch orders + partial unwind, presigner cache-first signing, handles CORRELATION_ARB type, generalized `platform_clients` dict), `sizing.py` (half-Kelly), `safety.py` (staleness, depth, gas, edge revalidation, inventory with `platform_filter`, confidence gate, circuit breaker, platform limits incl. Fanatics), `cross_platform.py` (dual-platform execution with unwind on failure), `cross_platform_v2.py` (generalized N-platform execution with fill state machine), `fill_state.py` (cross-platform order lifecycle state machine: PENDING→FILLED→UNWINDING→UNWOUND/STUCK), `tick_size.py` (price quantization to 0.01 or 0.001 tick grids), `maker_lifecycle.py` (GTC order tracking: age-based cancel, drift-based cancel, fill monitoring), `presigner.py` (`OrderPresigner`: LRU cache keyed by `PresignKey`, thread-safe, ±2 tick levels, prewarm top N from previous cycle)
- **`monitor/`** — `pnl.py` (NDJSON ledger), `logger.py` (structured JSON logs), `scan_tracker.py` (`ScanTracker`: research vs executable lane separation, per-cycle actionable/maker/sell breakdowns, 100-cycle memory cap), `status.py` (rolling `status.md` writer with embedded how-it-works guide, last 20 cycles), `display.py` (console pretty-printing with `resolve_opportunity_label` and scanner breakdown summaries)
- **`report/`** — `collector.py` (`ReportCollector`/`NullCollector`: funnel stages, strategy snapshots, scored opps with all 8 score factors, trades, safety rejections), `store.py` (SQLite WAL-mode persistence with thread-local connections), `server.py` (FastAPI + SSE dashboard at `--report-port`, `/live` endpoint, daemon thread), `readiness.py` (6-check health score 0-100 with GO/CAUTION/STOP recommendation), `import_ledger.py` (one-time `pnl_ledger.json` import into report DB). Zero overhead when `--report` not passed (NullCollector).
- **`state/`** — `checkpoint.py` (SQLite WAL-mode checkpoint manager: auto-save every N cycles, SIGTERM flush, atomic transactions). Serializable trackers: ArbTracker, SpikeDetector, MakerPersistenceGate, RealizedEVTracker, OFITracker.
- **`benchmark/`** — `evs.py` (expected value of scanning metric from logs), `weight_search.py` (scorer weight grid search), `latency_sim.py` (architecture latency simulation), `cross_platform.py` (platform breakdown analysis), `recorder.py` (`CycleRecorder`/`NullRecorder`: NDJSON cycle recording with schema v2, max file size rollover via `recording_max_mb`), `replay.py` (weight sweep replay engine: CLI `python -m benchmark.replay --input FILE [--sweep]`, P&L/Sharpe/win-rate metrics)
- **`pipeline/`** — `gas_utils.py` (gas estimation helpers)

### Data Flow

`Market` → `Event` → pre-filters (volume, TTL) → `BookService.prefetch()` (single REST batch) → 10 scanners emit `Opportunity` (with `LegOrder` legs) → split into executable lane + research lane → `rank_opportunities()` (8-factor composite score + optional ML blend) → Kelly sizing → safety checks (incl. confidence gate) → `execute_opportunity()` (presigner cache-first) → `TradeResult` → `PnLTracker` → `CheckpointManager.tick()` → repeat.

### BookFetcher Abstraction

All scanners depend on `BookFetcher = Callable[[list[str]], dict[str, OrderBook]]`, decoupling detection from data source. Three implementations: REST via `clob.get_orderbooks_parallel()`, cache layer via `BookCache.make_caching_fetcher()`, and WebSocket via `WSBridge`. In practice, `BookService` wraps the caching fetcher: `prefetch()` does one REST batch per cycle, then `make_fetcher()` returns a cache-only `BookFetcher` for all scanners. Scanners never know where orderbook data comes from.

### PlatformClient Protocol

Cross-platform extensibility uses two Protocols defined in `client/platform.py` and `scanner/platform_fees.py`:
- **`PlatformClient`** — exchange client interface (market discovery, orderbooks, order placement, positions, balance). Kalshi implements this fully. Fanatics is a stub (API TBD, all trading methods raise `NotImplementedError`; pipeline gracefully skips).
- **`PlatformFeeModel`** — fee calculation interface (taker fees, resolution fees, profit adjustment). Each exchange has its own fee schedule (`kalshi_fees.py`, `fanatics_fees.py`, etc.).

New exchanges plug in by: (1) implementing both protocols, (2) adding credentials to `config.py`, (3) registering in `active_platforms()` and the `run.py` initialization block. `active_platforms()` auto-detects Kalshi (key+path) and Fanatics (key+secret).

### NegRisk Event Grouping

NegRisk markets are grouped by `neg_risk_market_id` (not just `event_id`). A single event can have multiple outcome pools (e.g. moneyline vs spread vs totals for a sports event). Each pool becomes a separate `Event` with its own `neg_risk_market_id`. Markets without a `neg_risk_market_id` fall back to `event_id` grouping. `gamma.py:build_events()` handles this with a `nrm:` prefix key (non-negRisk uses `evt:` prefix). NegRisk completeness is validated against `get_event_market_counts()` to prevent false arbs from partial outcome sets. When `missing <= 2` outcomes, uses `payout_cap = 1.0 - (missing * 0.01)` with risk flags instead of skipping entirely.

**Large-event subset mode** (`negrisk_large_event_subset_enabled`, default True): events exceeding `max_legs_per_opportunity` (15) build a bounded basket of the cheapest-ask outcomes. Omitted tail must sum below `negrisk_large_event_tail_max_prob` (0.05). Uses `payout_cap` instead of $1.00. SELL arbs are skipped for subset events (path-dependent downside).

### Maker Strategy

`maker.py` detects wide bid-ask spreads where GTC limit orders at bid+1tick on both YES and NO sides cost < $1.00. Three-layer gate prevents phantom arbs:
1. **MakerPersistenceGate** — requires setup to persist N consecutive cycles before emitting.
2. **MakerExecutionModel** — queue-aware microstructure model using EWMA features (update rate, mid-move intensity, spread widening, queue imbalance) to estimate paired-fill probability and toxicity.
3. **RealizedEVTracker** (`realized_ev.py`) — Bayesian estimator tracking full-fill vs orphan-hedge outcomes to adjust expected profit.

Maker lifecycle (`executor/maker_lifecycle.py`) manages posted GTC orders across cycles: age-based cancellation, drift-based cancellation when book moves, fill monitoring.

### Strategy System

`StrategySelector` picks one of 4 modes per cycle based on `MarketState` (gas, spikes, momentum, win rate):
- **AGGRESSIVE** — low thresholds, 1.5x size (high win rate, low gas)
- **CONSERVATIVE** — high thresholds, 0.5x size (low win rate or high gas)
- **SPIKE_HUNT** — disables binary/latency, focuses on spike siblings + negrisk
- **LATENCY_FOCUS** — all scanners on, 0.5x ROI threshold (crypto momentum detected)

Priority: spikes > crypto momentum > gas conditions > win rate.

### Scorer

`rank_opportunities()` uses 8 weighted factors: profit (log-scaled, W=0.20), fill probability (depth ratio, W=0.20), capital efficiency (annualized ROI, W=0.15), urgency (spike=1.0 / latency=0.85 / steady=0.50, W=0.15), competition (trade count decay, W=0.00), persistence (arb confidence, W=0.10), realized EV (Bayesian fill quality, W=0.10), OFI divergence (order flow imbalance, W=0.10). Requires `ScoringContext` per opportunity for accurate scoring. Optional `MLScorer` augments hand-tuned weights when trained (disabled by default, needs 100+ labeled samples).

### Correlation Scanner

`CorrelationScanner` detects cross-event probability violations (e.g. complement pairs, parent-child, temporal sequences). Three `RelationType`s: `PARENT_CHILD`, `COMPLEMENT`, `TEMPORAL`. Aggregation strategies: `liquidity_weighted` (default), `median`, `top_liquidity`. Emits `CORRELATION_ARB` opportunities with `reason_code` and `risk_flags`.

Opportunities are split into **research lane** (flag-only metrics) and **executable lane** (taker BUY candidates) based on `correlation_execute_enabled` (default True). Per-cycle cap: `correlation_max_opps_per_cycle` (120) with `correlation_cap_min_buy_opps_per_cycle` (40) reserving BUY-side slots. Actionable BUY gates: `correlation_actionable_min_confidence` (0.30), `correlation_actionable_min_fill_score` (0.35). Structural BUY transforms (parent+child_no, earlier_no+later) enabled by `correlation_actionable_allow_structural_buy`.

### Performance Optimizations

- **Market data caching**: Gamma API results cached 60s; scanners skip reprocessing when data unchanged (~59 of 60 cycles reuse previous filter/group results).
- **Binary book pre-fetch**: all YES+NO books fetched into cache once before parallel scanners run, preventing redundant REST calls.
- **Event-driven scan skip**: when WS is healthy and no deltas arrive, skip full scan (force rescan every `ws_force_rescan_sec`).
- **Kalshi background cache**: daemon thread refreshes 123K+ Kalshi markets asynchronously; cross-platform scanner reads immutable snapshot without blocking.
- **Parallel orderbook fetch**: `get_orderbooks_parallel()` with configurable `book_fetch_workers` (default 8), sliding-window rate limiter prevents 429s.
- **WS auto-sharding**: `WSBridge` transparently upgrades from single `WSManager` to `WSPool` when token count exceeds 500/conn. `last_changed_tokens` set enables event-driven scan skip.
- **Stale-market cache**: `is_market_stale()` module-level cache keyed by `end_date` string with 60s TTL eliminates ~93% of datetime parsing for 15K+ markets/cycle.
- **Presigner prewarm**: `OrderPresigner` pre-signs top N opportunities from previous cycle before next cycle starts (`presigner_prewarm_top_n`, default 10).
- **Dry-run book cache**: `dry_run_book_cache_max_age_sec` (90s) allows longer cache reuse to reduce REST rate-limit pressure.

## Traps

- **Orderbook sort order**: py-clob-client SDK does NOT sort levels. `clob.py:_sort_book_levels()` enforces asks ascending, bids descending. Without this, `best_ask`/`best_bid` return worst prices and zero opportunities are found.
- **Resolution fee**: `fees.py` always deducts $0.02/set (2% of $1 payout) on ALL markets. 15-min crypto markets add dynamic taker fee up to 3.15% at 50/50 odds. `fees_v2.py` adds DCM (Daily Crypto Markets) parabolic taker fee: 0.10% at p=0.50, 0.01% at p=0.90.
- **Kalshi prices are cents**: Kalshi API uses 1-99 (cents), bot converts to 0.01-0.99 (dollars). Mismatch breaks all profit calculations. Kalshi YES asks are derived from NO bids via `(100 - P) / 100`.
- **Kalshi fees use `math.ceil()`**: $0.001 rounds up to $0.01. No resolution fee (unlike PM's 2%).
- **Tick sizes**: Markets use either 0.01 or 0.001 tick grids (`Market.min_tick_size`). Orders must be quantized via `tick_size.py:quantize_price()` before placement. `TickSizeExceededError` if price shifts more than half a tick.
- **Cross-platform execution order**: External platform filled first (fast REST ~50ms), PM second (on-chain ~2s). If PM fails after external fills, unwind logic sells external position at market; `CrossPlatformUnwindFailed` if that also fails.
- **Cross-platform matching risk**: fuzzy `token_set_ratio` matching (threshold 95%) can produce false positives with different settlement terms. Manual map (`cross_platform_map.json`) entries have confidence=1.0. Unverified fuzzy matches are logged but blocked from trading (confidence=0). Contract-level matching (`match_contracts()`) validates settlement equivalence when market metadata is available. Settlement keyword detection and year pattern mismatch guards prevent false positives. Negative TTL cache (`cross_platform_matching_negative_ttl_sec`, default 300s) prevents repeated matching attempts for known non-matches.
- **Cross-platform fee guard**: if `ext_fee / ext_vwap > 0.20`, the arb is rejected (fee eats too much edge). Cent-based platforms validated with `ext_cents in 1-99` and rounding drift > 0.5 cents rejected.
- **Cross-platform signal staleness**: `cross_platform_max_signal_age_sec` (1.5s) rejects cross-platform signals older than threshold. `cross_platform_inventory_pm_only` (True) restricts inventory checks to PM positions only.
- **Fanatics stub**: `fanatics.py` raises `NotImplementedError` on all trading methods. Pipeline skips gracefully. `fanatics_enabled` defaults `False`.
- **BookCache threading**: single-writer (WS thread) + single-reader (main loop) only. Not safe for concurrent reads.
- **CoinGecko rate limits**: free tier 429s fast (120s cooldown with 20% jitter). `GasOracle` falls back to default $0.50 POL/USD. Token ID is `polygon-ecosystem-token` (POL, not MATIC).
- **25K+ markets**: full REST scan takes 30s+ for binary alone. Use `--limit` for dev. `--limit` never truncates negRisk markets (they need complete outcome sets). `DRY_RUN_DEFAULT_LIMIT` auto-caps at 1200 markets.
- **`ALLOW_NON_POLYMARKET_APIS`**: defaults `true`. When `false`, latency scanner and cross-platform are force-disabled by `_enforce_polymarket_only_mode()` in `run.py`, and gas oracle uses fallback defaults (no RPC/CoinGecko calls).
- **Value scanner uniform fallacy**: `value_scanner_enabled` defaults `False`. It assumes uniform 1/N probability across outcomes, producing 100% false positives on markets with known favorites (e.g. PSG top 4 at $0.98). Cannot estimate true outcome probabilities. Defense-in-depth: `NEGRISK_VALUE` opps are stripped from `all_opps` when scanner is disabled.
- **Maker phantom arbs**: `maker_min_leg_price` (default 0.05) filters near-certain markets where the low-probability side will never fill. `maker_min_depth_sets` (default 15.0) filters micro-depth books. `maker_max_taker_cost` (default 1.08) rejects setups where taker crossing cost is too far from parity. `maker_max_spread_ticks` (default 8) caps maximum spread width.
- **Maker execution gates**: `maker_min_pair_fill_prob` (0.55), `maker_max_toxicity_score` (0.70), `maker_min_expected_ev_usd` (0.20) filter candidates through the MakerExecutionModel before emission.
- **CLOB HTTP/2 disabled**: py-clob-client's shared httpx client is patched to disable HTTP/2 (GOAWAY frame crashes) and set 15s timeout.
- **CLOB rate limiter**: sliding-window 25 calls / 10s on read endpoints. Parallel book fetches respect this globally.
- `neg_risk=True` markets excluded from binary scanner (handled by negrisk scanner).
- `SafetyCheckFailed` → skip opportunity. `CircuitBreakerTripped` → halt bot.
- Config from `.env` via pydantic-settings (frozen model). All fields/defaults in `config.py`.

## Run Modes

- `--dry-run` — unauthenticated CLOB client, public endpoints only, no wallet needed, implies `--scan-only`
- `--scan-only` — full detection pipeline but skips execution/sizing/safety, needs wallet
- Default (no flags) — paper trading with simulated fills, full pipeline
- `--live` — real FAK orders on Polymarket, **use cautiously**
- `--limit N` — cap binary markets for faster iteration (never truncates negRisk)
- `--json-log PATH` — machine-readable NDJSON runtime logs
- `--report` — enable pipeline dashboard (SQLite + SSE, default port 8787)

## Tests

`respx` mocks httpx, `MagicMock` for CLOB client. Factories: `_make_book()`, `_make_market()`, `_make_opp()`. 1503 tests. `asyncio_mode = "auto"` in pytest config.
