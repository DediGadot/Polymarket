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
```

PYTHONPATH required -- flat module imports, no package install.

## Architecture

Pipeline in `run.py`: **fetch markets → pre-filter → scan (10 scanners in parallel) → score (8-factor + optional ML) → size → safety → execute (presigned) → P&L → checkpoint → repeat**.

### Modules

- **`client/`** — `gamma.py` (market discovery REST), `clob.py` (orderbooks/orders via py-clob-client SDK, sliding-window rate limiter 25/10s, HTTP/2 disabled to avoid GOAWAY crashes), `cache.py` (thread-safe TTL cache wrapping Gamma API: 60s markets, 300s event counts), `gas.py` (cached Polygon gas + POL/USD oracle), `auth.py` (L2 HMAC from wallet key), `data.py` (position tracker via Data API for sell-side validation), `ws.py` (async WebSocket manager with retry/backoff), `ws_bridge.py` (async WS→sync bridge, feeds BookCache + SpikeDetector + OFITracker via daemon thread), `ws_pool.py` (WS connection sharding: 500 tokens/shard, merged output queues), `platform.py` (PlatformClient Protocol — see below), `kalshi.py` (Kalshi REST v2 client, cent→dollar conversion), `kalshi_auth.py` (RSA request signing), `kalshi_cache.py` (background daemon thread refreshing Kalshi market snapshot every 300s, non-blocking warm)
- **`scanner/`** — `models.py` (all frozen dataclasses + `BookFetcher` type alias). **Scanners**: `binary.py` (YES+NO < $1), `negrisk.py` (sum YES asks < $1 across outcomes), `latency.py` (15-min crypto lagging spot), `spike.py` (news-driven dislocations), `cross_platform.py` (Polymarket vs external platform price divergence, contract-level matching), `maker.py` (GTC limit orders inside bid-ask spread on both sides, paired-fill EV model), `stale_quote.py` (fast movers with lagging complementary quotes), `resolution.py` (near-resolution outcome sniping via OutcomeOracle), `correlation.py` (cross-event probability violation scanner, flag-only). **Support**: `depth.py` (VWAP sweep + worst fill price + slippage ceiling), `fees.py` (PM taker + 2% resolution fee), `kalshi_fees.py` (parabolic taker, no resolution fee), `platform_fees.py` (PlatformFeeModel Protocol), `scorer.py` (8-factor composite ranking with OFI), `strategy.py` (adaptive params: 4 modes), `book_cache.py` (WS-fed orderbook cache), `book_service.py` (centralized single-fetch-per-cycle service), `ofi.py` (Order Flow Imbalance tracker, leading indicator), `matching.py` (fuzzy event mapping across platforms), `confidence.py` (arb persistence tracking with serialization), `filters.py` (volume + time-to-resolution pre-filters), `validation.py` (boundary validators for price/size/gas — NaN, Inf, range checks), `labels.py` (human-readable opportunity labels from token→market question mapping), `realized_ev.py` (Bayesian realized-edge estimator for maker fill quality), `feature_engine.py` (fixed-width numpy feature extraction for ML), `ml_scorer.py` (GradientBoosting trade profitability classifier), `rl_strategy.py` (tabular Q-learning strategy selector, shadow mode)
- **`executor/`** — `engine.py` (FAK batch orders + partial unwind, presigner integration), `sizing.py` (half-Kelly), `safety.py` (staleness, depth, gas, edge revalidation, inventory, confidence gate, circuit breaker, platform limits), `cross_platform.py` (dual-platform execution with unwind on failure), `fill_state.py` (cross-platform order lifecycle state machine: PENDING→FILLED→UNWINDING→UNWOUND/STUCK), `tick_size.py` (price quantization to 0.01 or 0.001 tick grids), `maker_lifecycle.py` (GTC order tracking: age-based cancel, drift-based cancel, fill monitoring), `presigner.py` (pre-signed order template LRU cache, ±2 tick levels)
- **`monitor/`** — `pnl.py` (NDJSON ledger), `logger.py` (structured JSON logs), `scan_tracker.py` (scan-only session aggregator with actionable/maker/sell breakdowns), `status.py` (rolling `status.md` writer, last 20 cycles), `display.py` (console pretty-printing)
- **`report/`** — `collector.py` (pipeline telemetry: funnel stages, strategy snapshots, scored opps, trades, safety rejections), `store.py` (SQLite persistence), `server.py` (SSE-powered dashboard at `--report-port`), `readiness.py` (health/readiness endpoints). Zero overhead when `--report` not passed (NullCollector).
- **`state/`** — `checkpoint.py` (SQLite WAL-mode checkpoint manager: auto-save every N cycles, SIGTERM flush, atomic transactions). Serializable trackers: ArbTracker, SpikeDetector, MakerPersistenceGate, RealizedEVTracker, OFITracker.
- **`benchmark/`** — `evs.py` (expected value of scanning metric from logs), `weight_search.py` (scorer weight grid search), `latency_sim.py` (architecture latency simulation), `cross_platform.py` (platform breakdown analysis), `recorder.py` (NDJSON cycle recorder for offline replay, NullRecorder pattern), `replay.py` (weight sweep replay engine with P&L/Sharpe metrics)
- **`pipeline/`** — `gas_utils.py` (gas estimation helpers)

### Data Flow

`Market` → `Event` → pre-filters (volume, TTL) → scanners emit `Opportunity` (with `LegOrder` legs) → `rank_opportunities()` (7-factor composite score + realized EV) → Kelly sizing → safety checks (incl. confidence gate) → `execute_opportunity()` → `TradeResult` → `PnLTracker`.

### BookFetcher Abstraction

All scanners depend on `BookFetcher = Callable[[list[str]], dict[str, OrderBook]]`, decoupling detection from data source. Three implementations: REST via `clob.get_orderbooks_parallel()`, cache layer via `BookCache.make_caching_fetcher()`, and WebSocket via `WSBridge`. This means scanners never know where orderbook data comes from.

### PlatformClient Protocol

Cross-platform extensibility uses two Protocols defined in `client/platform.py` and `scanner/platform_fees.py`:
- **`PlatformClient`** — exchange client interface (market discovery, orderbooks, order placement, positions, balance). Kalshi and Fanatics implement this.
- **`PlatformFeeModel`** — fee calculation interface (taker fees, resolution fees, profit adjustment). Each exchange has its own fee schedule (`kalshi_fees.py`, etc.).

New exchanges plug in by: (1) implementing both protocols, (2) adding credentials to `config.py`, (3) registering in `active_platforms()` and the `run.py` initialization block.

### NegRisk Event Grouping

NegRisk markets are grouped by `neg_risk_market_id` (not just `event_id`). A single event can have multiple outcome pools (e.g. moneyline vs spread vs totals for a sports event). Each pool becomes a separate `Event` with its own `neg_risk_market_id`. Markets without a `neg_risk_market_id` fall back to `event_id` grouping. `gamma.py:build_events()` handles this with a `nrm:` prefix key. NegRisk completeness is validated against `get_event_market_counts()` to prevent false arbs from partial outcome sets.

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

### Performance Optimizations

- **Market data caching**: Gamma API results cached 60s; scanners skip reprocessing when data unchanged (~59 of 60 cycles reuse previous filter/group results).
- **Binary book pre-fetch**: all YES+NO books fetched into cache once before parallel scanners run, preventing redundant REST calls.
- **Event-driven scan skip**: when WS is healthy and no deltas arrive, skip full scan (force rescan every `ws_force_rescan_sec`).
- **Kalshi background cache**: daemon thread refreshes 123K+ Kalshi markets asynchronously; cross-platform scanner reads immutable snapshot without blocking.
- **Parallel orderbook fetch**: `get_orderbooks_parallel()` with configurable `book_fetch_workers` (default 8), sliding-window rate limiter prevents 429s.

## Traps

- **Orderbook sort order**: py-clob-client SDK does NOT sort levels. `clob.py:_sort_book_levels()` enforces asks ascending, bids descending. Without this, `best_ask`/`best_bid` return worst prices and zero opportunities are found.
- **Resolution fee**: `fees.py` always deducts $0.02/set (2% of $1 payout) on ALL markets. 15-min crypto markets add dynamic taker fee up to 3.15% at 50/50 odds.
- **Kalshi prices are cents**: Kalshi API uses 1-99 (cents), bot converts to 0.01-0.99 (dollars). Mismatch breaks all profit calculations. Kalshi YES asks are derived from NO bids via `(100 - P) / 100`.
- **Kalshi fees use `math.ceil()`**: $0.001 rounds up to $0.01. No resolution fee (unlike PM's 2%).
- **Tick sizes**: Markets use either 0.01 or 0.001 tick grids (`Market.min_tick_size`). Orders must be quantized via `tick_size.py:quantize_price()` before placement. `TickSizeExceededError` if price shifts more than half a tick.
- **Cross-platform execution order**: External platform filled first (fast REST ~50ms), PM second (on-chain ~2s). If PM fails after external fills, unwind logic sells external position at market; `CrossPlatformUnwindFailed` if that also fails.
- **Cross-platform matching risk**: fuzzy `token_set_ratio` matching (threshold 95%) can produce false positives with different settlement terms. Manual map (`cross_platform_map.json`) entries have confidence=1.0. Unverified fuzzy matches are logged but blocked from trading (confidence=0). Contract-level matching (`match_contracts()`) validates settlement equivalence when market metadata is available.
- **BookCache threading**: single-writer (WS thread) + single-reader (main loop) only. Not safe for concurrent reads.
- **CoinGecko rate limits**: free tier 429s fast. `GasOracle` falls back to default $0.50 POL/USD.
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

`respx` mocks httpx, `MagicMock` for CLOB client. Factories: `_make_book()`, `_make_market()`, `_make_opp()`. 1483 tests. `asyncio_mode = "auto"` in pytest config.
