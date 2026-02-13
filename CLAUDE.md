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
```

PYTHONPATH required -- flat module imports, no package install.

## Architecture

Pipeline in `run.py`: **fetch markets → pre-filter → scan (5 scanners) → score → size → safety → execute → P&L → repeat**.

### Modules

- **`client/`** — `gamma.py` (market discovery REST), `clob.py` (orderbooks/orders via py-clob-client SDK), `gas.py` (cached Polygon gas + POL/USD oracle), `auth.py` (L2 HMAC from wallet key), `data.py` (position tracker via Data API for sell-side validation), `ws.py` (async WebSocket manager with retry/backoff), `ws_bridge.py` (async WS→sync bridge, feeds BookCache + SpikeDetector via daemon thread), `kalshi.py` (Kalshi REST v2 client, cent→dollar conversion), `kalshi_auth.py` (RSA request signing)
- **`scanner/`** — `models.py` (all frozen dataclasses). Scanners: `binary.py` (YES+NO < $1), `negrisk.py` (sum YES asks < $1 across outcomes), `latency.py` (15-min crypto lagging spot), `spike.py` (news-driven dislocations), `cross_platform.py` (Polymarket vs Kalshi price divergence). Support: `depth.py` (VWAP sweep + worst fill price), `fees.py` (PM taker + 2% resolution fee), `kalshi_fees.py` (parabolic taker, no resolution fee), `scorer.py` (5-factor composite ranking), `strategy.py` (adaptive params: 4 modes), `book_cache.py` (WS-fed orderbook cache), `matching.py` (fuzzy event mapping across platforms), `confidence.py` (arb persistence tracking), `filters.py` (volume + time-to-resolution pre-filters)
- **`executor/`** — `engine.py` (FAK batch orders + partial unwind), `sizing.py` (half-Kelly), `safety.py` (staleness, depth, gas, edge revalidation, inventory, circuit breaker), `cross_platform.py` (dual-platform execution: Kalshi first, PM second, with Kalshi unwind on PM failure)
- **`monitor/`** — `pnl.py` (NDJSON ledger), `logger.py` (structured JSON logs), `scan_tracker.py` (scan-only session aggregator), `status.py` (rolling `status.md` writer, last 20 cycles), `display.py` (console pretty-printing)
- **`benchmark/`** — `evs.py` (expected value of scanning metric from logs), `weight_search.py` (scorer weight grid search), `latency_sim.py` (architecture latency simulation), `cross_platform.py` (platform breakdown analysis)

### Data Flow

`Market` → `Event` → pre-filters (volume, TTL) → scanners emit `Opportunity` (with `LegOrder` legs) → `rank_opportunities()` (5-factor composite score) → Kelly sizing → safety checks → `execute_opportunity()` → `TradeResult` → `PnLTracker`.

### BookFetcher Abstraction

All scanners depend on `BookFetcher = Callable[[list[str]], dict[str, OrderBook]]`, decoupling detection from data source. Three implementations: REST via `clob.get_orderbooks_parallel()`, cache layer via `BookCache.make_caching_fetcher()`, and WebSocket via `WSBridge`. This means scanners never know where orderbook data comes from.

### NegRisk Event Grouping

NegRisk markets are grouped by `neg_risk_market_id` (not just `event_id`). A single event can have multiple outcome pools (e.g. moneyline vs spread vs totals for a sports event). Each pool becomes a separate `Event` with its own `neg_risk_market_id`. Markets without a `neg_risk_market_id` fall back to `event_id` grouping. `gamma.py:build_events()` handles this with a `nrm:` prefix key. NegRisk completeness is validated against `get_event_market_counts()` to prevent false arbs from partial outcome sets.

### Strategy System

`StrategySelector` picks one of 4 modes per cycle based on `MarketState` (gas, spikes, momentum, win rate):
- **AGGRESSIVE** — low thresholds, 1.5x size (high win rate, low gas)
- **CONSERVATIVE** — high thresholds, 0.5x size (low win rate or high gas)
- **SPIKE_HUNT** — disables binary/latency, focuses on spike siblings + negrisk
- **LATENCY_FOCUS** — all scanners on, 0.5x ROI threshold (crypto momentum detected)

Priority: spikes > crypto momentum > gas conditions > win rate.

### Scorer

`rank_opportunities()` uses 5 weighted factors: profit (25%, log-scaled), fill probability (25%, depth ratio), capital efficiency (20%, annualized ROI), urgency (20%, spike=1.0 / latency=0.85 / steady=0.50), competition (10%, trade count decay). Requires `ScoringContext` per opportunity for accurate scoring.

## Traps

- **Orderbook sort order**: py-clob-client SDK does NOT sort levels. `clob.py:_sort_book_levels()` enforces asks ascending, bids descending. Without this, `best_ask`/`best_bid` return worst prices and zero opportunities are found.
- **Resolution fee**: `fees.py` always deducts $0.02/set (2% of $1 payout) on ALL markets. 15-min crypto markets add dynamic taker fee up to 3.15% at 50/50 odds.
- **Kalshi prices are cents**: Kalshi API uses 1-99 (cents), bot converts to 0.01-0.99 (dollars). Mismatch breaks all profit calculations. Kalshi YES asks are derived from NO bids via `(100 - P) / 100`.
- **Kalshi fees use `math.ceil()`**: $0.001 rounds up to $0.01. No resolution fee (unlike PM's 2%).
- **Cross-platform execution order**: Kalshi filled first (fast REST ~50ms), PM second (on-chain ~2s). If PM fails after Kalshi fills, unwind logic sells Kalshi position at market; `CrossPlatformUnwindFailed` if that also fails.
- **Cross-platform matching risk**: fuzzy `token_set_ratio` matching (threshold 95%) can produce false positives with different settlement terms. Manual map (`cross_platform_map.json`) entries have confidence=1.0. Unverified fuzzy matches are logged but blocked from trading (confidence=0).
- **BookCache threading**: single-writer (WS thread) + single-reader (main loop) only. Not safe for concurrent reads.
- **CoinGecko rate limits**: free tier 429s fast. `GasOracle` falls back to default $0.50 POL/USD.
- **25K+ markets**: full REST scan takes 30s+ for binary alone. Use `--limit` for dev. `--limit` never truncates negRisk markets (they need complete outcome sets).
- `neg_risk=True` markets excluded from binary scanner (handled by negrisk scanner).
- `SafetyCheckFailed` → skip opportunity. `CircuitBreakerTripped` → halt bot.
- Config from `.env` via pydantic-settings. All fields/defaults in `config.py`.

## Run Modes

- `--dry-run` — unauthenticated CLOB client, public endpoints only, no wallet needed
- `--scan-only` — full detection pipeline but skips execution/sizing/safety, needs wallet
- Default (no flags) — paper trading with simulated fills, full pipeline
- `--live` — real FAK orders on Polymarket, **use cautiously**

## Tests

`respx` mocks httpx, `MagicMock` for CLOB client. Factories: `_make_book()`, `_make_market()`, `_make_opp()`. 623 tests, 89% coverage.
