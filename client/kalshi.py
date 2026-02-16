"""
Kalshi REST API v2 client. Market discovery, orderbook fetching, order placement.

Kalshi API docs: https://trading-api.readme.io/reference
All prices are in cents (1-99). We convert to dollars (0.01-0.99) to match our OrderBook model.
"""

from __future__ import annotations

import logging
import random
import time
from dataclasses import dataclass

import httpx

from client.kalshi_auth import KalshiAuth
from scanner.models import OrderBook, PriceLevel, BookFetcher
from scanner.validation import validate_price, validate_size

logger = logging.getLogger(__name__)

# Kalshi API rate limits: 20 reads/sec basic tier
DEFAULT_TIMEOUT = 10.0
_MAX_READS_PER_SEC = 10  # stay well under 20/sec limit
_PAGE_DELAY_SEC = 1.0 / _MAX_READS_PER_SEC  # 100ms between paginated requests
_429_MAX_RETRIES = 3
_429_BACKOFF_SEC = 5.0
_429_JITTER_FRAC = 0.15


def dollars_to_cents(price: float) -> int:
    """
    Convert dollar price (0.01-0.99) to Kalshi cents (1-99).
    Fail-fast on out-of-range prices.
    """
    cents = round(price * 100)
    if cents < 1 or cents > 99:
        raise ValueError(
            f"Kalshi price out of range: ${price:.4f} -> {cents} cents (must be 1-99)"
        )
    return cents


@dataclass(frozen=True, slots=True)
class KalshiMarket:
    """Minimal representation of a Kalshi market (event + ticker)."""
    ticker: str
    event_ticker: str
    title: str
    subtitle: str
    yes_sub_title: str
    no_sub_title: str
    status: str  # "open", "closed", "settled"
    result: str  # "", "yes", "no"


class KalshiClient:
    """
    Kalshi REST API v2 client.

    Handles authentication, market discovery, orderbook fetching,
    and order placement. Converts Kalshi's cents-based pricing to
    our dollar-based OrderBook model.

    Satisfies the PlatformClient protocol for cross-platform integration.
    """

    @property
    def platform_name(self) -> str:
        return "kalshi"

    def __init__(
        self,
        auth: KalshiAuth,
        host: str = "https://trading-api.kalshi.com/trade-api/v2",
        demo: bool = False,
    ) -> None:
        self._auth = auth
        self._host = host.rstrip("/")
        if demo:
            self._host = "https://demo-api.kalshi.co/trade-api/v2"
        self._http = httpx.Client(timeout=DEFAULT_TIMEOUT)
        self._rate_limited_until: dict[str, float] = {}

    def _request(self, method: str, path: str, **kwargs) -> dict:
        """Make an authenticated request to Kalshi API. Retries on 429."""
        url = f"{self._host}{path}"
        # Sign with the full URL path (e.g. /trade-api/v2/markets), not the relative path.
        from urllib.parse import urlparse
        full_path = urlparse(url).path
        cooldown_key = f"{method}:{path}"

        now = time.time()
        blocked_until = self._rate_limited_until.get(cooldown_key, 0.0)
        if blocked_until > now:
            wait = blocked_until - now
            logger.debug(
                "Kalshi cooldown active on %s %s, sleeping %.1fs",
                method, path, wait,
            )
            time.sleep(wait)

        for attempt in range(_429_MAX_RETRIES + 1):
            headers = self._auth.sign_request(method, full_path)
            headers["Content-Type"] = "application/json"
            headers["Accept"] = "application/json"

            resp = self._http.request(method, url, headers=headers, **kwargs)
            if resp.status_code != 429:
                self._rate_limited_until.pop(cooldown_key, None)
                resp.raise_for_status()
                return resp.json()

            # 429 Too Many Requests — back off and retry
            retry_after = 0.0
            raw_retry_after = resp.headers.get("Retry-After")
            if raw_retry_after:
                try:
                    retry_after = max(0.0, float(raw_retry_after))
                except ValueError:
                    retry_after = 0.0
            wait = max(retry_after, _429_BACKOFF_SEC * (2 ** attempt))
            wait *= 1.0 + random.uniform(-_429_JITTER_FRAC, _429_JITTER_FRAC)
            wait = max(0.5, wait)
            self._rate_limited_until[cooldown_key] = time.time() + wait
            logger.warning(
                "Kalshi 429 rate limited on %s %s (attempt %d/%d, waiting %.1fs)",
                method, path, attempt + 1, _429_MAX_RETRIES + 1, wait,
            )
            time.sleep(wait)

        # All retries exhausted
        resp.raise_for_status()
        return resp.json()  # unreachable, raise_for_status throws

    # -- Market Discovery --

    def get_markets(
        self,
        event_ticker: str | None = None,
        status: str = "open",
        limit: int = 200,
        cursor: str | None = None,
    ) -> tuple[list[KalshiMarket], str | None]:
        """
        Fetch markets with optional filters. Returns (markets, next_cursor).
        """
        params: dict = {"limit": limit, "status": status}
        if event_ticker:
            params["event_ticker"] = event_ticker
        if cursor:
            params["cursor"] = cursor

        data = self._request("GET", "/markets", params=params)
        markets = [
            KalshiMarket(
                ticker=m["ticker"],
                event_ticker=m.get("event_ticker", ""),
                title=m.get("title", ""),
                subtitle=m.get("subtitle", ""),
                yes_sub_title=m.get("yes_sub_title", ""),
                no_sub_title=m.get("no_sub_title", ""),
                status=m.get("status", ""),
                result=m.get("result", ""),
            )
            for m in data.get("markets", [])
        ]
        next_cursor = data.get("cursor", None)
        # Empty cursor means no more pages
        if next_cursor == "":
            next_cursor = None
        return markets, next_cursor

    def get_all_markets(
        self,
        event_ticker: str | None = None,
        status: str = "open",
    ) -> list[KalshiMarket]:
        """Fetch all markets with pagination. Rate-limited to stay under API limits."""
        all_markets: list[KalshiMarket] = []
        cursor = None
        page = 0
        while True:
            markets, cursor = self.get_markets(
                event_ticker=event_ticker,
                status=status,
                cursor=cursor,
            )
            all_markets.extend(markets)
            page += 1
            if not cursor or not markets:
                break
            # Rate limit: pause between pages to stay under 20 reads/sec
            time.sleep(_PAGE_DELAY_SEC)
        logger.debug("Fetched %d Kalshi markets in %d pages", len(all_markets), page)
        return all_markets

    # -- Orderbook --

    def get_orderbook(self, ticker: str) -> OrderBook:
        """
        Fetch orderbook for a single market ticker.
        Converts Kalshi cents (1-99) to dollars (0.01-0.99).
        Returns our OrderBook model with sorted levels.
        """
        data = self._request("GET", f"/markets/{ticker}/orderbook")
        ob = data.get("orderbook", data)

        # Kalshi format: {"yes": [[price_cents, size], ...], "no": [[price_cents, size], ...]}
        # We model YES token: bids = yes bids, asks = yes asks
        # Kalshi gives "yes" and "no" arrays of [price, quantity] where price is in cents
        yes_bids_raw = ob.get("yes") or []
        no_bids_raw = ob.get("no") or []

        # Kalshi orderbook: "yes" contains bid levels for YES side (sorted desc by price)
        # To get asks for YES: NO bids at price P means YES ask at (100 - P) cents
        bids = tuple(sorted(
            (
                PriceLevel(
                    price=validate_price(lvl[0] / 100.0, context="Kalshi YES bid price"),
                    size=validate_size(float(lvl[1]), context="Kalshi YES bid size"),
                )
                for lvl in yes_bids_raw if lvl[1] > 0
            ),
            key=lambda level: level.price,
            reverse=True,
        ))

        asks = tuple(sorted(
            (
                PriceLevel(
                    price=validate_price((100 - lvl[0]) / 100.0, context="Kalshi YES ask price"),
                    size=validate_size(float(lvl[1]), context="Kalshi YES ask size"),
                )
                for lvl in no_bids_raw if lvl[1] > 0
            ),
            key=lambda level: level.price,
        ))

        return OrderBook(token_id=ticker, bids=bids, asks=asks)

    def get_orderbooks(self, tickers: list[str], max_workers: int = 4) -> dict[str, OrderBook]:
        """
        Fetch orderbooks for multiple tickers in parallel using ThreadPoolExecutor.

        Args:
            tickers: List of market tickers to fetch
            max_workers: Maximum number of parallel threads (default: 4)

        Returns:
            Dictionary mapping ticker to OrderBook ( skips failed tickers with warning)
        """
        if not tickers:
            return {}

        if len(tickers) == 1:
            # Single ticker - no need for threading overhead
            try:
                return {tickers[0]: self.get_orderbook(tickers[0])}
            except httpx.HTTPStatusError as e:
                logger.warning("Failed to fetch Kalshi orderbook for %s: %s", tickers[0], e)
                return {}

        from concurrent.futures import ThreadPoolExecutor, as_completed

        result: dict[str, OrderBook] = {}

        def _fetch_single(ticker: str) -> tuple[str, OrderBook | None]:
            """Fetch a single ticker, returning (ticker, book) or (ticker, None) on error."""
            try:
                return ticker, self.get_orderbook(ticker)
            except httpx.HTTPStatusError as e:
                logger.warning("Failed to fetch Kalshi orderbook for %s: %s", ticker, e)
                return ticker, None

        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            futures = {executor.submit(_fetch_single, ticker): ticker for ticker in tickers}
            for future in as_completed(futures):
                ticker, book = future.result()
                if book is not None:
                    result[ticker] = book

        return result

    @property
    def book_fetcher(self) -> BookFetcher:
        """Return a BookFetcher-compatible callable for use with scanners."""
        return self.get_orderbooks

    # -- Orders --

    def place_order(
        self,
        ticker: str,
        side: str,
        action: str = "buy",
        count: int = 1,
        type: str = "limit",
        yes_price: int | None = None,
        no_price: int | None = None,
        expiration_ts: int | None = None,
    ) -> dict:
        """
        Place an order on Kalshi.

        Args:
            ticker: Market ticker
            side: "yes" or "no"
            action: "buy" or "sell"
            count: Number of contracts
            type: "limit" or "market"
            yes_price: Limit price in cents for YES side (1-99)
            no_price: Limit price in cents for NO side (1-99)
            expiration_ts: Unix timestamp for order expiry (None = GTC)
        """
        body: dict = {
            "ticker": ticker,
            "side": side,
            "action": action,
            "count": count,
            "type": type,
        }
        if yes_price is not None:
            body["yes_price"] = yes_price
        if no_price is not None:
            body["no_price"] = no_price
        if expiration_ts is not None:
            body["expiration_ts"] = expiration_ts

        return self._request("POST", "/portfolio/orders", json=body)

    def batch_place_orders(self, orders: list[dict]) -> list[dict]:
        """
        Place multiple orders in a batch (max 20 per Kalshi docs).
        Each order dict has same format as place_order body.
        """
        if len(orders) > 20:
            raise ValueError(f"Kalshi batch limit is 20 orders, got {len(orders)}")
        return self._request("POST", "/portfolio/orders/batched", json={"orders": orders})

    def cancel_order(self, order_id: str) -> dict:
        """Cancel a single order by ID."""
        return self._request("DELETE", f"/portfolio/orders/{order_id}")

    def get_order(self, order_id: str) -> dict:
        """Get order status by ID."""
        return self._request("GET", f"/portfolio/orders/{order_id}")

    # -- Account --

    def get_positions(self) -> list[dict]:
        """Get current positions."""
        data = self._request("GET", "/portfolio/positions")
        return data.get("market_positions", [])

    def get_balance(self) -> float:
        """Get account balance in dollars."""
        data = self._request("GET", "/portfolio/balance")
        return data.get("balance", 0) / 100.0  # cents -> dollars

    def close(self) -> None:
        """Close the underlying HTTP client."""
        self._http.close()
