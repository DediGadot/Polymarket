"""
Stale quote sniping detector.
Detects when a WebSocket price update reveals a significant move in one token,
and the complementary token's orderbook hasn't caught up yet.

This is for binary (YES/NO) markets where the two tokens should sum to $1.
If YES moves up 5% but NO hasn't moved down, there's an arb opportunity.
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

from scanner.depth import effective_price, sweep_depth, worst_fill_price
from scanner.fees import MarketFeeModel
from scanner.models import (
    BookFetcher,
    Event,
    Market,
    Opportunity,
    OpportunityType,
    LegOrder,
    Side,
    OrderBook,
)

logger = logging.getLogger(__name__)

if TYPE_CHECKING:
    from scanner.book_cache import BookCache


@dataclass(frozen=True)
class StaleQuoteSignal:
    """Signal that a token moved significantly and complementary book should be checked."""
    moved_token_id: str
    stale_token_id: str  # Complementary token to check
    event_id: str
    market: Market  # The market containing both tokens
    move_pct: float
    old_price: float
    new_price: float
    timestamp: float = field(default_factory=time.time)


class StaleQuoteDetector:
    """
    Detects stale quote arbitrage opportunities from WebSocket price updates.

    Tracks prices per token and emits signals when:
    1. A token moves by > min_move_pct
    2. The token is not in cooldown
    3. Rate limit allows additional checks

    Rate limiting: max N REST book checks per second.
    """

    def __init__(
        self,
        min_move_pct: float = 3.0,
        max_staleness_ms: float = 500.0,
        cooldown_sec: float = 5.0,
        max_checks_per_sec: int = 10,
    ):
        """
        Args:
            min_move_pct: Minimum price move % to trigger signal
            max_staleness_ms: Maximum age of tracked price to consider valid (ms)
            cooldown_sec: Cooldown before emitting another signal for same token
            max_checks_per_sec: Rate limit for REST book checks
        """
        self._min_move_pct = min_move_pct
        self._max_staleness_ms = max_staleness_ms
        self._cooldown_sec = cooldown_sec
        self._max_checks_per_sec = max_checks_per_sec

        # token_id -> (price, timestamp)
        self._last_prices: dict[str, tuple[float, float]] = {}

        # token_id -> last signal timestamp
        self._cooldowns: dict[str, float] = {}

        # Rate limiting: check count in current window
        self._check_count: int = 0
        self._check_window_start: float = 0.0

    def on_price_update(
        self,
        token_id: str,
        price: float,
        timestamp: float,
        market: Market | None = None,
    ) -> StaleQuoteSignal | None:
        """
        Process a WebSocket price update.

        Args:
            token_id: Token that received price update
            price: New price (best bid or ask)
            timestamp: Update timestamp
            market: Market containing this token (needed to find complementary token)

        Returns:
            StaleQuoteSignal if significant move detected, None otherwise
        """
        now = time.time()

        # Get previous price for this token
        prev = self._last_prices.get(token_id)
        if prev is None:
            # First price for this token
            self._last_prices[token_id] = (price, timestamp)
            return None

        old_price, old_ts = prev

        # Check staleness of our tracked price
        staleness_ms = (timestamp - old_ts) * 1000
        if staleness_ms > self._max_staleness_ms:
            # Our tracked price is too old, update and skip
            self._last_prices[token_id] = (price, timestamp)
            return None

        # Compute % change
        if old_price <= 0:
            self._last_prices[token_id] = (price, timestamp)
            return None

        change_pct = abs(price - old_price) / old_price * 100

        # Update tracked price
        self._last_prices[token_id] = (price, timestamp)

        # Check if move exceeds threshold
        if change_pct < self._min_move_pct:
            return None

        # Check cooldown
        last_signal = self._cooldowns.get(token_id, 0)
        if now - last_signal < self._cooldown_sec:
            return None

        # Check rate limit
        if not self._allow_check(now):
            return None

        # Need market to find complementary token
        if market is None:
            return None

        # Determine complementary token
        if token_id == market.yes_token_id:
            stale_token_id = market.no_token_id
        elif token_id == market.no_token_id:
            stale_token_id = market.yes_token_id
        else:
            return None  # Not a YES/NO pair

        # Record cooldown
        self._cooldowns[token_id] = now

        return StaleQuoteSignal(
            moved_token_id=token_id,
            stale_token_id=stale_token_id,
            event_id=market.event_id,
            market=market,
            move_pct=change_pct,
            old_price=old_price,
            new_price=price,
            timestamp=now,
        )

    def _allow_check(self, now: float) -> bool:
        """Check if rate limit allows another REST book check."""
        # Reset window if more than 1 second passed
        if now - self._check_window_start >= 1.0:
            self._check_count = 0
            self._check_window_start = now

        if self._check_count >= self._max_checks_per_sec:
            return False

        self._check_count += 1
        return True

    def check_complementary_book(
        self,
        signal: StaleQuoteSignal,
        books: dict[str, OrderBook],
        fee_model: MarketFeeModel | None = None,
        gas_per_order: int = 100000,
        gas_price_gwei: float = 50.0,
        gas_cost_per_order: float = 0.005,
        min_profit_usd: float = 0.01,
        min_roi_pct: float = 1.0,
    ) -> Opportunity | None:
        """
        Check if the complementary token's book creates an arbitrage.

        Fetches the complementary token's book and checks if:
        - Combined cost of both sides < $1.00 (buy arb)
        - Combined proceeds > $1.00 (sell arb)

        Args:
            signal: Signal from on_price_update
            books: Dictionary of token_id -> OrderBook (must contain complementary token)
            fee_model: Fee model for profit calculation
            gas_per_order: Gas units per order
            gas_price_gwei: Gas price in gwei
            gas_cost_per_order: Estimated USD gas cost per order
            min_profit_usd: Minimum net profit (USD)
            min_roi_pct: Minimum ROI percentage

        Returns:
            Opportunity if arb exists, None otherwise
        """
        moved_book = books.get(signal.moved_token_id)
        stale_book = books.get(signal.stale_token_id)

        if not moved_book or not stale_book:
            return None

        # Check both buy and sell arbs
        opp = self._check_buy_arb(
            signal,
            moved_book,
            stale_book,
            fee_model,
            gas_per_order,
            gas_price_gwei,
            gas_cost_per_order,
            min_profit_usd,
            min_roi_pct,
        )
        if opp:
            return opp

        opp = self._check_sell_arb(
            signal,
            moved_book,
            stale_book,
            fee_model,
            gas_per_order,
            gas_price_gwei,
            gas_cost_per_order,
            min_profit_usd,
            min_roi_pct,
        )
        return opp

    def _check_buy_arb(
        self,
        signal: StaleQuoteSignal,
        moved_book: OrderBook,
        stale_book: OrderBook,
        fee_model: MarketFeeModel | None,
        gas_per_order: int,
        gas_price_gwei: float,
        gas_cost_per_order: float,
        min_profit_usd: float,
        min_roi_pct: float,
    ) -> Opportunity | None:
        """Check if buying both tokens costs less than $1.00."""
        yes_ask = moved_book.best_ask if moved_book.token_id == signal.market.yes_token_id else stale_book.best_ask
        no_ask = stale_book.best_ask if moved_book.token_id == signal.market.yes_token_id else moved_book.best_ask

        if not yes_ask or not no_ask:
            return None

        # Fast pre-check
        cost = yes_ask.price + no_ask.price
        if cost >= 1.0:
            return None

        # Check depth
        yes_depth = sweep_depth(
            moved_book if moved_book.token_id == signal.market.yes_token_id else stale_book,
            Side.BUY,
            max_price=yes_ask.price * 1.005,
        )
        no_depth = sweep_depth(
            stale_book if moved_book.token_id == signal.market.yes_token_id else moved_book,
            Side.BUY,
            max_price=no_ask.price * 1.005,
        )
        max_sets = min(yes_depth, no_depth)
        if max_sets <= 0:
            return None

        # VWAP pricing
        yes_vwap = effective_price(
            moved_book if moved_book.token_id == signal.market.yes_token_id else stale_book,
            Side.BUY,
            max_sets,
        )
        no_vwap = effective_price(
            stale_book if moved_book.token_id == signal.market.yes_token_id else moved_book,
            Side.BUY,
            max_sets,
        )
        if yes_vwap is None or no_vwap is None:
            return None

        cost = yes_vwap + no_vwap
        if cost >= 1.0:
            return None

        profit_per_set = 1.0 - cost

        # Worst-fill prices
        yes_worst = worst_fill_price(
            moved_book if moved_book.token_id == signal.market.yes_token_id else stale_book,
            Side.BUY,
            max_sets,
        )
        no_worst = worst_fill_price(
            stale_book if moved_book.token_id == signal.market.yes_token_id else moved_book,
            Side.BUY,
            max_sets,
        )
        if yes_worst is None or no_worst is None:
            return None

        # Create legs
        legs = (
            LegOrder(
                token_id=signal.market.yes_token_id,
                side=Side.BUY,
                price=yes_worst if moved_book.token_id == signal.market.yes_token_id else no_worst,
                size=max_sets,
                tick_size=signal.market.min_tick_size,
            ),
            LegOrder(
                token_id=signal.market.no_token_id,
                side=Side.BUY,
                price=no_worst if moved_book.token_id == signal.market.yes_token_id else yes_worst,
                size=max_sets,
                tick_size=signal.market.min_tick_size,
            ),
        )

        # Fee adjustment
        if fee_model:
            net_profit_per_set = fee_model.adjust_profit(profit_per_set, legs, market=signal.market)
        else:
            net_profit_per_set = profit_per_set

        gross_profit = profit_per_set * max_sets
        net_profit = net_profit_per_set * max_sets - gas_cost_per_order * 2
        required_capital = cost * max_sets
        roi_pct = (net_profit / required_capital * 100) if required_capital > 0 else 0

        if net_profit < min_profit_usd or roi_pct < min_roi_pct:
            return None

        logger.debug(
            "STALE QUOTE BUY: %s | moved=%s cost=%.4f profit/set=%.4f sets=%.1f net=$%.2f roi=%.2f%%",
            signal.market.question[:60], signal.moved_token_id, cost, profit_per_set, max_sets, net_profit, roi_pct,
        )

        return Opportunity(
            type=OpportunityType.STALE_QUOTE_ARB,
            event_id=signal.event_id,
            legs=legs,
            expected_profit_per_set=profit_per_set,
            net_profit_per_set=net_profit_per_set,
            max_sets=max_sets,
            gross_profit=gross_profit,
            estimated_gas_cost=gas_cost_per_order * 2,
            net_profit=net_profit,
            roi_pct=roi_pct,
            required_capital=required_capital,
        )

    def _check_sell_arb(
        self,
        signal: StaleQuoteSignal,
        moved_book: OrderBook,
        stale_book: OrderBook,
        fee_model: MarketFeeModel | None,
        gas_per_order: int,
        gas_price_gwei: float,
        gas_cost_per_order: float,
        min_profit_usd: float,
        min_roi_pct: float,
    ) -> Opportunity | None:
        """Check if selling both tokens yields more than $1.00."""
        yes_bid = moved_book.best_bid if moved_book.token_id == signal.market.yes_token_id else stale_book.best_bid
        no_bid = stale_book.best_bid if moved_book.token_id == signal.market.yes_token_id else moved_book.best_bid

        if not yes_bid or not no_bid:
            return None

        # Fast pre-check
        proceeds = yes_bid.price + no_bid.price
        if proceeds <= 1.0:
            return None

        # Check depth
        yes_depth = sweep_depth(
            moved_book if moved_book.token_id == signal.market.yes_token_id else stale_book,
            Side.SELL,
            max_price=yes_bid.price * 0.995,
        )
        no_depth = sweep_depth(
            stale_book if moved_book.token_id == signal.market.yes_token_id else moved_book,
            Side.SELL,
            max_price=no_bid.price * 0.995,
        )
        max_sets = min(yes_depth, no_depth)
        if max_sets <= 0:
            return None

        # VWAP pricing
        yes_vwap = effective_price(
            moved_book if moved_book.token_id == signal.market.yes_token_id else stale_book,
            Side.SELL,
            max_sets,
        )
        no_vwap = effective_price(
            stale_book if moved_book.token_id == signal.market.yes_token_id else moved_book,
            Side.SELL,
            max_sets,
        )
        if yes_vwap is None or no_vwap is None:
            return None

        proceeds = yes_vwap + no_vwap
        if proceeds <= 1.0:
            return None

        profit_per_set = proceeds - 1.0

        # Worst-fill prices
        yes_worst = worst_fill_price(
            moved_book if moved_book.token_id == signal.market.yes_token_id else stale_book,
            Side.SELL,
            max_sets,
        )
        no_worst = worst_fill_price(
            stale_book if moved_book.token_id == signal.market.yes_token_id else moved_book,
            Side.SELL,
            max_sets,
        )
        if yes_worst is None or no_worst is None:
            return None

        # Create legs
        legs = (
            LegOrder(
                token_id=signal.market.yes_token_id,
                side=Side.SELL,
                price=yes_worst if moved_book.token_id == signal.market.yes_token_id else no_worst,
                size=max_sets,
                tick_size=signal.market.min_tick_size,
            ),
            LegOrder(
                token_id=signal.market.no_token_id,
                side=Side.SELL,
                price=no_worst if moved_book.token_id == signal.market.yes_token_id else yes_worst,
                size=max_sets,
                tick_size=signal.market.min_tick_size,
            ),
        )

        # Fee adjustment (no resolution fee for sells)
        if fee_model:
            net_profit_per_set = fee_model.adjust_profit(
                profit_per_set, legs, market=signal.market, is_sell=True,
            )
        else:
            net_profit_per_set = profit_per_set

        gross_profit = profit_per_set * max_sets
        net_profit = net_profit_per_set * max_sets - gas_cost_per_order * 2
        required_capital = 1.0 * max_sets
        roi_pct = (net_profit / required_capital * 100) if required_capital > 0 else 0

        if net_profit < min_profit_usd or roi_pct < min_roi_pct:
            return None

        logger.debug(
            "STALE QUOTE SELL: %s | moved=%s proceeds=%.4f profit/set=%.4f sets=%.1f net=$%.2f roi=%.2f%%",
            signal.market.question[:60], signal.moved_token_id, proceeds, profit_per_set, max_sets, net_profit, roi_pct,
        )

        return Opportunity(
            type=OpportunityType.STALE_QUOTE_ARB,
            event_id=signal.event_id,
            legs=legs,
            expected_profit_per_set=profit_per_set,
            net_profit_per_set=net_profit_per_set,
            max_sets=max_sets,
            gross_profit=gross_profit,
            estimated_gas_cost=gas_cost_per_order * 2,
            net_profit=net_profit,
            roi_pct=roi_pct,
            required_capital=required_capital,
        )


def scan_stale_quote_signals(
    detector: StaleQuoteDetector,
    book_fetcher: BookFetcher,
    signals: list[StaleQuoteSignal],
    fee_model: MarketFeeModel | None = None,
    gas_per_order: int = 100000,
    gas_price_gwei: float = 50.0,
    gas_cost_per_order: float = 0.005,
    min_profit_usd: float = 0.01,
    min_roi_pct: float = 1.0,
) -> list[Opportunity]:
    """
    Process multiple stale quote signals and return opportunities.

    Fetches complementary books and checks for arb opportunities.

    Args:
        detector: StaleQuoteDetector instance
        book_fetcher: Function to fetch orderbooks
        signals: List of signals to process
        fee_model: Fee model
        gas_per_order: Gas units per order
        gas_price_gwei: Gas price in gwei
        gas_cost_per_order: Estimated USD gas cost per order
        min_profit_usd: Minimum net profit (USD)
        min_roi_pct: Minimum ROI percentage

    Returns:
        List of opportunities sorted by ROI descending
    """
    if not signals:
        return []

    # Collect all token IDs to fetch
    token_ids = set()
    for signal in signals:
        token_ids.add(signal.moved_token_id)
        token_ids.add(signal.stale_token_id)

    # Fetch all books
    books = book_fetcher(list(token_ids))

    opportunities: list[Opportunity] = []
    for signal in signals:
        opp = detector.check_complementary_book(
            signal,
            books,
            fee_model=fee_model,
            gas_per_order=gas_per_order,
            gas_price_gwei=gas_price_gwei,
            gas_cost_per_order=gas_cost_per_order,
            min_profit_usd=min_profit_usd,
            min_roi_pct=min_roi_pct,
        )
        if opp:
            opportunities.append(opp)

    opportunities.sort(key=lambda o: o.roi_pct, reverse=True)
    return opportunities
