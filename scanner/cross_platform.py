"""
Cross-platform arbitrage scanner.

Detects when the same event is priced differently across Polymarket and Kalshi,
creating a guaranteed profit: buy YES cheap on one platform, buy NO cheap on the other,
total cost < $1, guaranteed $1 payout at resolution.
"""

from __future__ import annotations

import logging

from client.gas import GasOracle
from scanner.depth import effective_price, sweep_depth, worst_fill_price
from scanner.fees import MarketFeeModel
from scanner.kalshi_fees import KalshiFeeModel
from scanner.matching import MatchedEvent
from scanner.models import (
    LegOrder,
    Opportunity,
    OpportunityType,
    OrderBook,
    Side,
)

logger = logging.getLogger(__name__)


def scan_cross_platform(
    matched_events: list[MatchedEvent],
    pm_books: dict[str, OrderBook],
    kalshi_books: dict[str, OrderBook],
    min_profit_usd: float,
    min_roi_pct: float,
    gas_per_order: int,
    gas_oracle: GasOracle | None = None,
    gas_price_gwei: float = 30.0,
    pm_fee_model: MarketFeeModel | None = None,
    kalshi_fee_model: KalshiFeeModel | None = None,
    min_confidence: float = 0.90,
) -> list[Opportunity]:
    """
    Scan matched events for cross-platform arbitrage opportunities.

    For each matched binary event, checks two directions:
      1. Buy PM YES + Buy Kalshi NO  (if PM_yes_ask + Kalshi_no_ask < $1)
      2. Buy PM NO  + Buy Kalshi YES (if PM_no_ask + Kalshi_yes_ask < $1)

    Returns list of profitable Opportunity objects sorted by ROI descending.
    """
    opportunities: list[Opportunity] = []

    for match in matched_events:
        if match.confidence < min_confidence:
            logger.debug(
                "Skipping match %s -> %s (confidence=%.2f < %.2f)",
                match.pm_event_id, match.kalshi_event_ticker,
                match.confidence, min_confidence,
            )
            continue

        # For cross-platform arb, we need a 1:1 binary mapping
        # Each PM binary market maps to one Kalshi ticker
        if len(match.pm_markets) == 0 or len(match.kalshi_tickers) == 0:
            continue

        # Use first PM binary market and first Kalshi ticker
        pm_market = match.pm_markets[0]
        kalshi_ticker = match.kalshi_tickers[0]

        pm_yes_book = pm_books.get(pm_market.yes_token_id)
        pm_no_book = pm_books.get(pm_market.no_token_id)
        kalshi_book = kalshi_books.get(kalshi_ticker)

        if not pm_yes_book or not pm_no_book or not kalshi_book:
            continue

        # Direction 1: Buy PM YES + Buy Kalshi NO
        # Kalshi NO ask price = 1 - kalshi_yes_bid? No -- Kalshi book is already
        # modeled as YES token. To buy NO on Kalshi, we need kalshi NO ask.
        # In our OrderBook model for Kalshi, asks are YES asks.
        # To get Kalshi NO ask: it's (1 - kalshi YES bid price).
        # Actually, we can think of it as: if we want NO on Kalshi,
        # we buy it at (1 - best_bid) equivalent. But the orderbook structure
        # already gives us asks from the NO side.
        #
        # Simpler approach: Kalshi's YES ask represents the cost to go YES.
        # The NO cost = 1 - YES bid price (selling YES is equiv to buying NO).
        # But for the depth-aware approach, we'd need the actual NO orderbook.
        #
        # For binary markets on both platforms:
        # PM has separate YES and NO books.
        # Kalshi: our model has YES bids and YES asks.
        # To buy Kalshi NO: effectively we sell Kalshi YES. So Kalshi NO cost = 1 - kalshi_yes_bid.
        # But actually, we should just use the Kalshi book directly. The asks represent YES cost.
        #
        # Direction 1: PM YES ask + (1 - Kalshi YES bid) < 1
        #   => PM YES ask < Kalshi YES bid (buy PM YES cheap, sell Kalshi YES high equiv)
        #   Actually this doesn't work for arb -- we need to BUY on both sides.
        #
        # Correct formulation:
        # Direction 1: Buy PM YES + Buy Kalshi NO
        #   PM YES cost = pm_yes_ask
        #   Kalshi NO cost = 1 - kalshi_yes_bid (NO ask = complement of YES bid)
        #   Total cost < 1 is arb
        #
        # Direction 2: Buy PM NO + Buy Kalshi YES
        #   PM NO cost = pm_no_ask
        #   Kalshi YES cost = kalshi_yes_ask
        #   Total cost < 1 is arb

        # Direction 1: Buy PM YES + Buy Kalshi NO (= sell Kalshi YES)
        opp = _check_cross_platform_arb(
            event_id=match.pm_event_id,
            pm_book=pm_yes_book,
            pm_side=Side.BUY,
            pm_token_id=pm_market.yes_token_id,
            kalshi_book=kalshi_book,
            kalshi_side=Side.SELL,  # Selling YES on Kalshi = buying NO
            kalshi_ticker=kalshi_ticker,
            min_profit_usd=min_profit_usd,
            min_roi_pct=min_roi_pct,
            gas_per_order=gas_per_order,
            gas_oracle=gas_oracle,
            gas_price_gwei=gas_price_gwei,
            pm_market=pm_market,
            pm_fee_model=pm_fee_model,
            kalshi_fee_model=kalshi_fee_model,
        )
        if opp:
            opportunities.append(opp)

        # Direction 2: Buy PM NO + Buy Kalshi YES
        opp = _check_cross_platform_arb(
            event_id=match.pm_event_id,
            pm_book=pm_no_book,
            pm_side=Side.BUY,
            pm_token_id=pm_market.no_token_id,
            kalshi_book=kalshi_book,
            kalshi_side=Side.BUY,  # Buying YES on Kalshi
            kalshi_ticker=kalshi_ticker,
            min_profit_usd=min_profit_usd,
            min_roi_pct=min_roi_pct,
            gas_per_order=gas_per_order,
            gas_oracle=gas_oracle,
            gas_price_gwei=gas_price_gwei,
            pm_market=pm_market,
            pm_fee_model=pm_fee_model,
            kalshi_fee_model=kalshi_fee_model,
        )
        if opp:
            opportunities.append(opp)

    opportunities.sort(key=lambda o: o.roi_pct, reverse=True)
    return opportunities


def _check_cross_platform_arb(
    event_id: str,
    pm_book: OrderBook,
    pm_side: Side,
    pm_token_id: str,
    kalshi_book: OrderBook,
    kalshi_side: Side,
    kalshi_ticker: str,
    min_profit_usd: float,
    min_roi_pct: float,
    gas_per_order: int,
    gas_oracle: GasOracle | None,
    gas_price_gwei: float,
    pm_market: object | None = None,
    pm_fee_model: MarketFeeModel | None = None,
    kalshi_fee_model: KalshiFeeModel | None = None,
) -> Opportunity | None:
    """
    Check one direction of cross-platform arb.

    PM leg: BUY at pm_book ask (or SELL at pm_book bid)
    Kalshi leg: BUY at kalshi_book ask (or SELL at kalshi_book bid)

    For arb: total cost < $1 when buying complementary outcomes across platforms.
    """
    # Get best prices
    if pm_side == Side.BUY:
        if not pm_book.best_ask:
            return None
        pm_price = pm_book.best_ask.price
    else:
        if not pm_book.best_bid:
            return None
        pm_price = pm_book.best_bid.price

    if kalshi_side == Side.BUY:
        if not kalshi_book.best_ask:
            return None
        kalshi_price = kalshi_book.best_ask.price
    else:
        if not kalshi_book.best_bid:
            return None
        kalshi_price = kalshi_book.best_bid.price

    # For BUY+BUY on complementary outcomes: total cost must < $1
    # For BUY PM + SELL Kalshi: PM cost + (1 - Kalshi sell price) < $1
    #   => PM cost < Kalshi sell price
    if kalshi_side == Side.SELL:
        total_cost = pm_price + (1.0 - kalshi_price)
    else:
        total_cost = pm_price + kalshi_price

    if total_cost >= 1.0:
        return None

    profit_per_set = 1.0 - total_cost

    # Depth-aware sizing
    pm_slippage = 1.005 if pm_side == Side.BUY else 0.995
    kalshi_slippage = 1.005 if kalshi_side == Side.BUY else 0.995
    pm_depth = sweep_depth(pm_book, pm_side, max_price=pm_price * pm_slippage)
    kalshi_depth = sweep_depth(kalshi_book, kalshi_side, max_price=kalshi_price * kalshi_slippage)
    max_sets = min(pm_depth, kalshi_depth)
    if max_sets <= 0:
        return None

    # VWAP-aware cost
    pm_vwap = effective_price(pm_book, pm_side, max_sets)
    kalshi_vwap = effective_price(kalshi_book, kalshi_side, max_sets)
    if pm_vwap is None or kalshi_vwap is None:
        return None

    if kalshi_side == Side.SELL:
        total_cost = pm_vwap + (1.0 - kalshi_vwap)
    else:
        total_cost = pm_vwap + kalshi_vwap

    if total_cost >= 1.0:
        return None

    profit_per_set = 1.0 - total_cost

    # Worst-fill prices for execution limits
    pm_worst = worst_fill_price(pm_book, pm_side, max_sets)
    kalshi_worst = worst_fill_price(kalshi_book, kalshi_side, max_sets)
    if pm_worst is None or kalshi_worst is None:
        return None

    # Kalshi prices must survive cent rounding without meaningful drift
    kalshi_cents = round(kalshi_worst * 100)
    if kalshi_cents < 1 or kalshi_cents > 99:
        return None  # Price outside Kalshi's valid range
    if abs(kalshi_cents / 100.0 - kalshi_worst) > 0.005:
        return None  # >0.5 cent drift from rounding — profit estimate unreliable

    # Gas cost: only Polymarket leg (on-chain), Kalshi is centralized
    if gas_oracle:
        gas_cost_usd = gas_oracle.estimate_cost_usd(1, gas_per_order)
    else:
        gas_cost_wei = 1 * gas_per_order * gas_price_gwei * 1e9
        gas_cost_matic = gas_cost_wei / 1e18
        gas_cost_usd = gas_cost_matic * 0.50

    # Fee adjustment
    net_profit_per_set = profit_per_set

    # PM fees: taker + resolution
    if pm_fee_model and pm_market:
        pm_leg = LegOrder(
            token_id=pm_token_id, side=pm_side,
            price=pm_vwap, size=max_sets, platform="polymarket",
        )
        pm_fee_adj = pm_fee_model.adjust_profit(profit_per_set, (pm_leg,), market=pm_market)
        pm_fee_deducted = profit_per_set - pm_fee_adj
        net_profit_per_set -= pm_fee_deducted

    # Kalshi fees: taker only (no resolution fee)
    if kalshi_fee_model:
        kalshi_fee = kalshi_fee_model.taker_fee_per_contract(kalshi_vwap)
        # Guard: reject arbs where Kalshi fee is disproportionate to contract price
        # At extreme prices, ceil() rounding creates a $0.01 floor (e.g., 50% at $0.02)
        fee_rate = kalshi_fee / kalshi_vwap if kalshi_vwap > 0 else 1.0
        if fee_rate > 0.20:
            return None
        net_profit_per_set -= kalshi_fee

    gross_profit = profit_per_set * max_sets
    net_profit = net_profit_per_set * max_sets - gas_cost_usd
    required_capital = total_cost * max_sets
    roi_pct = (net_profit / required_capital * 100) if required_capital > 0 else 0

    if net_profit < min_profit_usd or roi_pct < min_roi_pct:
        return None

    legs = (
        LegOrder(
            token_id=pm_token_id, side=pm_side,
            price=pm_worst, size=max_sets, platform="polymarket",
        ),
        LegOrder(
            token_id=kalshi_ticker, side=kalshi_side,
            price=kalshi_worst, size=max_sets, platform="kalshi",
        ),
    )

    logger.info(
        "CROSS-PLATFORM ARB: event=%s | pm_%s=%.4f kalshi_%s=%.4f | cost=%.4f profit/set=%.4f sets=%.1f net=$%.2f roi=%.2f%%",
        event_id,
        "yes" if pm_side == Side.BUY else "no", pm_vwap,
        "yes" if kalshi_side == Side.BUY else "no", kalshi_vwap,
        total_cost, profit_per_set, max_sets, net_profit, roi_pct,
    )

    return Opportunity(
        type=OpportunityType.CROSS_PLATFORM_ARB,
        event_id=event_id,
        legs=legs,
        expected_profit_per_set=profit_per_set,
        net_profit_per_set=net_profit_per_set,
        max_sets=max_sets,
        gross_profit=gross_profit,
        estimated_gas_cost=gas_cost_usd,
        net_profit=net_profit,
        roi_pct=roi_pct,
        required_capital=required_capital,
    )
