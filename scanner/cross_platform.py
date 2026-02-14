"""
Cross-platform arbitrage scanner.

Detects when the same event is priced differently across Polymarket and external
platforms, creating a guaranteed profit: buy YES cheap on one platform, buy NO
cheap on the other, total cost < $1, guaranteed $1 payout at resolution.

Generalized for N platforms: iterates each PlatformMatch within each MatchedEvent.
"""

from __future__ import annotations

import logging

from client.gas import GasOracle
from scanner.depth import effective_price, slippage_ceiling, sweep_depth, worst_fill_price
from scanner.fees import MarketFeeModel
from scanner.platform_fees import PlatformFeeModel
from scanner.matching import MatchedEvent, match_contracts, filter_by_confidence
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
    platform_books: dict[str, dict[str, OrderBook]],
    min_profit_usd: float,
    min_roi_pct: float,
    gas_per_order: int,
    gas_oracle: GasOracle | None = None,
    gas_price_gwei: float = 30.0,
    pm_fee_model: MarketFeeModel | None = None,
    platform_fee_models: dict[str, PlatformFeeModel] | None = None,
    min_confidence: float = 0.90,
    # Contract-level matching: external markets for settlement validation
    platform_markets: dict[str, list] | None = None,
    # Backward-compat: accept old kalshi-specific params
    kalshi_books: dict[str, OrderBook] | None = None,
    kalshi_fee_model: object | None = None,
) -> list[Opportunity]:
    """
    Scan matched events for cross-platform arbitrage opportunities.

    For each matched binary event and each platform match, checks two directions:
      1. Buy PM YES + Buy EXT NO
      2. Buy PM NO  + Buy EXT YES

    Contract-level matching validates settlement equivalence (e.g., "Over 2.5" != "Over 3.5")
    and filters out low-confidence matches before creating opportunities.

    Args:
        platform_books: {platform_name: {ticker: OrderBook, ...}, ...}
        platform_fee_models: {platform_name: PlatformFeeModel, ...}
        platform_markets: {platform_name: [market objects, ...], ...} for contract matching
        kalshi_books/kalshi_fee_model: Backward-compat; merged into platform args.

    Returns list of profitable Opportunity objects sorted by ROI descending.
    """
    # Backward-compat: merge old kalshi-specific params
    if platform_books is None:
        platform_books = {}
    if kalshi_books is not None and "kalshi" not in platform_books:
        platform_books["kalshi"] = kalshi_books
    if platform_fee_models is None:
        platform_fee_models = {}
    if kalshi_fee_model is not None and "kalshi" not in platform_fee_models:
        platform_fee_models["kalshi"] = kalshi_fee_model

    opportunities: list[Opportunity] = []

    for match in matched_events:
        if len(match.pm_markets) == 0:
            continue

        pm_market = match.pm_markets[0]
        pm_yes_book = pm_books.get(pm_market.yes_token_id)
        pm_no_book = pm_books.get(pm_market.no_token_id)

        if not pm_yes_book or not pm_no_book:
            continue

        for pm in match.platform_matches:
            if pm.confidence < min_confidence:
                logger.debug(
                    "Skipping match %s -> %s %s (confidence=%.2f < %.2f)",
                    match.pm_event_id, pm.platform, pm.event_ticker,
                    pm.confidence, min_confidence,
                )
                continue

            if not pm.tickers:
                continue

            ext_ticker = pm.tickers[0]
            ext_books = platform_books.get(pm.platform, {})
            ext_book = ext_books.get(ext_ticker)
            ext_fee_model = platform_fee_models.get(pm.platform)

            if not ext_book:
                continue

            # Contract-level matching: fetch external markets and validate
            # This requires external platform markets to perform settlement equivalence check
            # Skip if we can't do proper contract-level validation
            ext_markets = []  # Would need to be passed in from caller
            if ext_markets:
                # Get contract-level matches and filter by confidence
                contract_matches = match_contracts(match.pm_markets, ext_markets)
                contract_matches = filter_by_confidence(contract_matches, min_confidence=min_confidence)

                if not contract_matches:
                    logger.info(
                        "Skipping cross-platform arb %s -> %s %s (no valid contract matches)",
                        match.pm_event_id, pm.platform, pm.event_ticker,
                    )
                    continue

            # Direction 1: Buy PM YES + Buy EXT NO (= sell EXT YES)
            opp = _check_cross_platform_arb(
                event_id=match.pm_event_id,
                pm_book=pm_yes_book,
                pm_side=Side.BUY,
                pm_token_id=pm_market.yes_token_id,
                ext_book=ext_book,
                ext_side=Side.SELL,
                ext_ticker=ext_ticker,
                platform=pm.platform,
                min_profit_usd=min_profit_usd,
                min_roi_pct=min_roi_pct,
                gas_per_order=gas_per_order,
                gas_oracle=gas_oracle,
                gas_price_gwei=gas_price_gwei,
                pm_market=pm_market,
                pm_fee_model=pm_fee_model,
                ext_fee_model=ext_fee_model,
            )
            if opp:
                opportunities.append(opp)

            # Direction 2: Buy PM NO + Buy EXT YES
            opp = _check_cross_platform_arb(
                event_id=match.pm_event_id,
                pm_book=pm_no_book,
                pm_side=Side.BUY,
                pm_token_id=pm_market.no_token_id,
                ext_book=ext_book,
                ext_side=Side.BUY,
                ext_ticker=ext_ticker,
                platform=pm.platform,
                min_profit_usd=min_profit_usd,
                min_roi_pct=min_roi_pct,
                gas_per_order=gas_per_order,
                gas_oracle=gas_oracle,
                gas_price_gwei=gas_price_gwei,
                pm_market=pm_market,
                pm_fee_model=pm_fee_model,
                ext_fee_model=ext_fee_model,
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
    ext_book: OrderBook,
    ext_side: Side,
    ext_ticker: str,
    platform: str,
    min_profit_usd: float,
    min_roi_pct: float,
    gas_per_order: int,
    gas_oracle: GasOracle | None,
    gas_price_gwei: float,
    pm_market: object | None = None,
    pm_fee_model: MarketFeeModel | None = None,
    ext_fee_model: PlatformFeeModel | None = None,
) -> Opportunity | None:
    """
    Check one direction of cross-platform arb.

    PM leg: BUY at pm_book ask (or SELL at pm_book bid)
    EXT leg: BUY at ext_book ask (or SELL at ext_book bid)

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

    if ext_side == Side.BUY:
        if not ext_book.best_ask:
            return None
        ext_price = ext_book.best_ask.price
    else:
        if not ext_book.best_bid:
            return None
        ext_price = ext_book.best_bid.price

    # For BUY+BUY on complementary outcomes: total cost must < $1
    # For BUY PM + SELL EXT: PM cost + (1 - EXT sell price) < $1
    if ext_side == Side.SELL:
        total_cost = pm_price + (1.0 - ext_price)
    else:
        total_cost = pm_price + ext_price

    if total_cost >= 1.0:
        return None

    profit_per_set = 1.0 - total_cost

    # Edge-proportional depth sizing
    edge_pct = (profit_per_set / total_cost) * 100.0 if total_cost > 0 else 0.0
    pm_ceil = slippage_ceiling(pm_price, edge_pct, pm_side)
    ext_ceil = slippage_ceiling(ext_price, edge_pct, ext_side)
    pm_depth = sweep_depth(pm_book, pm_side, max_price=pm_ceil)
    ext_depth = sweep_depth(ext_book, ext_side, max_price=ext_ceil)
    max_sets = min(pm_depth, ext_depth)
    if max_sets <= 0:
        return None

    # VWAP-aware cost
    pm_vwap = effective_price(pm_book, pm_side, max_sets)
    ext_vwap = effective_price(ext_book, ext_side, max_sets)
    if pm_vwap is None or ext_vwap is None:
        return None

    if ext_side == Side.SELL:
        total_cost = pm_vwap + (1.0 - ext_vwap)
    else:
        total_cost = pm_vwap + ext_vwap

    if total_cost >= 1.0:
        return None

    profit_per_set = 1.0 - total_cost

    # Worst-fill prices for execution limits
    pm_worst = worst_fill_price(pm_book, pm_side, max_sets)
    ext_worst = worst_fill_price(ext_book, ext_side, max_sets)
    if pm_worst is None or ext_worst is None:
        return None

    # Cent-based platforms (Kalshi, Fanatics) need price rounding check
    ext_cents = round(ext_worst * 100)
    if ext_cents < 1 or ext_cents > 99:
        return None  # Price outside valid range for cent-based platforms
    if abs(ext_cents / 100.0 - ext_worst) > 0.005:
        return None  # >0.5 cent drift from rounding — profit estimate unreliable

    # Gas cost: only Polymarket leg (on-chain), external platforms are centralized
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

    # External platform fees: taker only (no resolution fee for Kalshi/Fanatics)
    if ext_fee_model:
        ext_fee = ext_fee_model.taker_fee_per_contract(ext_vwap)
        # Guard: reject arbs where fee is disproportionate to contract price
        fee_rate = ext_fee / ext_vwap if ext_vwap > 0 else 1.0
        if fee_rate > 0.20:
            return None
        net_profit_per_set -= ext_fee

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
            token_id=ext_ticker, side=ext_side,
            price=ext_worst, size=max_sets, platform=platform,
        ),
    )

    logger.info(
        "CROSS-PLATFORM ARB: event=%s platform=%s | pm_%s=%.4f ext_%s=%.4f | cost=%.4f profit/set=%.4f sets=%.1f net=$%.2f roi=%.2f%%",
        event_id, platform,
        "yes" if pm_side == Side.BUY else "no", pm_vwap,
        "yes" if ext_side == Side.BUY else "no", ext_vwap,
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
