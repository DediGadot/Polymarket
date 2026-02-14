"""
Trade execution engine. Handles order placement, fill tracking, and partial fill unwinding.
"""

from __future__ import annotations

import logging
import time

from py_clob_client.client import ClobClient
from py_clob_client.clob_types import OrderType

from client.clob import (
    create_limit_order,
    create_market_order,
    post_order,
    post_orders,
    cancel_order,
)
from client.platform import PlatformClient
from executor.tick_size import quantize_price
from scanner.models import (
    Opportunity,
    OpportunityType,
    LegOrder,
    Side,
    TradeResult,
)

logger = logging.getLogger(__name__)

# Max orders per batch on Polymarket CLOB
MAX_BATCH_SIZE = 15


class UnwindFailed(Exception):
    """Raised when partial fill unwind fails. Caller must log stuck positions."""
    pass


def execute_opportunity(
    client: ClobClient,
    opportunity: Opportunity,
    size: float,
    paper_trading: bool = False,
    use_fak: bool = True,
    order_timeout_sec: float = 5.0,
    kalshi_client: PlatformClient | None = None,
    platform_clients: dict[str, PlatformClient] | None = None,
    cross_platform_deadline_sec: float = 5.0,
) -> TradeResult:
    """
    Execute an arbitrage opportunity. Places all leg orders, tracks fills.
    Returns a TradeResult with execution details.

    For paper trading, simulates execution without placing real orders.
    use_fak: Use Fill-and-Kill orders instead of GTC (immediate fill or cancel).
    platform_clients: Dict of platform_name -> PlatformClient for cross-platform arbs.
    kalshi_client: Backward-compat alias; merged into platform_clients if provided.
    """
    start_time = time.time()
    order_type = OrderType.FAK if use_fak else OrderType.GTC

    if paper_trading:
        return _paper_execute(opportunity, size, start_time)

    if opportunity.type == OpportunityType.BINARY_REBALANCE:
        return _execute_binary(client, opportunity, size, start_time, order_type, order_timeout_sec)
    elif opportunity.type == OpportunityType.NEGRISK_REBALANCE:
        return _execute_negrisk(client, opportunity, size, start_time, order_type, order_timeout_sec)
    elif opportunity.type == OpportunityType.LATENCY_ARB:
        return _execute_single_leg(client, opportunity, size, start_time, order_type)
    elif opportunity.type == OpportunityType.SPIKE_LAG:
        return _execute_negrisk(client, opportunity, size, start_time, order_type, order_timeout_sec)
    elif opportunity.type == OpportunityType.CROSS_PLATFORM_ARB:
        from executor.cross_platform import execute_cross_platform
        # Build platform_clients dict, merging backward-compat kalshi_client
        all_clients = dict(platform_clients) if platform_clients else {}
        if kalshi_client is not None and "kalshi" not in all_clients:
            all_clients["kalshi"] = kalshi_client
        if not all_clients:
            raise ValueError("platform_clients required for CROSS_PLATFORM_ARB execution")
        return execute_cross_platform(
            client, all_clients, opportunity, size,
            paper_trading=paper_trading, use_fak=use_fak,
            deadline_sec=cross_platform_deadline_sec,
        )
    else:
        raise ValueError(f"Unknown opportunity type: {opportunity.type}")


def _paper_execute(
    opportunity: Opportunity,
    size: float,
    start_time: float,
) -> TradeResult:
    """Simulate execution for paper trading mode."""
    elapsed_ms = (time.time() - start_time) * 1000

    fill_prices = tuple(leg.price for leg in opportunity.legs)
    fill_sizes = tuple(size for _ in opportunity.legs)
    net_pnl = opportunity.net_profit_per_set * size - opportunity.estimated_gas_cost

    logger.info(
        "[PAPER] Executed %s: %d legs, size=%.1f, pnl=$%.2f",
        opportunity.type.value, len(opportunity.legs), size, net_pnl,
    )

    return TradeResult(
        opportunity=opportunity,
        order_ids=tuple("paper_" + str(i) for i in range(len(opportunity.legs))),
        fill_prices=fill_prices,
        fill_sizes=fill_sizes,
        fees=0.0,
        gas_cost=opportunity.estimated_gas_cost,
        net_pnl=net_pnl,
        execution_time_ms=elapsed_ms,
        fully_filled=True,
    )


def _execute_binary(
    client: ClobClient,
    opportunity: Opportunity,
    size: float,
    start_time: float,
    order_type: OrderType = OrderType.FAK,
    order_timeout_sec: float = 5.0,
) -> TradeResult:
    """
    Execute a binary rebalancing trade.
    Places 2 orders (YES + NO) via batch endpoint.
    """
    assert len(opportunity.legs) == 2, f"Binary arb must have 2 legs, got {len(opportunity.legs)}"

    # Build signed orders for both legs
    signed_orders = []
    for leg in opportunity.legs:
        neg_risk = False  # binary markets are not negRisk
        tick_size = float(leg.tick_size) if hasattr(leg, 'tick_size') and leg.tick_size else 0.01
        quantized_price = quantize_price(leg.price, tick_size)
        signed = create_limit_order(
            client,
            token_id=leg.token_id,
            side=leg.side,
            price=quantized_price,
            size=size,
            neg_risk=neg_risk,
        )
        signed_orders.append((signed, order_type))

    # Submit both as a batch
    responses = post_orders(client, signed_orders)

    # Track results
    order_ids: list[str] = []
    fill_prices: list[float] = []
    fill_sizes: list[float] = []
    all_filled = True

    for i, resp in enumerate(responses):
        oid = resp.get("orderID", resp.get("order_id", f"unknown_{i}"))
        order_ids.append(oid)

        # Check fill status
        status = str(resp.get("status", ""))
        is_partial = status.lower() == "partial"
        filled = _order_is_filled(status)
        if not filled and order_type == OrderType.GTC and not oid.startswith("unknown_"):
            filled = _wait_for_fill(client, oid, order_timeout_sec)
            if filled:
                is_partial = False

        if filled:
            fill_prices.append(opportunity.legs[i].price)
            fill_sizes.append(size)
        elif is_partial:
            partial_size = _filled_size_from_response(resp, requested_size=size)
            if partial_size <= 0 and not oid.startswith("unknown_"):
                partial_size = _fetch_filled_size(client, oid, requested_size=size)
            if partial_size <= 0:
                raise UnwindFailed(
                    f"Order {oid} reported partial fill but matched size is unknown"
                )
            fill_prices.append(opportunity.legs[i].price if partial_size > 0 else 0.0)
            fill_sizes.append(partial_size)
            all_filled = False
        else:
            fill_prices.append(0.0)
            fill_sizes.append(0.0)
            all_filled = False

    elapsed_ms = (time.time() - start_time) * 1000

    if not all_filled:
        logger.warning("Partial fill on binary arb, unwinding...")
        _unwind_partial(client, order_ids, opportunity.legs, fill_sizes, neg_risk=False)

    # Compute P&L
    buy_notional = sum(
        fp * fs
        for leg, fp, fs in zip(opportunity.legs, fill_prices, fill_sizes)
        if fs > 0 and leg.side == Side.BUY
    )
    sell_notional = sum(
        fp * fs
        for leg, fp, fs in zip(opportunity.legs, fill_prices, fill_sizes)
        if fs > 0 and leg.side == Side.SELL
    )
    expected_payout = min(fill_sizes) if all_filled else 0  # $1 per set at resolution
    if all_filled:
        if buy_notional > 0 and sell_notional == 0:
            net_pnl = expected_payout - buy_notional - opportunity.estimated_gas_cost
        elif sell_notional > 0 and buy_notional == 0:
            net_pnl = sell_notional - expected_payout - opportunity.estimated_gas_cost
        else:
            # Defensive fallback for mixed-side opportunities.
            net_pnl = sell_notional - buy_notional - opportunity.estimated_gas_cost
    else:
        # Partial fills were unwound; track conservative loss from executed notional.
        net_pnl = -(buy_notional + sell_notional)

    logger.info(
        "Binary execution: filled=%s orders=%s elapsed=%.0fms pnl=$%.2f",
        all_filled, order_ids, elapsed_ms, net_pnl,
    )

    return TradeResult(
        opportunity=opportunity,
        order_ids=tuple(order_ids),
        fill_prices=tuple(fill_prices),
        fill_sizes=tuple(fill_sizes),
        fees=0.0,
        gas_cost=opportunity.estimated_gas_cost,
        net_pnl=net_pnl,
        execution_time_ms=elapsed_ms,
        fully_filled=all_filled,
    )


def _execute_negrisk(
    client: ClobClient,
    opportunity: Opportunity,
    size: float,
    start_time: float,
    order_type: OrderType = OrderType.FAK,
    order_timeout_sec: float = 5.0,
) -> TradeResult:
    """
    Execute a NegRisk rebalancing trade.
    Places N orders (one per outcome) via batch endpoint.
    May need multiple batches if N > 15.
    """
    legs = opportunity.legs

    # Build all signed orders
    signed_orders = []
    for leg in legs:
        tick_size = float(leg.tick_size) if hasattr(leg, 'tick_size') and leg.tick_size else 0.01
        quantized_price = quantize_price(leg.price, tick_size)
        signed = create_limit_order(
            client,
            token_id=leg.token_id,
            side=leg.side,
            price=quantized_price,
            size=size,
            neg_risk=True,
        )
        signed_orders.append((signed, order_type))

    # Submit in batches of MAX_BATCH_SIZE
    all_responses = []
    for batch_start in range(0, len(signed_orders), MAX_BATCH_SIZE):
        batch = signed_orders[batch_start:batch_start + MAX_BATCH_SIZE]
        responses = post_orders(client, batch)
        all_responses.extend(responses)

    # Track results
    order_ids = []
    fill_prices = []
    fill_sizes = []
    all_filled = True

    for i, resp in enumerate(all_responses):
        oid = resp.get("orderID", resp.get("order_id", f"unknown_{i}"))
        order_ids.append(oid)

        status = str(resp.get("status", ""))
        is_partial = status.lower() == "partial"
        filled = _order_is_filled(status)
        if not filled and order_type == OrderType.GTC and not oid.startswith("unknown_"):
            filled = _wait_for_fill(client, oid, order_timeout_sec)
            if filled:
                is_partial = False

        if filled:
            fill_prices.append(legs[i].price)
            fill_sizes.append(size)
        elif is_partial:
            partial_size = _filled_size_from_response(resp, requested_size=size)
            if partial_size <= 0 and not oid.startswith("unknown_"):
                partial_size = _fetch_filled_size(client, oid, requested_size=size)
            if partial_size <= 0:
                raise UnwindFailed(
                    f"Order {oid} reported partial fill but matched size is unknown"
                )
            fill_prices.append(legs[i].price if partial_size > 0 else 0.0)
            fill_sizes.append(partial_size)
            all_filled = False
        else:
            fill_prices.append(0.0)
            fill_sizes.append(0.0)
            all_filled = False

    elapsed_ms = (time.time() - start_time) * 1000

    if not all_filled:
        logger.warning("Partial fill on negrisk arb (%d/%d legs), unwinding...",
                       sum(1 for s in fill_sizes if s > 0), len(legs))
        _unwind_partial(client, order_ids, legs, fill_sizes, neg_risk=True)

    buy_notional = sum(
        fp * fs
        for leg, fp, fs in zip(legs, fill_prices, fill_sizes)
        if fs > 0 and leg.side == Side.BUY
    )
    sell_notional = sum(
        fp * fs
        for leg, fp, fs in zip(legs, fill_prices, fill_sizes)
        if fs > 0 and leg.side == Side.SELL
    )
    expected_payout = size if all_filled else 0  # $1 per set at resolution
    if all_filled:
        if buy_notional > 0 and sell_notional == 0:
            net_pnl = expected_payout - buy_notional - opportunity.estimated_gas_cost
        elif sell_notional > 0 and buy_notional == 0:
            net_pnl = sell_notional - expected_payout - opportunity.estimated_gas_cost
        else:
            # Defensive fallback for mixed-side opportunities.
            net_pnl = sell_notional - buy_notional - opportunity.estimated_gas_cost
    else:
        # Partial fills were unwound; track conservative loss from executed notional.
        net_pnl = -(buy_notional + sell_notional)

    logger.info(
        "NegRisk execution: %d legs filled=%s elapsed=%.0fms pnl=$%.2f",
        len(legs), all_filled, elapsed_ms, net_pnl,
    )

    return TradeResult(
        opportunity=opportunity,
        order_ids=tuple(order_ids),
        fill_prices=tuple(fill_prices),
        fill_sizes=tuple(fill_sizes),
        fees=0.0,
        gas_cost=opportunity.estimated_gas_cost,
        net_pnl=net_pnl,
        execution_time_ms=elapsed_ms,
        fully_filled=all_filled,
    )


def _execute_single_leg(
    client: ClobClient,
    opportunity: Opportunity,
    size: float,
    start_time: float,
    order_type: OrderType = OrderType.FAK,
) -> TradeResult:
    """
    Execute a single-leg trade (latency arb, spike lag).
    Places one order for the single leg.
    """
    assert len(opportunity.legs) == 1, f"Single-leg trade must have 1 leg, got {len(opportunity.legs)}"
    leg = opportunity.legs[0]

    tick_size = float(leg.tick_size) if hasattr(leg, 'tick_size') and leg.tick_size else 0.01
    quantized_price = quantize_price(leg.price, tick_size)
    signed = create_limit_order(
        client,
        token_id=leg.token_id,
        side=leg.side,
        price=quantized_price,
        size=size,
        neg_risk=False,
    )
    resp = post_order(client, signed, order_type)

    oid = resp.get("orderID", resp.get("order_id", "unknown_0"))
    status = resp.get("status", "")
    status_lower = str(status).lower()
    filled = status_lower in ("matched", "filled")
    partial = status_lower == "partial"

    elapsed_ms = (time.time() - start_time) * 1000

    if filled:
        fill_size = size
    elif partial:
        fill_size = _filled_size_from_response(resp, requested_size=size)
        if fill_size <= 0 and not oid.startswith("unknown_"):
            fill_size = _fetch_filled_size(client, oid, requested_size=size)
    else:
        fill_size = 0.0

    fill_price = leg.price if fill_size > 0 else 0.0
    net_pnl = (
        opportunity.net_profit_per_set * fill_size - opportunity.estimated_gas_cost
        if fill_size > 0 else 0.0
    )

    logger.info(
        "Single-leg execution: type=%s filled=%s order=%s elapsed=%.0fms pnl=$%.2f",
        opportunity.type.value, filled and not partial, oid, elapsed_ms, net_pnl,
    )

    return TradeResult(
        opportunity=opportunity,
        order_ids=(oid,),
        fill_prices=(fill_price,),
        fill_sizes=(fill_size,),
        fees=0.0,
        gas_cost=opportunity.estimated_gas_cost,
        net_pnl=net_pnl,
        execution_time_ms=elapsed_ms,
        fully_filled=filled and not partial,
    )


def _unwind_partial(
    client: ClobClient,
    order_ids: list[str],
    legs: tuple[LegOrder, ...],
    fill_sizes: list[float],
    neg_risk: bool,
) -> None:
    """
    Unwind a partially filled multi-leg trade.
    1. Cancel all unfilled orders.
    2. Market-sell any filled positions to minimize exposure.
    Raises UnwindFailed if any unwind order fails -- caller must log stuck positions.
    """
    stuck_positions: list[dict] = []

    for i, (oid, leg, filled) in enumerate(zip(order_ids, legs, fill_sizes)):
        if filled == 0:
            # Cancel unfilled order
            try:
                cancel_order(client, oid)
                logger.info("Cancelled unfilled order %s", oid)
            except Exception as e:
                logger.error("Failed to cancel order %s: %s", oid, e)
        else:
            # Market-sell the filled position
            opposite_side = Side.SELL if leg.side == Side.BUY else Side.BUY
            try:
                # SDK semantics: BUY market orders use dollar notional, SELL uses share count.
                unwind_amount = (
                    max(0.01, filled * leg.price)
                    if opposite_side == Side.BUY
                    else filled
                )
                unwind_order = create_market_order(
                    client,
                    token_id=leg.token_id,
                    side=opposite_side,
                    amount=unwind_amount,
                    neg_risk=neg_risk,
                )
                resp = post_order(client, unwind_order, OrderType.FOK)
                logger.info("Unwound position %s: %s", leg.token_id, resp)
            except Exception as e:
                logger.error("Failed to unwind position %s: %s", leg.token_id, e)
                stuck_positions.append({
                    "token_id": leg.token_id,
                    "side": leg.side.value,
                    "size": filled,
                    "error": str(e),
                })

    if stuck_positions:
        raise UnwindFailed(
            f"Failed to unwind {len(stuck_positions)} position(s): {stuck_positions}"
        )


def _order_is_filled(status: str) -> bool:
    """Normalize CLOB order status values to a filled/not-filled boolean."""
    return status.lower() in ("matched", "filled")


def _filled_size_from_response(resp: dict, requested_size: float) -> float:
    """
    Best-effort extraction of filled size from an order response payload.
    Returns 0.0 if no known filled-size field is present or parseable.
    """
    for key in (
        "filled_size",
        "filledSize",
        "size_filled",
        "sizeFilled",
        "matched_size",
        "matchedSize",
        "size_matched",
        "filled",
        "fill_size",
        "filledQuantity",
        "quantity_filled",
    ):
        if key not in resp:
            continue
        try:
            parsed = float(resp[key])
        except (TypeError, ValueError):
            continue
        if parsed <= 0:
            return 0.0
        return min(parsed, requested_size)
    return 0.0


def _fetch_filled_size(client: ClobClient, order_id: str, requested_size: float) -> float:
    """Query order status and extract matched size for partial fills."""
    try:
        resp = client.get_order(order_id) or {}
    except Exception as e:
        logger.warning("Failed to fetch filled size for %s: %s", order_id, e)
        return 0.0
    return _filled_size_from_response(resp, requested_size=requested_size)


def _wait_for_fill(
    client: ClobClient,
    order_id: str,
    timeout_sec: float,
    poll_interval_sec: float = 0.1,
) -> bool:
    """
    Poll order status until filled or timeout.
    Used for GTC mode where orders may not be immediately matched.
    """
    if timeout_sec <= 0:
        return False

    deadline = time.time() + timeout_sec
    while time.time() < deadline:
        try:
            resp = client.get_order(order_id)
            status = str((resp or {}).get("status", ""))
        except Exception as e:
            logger.warning("Order status poll failed for %s: %s", order_id, e)
            return False

        if _order_is_filled(status):
            return True

        # Terminal non-filled states.
        if status.lower() in ("cancelled", "canceled", "expired", "rejected"):
            return False

        remaining = deadline - time.time()
        if remaining <= 0:
            break
        time.sleep(min(poll_interval_sec, remaining))

    return False
