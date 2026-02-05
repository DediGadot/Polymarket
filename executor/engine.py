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


def execute_opportunity(
    client: ClobClient,
    opportunity: Opportunity,
    size: float,
    paper_trading: bool = False,
) -> TradeResult:
    """
    Execute an arbitrage opportunity. Places all leg orders, tracks fills.
    Returns a TradeResult with execution details.

    For paper trading, simulates execution without placing real orders.
    """
    start_time = time.time()

    if paper_trading:
        return _paper_execute(opportunity, size, start_time)

    if opportunity.type == OpportunityType.BINARY_REBALANCE:
        return _execute_binary(client, opportunity, size, start_time)
    elif opportunity.type == OpportunityType.NEGRISK_REBALANCE:
        return _execute_negrisk(client, opportunity, size, start_time)
    else:
        raise ValueError(f"Unknown opportunity type: {opportunity.type}")


def _paper_execute(
    opportunity: Opportunity,
    size: float,
    start_time: float,
) -> TradeResult:
    """Simulate execution for paper trading mode."""
    elapsed_ms = (time.time() - start_time) * 1000

    fill_prices = [leg.price for leg in opportunity.legs]
    fill_sizes = [size] * len(opportunity.legs)

    cost_per_set = sum(leg.price for leg in opportunity.legs if leg.side == Side.BUY)
    net_pnl = opportunity.expected_profit_per_set * size - opportunity.estimated_gas_cost

    logger.info(
        "[PAPER] Executed %s: %d legs, size=%.1f, pnl=$%.2f",
        opportunity.type.value, len(opportunity.legs), size, net_pnl,
    )

    return TradeResult(
        opportunity=opportunity,
        order_ids=["paper_" + str(i) for i in range(len(opportunity.legs))],
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
        signed = create_limit_order(
            client,
            token_id=leg.token_id,
            side=leg.side,
            price=leg.price,
            size=size,
            neg_risk=neg_risk,
        )
        signed_orders.append((signed, OrderType.GTC))

    # Submit both as a batch
    responses = post_orders(client, signed_orders)

    # Track results
    order_ids = []
    fill_prices = []
    fill_sizes = []
    all_filled = True

    for i, resp in enumerate(responses):
        oid = resp.get("orderID", resp.get("order_id", f"unknown_{i}"))
        order_ids.append(oid)

        # Check fill status
        status = resp.get("status", "")
        if status == "matched" or status == "filled":
            fill_prices.append(opportunity.legs[i].price)
            fill_sizes.append(size)
        else:
            fill_prices.append(0.0)
            fill_sizes.append(0.0)
            all_filled = False

    elapsed_ms = (time.time() - start_time) * 1000

    if not all_filled:
        logger.warning("Partial fill on binary arb, unwinding...")
        _unwind_partial(client, order_ids, opportunity.legs, fill_sizes)

    # Compute P&L
    total_cost = sum(fp * fs for fp, fs in zip(fill_prices, fill_sizes) if fs > 0)
    expected_payout = min(fill_sizes) if all_filled else 0  # $1 per set at resolution
    net_pnl = expected_payout - total_cost - opportunity.estimated_gas_cost if all_filled else -total_cost

    logger.info(
        "Binary execution: filled=%s orders=%s elapsed=%.0fms pnl=$%.2f",
        all_filled, order_ids, elapsed_ms, net_pnl,
    )

    return TradeResult(
        opportunity=opportunity,
        order_ids=order_ids,
        fill_prices=fill_prices,
        fill_sizes=fill_sizes,
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
        signed = create_limit_order(
            client,
            token_id=leg.token_id,
            side=leg.side,
            price=leg.price,
            size=size,
            neg_risk=True,
        )
        signed_orders.append((signed, OrderType.GTC))

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

        status = resp.get("status", "")
        if status == "matched" or status == "filled":
            fill_prices.append(legs[i].price)
            fill_sizes.append(size)
        else:
            fill_prices.append(0.0)
            fill_sizes.append(0.0)
            all_filled = False

    elapsed_ms = (time.time() - start_time) * 1000

    if not all_filled:
        logger.warning("Partial fill on negrisk arb (%d/%d legs), unwinding...",
                       sum(1 for s in fill_sizes if s > 0), len(legs))
        _unwind_partial(client, order_ids, legs, fill_sizes)

    total_cost = sum(fp * fs for fp, fs in zip(fill_prices, fill_sizes) if fs > 0)
    expected_payout = size if all_filled else 0  # $1 per set at resolution
    net_pnl = expected_payout - total_cost - opportunity.estimated_gas_cost if all_filled else -total_cost

    logger.info(
        "NegRisk execution: %d legs filled=%s elapsed=%.0fms pnl=$%.2f",
        len(legs), all_filled, elapsed_ms, net_pnl,
    )

    return TradeResult(
        opportunity=opportunity,
        order_ids=order_ids,
        fill_prices=fill_prices,
        fill_sizes=fill_sizes,
        fees=0.0,
        gas_cost=opportunity.estimated_gas_cost,
        net_pnl=net_pnl,
        execution_time_ms=elapsed_ms,
        fully_filled=all_filled,
    )


def _unwind_partial(
    client: ClobClient,
    order_ids: list[str],
    legs: tuple[LegOrder, ...],
    fill_sizes: list[float],
) -> None:
    """
    Unwind a partially filled multi-leg trade.
    1. Cancel all unfilled orders.
    2. Market-sell any filled positions to minimize exposure.
    """
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
                unwind_order = create_market_order(
                    client,
                    token_id=leg.token_id,
                    side=opposite_side,
                    amount=filled,
                    neg_risk=True,
                )
                resp = post_order(client, unwind_order, OrderType.FOK)
                logger.info("Unwound position %s: %s", leg.token_id, resp)
            except Exception as e:
                logger.error("Failed to unwind position %s: %s", leg.token_id, e)
