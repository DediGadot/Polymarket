"""
Cross-platform execution engine.

Handles placing orders on both Polymarket and Kalshi for cross-platform arb.
Places Kalshi leg first (faster confirmation ~50ms) then Polymarket (on-chain ~2s).
If Polymarket fails, attempts to unwind Kalshi position with tracked loss.
"""

from __future__ import annotations

import logging
import time

from py_clob_client.client import ClobClient
from py_clob_client.clob_types import OrderType

from client.clob import create_limit_order, post_order
from client.kalshi import KalshiClient, dollars_to_cents
from scanner.models import (
    LegOrder,
    Opportunity,
    Side,
    TradeResult,
)

logger = logging.getLogger(__name__)

# Conservative estimate: 1 cent spread + fees per contract on unwind
_UNWIND_LOSS_PER_CONTRACT = 0.02


class CrossPlatformUnwindFailed(Exception):
    """Raised when cross-platform unwind fails. Stuck positions on one platform."""
    pass


def execute_cross_platform(
    pm_client: ClobClient,
    kalshi_client: KalshiClient,
    opportunity: Opportunity,
    size: float,
    paper_trading: bool = False,
    use_fak: bool = True,
    deadline_sec: float = 5.0,
) -> TradeResult:
    """
    Execute a cross-platform arbitrage opportunity.

    Strategy:
    1. Place Kalshi leg first (REST, ~50ms confirmation)
    2. If Kalshi fills: place Polymarket leg (on-chain, ~2s)
    3. If Polymarket fails: unwind Kalshi position (track loss)
    4. If both fill: compute combined P&L

    Args:
        deadline_sec: Max total execution time. Abort PM leg if exceeded.
    """
    start_time = time.time()
    deadline = start_time + deadline_sec

    if paper_trading:
        return _paper_execute_cross_platform(opportunity, size, start_time)

    # Kalshi contracts are integer-count; enforce a common executable size
    # across both platforms to avoid residual unhedged exposure.
    exec_size = int(size)
    if exec_size <= 0:
        elapsed_ms = (time.time() - start_time) * 1000
        logger.info("Cross-platform execution skipped: requested size %.4f rounds to 0 contracts", size)
        return TradeResult(
            opportunity=opportunity,
            order_ids=[],
            fill_prices=[],
            fill_sizes=[],
            fees=0.0,
            gas_cost=0.0,
            net_pnl=0.0,
            execution_time_ms=elapsed_ms,
            fully_filled=False,
        )

    # Separate legs by platform
    pm_legs = [leg for leg in opportunity.legs if leg.platform == "polymarket"]
    kalshi_legs = [leg for leg in opportunity.legs if leg.platform == "kalshi"]

    if not pm_legs or not kalshi_legs:
        raise ValueError(
            f"Cross-platform opportunity must have both PM and Kalshi legs, "
            f"got {len(pm_legs)} PM and {len(kalshi_legs)} Kalshi"
        )

    order_ids: list[str] = []
    fill_prices: list[float] = []
    fill_sizes: list[float] = []

    # Step 1: Place Kalshi leg(s)
    # Our Kalshi OrderBook model is YES-token-based:
    #   BUY leg  → side="yes", action="buy"  (buying YES tokens)
    #   SELL leg → side="yes", action="sell"  (selling YES tokens = buying NO)
    kalshi_filled = True
    for leg in kalshi_legs:
        try:
            price_cents = dollars_to_cents(leg.price)
            kalshi_action = "buy" if leg.side == Side.BUY else "sell"
            result = kalshi_client.place_order(
                ticker=leg.token_id,
                side="yes",
                action=kalshi_action,
                count=exec_size,
                type="limit",
                yes_price=price_cents,
            )
            order_data = result.get("order", result)
            oid = order_data.get("order_id", "unknown_kalshi")
            status = order_data.get("status", "")
            order_ids.append(oid)

            if status in ("executed", "filled"):
                fill_prices.append(leg.price)
                fill_sizes.append(float(exec_size))
            elif status == "resting":
                # Order placed but not yet filled — poll for fill
                filled = _wait_for_kalshi_fill(kalshi_client, oid, timeout_sec=2.0)
                if filled:
                    fill_prices.append(leg.price)
                    fill_sizes.append(float(exec_size))
                else:
                    kalshi_client.cancel_order(oid)
                    fill_prices.append(0.0)
                    fill_sizes.append(0.0)
                    kalshi_filled = False
            else:
                fill_prices.append(0.0)
                fill_sizes.append(0.0)
                kalshi_filled = False
        except Exception as e:
            logger.error("Kalshi order failed for %s: %s", leg.token_id, e)
            order_ids.append("failed_kalshi")
            fill_prices.append(0.0)
            fill_sizes.append(0.0)
            kalshi_filled = False

    if not kalshi_filled:
        elapsed_ms = (time.time() - start_time) * 1000
        logger.warning("Kalshi leg failed, aborting cross-platform arb")
        # Cancel any resting Kalshi orders
        for oid in order_ids:
            if oid not in ("unknown_kalshi", "failed_kalshi"):
                try:
                    kalshi_client.cancel_order(oid)
                except Exception as e:
                    logger.error("Failed to cancel Kalshi order %s: %s", oid, e)

        return TradeResult(
            opportunity=opportunity,
            order_ids=order_ids,
            fill_prices=fill_prices,
            fill_sizes=fill_sizes,
            fees=0.0,
            gas_cost=0.0,
            net_pnl=0.0,
            execution_time_ms=elapsed_ms,
            fully_filled=False,
        )

    # Check deadline before proceeding to PM leg
    if time.time() > deadline:
        elapsed_ms = (time.time() - start_time) * 1000
        logger.warning(
            "Cross-platform deadline exceeded (%.0fms > %.0fms), unwinding Kalshi",
            elapsed_ms, deadline_sec * 1000,
        )
        unwind_loss = _unwind_kalshi(kalshi_client, kalshi_legs, float(exec_size))
        return TradeResult(
            opportunity=opportunity,
            order_ids=order_ids,
            fill_prices=fill_prices,
            fill_sizes=fill_sizes,
            fees=0.0,
            gas_cost=opportunity.estimated_gas_cost,
            net_pnl=-unwind_loss,
            execution_time_ms=elapsed_ms,
            fully_filled=False,
        )

    # Step 2: Place Polymarket leg(s)
    pm_filled = True
    order_type = OrderType.FAK if use_fak else OrderType.GTC

    for leg in pm_legs:
        try:
            signed = create_limit_order(
                pm_client,
                token_id=leg.token_id,
                side=leg.side,
                price=leg.price,
                size=float(exec_size),
                neg_risk=False,
            )
            resp = post_order(pm_client, signed, order_type)
            oid = resp.get("orderID", resp.get("order_id", "unknown_pm"))
            status = str(resp.get("status", ""))
            order_ids.append(oid)

            if status.lower() in ("matched", "filled"):
                fill_prices.append(leg.price)
                fill_sizes.append(float(exec_size))
            else:
                fill_prices.append(0.0)
                fill_sizes.append(0.0)
                pm_filled = False
        except Exception as e:
            logger.error("Polymarket order failed for %s: %s", leg.token_id, e)
            order_ids.append("failed_pm")
            fill_prices.append(0.0)
            fill_sizes.append(0.0)
            pm_filled = False

    elapsed_ms = (time.time() - start_time) * 1000

    if not pm_filled:
        # Unwind Kalshi: sell the filled Kalshi position at market
        logger.warning("Polymarket leg failed, unwinding Kalshi position...")
        unwind_loss = _unwind_kalshi(kalshi_client, kalshi_legs, float(exec_size))

        return TradeResult(
            opportunity=opportunity,
            order_ids=order_ids,
            fill_prices=fill_prices,
            fill_sizes=fill_sizes,
            fees=0.0,
            gas_cost=opportunity.estimated_gas_cost,
            net_pnl=-unwind_loss,
            execution_time_ms=elapsed_ms,
            fully_filled=False,
        )

    # Both filled: compute P&L
    net_pnl = opportunity.net_profit_per_set * exec_size - opportunity.estimated_gas_cost

    logger.info(
        "Cross-platform execution: filled=True orders=%s elapsed=%.0fms pnl=$%.2f",
        order_ids, elapsed_ms, net_pnl,
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
        fully_filled=True,
    )


def _paper_execute_cross_platform(
    opportunity: Opportunity,
    size: float,
    start_time: float,
) -> TradeResult:
    """Simulate cross-platform execution for paper trading."""
    elapsed_ms = (time.time() - start_time) * 1000
    fill_prices = [leg.price for leg in opportunity.legs]
    fill_sizes = [size] * len(opportunity.legs)
    net_pnl = opportunity.net_profit_per_set * size - opportunity.estimated_gas_cost

    logger.info(
        "[PAPER] Cross-platform: %d legs, size=%.1f, pnl=$%.2f",
        len(opportunity.legs), size, net_pnl,
    )

    return TradeResult(
        opportunity=opportunity,
        order_ids=[f"paper_xp_{i}" for i in range(len(opportunity.legs))],
        fill_prices=fill_prices,
        fill_sizes=fill_sizes,
        fees=0.0,
        gas_cost=opportunity.estimated_gas_cost,
        net_pnl=net_pnl,
        execution_time_ms=elapsed_ms,
        fully_filled=True,
    )


def _wait_for_kalshi_fill(
    kalshi_client: KalshiClient,
    order_id: str,
    timeout_sec: float = 2.0,
    poll_interval: float = 0.1,
) -> bool:
    """Poll Kalshi order status until filled or timeout."""
    deadline = time.time() + timeout_sec
    while time.time() < deadline:
        try:
            order = kalshi_client.get_order(order_id)
            status = order.get("order", order).get("status", "")
            if status in ("executed", "filled"):
                return True
            if status in ("canceled", "cancelled", "expired"):
                return False
        except Exception:
            return False
        time.sleep(poll_interval)
    return False


def _unwind_kalshi(
    kalshi_client: KalshiClient,
    kalshi_legs: list[LegOrder],
    size: float,
) -> float:
    """
    Unwind Kalshi positions by placing opposite market orders.
    Returns estimated unwind loss in dollars.
    Raises CrossPlatformUnwindFailed if any unwind fails.
    """
    stuck: list[dict] = []
    total_unwind_loss = 0.0

    for leg in kalshi_legs:
        opposite_action = "sell" if leg.side == Side.BUY else "buy"
        try:
            kalshi_client.place_order(
                ticker=leg.token_id,
                side="yes",
                action=opposite_action,
                count=int(size),
                type="market",
            )
            # Estimate loss: spread + fees per contract
            total_unwind_loss += _UNWIND_LOSS_PER_CONTRACT * size
            logger.info("Unwound Kalshi position: %s %s %d", opposite_action, leg.token_id, int(size))
        except Exception as e:
            logger.error("Failed to unwind Kalshi %s: %s", leg.token_id, e)
            # Stuck position: estimate full cost as loss
            total_unwind_loss += leg.price * size
            stuck.append({"ticker": leg.token_id, "side": leg.side.value, "size": size, "error": str(e)})

    if stuck:
        raise CrossPlatformUnwindFailed(
            f"Failed to unwind {len(stuck)} Kalshi position(s): {stuck}"
        )

    return total_unwind_loss
