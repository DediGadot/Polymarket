"""
Cross-platform execution engine.

Handles placing orders on both Polymarket and an external platform for cross-platform arb.
Places external leg first (faster confirmation ~50ms) then Polymarket (on-chain ~2s).
If Polymarket fails, attempts to unwind external position with tracked loss.

Generalized for N platforms: determines external platform from leg metadata.
"""

from __future__ import annotations

import json
import logging
import time
from pathlib import Path

from py_clob_client.client import ClobClient
from py_clob_client.clob_types import OrderType

from client.clob import create_limit_order, post_order
from client.platform import PlatformClient
from executor.fill_state import (
    FillState,
    can_transition_to,
    transition_to,
    is_terminal_state,
    is_final_state,
)
from scanner.models import (
    LegOrder,
    Opportunity,
    Side,
    TradeResult,
)

logger = logging.getLogger(__name__)

# Conservative estimate: 1 cent spread + fees per contract on unwind
_UNWIND_LOSS_PER_CONTRACT = 0.02

# State machine retry config
_MAX_UNWIND_RETRIES = 3
_UNWIND_BACKOFF_SEC = 0.5
_STUCK_POSITIONS_FILE = "stuck_positions.json"


class CrossPlatformUnwindFailed(Exception):
    """Raised when cross-platform unwind fails. Stuck positions on one platform."""
    pass


def _persist_stuck(position_data: dict) -> None:
    """Persist a stuck position to stuck_positions.json."""
    try:
        stuck_file = Path(_STUCK_POSITIONS_FILE)
        existing = []
        if stuck_file.exists():
            try:
                existing = json.loads(stuck_file.read_text())
            except (json.JSONDecodeError, IOError):
                existing = []

        all_positions = existing + [position_data]
        stuck_file.write_text(json.dumps(all_positions, indent=2))
        logger.info("Persisted stuck position: %s @ %s", position_data.get("ticker"), position_data.get("platform"))
    except Exception as e:
        logger.error("Failed to persist stuck position: %s", e)


def _load_stuck_positions() -> list[dict]:
    """Load stuck positions from previous run on startup."""
    try:
        stuck_file = Path(_STUCK_POSITIONS_FILE)
        if stuck_file.exists():
            positions = json.loads(stuck_file.read_text())
            if positions:
                logger.warning(
                    "Found %d stuck position(s) from previous run: %s",
                    len(positions), _STUCK_POSITIONS_FILE,
                )
                for pos in positions:
                    logger.warning("  Stuck: %s %s @ %s", pos["ticker"], pos["side"], pos["platform"])
                return positions
    except Exception as e:
        logger.error("Failed to read stuck positions: %s", e)
    return []


def _dollars_to_cents(price: float) -> int:
    """Convert dollar price (0.01-0.99) to cents (1-99) for cent-based platforms."""
    cents = round(price * 100)
    if cents < 1 or cents > 99:
        raise ValueError(f"Price out of range: ${price:.4f} -> {cents} cents (must be 1-99)")
    return cents


def execute_cross_platform(
    pm_client: ClobClient,
    platform_clients: dict[str, PlatformClient],
    opportunity: Opportunity,
    size: float,
    paper_trading: bool = False,
    use_fak: bool = True,
    deadline_sec: float = 5.0,
) -> TradeResult:
    """
    Execute a cross-platform arbitrage opportunity.

    Strategy:
    1. Place external platform leg first (REST, ~50ms confirmation)
    2. If external fills: place Polymarket leg (on-chain, ~2s)
    3. If Polymarket fails: unwind external position (track loss)
    4. If both fill: compute combined P&L

    Args:
        platform_clients: {platform_name: PlatformClient, ...}
        deadline_sec: Max total execution time. Abort PM leg if exceeded.
    """
    start_time = time.time()
    deadline = start_time + deadline_sec

    if paper_trading:
        return _paper_execute_cross_platform(opportunity, size, start_time)

    # External contracts are integer-count; enforce a common executable size
    exec_size = int(size)
    if exec_size <= 0:
        elapsed_ms = (time.time() - start_time) * 1000
        logger.info("Cross-platform execution skipped: requested size %.4f rounds to 0 contracts", size)
        return TradeResult(
            opportunity=opportunity,
            order_ids=(),
            fill_prices=(),
            fill_sizes=(),
            fees=0.0,
            gas_cost=0.0,
            net_pnl=0.0,
            execution_time_ms=elapsed_ms,
            fully_filled=False,
        )

    # Separate legs by platform
    pm_legs = [leg for leg in opportunity.legs if leg.platform in ("polymarket", "")]
    ext_legs = [leg for leg in opportunity.legs if leg.platform not in ("polymarket", "")]

    if not pm_legs or not ext_legs:
        raise ValueError(
            f"Cross-platform opportunity must have both PM and external legs, "
            f"got {len(pm_legs)} PM and {len(ext_legs)} external"
        )

    # Determine external platform from first ext leg
    ext_platform = ext_legs[0].platform
    ext_client = platform_clients.get(ext_platform)
    if ext_client is None:
        raise ValueError(f"No client for external platform '{ext_platform}'")

    order_ids: list[str] = []
    fill_prices: list[float] = []
    fill_sizes: list[float] = []

    # Step 1: Place external platform leg(s)
    ext_filled = True
    for leg in ext_legs:
        try:
            price_cents = _dollars_to_cents(leg.price)
            ext_action = "buy" if leg.side == Side.BUY else "sell"
            result = ext_client.place_order(
                ticker=leg.token_id,
                side="yes",
                action=ext_action,
                count=exec_size,
                type="limit",
                yes_price=price_cents,
            )
            order_data = result.get("order", result)
            oid = order_data.get("order_id", f"unknown_{ext_platform}")
            status = order_data.get("status", "")
            order_ids.append(oid)

            if status in ("executed", "filled"):
                fill_prices.append(leg.price)
                fill_sizes.append(float(exec_size))
            elif status == "resting":
                filled = _wait_for_ext_fill(ext_client, oid, timeout_sec=2.0)
                if filled:
                    fill_prices.append(leg.price)
                    fill_sizes.append(float(exec_size))
                else:
                    ext_client.cancel_order(oid)
                    fill_prices.append(0.0)
                    fill_sizes.append(0.0)
                    ext_filled = False
            else:
                fill_prices.append(0.0)
                fill_sizes.append(0.0)
                ext_filled = False
        except Exception as e:
            logger.error("%s order failed for %s: %s", ext_platform, leg.token_id, e)
            order_ids.append(f"failed_{ext_platform}")
            fill_prices.append(0.0)
            fill_sizes.append(0.0)
            ext_filled = False

    if not ext_filled:
        elapsed_ms = (time.time() - start_time) * 1000
        logger.warning("%s leg failed, aborting cross-platform arb", ext_platform)
        for oid in order_ids:
            if not oid.startswith(("unknown_", "failed_")):
                try:
                    ext_client.cancel_order(oid)
                except Exception as e:
                    logger.error("Failed to cancel %s order %s: %s", ext_platform, oid, e)

        return TradeResult(
            opportunity=opportunity,
            order_ids=tuple(order_ids),
            fill_prices=tuple(fill_prices),
            fill_sizes=tuple(fill_sizes),
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
            "Cross-platform deadline exceeded (%.0fms > %.0fms), unwinding %s",
            elapsed_ms, deadline_sec * 1000, ext_platform,
        )
        unwind_loss = _unwind_platform(ext_client, ext_platform, ext_legs, float(exec_size))
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
            from executor.tick_size import quantize_price
            tick_size = float(leg.tick_size) if hasattr(leg, 'tick_size') and leg.tick_size else 0.01
            quantized_price = quantize_price(leg.price, tick_size)
            signed = create_limit_order(
                pm_client,
                token_id=leg.token_id,
                side=leg.side,
                price=quantized_price,
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
        logger.warning("Polymarket leg failed, unwinding %s position...", ext_platform)
        unwind_loss = _unwind_platform(ext_client, ext_platform, ext_legs, float(exec_size))

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


def _wait_for_ext_fill(
    ext_client: object,
    order_id: str,
    timeout_sec: float = 2.0,
    poll_interval: float = 0.1,
) -> bool:
    """Poll external platform order status until filled or timeout."""
    deadline = time.time() + timeout_sec
    while time.time() < deadline:
        try:
            order = ext_client.get_order(order_id)
            status = order.get("order", order).get("status", "")
            if status in ("executed", "filled"):
                return True
            if status in ("canceled", "cancelled", "expired"):
                return False
        except Exception:
            return False
        time.sleep(poll_interval)
    return False


def _unwind_platform(
    ext_client: PlatformClient,
    platform: str,
    ext_legs: list[LegOrder],
    size: float,
) -> float:
    """
    Unwind external platform positions by placing opposite market orders.
    Returns estimated unwind loss in dollars.
    Raises CrossPlatformUnwindFailed if any unwind fails.
    Implements retry logic: 3 attempts with 0.5s backoff.
    """
    stuck: list[dict] = []
    total_unwind_loss = 0.0

    for attempt in range(_MAX_UNWIND_RETRIES):
        for leg in ext_legs:
            opposite_action = "sell" if leg.side == Side.BUY else "buy"
            try:
                ext_client.place_order(
                    ticker=leg.token_id,
                    side="yes",
                    action=opposite_action,
                    count=int(size),
                    type="market",
                )
                total_unwind_loss += _UNWIND_LOSS_PER_CONTRACT * size
                logger.info("Unwound %s position: %s %s %d (attempt %d/%d)",
                                  platform, opposite_action, leg.token_id, int(size),
                                  attempt + 1, _MAX_UNWIND_RETRIES)
                # Unwind successful - exit retry loop
                return total_unwind_loss
            except Exception as e:
                logger.warning("Unwind attempt %d/%d failed for %s: %s",
                                 attempt + 1, _MAX_UNWIND_RETRIES, leg.token_id, e)
                # Backoff before next retry
                if attempt < _MAX_UNWIND_RETRIES - 1:
                    time.sleep(_UNWIND_BACKOFF_SEC * (2 ** attempt))
                # Last attempt - mark as stuck
                if attempt == _MAX_UNWIND_RETRIES - 1:
                    logger.error("Failed to unwind %s after %d attempts: %s",
                                   leg.token_id, _MAX_UNWIND_RETRIES, e)
                    total_unwind_loss += leg.price * size
                    stuck.append({
                        "ticker": leg.token_id,
                        "side": leg.side.value,
                        "size": size,
                        "platform": platform,
                        "error": str(e),
                        "attempts": attempt + 1,
                    })

    if stuck:
        # Persist stuck positions before raising
        for pos in stuck:
            _persist_stuck(pos)

        raise CrossPlatformUnwindFailed(
            f"Failed to unwind {len(stuck)} {platform} position(s): {stuck}"
        )

    return total_unwind_loss
