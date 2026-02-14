"""
Cross-platform execution engine with state machine for fill tracking.

Handles placing orders on both Polymarket and an external platform for cross-platform arb.
Places external leg first (faster REST ~50ms) then Polymarket (on-chain ~2s).
If Polymarket fails, attempts to unwind external position with tracked loss.

Generalized for N platforms: determines external platform from leg metadata.
"""

from __future__ import annotations

import json
import logging
import time
from pathlib import Path
from typing import Literal

from py_clob_client.client import ClobClient
from py_clob_client.clob_types import (
    OrderArgs,
    MarketOrderArgs,
    OrderType,
    BookParams,
    PartialCreateOrderOptions,
)
from py_clob_client.order_builder.constants import BUY, SELL

from client.clob import create_limit_order, post_order, post_orders
from client.platform import PlatformClient
from executor.fill_state import (
    FillState,
    can_transition_to,
    transition_to,
    is_terminal_state,
    get_progress_states,
    get_failure_states,
    CrossPlatformUnwindFailed,
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
    Execute a cross-platform arbitrage opportunity using state machine.

    Strategy:
    1. Place external platform leg first (REST, ~50ms confirmation)
    2. If external fills: place Polymarket leg (on-chain, ~2s)
    3. If Polymarket fails: unwind external position (track loss)
    4. If both fill: compute combined P&L

    State Machine:
    - Tracks fill state for each leg (PENDING -> FILLED/REJECTED/PARTIAL/RESTING)
    - On unwind: tracks UNWINDING -> UNWOUND/STUCK
    - Stuck positions persisted to JSON for recovery

    Args:
        platform_clients: {platform_name: PlatformClient, ...}
        deadline_sec: Max total execution time. Abort PM leg if exceeded.

    Returns:
        TradeResult with execution details.
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
    pm_legs = [leg for leg in opportunity.legs if leg.platform in ("polymarket", "")]
    ext_legs = [leg for leg in opportunity.legs if leg.platform not in ("polymarket", "")]

    if not pm_legs or not ext_legs:
        raise ValueError(
            f"Cross-platform opportunity must have both PM and external legs, "
            f"got {len(pm_legs)} PM and {len(ext_legs)} external"
        )

    # Initialize state tracking for each leg
    # Format: {token_id: {"state": FillState, "attempts": int, "last_error": str|None}}
    leg_states: dict[str, dict[str, FillState | int | str | None]] = {}
    for leg in pm_legs + ext_legs:
        leg_states[leg.token_id] = {"state": FillState.PENDING, "attempts": 0, "last_error": None}

    # Determine external platform from first ext leg
    ext_platform = ext_legs[0].platform
    ext_client = platform_clients.get(ext_platform)
    if ext_client is None:
        raise ValueError(f"No client for external platform '{ext_platform}'")

    order_ids: list[str] = []
    fill_prices: list[float] = []
    fill_sizes: list[float] = []
    fees = 0.0

    # ========================================================================
    # STEP 1: Place external platform leg(s) with state machine
    # ========================================================================
    ext_filled = _execute_external_legs_with_state_machine(
        ext_legs, ext_client, leg_states, exec_size,
        start_time, deadline,
    )

    if not ext_filled:
        elapsed_ms = (time.time() - start_time) * 1000
        logger.warning("External leg(s) failed, aborting cross-platform arb")
        # Cancel any resting orders
        for token_id, state_info in leg_states.items():
            if state_info["state"] == FillState.RESTING:
                try:
                    ext_client.cancel_order(token_id)
                except Exception as e:
                    logger.error("Failed to cancel resting order %s: %s", token_id, e)

        return TradeResult(
            opportunity=opportunity,
            order_ids=order_ids,
            fill_prices=fill_prices,
            fill_sizes=fill_sizes,
            fees=fees,
            gas_cost=0.0,
            net_pnl=0.0,
            execution_time_ms=elapsed_ms,
            fully_filled=False,
        )

    # Check deadline before proceeding to PM leg
    if time.time() > deadline:
        elapsed_ms = (time.time() - start_time) * 1000
        logger.warning(
            "Cross-platform deadline exceeded (%.0fms > %.0fms), unwinding external",
            elapsed_ms, deadline_sec * 1000, ext_platform,
        )
        unwind_loss = _unwind_external_with_retry(
            ext_client, ext_platform, ext_legs, leg_states, exec_size,
        )
        return TradeResult(
            opportunity=opportunity,
            order_ids=order_ids,
            fill_prices=fill_prices,
            fill_sizes=fill_sizes,
            fees=fees,
            gas_cost=opportunity.estimated_gas_cost,
            net_pnl=-unwind_loss,
            execution_time_ms=elapsed_ms,
            fully_filled=False,
        )

    # ========================================================================
    # STEP 2: Place Polymarket leg(s) with state machine
    # ========================================================================
    pm_filled = _execute_pm_legs_with_state_machine(
        pm_client, pm_legs, leg_states, exec_size,
        start_time, use_fak,
    )

    elapsed_ms = (time.time() - start_time) * 1000

    if not pm_filled:
        logger.warning("Polymarket leg(s) failed, unwinding external...")
        unwind_loss = _unwind_external_with_retry(
            ext_client, ext_platform, ext_legs, leg_states, exec_size,
        )
        return TradeResult(
            opportunity=opportunity,
            order_ids=order_ids,
            fill_prices=fill_prices,
            fill_sizes=fill_sizes,
            fees=fees,
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
        fees=fees,
        gas_cost=opportunity.estimated_gas_cost,
        net_pnl=net_pnl,
        execution_time_ms=elapsed_ms,
        fully_filled=True,
    )


def _execute_external_legs_with_state_machine(
    ext_legs: list[LegOrder],
    ext_client: PlatformClient,
    leg_states: dict[str, dict[str, FillState | int | str | None]],
    exec_size: int,
    start_time: float,
    deadline: float,
) -> bool:
    """
    Execute external platform legs with state machine tracking.

    Returns:
        True if all legs filled successfully, False otherwise.
    """
    all_filled = True

    for leg in ext_legs:
        token_id = leg.token_id
        state_info = leg_states[token_id]

        # Can only proceed if not in a terminal state
        if is_terminal_state(state_info["state"]):
            logger.debug(
                "Skipping %s: already in terminal state %s",
                token_id, state_info["state"].value,
            )
            continue

        # Transition from PENDING to RESTING
        if state_info["state"] == FillState.PENDING:
            try:
                state_info["state"] = transition_to(
                    state_info["state"], FillState.RESTING
                )
            except ValueError as e:
                logger.error("Invalid state transition for %s: %s", token_id, e)
                state_info["state"] = FillState.REJECTED
                state_info["last_error"] = str(e)
                all_filled = False
                continue

        # Place the order
        try:
            result = ext_client.place_order(
                ticker=token_id,
                side="yes",
                action="buy",
                count=exec_size,
                type="limit",
            )
            order_data = result.get("order", result)
            oid = order_data.get("order_id", f"unknown_{token_id}")
            status = str(order_data.get("status", ""))

            logger.debug("External order placed: %s status=%s", oid, status)

            if status.lower() in ("executed", "filled"):
                state_info["state"] = transition_to(
                    FillState.RESTING, FillState.FILLED
                )
            elif status == "resting":
                state_info["state"] = FillState.RESTING  # Already resting
            elif status in ("rejected", "canceled", "expired"):
                state_info["state"] = transition_to(
                    FillState.RESTING, FillState.REJECTED
                )
                state_info["last_error"] = status
                all_filled = False

        except Exception as e:
            logger.error("External order failed for %s: %s", token_id, e)
            state_info["state"] = FillState.REJECTED
            state_info["last_error"] = str(e)
            all_filled = False

    return all_filled


def _execute_pm_legs_with_state_machine(
    pm_client: ClobClient,
    pm_legs: list[LegOrder],
    leg_states: dict[str, dict[str, FillState | int | str | None]],
    exec_size: int,
    start_time: float,
    use_fak: bool,
) -> bool:
    """
    Execute Polymarket legs with state machine tracking.

    Returns:
        True if all legs filled successfully, False otherwise.
    """
    all_filled = True
    order_type = OrderType.FAK if use_fak else OrderType.GTC

    for leg in pm_legs:
        token_id = leg.token_id
        state_info = leg_states[token_id]

        # Can only proceed if not in a terminal state
        if is_terminal_state(state_info["state"]):
            logger.debug(
                "Skipping %s: already in terminal state %s",
                token_id, state_info["state"].value,
            )
            continue

        # Transition from PENDING to RESTING
        if state_info["state"] == FillState.PENDING:
            try:
                state_info["state"] = transition_to(
                    state_info["state"], FillState.RESTING
                )
            except ValueError as e:
                logger.error("Invalid state transition for %s: %s", token_id, e)
                state_info["state"] = FillState.REJECTED
                state_info["last_error"] = str(e)
                all_filled = False
                continue

        # Create and post the order
        try:
            signed = create_limit_order(
                pm_client,
                token_id=token_id,
                side=leg.side,
                price=leg.price,
                size=float(exec_size),
                neg_risk=False,
            )
            resp = post_order(pm_client, signed, order_type)
            oid = resp.get("orderID", resp.get("order_id", f"unknown_pm"))
            status = str(resp.get("status", ""))

            logger.debug("PM order posted: %s status=%s", oid, status)

            if status.lower() in ("matched", "filled"):
                state_info["state"] = transition_to(
                    FillState.RESTING, FillState.FILLED
                )
            elif status in ("open", "resting"):
                state_info["state"] = FillState.RESTING
            else:
                # Rejected or other failure
                state_info["state"] = transition_to(
                    FillState.RESTING, FillState.REJECTED
                )
                state_info["last_error"] = status
                all_filled = False

        except Exception as e:
            logger.error("PM order failed for %s: %s", token_id, e)
            state_info["state"] = FillState.REJECTED
            state_info["last_error"] = str(e)
            all_filled = False

    return all_filled


def _unwind_external_with_retry(
    ext_client: PlatformClient,
    ext_platform: str,
    ext_legs: list[LegOrder],
    leg_states: dict[str, dict[str, FillState | int | str | None]],
    exec_size: int,
) -> float:
    """
    Unwind external platform positions with retry logic.

    Implements:
    - 3 retry attempts with 0.5s exponential backoff
    - State machine transitions (FILLED -> UNWINDING -> UNWOUND/STUCK)
    - Persistence of stuck positions to JSON file

    Returns:
        Total unwind loss in dollars

    Raises:
        CrossPlatformUnwindFailed: if all retries fail
    """
    total_unwind_loss = 0.0
    stuck_positions = []

    for leg in ext_legs:
        token_id = leg.token_id
        state_info = leg_states[token_id]

        # Transition FILLED -> UNWINDING
        if state_info["state"] == FillState.FILLED:
            try:
                state_info["state"] = transition_to(
                    FillState.FILLED, FillState.UNWINDING
                )
            except ValueError:
                state_info["state"] = FillState.STUCK
                stuck_positions.append({
                    "ticker": token_id,
                    "side": leg.side.value,
                    "size": exec_size,
                    "price": leg.price,
                    "platform": ext_platform,
                    "error": "Invalid state transition",
                })
                continue

        # Attempt unwind with retry logic
        unwind_success = False
        for attempt in range(_MAX_UNWIND_RETRIES):
            state_info["attempts"] = attempt + 1

            try:
                # Place opposite order to unwind
                opposite_action = "sell" if leg.side == Side.BUY else "buy"
                ext_client.place_order(
                    ticker=token_id,
                    side="yes",
                    action=opposite_action,
                    count=int(exec_size),
                    type="market",
                )
                state_info["state"] = transition_to(
                    FillState.UNWINDING, FillState.UNWOUND
                )
                total_unwind_loss += _UNWIND_LOSS_PER_CONTRACT * exec_size
                unwind_success = True
                logger.info(
                    "Unwind success for %s: attempt %d/%d",
                    token_id, attempt + 1, _MAX_UNWIND_RETRIES,
                )
                break

            except Exception as e:
                logger.warning(
                    "Unwind attempt %d/%d failed for %s: %s",
                    attempt + 1, _MAX_UNWIND_RETRIES, token_id, e,
                )
                state_info["last_error"] = str(e)

                # Backoff before retry
                if attempt < _MAX_UNWIND_RETRIES - 1:
                    time.sleep(_UNWIND_BACKOFF_SEC * (2 ** attempt))

        if not unwind_success:
            state_info["state"] = FillState.STUCK
            stuck_positions.append({
                "ticker": token_id,
                "side": leg.side.value,
                "size": exec_size,
                "price": leg.price,
                "platform": ext_platform,
                "error": state_info.get("last_error", "Max retries exceeded"),
            })

    # Persist stuck positions if any
    if stuck_positions:
        _persist_stuck_positions(stuck_positions)

    if len(stuck_positions) > 0:
        raise CrossPlatformUnwindFailed(
            f"Failed to unwind {len(stuck_positions)} {ext_platform} position(s)"
        )

    return total_unwind_loss


def _persist_stuck_positions(positions: list[dict]) -> None:
    """
    Persist stuck positions to JSON file for recovery on next startup.

    Args:
        positions: List of stuck position dicts
    """
    try:
        stuck_file = Path(_STUCK_POSITIONS_FILE)
        existing = []
        if stuck_file.exists():
            try:
                existing = json.loads(stuck_file.read_text())
            except (json.JSONDecodeError, IOError):
                existing = []

        all_positions = existing + positions
        stuck_file.write_text(json.dumps(all_positions, indent=2))
        logger.info("Persisted %d stuck position(s) to %s", len(positions), _STUCK_POSITIONS_FILE)
    except Exception as e:
        logger.error("Failed to persist stuck positions: %s", e)


def check_stuck_positions_on_startup() -> list[dict]:
    """
    Check for and log any stuck positions from previous run.

    Returns:
        List of stuck position dicts, or empty list if none.
    """
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
