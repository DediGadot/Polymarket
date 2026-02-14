"""
Cross-platform fill state machine.

Defines states for cross-platform order execution and valid transitions between them.
Used by executor/cross_platform.py to track order lifecycle and handle retries.
"""

from __future__ import annotations

import logging
from enum import Enum
from typing import Literal

logger = logging.getLogger(__name__)


class FillState(str, Enum):
    """
    States for a fill in cross-platform execution.

    State flow:
    - PENDING: Initial state, order placed but not yet confirmed
    - FILLED: Order successfully filled at requested price/size
    - PARTIAL: Order partially filled (rest of order working)
    - REJECTED: Order rejected by exchange (price outside limits, etc)
    - RESTING: Limit order placed and waiting for fill
    - UNWINDING: Attempting to unwind a position on external platform
    - UNWOUND: Position successfully unwound
    - STUCK: Failed to unwind after max retries (manual intervention needed)
    """
    PENDING = "pending"
    FILLED = "filled"
    PARTIAL = "partial"
    REJECTED = "rejected"
    RESTING = "resting"
    UNWINDING = "unwinding"
    UNWOUND = "unwound"
    STUCK = "stuck"

    def __new__(cls, value):
        # Create enum member with value = name for string equality
        obj = str.__new__(cls, value)
        obj._value_ = value  # Store original value
        obj._name_ = value  # Set name to match value
        return obj

    @property
    def value(self):
        # Return value (same as name for string enum)
        return self._value_

    @property
    def name(self):
        # Return name (same as value for string enum)
        return self._name_


# Valid state transitions: {from_state: set[valid_to_states]}
_VALID_TRANSITIONS: dict[FillState, set[FillState]] = {
    FillState.PENDING: {FillState.FILLED, FillState.REJECTED, FillState.RESTING, FillState.PARTIAL, FillState.STUCK},
    FillState.RESTING: {FillState.FILLED, FillState.REJECTED, FillState.PARTIAL, FillState.RESTING, FillState.STUCK},
    FillState.PARTIAL: {FillState.FILLED, FillState.PARTIAL, FillState.RESTING, FillState.STUCK},
    FillState.FILLED: {FillState.UNWINDING},  # After fill, can only go to unwinding
    FillState.REJECTED: {FillState.UNWINDING},  # After rejection, need to unwind other leg
    FillState.UNWINDING: {FillState.UNWOUND, FillState.STUCK},
    FillState.UNWOUND: set(),  # Terminal state
    FillState.STUCK: set(),  # Terminal state
}


def can_transition_to(from_state: FillState, to_state: FillState) -> bool:
    """
    Check if a state transition is valid.

    Args:
        from_state: Current state
        to_state: Target state to transition to

    Returns:
        True if transition is valid, False otherwise

    Raises:
        ValueError: If from_state or to_state is not a valid FillState
    """
    if not isinstance(from_state, FillState):
        raise ValueError(f"Invalid from_state: {from_state}")
    if not isinstance(to_state, FillState):
        raise ValueError(f"Invalid to_state: {to_state}")

    valid_targets = _VALID_TRANSITIONS.get(from_state, set())
    return to_state in valid_targets


def transition_to(
    from_state: FillState,
    to_state: FillState,
) -> FillState:
    """
    Perform a state transition, returning the new state.

    Args:
        from_state: Current state
        to_state: Target state to transition to

    Returns:
        The new state (to_state)

    Raises:
        ValueError: If transition is invalid
    """
    if not can_transition_to(from_state, to_state):
        raise ValueError(
            f"Invalid state transition: {from_state.value} -> {to_state.value}"
        )
    logger.debug("State transition: %s -> %s", from_state.value, to_state.value)
    return to_state


def is_terminal_state(state: FillState) -> bool:
    """
    Check if a state is terminal (no further transitions possible).

    Terminal states: FILLED (if not unwinding), UNWOUND, REJECTED, STUCK

    Args:
        state: State to check

    Returns:
        True if state is terminal, False otherwise
    """
    return state in {
        FillState.FILLED,  # Terminal unless followed by unwinding
        FillState.REJECTED,
        FillState.UNWOUND,
        FillState.STUCK,
    }


def is_final_state(state: FillState) -> bool:
    """
    Check if a state represents final execution status.

    Final states: FILLED (complete), REJECTED, STUCK

    Args:
        state: State to check

    Returns:
        True if state is final, False otherwise
    """
    return state in {
        FillState.FILLED,
        FillState.REJECTED,
        FillState.STUCK,
    }


def get_progress_states() -> set[FillState]:
    """
    Return set of states that indicate work in progress.

    Returns:
        Set of states: {PENDING, RESTING, PARTIAL, UNWINDING}
    """
    return {
        FillState.PENDING,
        FillState.RESTING,
        FillState.PARTIAL,
        FillState.UNWINDING,
    }


def get_failure_states() -> set[FillState]:
    """
    Return set of states that indicate failure/rejection.

    Returns:
        Set of states: {REJECTED, STUCK}
    """
    return {FillState.REJECTED, FillState.STUCK}
