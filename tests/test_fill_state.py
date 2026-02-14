"""
Unit tests for executor/fill_state.py -- cross-platform fill state machine.

Tests FillState enum, state transitions, and invalid transitions.
"""

import pytest

from executor.fill_state import (
    FillState,
    can_transition_to,
    transition_to,
    is_terminal_state,
)


class TestFillStateEnum:
    """Test FillState enum definition and values."""

    def test_all_states_defined(self):
        """All required states should be defined."""
        required_states = {
            "PENDING",
            "FILLED",
            "PARTIAL",
            "REJECTED",
            "RESTING",
            "UNWINDING",
            "UNWOUND",
            "STUCK",
        }
        actual_states = {s.name for s in FillState}
        assert actual_states == required_states

    def test_state_values_are_strings(self):
        """State enum values should be readable strings."""
        for state in FillState:
            assert isinstance(state.value, str)
            assert len(state.value) > 0


class TestStateTransitions:
    """Test valid state transitions."""

    def test_pending_to_filled(self):
        """PENDING -> FILLED is valid after successful fill."""
        assert can_transition_to(FillState.PENDING, FillState.FILLED)

    def test_pending_to_rejected(self):
        """PENDING -> REJECTED is valid after order rejection."""
        assert can_transition_to(FillState.PENDING, FillState.REJECTED)

    def test_pending_to_resting(self):
        """PENDING -> RESTING is valid for resting limit orders."""
        assert can_transition_to(FillState.PENDING, FillState.RESTING)

    def test_resting_to_filled(self):
        """RESTING -> FILLED is valid when resting order fills."""
        assert can_transition_to(FillState.RESTING, FillState.FILLED)

    def test_resting_to_partial(self):
        """RESTING -> PARTIAL is valid for partial fills."""
        assert can_transition_to(FillState.RESTING, FillState.PARTIAL)

    def test_partial_to_filled(self):
        """PARTIAL -> FILLED is valid when fully filled."""
        assert can_transition_to(FillState.PARTIAL, FillState.FILLED)

    def test_filled_to_unwinding(self):
        """FILLED -> UNWINDING is valid for cross-platform unwind."""
        assert can_transition_to(FillState.FILLED, FillState.UNWINDING)

    def test_unwinding_to_unwound(self):
        """UNWINDING -> UNWOUND is valid after successful unwind."""
        assert can_transition_to(FillState.UNWINDING, FillState.UNWOUND)

    def test_unwinding_to_stuck(self):
        """UNWINDING -> STUCK is valid after multiple failed retries."""
        assert can_transition_to(FillState.UNWINDING, FillState.STUCK)

    def test_any_to_stuck_on_max_retries(self):
        """Most states can transition to STUCK after max retries."""
        # After max retries, many states can go to STUCK
        for state in [FillState.PENDING, FillState.RESTING, FillState.UNWINDING]:
            assert can_transition_to(state, FillState.STUCK)


class TestInvalidTransitions:
    """Test invalid state transitions."""

    def test_filled_cannot_go_to_pending(self):
        """FILLED -> PENDING is invalid (already executed)."""
        assert not can_transition_to(FillState.FILLED, FillState.PENDING)

    def test_unwound_cannot_go_to_pending(self):
        """UNWOUND -> PENDING is invalid (already unwound)."""
        assert not can_transition_to(FillState.UNWOUND, FillState.PENDING)

    def test_rejected_cannot_go_to_filled(self):
        """REJECTED -> FILLED is invalid (order was rejected)."""
        assert not can_transition_to(FillState.REJECTED, FillState.FILLED)

    def test_stuck_cannot_go_to_filled(self):
        """STUCK -> FILLED is invalid (stuck means failed)."""
        assert not can_transition_to(FillState.STUCK, FillState.FILLED)

    def test_unwound_cannot_go_to_unwinding(self):
        """UNWOUND -> UNWINDING is invalid (already unwound)."""
        assert not can_transition_to(FillState.UNWOUND, FillState.UNWINDING)

    def test_invalid_state_raises(self):
        """Transition from terminal states back to active states should raise."""
        with pytest.raises(ValueError, match="Invalid state transition"):
            transition_to(FillState.UNWOUND, FillState.PENDING)


class TestTerminalStates:
    """Test terminal state identification."""

    def test_filled_is_terminal(self):
        """FILLED is a terminal state for single-leg orders."""
        assert is_terminal_state(FillState.FILLED)

    def test_unwound_is_terminal(self):
        """UNWOUND is terminal (unwind complete)."""
        assert is_terminal_state(FillState.UNWOUND)

    def test_rejected_is_terminal(self):
        """REJECTED is terminal (order rejected)."""
        assert is_terminal_state(FillState.REJECTED)

    def test_stuck_is_terminal(self):
        """STUCK is terminal (manual intervention required)."""
        assert is_terminal_state(FillState.STUCK)

    def test_pending_is_not_terminal(self):
        """PENDING is not terminal (waiting for fill)."""
        assert not is_terminal_state(FillState.PENDING)

    def test_unwinding_is_not_terminal(self):
        """UNWINDING is not terminal (in progress)."""
        assert not is_terminal_state(FillState.UNWINDING)


class TestTransitionToFunction:
    """Test transition_to function behavior."""

    def test_returns_target_state(self):
        """transition_to should return the target state."""
        result = transition_to(FillState.PENDING, FillState.FILLED)
        assert result == FillState.FILLED

    def test_raises_on_invalid_transition(self):
        """transition_to should raise ValueError on invalid transition."""
        with pytest.raises(ValueError, match="Invalid state transition"):
            transition_to(FillState.STUCK, FillState.PENDING)

    def test_allows_valid_transitions(self):
        """transition_to should allow all defined valid transitions."""
        # PENDING can go to FILLED, REJECTED, RESTING
        assert transition_to(FillState.PENDING, FillState.FILLED) == FillState.FILLED
        assert transition_to(FillState.PENDING, FillState.REJECTED) == FillState.REJECTED
        assert transition_to(FillState.PENDING, FillState.RESTING) == FillState.RESTING


class TestStateStringRepresentation:
    """Test state string representations."""

    def test_state_name_is_value(self):
        """State name should match its value (case-insensitive for string enum)."""
        for state in FillState:
            # For string enums, name is uppercase ('PENDING') and value is lowercase ('pending')
            assert state.name.lower() == state.value

    def test_states_are_hashable(self):
        """States should be usable in sets/dicts."""
        state_set = {FillState.PENDING, FillState.FILLED, FillState.STUCK}
        assert len(state_set) == 3
        assert FillState.PENDING in state_set
