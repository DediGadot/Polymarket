"""Tests for scanner.rl_strategy — tabular Q-learning strategy selector."""

from __future__ import annotations

import random

import pytest

from scanner.rl_strategy import (
    ACTIONS,
    DiscreteState,
    RLStrategySelector,
    discretize_state,
)
from scanner.strategy import MarketState, StrategyMode


# -------------------------------------------------------------------
# Helpers
# -------------------------------------------------------------------

def _make_state(
    gas: float = 25.0,
    spikes: int = 0,
    momentum: bool = False,
    win_rate: float = 0.5,
) -> MarketState:
    return MarketState(
        gas_price_gwei=gas,
        active_spike_count=spikes,
        has_crypto_momentum=momentum,
        recent_win_rate=win_rate,
    )


# -------------------------------------------------------------------
# 1. discretize_state
# -------------------------------------------------------------------


class TestDiscretizeState:
    def test_low_gas_average_winrate(self):
        ds = discretize_state(_make_state(gas=10.0, win_rate=0.45))
        assert ds == DiscreteState("low", False, False, "average")

    def test_med_gas_with_spike(self):
        ds = discretize_state(_make_state(gas=50.0, spikes=2))
        assert ds == DiscreteState("med", True, False, "average")

    def test_high_gas_momentum_winning(self):
        ds = discretize_state(
            _make_state(gas=100.0, momentum=True, win_rate=0.75),
        )
        assert ds == DiscreteState("high", False, True, "winning")

    def test_losing_bucket(self):
        ds = discretize_state(_make_state(win_rate=0.1))
        assert ds.win_rate_bucket == "losing"

    def test_boundary_gas_30_is_med(self):
        ds = discretize_state(_make_state(gas=30.0))
        assert ds.gas_level == "med"

    def test_boundary_gas_80_is_high(self):
        ds = discretize_state(_make_state(gas=80.0))
        assert ds.gas_level == "high"

    def test_boundary_winrate_0_is_losing(self):
        ds = discretize_state(_make_state(win_rate=0.0))
        assert ds.win_rate_bucket == "losing"

    def test_boundary_winrate_0_6_is_winning(self):
        ds = discretize_state(_make_state(win_rate=0.6))
        assert ds.win_rate_bucket == "winning"


# -------------------------------------------------------------------
# 2. select returns valid action
# -------------------------------------------------------------------


class TestSelectValid:
    def test_returns_strategy_mode(self):
        rl = RLStrategySelector()
        action = rl.select(_make_state())
        assert isinstance(action, StrategyMode)
        assert action in ACTIONS

    def test_returns_valid_across_states(self):
        rl = RLStrategySelector()
        for gas in (5, 50, 150):
            for spikes in (0, 3):
                for mom in (False, True):
                    for wr in (0.1, 0.5, 0.8):
                        action = rl.select(
                            _make_state(gas=gas, spikes=spikes, momentum=mom, win_rate=wr),
                        )
                        assert action in ACTIONS


# -------------------------------------------------------------------
# 3. Epsilon-greedy exploration
# -------------------------------------------------------------------


class TestEpsilonGreedy:
    def test_epsilon_one_explores(self):
        """With epsilon=1.0 every action should be random."""
        rl = RLStrategySelector(epsilon_start=1.0)
        random.seed(42)
        actions = {rl.select(_make_state()) for _ in range(200)}
        # Should see more than one action with very high probability
        assert len(actions) > 1

    def test_epsilon_zero_is_greedy(self):
        """With epsilon=0.0 the same state should always yield the same action."""
        rl = RLStrategySelector(epsilon_start=0.0)
        state = _make_state()
        first = rl.select(state)
        for _ in range(50):
            assert rl.select(state) == first


# -------------------------------------------------------------------
# 4. Q-table update increases value on positive reward
# -------------------------------------------------------------------


class TestQTableUpdate:
    def test_positive_reward_increases_q(self):
        rl = RLStrategySelector(learning_rate=0.5)
        state = _make_state()
        action = StrategyMode.AGGRESSIVE
        q_before = rl.get_q_values(state)[action.value]

        rl.update(state, action, reward=10.0)
        q_after = rl.get_q_values(state)[action.value]
        assert q_after > q_before

    def test_negative_reward_decreases_q(self):
        rl = RLStrategySelector(learning_rate=0.5)
        state = _make_state()
        action = StrategyMode.AGGRESSIVE
        # Start with a positive Q so we can see decrease
        rl.update(state, action, reward=5.0)
        q_before = rl.get_q_values(state)[action.value]

        rl.update(state, action, reward=-5.0)
        q_after = rl.get_q_values(state)[action.value]
        assert q_after < q_before

    def test_td_update_with_next_state(self):
        rl = RLStrategySelector(learning_rate=0.5, discount_factor=0.9)
        s1 = _make_state(gas=10)
        s2 = _make_state(gas=50)
        action = StrategyMode.CONSERVATIVE

        # Seed s2 with known value so TD target is deterministic
        rl._q_table[(discretize_state(s2), StrategyMode.AGGRESSIVE)] = 2.0

        rl.update(s1, action, reward=1.0, next_market_state=s2)
        # TD target = 1.0 + 0.9 * 2.0 = 2.8  (max next Q = 2.0)
        # Q = 0 + 0.5 * (2.8 - 0) = 1.4
        q = rl.get_q_values(s1)[action.value]
        assert abs(q - 1.4) < 1e-9


# -------------------------------------------------------------------
# 5. Epsilon decay
# -------------------------------------------------------------------


class TestEpsilonDecay:
    def test_epsilon_decreases_after_updates(self):
        rl = RLStrategySelector(epsilon_start=0.3, epsilon_decay=0.99)
        eps_before = rl._epsilon
        state = _make_state()
        for _ in range(20):
            rl.update(state, StrategyMode.AGGRESSIVE, reward=0.0)
        assert rl._epsilon < eps_before

    def test_epsilon_floor(self):
        rl = RLStrategySelector(
            epsilon_start=0.3, epsilon_min=0.05, epsilon_decay=0.5,
        )
        state = _make_state()
        for _ in range(200):
            rl.update(state, StrategyMode.AGGRESSIVE, reward=0.0)
        assert rl._epsilon == pytest.approx(0.05)


# -------------------------------------------------------------------
# 7. Greedy selection follows highest Q
# -------------------------------------------------------------------


class TestGreedySelection:
    def test_picks_highest_q_action(self):
        rl = RLStrategySelector(epsilon_start=0.0)
        state = _make_state()
        ds = discretize_state(state)

        # Manually seed Q-values
        rl._q_table[(ds, StrategyMode.AGGRESSIVE)] = 1.0
        rl._q_table[(ds, StrategyMode.CONSERVATIVE)] = 5.0
        rl._q_table[(ds, StrategyMode.SPIKE_HUNT)] = 3.0
        rl._q_table[(ds, StrategyMode.LATENCY_FOCUS)] = 2.0

        assert rl.select(state) == StrategyMode.CONSERVATIVE

    def test_tiebreak_is_deterministic(self):
        rl = RLStrategySelector(epsilon_start=0.0)
        state = _make_state()
        # All Q = 0 (default), selection should be consistent
        first = rl.select(state)
        for _ in range(20):
            assert rl.select(state) == first


# -------------------------------------------------------------------
# 8. log_comparison and agreement_rate
# -------------------------------------------------------------------


class TestLogComparison:
    def test_agreement_rate_all_agree(self):
        rl = RLStrategySelector()
        for _ in range(10):
            rl.log_comparison(StrategyMode.AGGRESSIVE, StrategyMode.AGGRESSIVE, 1.0)
        assert rl.agreement_rate() == pytest.approx(1.0)

    def test_agreement_rate_none_agree(self):
        rl = RLStrategySelector()
        for _ in range(10):
            rl.log_comparison(StrategyMode.AGGRESSIVE, StrategyMode.CONSERVATIVE, 1.0)
        assert rl.agreement_rate() == pytest.approx(0.0)

    def test_agreement_rate_partial(self):
        rl = RLStrategySelector()
        for _ in range(6):
            rl.log_comparison(StrategyMode.AGGRESSIVE, StrategyMode.AGGRESSIVE, 1.0)
        for _ in range(4):
            rl.log_comparison(StrategyMode.AGGRESSIVE, StrategyMode.SPIKE_HUNT, -1.0)
        assert rl.agreement_rate() == pytest.approx(0.6)

    def test_empty_agreement_rate(self):
        rl = RLStrategySelector()
        assert rl.agreement_rate() == 0.0

    def test_max_comparisons_capped(self):
        rl = RLStrategySelector()
        rl._max_comparisons = 5
        for _ in range(10):
            rl.log_comparison(StrategyMode.AGGRESSIVE, StrategyMode.AGGRESSIVE, 0.0)
        assert len(rl._comparisons) == 5


# -------------------------------------------------------------------
# 9. Serialization round-trip
# -------------------------------------------------------------------


class TestSerialization:
    def test_round_trip_preserves_q_table(self):
        rl = RLStrategySelector(
            learning_rate=0.2,
            discount_factor=0.9,
            epsilon_start=0.15,
            epsilon_min=0.01,
            epsilon_decay=0.99,
        )
        state = _make_state(gas=50, spikes=1, momentum=True, win_rate=0.7)
        rl.update(state, StrategyMode.SPIKE_HUNT, reward=5.0)
        rl.update(state, StrategyMode.AGGRESSIVE, reward=-2.0)

        data = rl.to_dict()
        restored = RLStrategySelector.from_dict(data)

        # Q-values match
        orig_q = rl.get_q_values(state)
        rest_q = restored.get_q_values(state)
        for k in orig_q:
            assert orig_q[k] == pytest.approx(rest_q[k])

        # Hyperparams match
        assert restored._lr == pytest.approx(0.2)
        assert restored._gamma == pytest.approx(0.9)
        assert restored._epsilon_min == pytest.approx(0.01)
        assert restored._epsilon_decay == pytest.approx(0.99)
        assert restored._total_updates == rl._total_updates

    def test_from_dict_ignores_bad_keys(self):
        data = {
            "q_table": {
                "bad_key": 1.0,
                "low|True|False|average|aggressive": 2.5,
            },
        }
        rl = RLStrategySelector.from_dict(data)
        # Good key restored, bad key skipped
        ds = DiscreteState("low", True, False, "average")
        assert rl._q_table[(ds, StrategyMode.AGGRESSIVE)] == pytest.approx(2.5)
        assert rl._total_updates == 0

    def test_from_dict_ignores_bad_action(self):
        data = {
            "q_table": {"low|False|False|average|nonexistent_mode": 9.9},
        }
        rl = RLStrategySelector.from_dict(data)
        assert len(rl._q_table) == 0

    def test_empty_dict_creates_fresh(self):
        rl = RLStrategySelector.from_dict({})
        assert rl._total_updates == 0
        assert len(rl._q_table) == 0


# -------------------------------------------------------------------
# 10. Safety override guard (conceptual test)
# -------------------------------------------------------------------


class TestSafetyOverride:
    def test_caller_can_override_to_conservative(self):
        """
        Demonstrates the safety pattern: if RL picks a mode that the
        caller deems unsafe, the caller overrides to CONSERVATIVE.
        This is enforced in run.py, not in rl_strategy.py itself.
        """
        rl = RLStrategySelector(epsilon_start=0.0)
        state = _make_state()
        ds = discretize_state(state)

        # Make RL favor AGGRESSIVE
        rl._q_table[(ds, StrategyMode.AGGRESSIVE)] = 100.0
        rl_choice = rl.select(state)
        assert rl_choice == StrategyMode.AGGRESSIVE

        # Caller safety guard
        safety_violated = True  # hypothetical safety check
        final = StrategyMode.CONSERVATIVE if safety_violated else rl_choice
        assert final == StrategyMode.CONSERVATIVE


# -------------------------------------------------------------------
# 11. get_q_values
# -------------------------------------------------------------------


class TestGetQValues:
    def test_returns_all_four_actions(self):
        rl = RLStrategySelector()
        qv = rl.get_q_values(_make_state())
        assert set(qv.keys()) == {m.value for m in StrategyMode}

    def test_default_values_are_zero(self):
        rl = RLStrategySelector()
        qv = rl.get_q_values(_make_state())
        for v in qv.values():
            assert v == pytest.approx(0.0)

    def test_reflects_updates(self):
        rl = RLStrategySelector(learning_rate=1.0)
        state = _make_state()
        rl.update(state, StrategyMode.LATENCY_FOCUS, reward=7.0)
        qv = rl.get_q_values(state)
        assert qv["latency_focus"] == pytest.approx(7.0)


# -------------------------------------------------------------------
# 12. Shadow mode workflow (end-to-end)
# -------------------------------------------------------------------


class TestShadowModeWorkflow:
    def test_full_cycle(self):
        rl = RLStrategySelector(epsilon_start=0.0)
        state = _make_state(gas=20, win_rate=0.55)

        # Step 1: RL selects an action
        rl_action = rl.select(state)
        assert isinstance(rl_action, StrategyMode)

        # Step 2: Heuristic picks (simulated)
        heuristic_action = StrategyMode.AGGRESSIVE

        # Step 3: Execute cycle, observe P&L
        cycle_pnl = 2.50

        # Step 4: Update RL from outcome
        next_state = _make_state(gas=25, win_rate=0.58)
        rl.update(state, rl_action, reward=cycle_pnl, next_market_state=next_state)

        # Step 5: Log comparison
        rl.log_comparison(heuristic_action, rl_action, cycle_pnl)

        # Verify state
        assert rl._total_updates == 1
        assert len(rl._comparisons) == 1
        assert rl.stats["total_updates"] == 1

    def test_multi_cycle_learning(self):
        """RL should learn to prefer an action that consistently earns reward."""
        rl = RLStrategySelector(
            learning_rate=0.5, epsilon_start=0.0, discount_factor=0.0,
        )
        state = _make_state()

        # Train: AGGRESSIVE always gets +10, CONSERVATIVE always gets -5
        for _ in range(20):
            rl.update(state, StrategyMode.AGGRESSIVE, reward=10.0)
            rl.update(state, StrategyMode.CONSERVATIVE, reward=-5.0)

        # With epsilon=0, greedy should now pick AGGRESSIVE
        assert rl.select(state) == StrategyMode.AGGRESSIVE

    def test_stats_populated(self):
        rl = RLStrategySelector()
        state = _make_state()
        rl.update(state, StrategyMode.SPIKE_HUNT, reward=1.0)
        rl.log_comparison(StrategyMode.SPIKE_HUNT, StrategyMode.SPIKE_HUNT, 1.0)

        stats = rl.stats
        assert stats["total_updates"] == 1
        assert stats["comparisons_logged"] == 1
        assert stats["agreement_rate"] == pytest.approx(1.0)
        assert stats["epsilon"] < 0.3  # decayed once
        assert stats["q_table_size"] >= 1
