"""
Tabular Q-learning strategy selector. Learns optimal strategy mode per
market state from observed cycle P&L.

EXPERIMENTAL: Shadow mode only -- logs what RL would choose vs heuristic,
does not control actual strategy selection until validated.
"""

from __future__ import annotations

import logging
import random
from collections import defaultdict
from dataclasses import dataclass

from scanner.strategy import MarketState, StrategyMode

logger = logging.getLogger(__name__)

# Discretization buckets: (lo_inclusive, hi_exclusive, label)
_GAS_LEVELS = [(0, 30, "low"), (30, 80, "med"), (80, float("inf"), "high")]
_WIN_RATE_BUCKETS = [
    (0, 0.3, "losing"),
    (0.3, 0.6, "average"),
    (0.6, 1.0, "winning"),
]

ACTIONS = list(StrategyMode)


@dataclass(frozen=True)
class DiscreteState:
    """Discretized market state for Q-table lookup."""

    gas_level: str  # "low", "med", "high"
    spike_active: bool
    has_momentum: bool
    win_rate_bucket: str  # "losing", "average", "winning"


def discretize_state(market_state: MarketState) -> DiscreteState:
    """Convert continuous MarketState into discrete Q-table key."""
    gas = market_state.gas_price_gwei
    gas_level = "high"  # default for out-of-range
    for lo, hi, label in _GAS_LEVELS:
        if lo <= gas < hi:
            gas_level = label
            break

    spike_active = market_state.active_spike_count > 0

    wr = market_state.recent_win_rate
    win_bucket = "average"  # default
    for lo, hi, label in _WIN_RATE_BUCKETS:
        if lo <= wr < hi:
            win_bucket = label
            break

    return DiscreteState(
        gas_level=gas_level,
        spike_active=spike_active,
        has_momentum=market_state.has_crypto_momentum,
        win_rate_bucket=win_bucket,
    )


class RLStrategySelector:
    """
    Tabular Q-learning agent for strategy mode selection.

    State space: ~3 x 2 x 2 x 3 = 36 discrete states
    Action space: 4 (AGGRESSIVE, CONSERVATIVE, SPIKE_HUNT, LATENCY_FOCUS)

    Usage (shadow mode)::

        rl = RLStrategySelector()
        rl_action = rl.select(market_state)        # what RL would choose
        heuristic_action = heuristic.select(state)  # what heuristic chose
        # ... execute with heuristic_action ...
        rl.update(market_state, rl_action, cycle_pnl)
        rl.log_comparison(heuristic_action, rl_action, cycle_pnl)
    """

    def __init__(
        self,
        learning_rate: float = 0.1,
        discount_factor: float = 0.95,
        epsilon_start: float = 0.3,
        epsilon_min: float = 0.05,
        epsilon_decay: float = 0.995,
    ) -> None:
        self._lr = learning_rate
        self._gamma = discount_factor
        self._epsilon = epsilon_start
        self._epsilon_min = epsilon_min
        self._epsilon_decay = epsilon_decay

        # Q-table: (DiscreteState, StrategyMode) -> float
        self._q_table: dict[tuple[DiscreteState, StrategyMode], float] = defaultdict(
            float,
        )

        self._total_updates = 0
        self._comparisons: list[dict] = []
        self._max_comparisons = 1000

    # ------------------------------------------------------------------
    # Core RL interface
    # ------------------------------------------------------------------

    def select(self, market_state: MarketState) -> StrategyMode:
        """Select strategy mode via epsilon-greedy policy."""
        state = discretize_state(market_state)

        if random.random() < self._epsilon:
            return random.choice(ACTIONS)

        # Greedy: highest Q-value wins
        q_values = {a: self._q_table[(state, a)] for a in ACTIONS}
        return max(q_values, key=q_values.get)  # type: ignore[arg-type]

    def update(
        self,
        market_state: MarketState,
        action: StrategyMode,
        reward: float,
        next_market_state: MarketState | None = None,
    ) -> None:
        """
        Update Q-table from observed reward.

        Uses TD(0) when *next_market_state* is provided, else treats
        *reward* as a terminal return.
        """
        state = discretize_state(market_state)
        key = (state, action)

        if next_market_state is not None:
            next_state = discretize_state(next_market_state)
            max_next_q = max(self._q_table[(next_state, a)] for a in ACTIONS)
            td_target = reward + self._gamma * max_next_q
        else:
            td_target = reward

        old_q = self._q_table[key]
        self._q_table[key] = old_q + self._lr * (td_target - old_q)

        self._epsilon = max(self._epsilon_min, self._epsilon * self._epsilon_decay)
        self._total_updates += 1

    # ------------------------------------------------------------------
    # Shadow-mode comparison logging
    # ------------------------------------------------------------------

    def log_comparison(
        self,
        heuristic_action: StrategyMode,
        rl_action: StrategyMode,
        cycle_pnl: float,
    ) -> None:
        """Log a comparison between heuristic and RL choices."""
        entry = {
            "heuristic": heuristic_action.value,
            "rl": rl_action.value,
            "agreed": heuristic_action == rl_action,
            "cycle_pnl": cycle_pnl,
        }
        self._comparisons.append(entry)
        if len(self._comparisons) > self._max_comparisons:
            self._comparisons = self._comparisons[-self._max_comparisons:]

        if not entry["agreed"]:
            logger.debug(
                "RL shadow: heuristic=%s rl=%s pnl=$%.2f",
                heuristic_action.value,
                rl_action.value,
                cycle_pnl,
            )

    # ------------------------------------------------------------------
    # Introspection helpers
    # ------------------------------------------------------------------

    def get_q_values(self, market_state: MarketState) -> dict[str, float]:
        """Get Q-values for all actions in the given state."""
        state = discretize_state(market_state)
        return {a.value: self._q_table[(state, a)] for a in ACTIONS}

    def agreement_rate(self) -> float:
        """Fraction of logged cycles where RL and heuristic agreed."""
        if not self._comparisons:
            return 0.0
        agreed = sum(1 for c in self._comparisons if c["agreed"])
        return agreed / len(self._comparisons)

    @property
    def stats(self) -> dict:
        """Summary statistics for monitoring."""
        return {
            "epsilon": self._epsilon,
            "total_updates": self._total_updates,
            "q_table_size": len(self._q_table),
            "agreement_rate": self.agreement_rate(),
            "comparisons_logged": len(self._comparisons),
        }

    # ------------------------------------------------------------------
    # Serialization (checkpoint persistence)
    # ------------------------------------------------------------------

    def to_dict(self) -> dict:
        """Serialize for checkpoint persistence."""
        q_data: dict[str, float] = {}
        for (state, action), value in self._q_table.items():
            key_str = (
                f"{state.gas_level}|{state.spike_active}|"
                f"{state.has_momentum}|{state.win_rate_bucket}|{action.value}"
            )
            q_data[key_str] = value

        return {
            "learning_rate": self._lr,
            "discount_factor": self._gamma,
            "epsilon": self._epsilon,
            "epsilon_min": self._epsilon_min,
            "epsilon_decay": self._epsilon_decay,
            "q_table": q_data,
            "total_updates": self._total_updates,
        }

    @classmethod
    def from_dict(cls, data: dict) -> RLStrategySelector:
        """Restore from checkpoint data."""
        rl = cls(
            learning_rate=data.get("learning_rate", 0.1),
            discount_factor=data.get("discount_factor", 0.95),
            epsilon_start=data.get("epsilon", 0.3),
            epsilon_min=data.get("epsilon_min", 0.05),
            epsilon_decay=data.get("epsilon_decay", 0.995),
        )
        rl._epsilon = data.get("epsilon", 0.3)
        rl._total_updates = data.get("total_updates", 0)

        for key_str, value in data.get("q_table", {}).items():
            parts = key_str.split("|")
            if len(parts) != 5:
                continue
            gas_level, spike_str, momentum_str, win_bucket, action_str = parts
            state = DiscreteState(
                gas_level=gas_level,
                spike_active=spike_str == "True",
                has_momentum=momentum_str == "True",
                win_rate_bucket=win_bucket,
            )
            try:
                action = StrategyMode(action_str)
            except ValueError:
                continue
            rl._q_table[(state, action)] = value

        return rl
