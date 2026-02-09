"""
Adaptive strategy selection. Tunes scan parameters per cycle based on
current market conditions (gas prices, spread tightness, active spikes).

Modes:
- AGGRESSIVE: Low gas + wide spreads → lower thresholds, bigger sizes
- CONSERVATIVE: High gas + tight spreads → higher thresholds, smaller sizes
- SPIKE_HUNT: Active spike → disable steady-state, focus on spike + siblings
- LATENCY_FOCUS: Crypto market momentum detected → prioritize latency scanner
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from enum import Enum

logger = logging.getLogger(__name__)


class StrategyMode(Enum):
    AGGRESSIVE = "aggressive"
    CONSERVATIVE = "conservative"
    SPIKE_HUNT = "spike_hunt"
    LATENCY_FOCUS = "latency_focus"


@dataclass(frozen=True)
class ScanParams:
    """Tunable parameters for a single scan cycle."""
    mode: StrategyMode
    min_profit_usd: float
    min_roi_pct: float
    target_size_usd: float
    binary_enabled: bool = True
    negrisk_enabled: bool = True
    latency_enabled: bool = True
    spike_enabled: bool = True


@dataclass(frozen=True)
class MarketState:
    """Current market conditions used to select strategy."""
    gas_price_gwei: float = 30.0
    avg_spread_pct: float = 2.0  # average bid-ask spread across scanned markets
    active_spike_count: int = 0
    has_crypto_momentum: bool = False
    recent_win_rate: float = 0.50


# Thresholds for mode selection
HIGH_GAS_GWEI = 100.0
LOW_GAS_GWEI = 20.0
WIDE_SPREAD_PCT = 3.0
TIGHT_SPREAD_PCT = 1.0


class StrategySelector:
    """
    Selects strategy mode and tunes scan parameters based on market conditions.
    """

    def __init__(
        self,
        base_min_profit: float = 0.50,
        base_min_roi: float = 2.0,
        base_target_size: float = 100.0,
    ):
        self._base_profit = base_min_profit
        self._base_roi = base_min_roi
        self._base_size = base_target_size

    def select(self, state: MarketState) -> ScanParams:
        """Select strategy mode and return tuned ScanParams."""
        mode = self._pick_mode(state)

        if mode == StrategyMode.SPIKE_HUNT:
            return ScanParams(
                mode=mode,
                min_profit_usd=self._base_profit * 0.5,  # lower threshold for spike opps
                min_roi_pct=self._base_roi * 0.5,
                target_size_usd=self._base_size * 1.5,
                binary_enabled=False,  # focus on spike siblings
                negrisk_enabled=True,
                latency_enabled=False,
                spike_enabled=True,
            )

        if mode == StrategyMode.LATENCY_FOCUS:
            return ScanParams(
                mode=mode,
                min_profit_usd=self._base_profit * 0.8,
                min_roi_pct=self._base_roi * 0.5,
                target_size_usd=self._base_size,
                binary_enabled=True,
                negrisk_enabled=True,
                latency_enabled=True,
                spike_enabled=True,
            )

        if mode == StrategyMode.AGGRESSIVE:
            return ScanParams(
                mode=mode,
                min_profit_usd=self._base_profit * 0.7,
                min_roi_pct=self._base_roi * 0.7,
                target_size_usd=self._base_size * 1.5,
                binary_enabled=True,
                negrisk_enabled=True,
                latency_enabled=True,
                spike_enabled=True,
            )

        # CONSERVATIVE
        return ScanParams(
            mode=mode,
            min_profit_usd=self._base_profit * 1.5,
            min_roi_pct=self._base_roi * 1.5,
            target_size_usd=self._base_size * 0.5,
            binary_enabled=True,
            negrisk_enabled=True,
            latency_enabled=True,
            spike_enabled=True,
        )

    def _pick_mode(self, state: MarketState) -> StrategyMode:
        """Determine the best mode given current conditions."""
        # Priority 1: Active spikes override everything
        if state.active_spike_count > 0:
            return StrategyMode.SPIKE_HUNT

        # Priority 2: Crypto momentum detected
        if state.has_crypto_momentum:
            return StrategyMode.LATENCY_FOCUS

        # Priority 3: Market conditions
        low_gas = state.gas_price_gwei < LOW_GAS_GWEI
        wide_spreads = state.avg_spread_pct > WIDE_SPREAD_PCT

        if low_gas and wide_spreads:
            return StrategyMode.AGGRESSIVE

        high_gas = state.gas_price_gwei > HIGH_GAS_GWEI
        tight_spreads = state.avg_spread_pct < TIGHT_SPREAD_PCT

        if high_gas or tight_spreads:
            return StrategyMode.CONSERVATIVE

        # Default: aggressive if winning, conservative if losing
        if state.recent_win_rate >= 0.50:
            return StrategyMode.AGGRESSIVE
        return StrategyMode.CONSERVATIVE
