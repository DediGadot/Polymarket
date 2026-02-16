"""
Maker strategy scanner for binary markets.

Detects wide bid-ask spreads where placing GTC limit orders at bid+1tick
on both YES and NO sides costs less than $1.00. If both fill, guaranteed
$1.00 payout at resolution.

Unlike taker arbs (FAK), this uses GTC limit orders that improve the spread,
earning the maker rebate and capturing wider edges. The tradeoff is fill
uncertainty — orders may not fill or may partially fill.
"""

from __future__ import annotations

import logging
import math
import time
from dataclasses import dataclass, field

from scanner.fees import MarketFeeModel
from scanner.models import (
    LegOrder,
    Market,
    Opportunity,
    OpportunityType,
    OrderBook,
    Side,
    is_market_stale,
)

logger = logging.getLogger(__name__)


@dataclass
class MakerPersistenceGate:
    """
    Require a maker setup to persist for N consecutive scan cycles before
    emitting it as an actionable candidate.
    """

    min_consecutive_cycles: int = 3
    _streaks: dict[str, int] = field(default_factory=dict)
    _viable_this_cycle: set[str] = field(default_factory=set)

    def begin_cycle(self) -> None:
        self._viable_this_cycle.clear()

    def mark_viable(self, market_key: str) -> bool:
        streak = self._streaks.get(market_key, 0) + 1
        self._streaks[market_key] = streak
        self._viable_this_cycle.add(market_key)
        return streak >= self.min_consecutive_cycles

    def to_dict(self) -> dict:
        """Serialize gate state to a JSON-safe dict."""
        return {
            "min_consecutive_cycles": self.min_consecutive_cycles,
            "streaks": dict(self._streaks),
        }

    @classmethod
    def from_dict(cls, data: dict) -> MakerPersistenceGate:
        """Restore gate from a serialized dict."""
        gate = cls(min_consecutive_cycles=data.get("min_consecutive_cycles", 3))
        gate._streaks = dict(data.get("streaks", {}))
        return gate

    def end_cycle(self, universe_market_keys: set[str]) -> None:
        # Reset streaks for markets that failed viability this cycle.
        for key in universe_market_keys:
            if key not in self._viable_this_cycle and key in self._streaks:
                self._streaks[key] = 0

        # Prune cold keys to keep memory bounded.
        stale = [k for k, v in self._streaks.items() if v <= 0 and k not in universe_market_keys]
        for k in stale:
            del self._streaks[k]


@dataclass
class MakerExecutionSignal:
    """Execution-quality estimate for a maker pair in the current market state."""

    pair_fill_prob: float
    toxicity_score: float
    orphan_loss_per_set: float
    expected_net_profit: float
    expected_roi_pct: float
    update_rate_hz: float
    spread_ticks_avg: float


@dataclass
class _MakerMicroState:
    """Per-market microstructure state with lightweight EWMA features."""

    last_ts: float
    yes_bid: float
    yes_ask: float
    no_bid: float
    no_ask: float
    ewma_update_hz: float = 0.0
    ewma_mid_move_ticks_per_sec: float = 0.0
    ewma_spread_ticks: float = 0.0
    ewma_spread_widen_ticks: float = 0.0
    ewma_queue_imbalance: float = 0.0


class MakerExecutionModel:
    """
    Queue-aware execution-quality model for paired maker fills.

    This model intentionally uses conservative, explainable signals:
    - quote update rate / micro-move intensity (fill opportunity)
    - spread regime (wider spreads reduce paired-fill odds)
    - queue imbalance (one-sided books are more toxic)
    - spread widening / volatility (adverse-selection risk)
    """

    def __init__(self, *, alpha: float = 0.25) -> None:
        self._alpha = max(0.05, min(0.95, alpha))
        self._state: dict[str, _MakerMicroState] = {}

    def _ewma(self, prev: float, value: float) -> float:
        return (1.0 - self._alpha) * prev + self._alpha * value

    @staticmethod
    def _clamp(value: float, lo: float, hi: float) -> float:
        return max(lo, min(hi, value))

    @staticmethod
    def _sigmoid(x: float) -> float:
        return 1.0 / (1.0 + math.exp(-x))

    def evaluate(
        self,
        *,
        market_key: str,
        tick_size: float,
        yes_book: OrderBook,
        no_book: OrderBook,
        market_volume: float,
        min_depth_sets: float,
        net_profit_per_set: float,
        max_sets: float,
        gas_cost_per_order: float,
    ) -> MakerExecutionSignal:
        """Return paired-fill probability, toxicity, and expected realized EV."""
        now = time.time()
        yes_bid = yes_book.best_bid.price
        yes_ask = yes_book.best_ask.price
        no_bid = no_book.best_bid.price
        no_ask = no_book.best_ask.price
        yes_bid_size = yes_book.best_bid.size
        no_bid_size = no_book.best_bid.size
        yes_ask_size = yes_book.best_ask.size
        no_ask_size = no_book.best_ask.size
        spread_ticks_avg = (
            (yes_ask - yes_bid) / tick_size + (no_ask - no_bid) / tick_size
        ) / 2.0

        state = self._state.get(market_key)
        if state is None:
            state = _MakerMicroState(
                last_ts=now,
                yes_bid=yes_bid,
                yes_ask=yes_ask,
                no_bid=no_bid,
                no_ask=no_ask,
                ewma_spread_ticks=spread_ticks_avg,
            )
            self._state[market_key] = state
            dt = 1.0
            update_hz = 0.0
            mid_move_ticks_per_sec = 0.0
            spread_widen_ticks = 0.0
        else:
            dt = max(0.05, now - state.last_ts)
            changed = (
                yes_bid != state.yes_bid
                or yes_ask != state.yes_ask
                or no_bid != state.no_bid
                or no_ask != state.no_ask
            )
            update_hz = (1.0 / dt) if changed else 0.0

            prev_mid_yes = (state.yes_bid + state.yes_ask) / 2.0
            prev_mid_no = (state.no_bid + state.no_ask) / 2.0
            curr_mid_yes = (yes_bid + yes_ask) / 2.0
            curr_mid_no = (no_bid + no_ask) / 2.0
            mid_move_ticks = (abs(curr_mid_yes - prev_mid_yes) + abs(curr_mid_no - prev_mid_no)) / tick_size
            mid_move_ticks_per_sec = mid_move_ticks / dt
            spread_widen_ticks = max(0.0, spread_ticks_avg - state.ewma_spread_ticks)

        queue_imbalance = abs(
            (yes_bid_size - no_bid_size) / max(1.0, yes_bid_size + no_bid_size)
        )

        state.ewma_update_hz = self._ewma(state.ewma_update_hz, update_hz)
        state.ewma_mid_move_ticks_per_sec = self._ewma(state.ewma_mid_move_ticks_per_sec, mid_move_ticks_per_sec)
        state.ewma_spread_widen_ticks = self._ewma(state.ewma_spread_widen_ticks, spread_widen_ticks)
        state.ewma_spread_ticks = self._ewma(state.ewma_spread_ticks, spread_ticks_avg)
        state.ewma_queue_imbalance = self._ewma(state.ewma_queue_imbalance, queue_imbalance)
        state.last_ts = now
        state.yes_bid = yes_bid
        state.yes_ask = yes_ask
        state.no_bid = no_bid
        state.no_ask = no_ask

        activity_score = self._clamp(state.ewma_update_hz / 2.5, 0.0, 1.0)
        move_score = self._clamp(state.ewma_mid_move_ticks_per_sec / 8.0, 0.0, 1.0)
        widen_score = self._clamp(state.ewma_spread_widen_ticks / 3.0, 0.0, 1.0)
        imbalance_score = self._clamp(state.ewma_queue_imbalance, 0.0, 1.0)
        spread_penalty = self._clamp((state.ewma_spread_ticks - 2.0) / 8.0, 0.0, 1.0)

        toxicity_score = self._clamp(
            0.35 * move_score
            + 0.25 * widen_score
            + 0.25 * imbalance_score
            + 0.15 * activity_score,
            0.0,
            1.0,
        )

        depth_quality = self._clamp(
            min(yes_bid_size, no_bid_size, yes_ask_size, no_ask_size) / max(1.0, min_depth_sets),
            0.0,
            1.0,
        )
        volume_quality = self._clamp(math.log10(max(1.0, market_volume)) / 5.0, 0.0, 1.0)
        symmetry = self._clamp(1.0 - abs((yes_bid + no_bid) - 1.0) / 0.20, 0.0, 1.0)

        z = (
            -0.10
            + 1.20 * activity_score
            + 0.80 * depth_quality
            + 0.55 * volume_quality
            + 0.35 * symmetry
            - 1.25 * spread_penalty
            - 1.65 * toxicity_score
        )
        pair_fill_prob = self._clamp(self._sigmoid(z), 0.01, 0.98)

        avg_spread = ((yes_ask - yes_bid) + (no_ask - no_bid)) / 2.0
        # One-leg orphan loss is typically the unwind slippage from our posted
        # price to immediate hedge plus urgency premium, not the full spread.
        orphan_loss_per_set = max(
            tick_size * 2.0,
            avg_spread * 0.20 + tick_size * 1.5,
        )

        full_fill_net = net_profit_per_set * max_sets - (gas_cost_per_order * 2.0)
        orphan_loss_total = orphan_loss_per_set * max_sets + (gas_cost_per_order * 3.0)
        expected_net_profit = pair_fill_prob * full_fill_net - (1.0 - pair_fill_prob) * orphan_loss_total
        required_capital = max(1e-9, (yes_bid + tick_size + no_bid + tick_size) * max_sets)
        expected_roi_pct = expected_net_profit / required_capital * 100.0

        return MakerExecutionSignal(
            pair_fill_prob=pair_fill_prob,
            toxicity_score=toxicity_score,
            orphan_loss_per_set=orphan_loss_per_set,
            expected_net_profit=expected_net_profit,
            expected_roi_pct=expected_roi_pct,
            update_rate_hz=state.ewma_update_hz,
            spread_ticks_avg=state.ewma_spread_ticks,
        )


def scan_maker_opportunities(
    markets: list[Market],
    books: dict[str, OrderBook],
    fee_model: MarketFeeModel | None = None,
    min_edge_usd: float = 0.01,
    gas_cost_per_order: float = 0.005,
    min_spread_ticks: int = 2,
    min_leg_price: float = 0.05,
    min_depth_sets: float = 5.0,
    min_volume: float = 0.0,
    max_taker_cost: float = 1.03,
    max_spread_ticks: int = 20,
    persistence_gate: MakerPersistenceGate | None = None,
    execution_model: MakerExecutionModel | None = None,
    min_pair_fill_prob: float = 0.0,
    max_toxicity_score: float = 1.0,
    min_expected_ev_usd: float = 0.0,
) -> list[Opportunity]:
    """
    Scan binary markets for maker spread capture opportunities.

    For each binary market, checks if placing GTC limit orders at bid+1tick
    on both YES and NO sides results in a combined cost below $1.00.

    Args:
        markets: Binary markets to scan.
        books: Pre-fetched orderbooks keyed by token_id.
        fee_model: Fee model for cost calculations.
        min_edge_usd: Minimum net profit after gas.
        gas_cost_per_order: Estimated gas cost per order.
        min_spread_ticks: Minimum spread (in ticks) required to consider market.

    Returns:
        List of Opportunities sorted by net profit descending.
    """
    opps: list[Opportunity] = []
    universe_market_keys: set[str] = set()
    if persistence_gate is not None:
        persistence_gate.begin_cycle()

    for market in markets:
        # Skip negRisk (handled by negrisk scanner)
        if market.neg_risk:
            continue

        # Skip inactive or stale markets
        if not market.active or is_market_stale(market):
            continue

        # Skip low-volume markets (persistent wide spreads on illiquid markets)
        if min_volume > 0 and market.volume < min_volume:
            continue

        market_key = market.condition_id or market.event_id
        universe_market_keys.add(market_key)

        yes_book = books.get(market.yes_token_id)
        no_book = books.get(market.no_token_id)

        if not yes_book or not no_book:
            continue

        opp = _check_maker_arb(
            market, yes_book, no_book,
            fee_model=fee_model,
            min_edge_usd=min_edge_usd,
            gas_cost_per_order=gas_cost_per_order,
            min_spread_ticks=min_spread_ticks,
            min_leg_price=min_leg_price,
            min_depth_sets=min_depth_sets,
            max_taker_cost=max_taker_cost,
            max_spread_ticks=max_spread_ticks,
            market_key=market_key,
            persistence_gate=persistence_gate,
            execution_model=execution_model,
            min_pair_fill_prob=min_pair_fill_prob,
            max_toxicity_score=max_toxicity_score,
            min_expected_ev_usd=min_expected_ev_usd,
        )
        if opp:
            opps.append(opp)

    if persistence_gate is not None:
        persistence_gate.end_cycle(universe_market_keys)

    opps.sort(key=lambda o: o.net_profit, reverse=True)
    return opps


def _check_maker_arb(
    market: Market,
    yes_book: OrderBook,
    no_book: OrderBook,
    fee_model: MarketFeeModel | None,
    min_edge_usd: float,
    gas_cost_per_order: float,
    min_spread_ticks: int,
    min_leg_price: float = 0.05,
    min_depth_sets: float = 5.0,
    max_taker_cost: float = 1.03,
    max_spread_ticks: int = 20,
    market_key: str = "",
    persistence_gate: MakerPersistenceGate | None = None,
    execution_model: MakerExecutionModel | None = None,
    min_pair_fill_prob: float = 0.0,
    max_toxicity_score: float = 1.0,
    min_expected_ev_usd: float = 0.0,
) -> Opportunity | None:
    """
    Check if placing GTC limit orders inside the spread is profitable.

    Strategy: post BUY YES at (yes_bid + 1tick) and BUY NO at (no_bid + 1tick).
    If both fill, cost = yes_price + no_price. If cost < $1.00, profit = $1 - cost.
    """
    if not yes_book.best_bid or not no_book.best_bid:
        return None
    if not yes_book.best_ask or not no_book.best_ask:
        return None

    # Filter near-certain markets: if either side's best ask is below
    # min_leg_price, the low-probability side will never fill a GTC order.
    if min_leg_price > 0:
        if yes_book.best_ask and yes_book.best_ask.price < min_leg_price:
            return None
        if no_book.best_ask and no_book.best_ask.price < min_leg_price:
            return None

    tick_size = float(market.min_tick_size)

    yes_bid = yes_book.best_bid.price
    no_bid = no_book.best_bid.price

    # Check spread width on both sides
    yes_spread_ticks = round((yes_book.best_ask.price - yes_bid) / tick_size)
    no_spread_ticks = round((no_book.best_ask.price - no_bid) / tick_size)
    if yes_spread_ticks < min_spread_ticks or no_spread_ticks < min_spread_ticks:
        return None
    if max_spread_ticks > 0:
        if yes_spread_ticks > max_spread_ticks or no_spread_ticks > max_spread_ticks:
            return None

    # Our limit order prices: 1 tick inside the best bid
    yes_price = round(yes_bid + tick_size, 6)
    no_price = round(no_bid + tick_size, 6)

    # Combined cost must be < $1.00
    total_cost = yes_price + no_price
    if total_cost >= 1.0:
        return None

    # Realism gate: if crossing the spread is far above parity, paired maker
    # fills are usually not actionable in practice.
    taker_cost = yes_book.best_ask.price + no_book.best_ask.price
    if taker_cost > max_taker_cost:
        return None

    profit_per_set = 1.0 - total_cost

    # Depth is limited by the smaller side's bid depth
    yes_depth = yes_book.best_bid.size
    no_depth = no_book.best_bid.size
    max_sets = min(yes_depth, no_depth)
    if max_sets <= 0:
        return None

    # Filter micro-depth phantom arbs
    if max_sets < min_depth_sets:
        return None

    # Gas cost: 2 orders (YES + NO)
    gas_cost = gas_cost_per_order * 2

    # Fee adjustment
    legs = (
        LegOrder(
            token_id=market.yes_token_id,
            side=Side.BUY,
            price=yes_price,
            size=max_sets,
            tick_size=market.min_tick_size,
        ),
        LegOrder(
            token_id=market.no_token_id,
            side=Side.BUY,
            price=no_price,
            size=max_sets,
            tick_size=market.min_tick_size,
        ),
    )

    if fee_model:
        net_profit_per_set = fee_model.adjust_profit(profit_per_set, legs, market=market)
    else:
        net_profit_per_set = profit_per_set

    gross_profit = profit_per_set * max_sets
    net_profit = net_profit_per_set * max_sets - gas_cost
    required_capital = total_cost * max_sets
    roi_pct = (net_profit / required_capital * 100) if required_capital > 0 else 0

    pair_fill_prob = 1.0
    toxicity_score = 0.0
    expected_net_profit = net_profit
    expected_roi_pct = roi_pct
    orphan_loss_per_set = 0.0
    update_rate_hz = 0.0
    spread_ticks_avg = (yes_spread_ticks + no_spread_ticks) / 2.0
    if execution_model is not None:
        signal = execution_model.evaluate(
            market_key=market_key or market.condition_id or market.event_id,
            tick_size=tick_size,
            yes_book=yes_book,
            no_book=no_book,
            market_volume=market.volume,
            min_depth_sets=min_depth_sets,
            net_profit_per_set=net_profit_per_set,
            max_sets=max_sets,
            gas_cost_per_order=gas_cost_per_order,
        )
        pair_fill_prob = signal.pair_fill_prob
        toxicity_score = signal.toxicity_score
        expected_net_profit = signal.expected_net_profit
        expected_roi_pct = signal.expected_roi_pct
        orphan_loss_per_set = signal.orphan_loss_per_set
        update_rate_hz = signal.update_rate_hz
        spread_ticks_avg = signal.spread_ticks_avg

        if pair_fill_prob < min_pair_fill_prob:
            return None
        if toxicity_score > max_toxicity_score:
            return None
        if expected_net_profit < min_expected_ev_usd:
            return None

    effective_net_profit = expected_net_profit
    effective_roi_pct = expected_roi_pct
    if effective_net_profit < min_edge_usd:
        return None

    if persistence_gate is not None:
        key = market_key or market.condition_id or market.event_id
        if not persistence_gate.mark_viable(key):
            return None

    logger.debug(
        "MAKER SPREAD: %s | yes_bid=%.4f no_bid=%.4f cost=%.4f taker=%.4f spread=(%dt,%dt) "
        "profit/set=%.4f sets=%.0f net=$%.4f roi=%.2f%% fill_p=%.2f tox=%.2f exp_net=$%.4f "
        "orphan/set=%.4f upd_hz=%.2f spread_avg=%.2f",
        market.question[:50], yes_price, no_price, total_cost, taker_cost,
        yes_spread_ticks, no_spread_ticks, profit_per_set, max_sets, net_profit, roi_pct,
        pair_fill_prob, toxicity_score, expected_net_profit, orphan_loss_per_set,
        update_rate_hz, spread_ticks_avg,
    )

    return Opportunity(
        type=OpportunityType.MAKER_REBALANCE,
        event_id=market.event_id,
        legs=legs,
        expected_profit_per_set=profit_per_set,
        net_profit_per_set=net_profit_per_set,
        max_sets=max_sets,
        gross_profit=gross_profit,
        estimated_gas_cost=gas_cost,
        net_profit=effective_net_profit,
        roi_pct=effective_roi_pct,
        required_capital=required_capital,
        pair_fill_prob=pair_fill_prob,
        toxicity_score=toxicity_score,
        expected_realized_net=expected_net_profit,
        quote_theoretical_net=net_profit,
    )
