"""
Go-live readiness composite score (0-100) with traffic light recommendation.

Six weighted checks determine whether the bot is ready for live trading.
"""

from __future__ import annotations

from typing import Any

from report.store import ReportStore


# Check weights (must sum to 1.0)
W_WIN_RATE = 0.25
W_SAFETY_PASS = 0.20
W_CIRCUIT_BREAKER = 0.15
W_PNL_TREND = 0.20
W_EXEC_SPEED = 0.10
W_OPP_CONSISTENCY = 0.10


def compute_readiness(store: ReportStore, session_id: int) -> dict[str, Any]:
    """Compute go-live readiness score for a session.

    Returns:
        {
            "score": 0-100,
            "recommendation": "GO" | "CAUTION" | "STOP",
            "checks": [
                {"name": str, "score": 0-1, "weight": float, "detail": str, "passed": bool},
                ...
            ],
            "summary": str,
            "simulated_only": bool,
        }
    """
    trades = store.get_trades(session_id)
    sim_count = sum(1 for t in trades if t.get("simulated"))
    real_trades = [t for t in trades if not t.get("simulated")]
    real_count = len(real_trades)
    simulated_only = len(trades) > 0 and real_count == 0

    checks = [
        _check_win_rate(real_trades, simulated_count=sim_count),
        _check_safety_pass_rate(store, session_id),
        _check_circuit_breaker(store, session_id, real_trades, simulated_count=sim_count),
        _check_pnl_trend(real_trades, simulated_count=sim_count),
        _check_execution_speed(real_trades, simulated_count=sim_count),
        _check_opportunity_consistency(store, session_id),
    ]

    composite = sum(c["score"] * c["weight"] for c in checks) * 100
    composite = max(0.0, min(100.0, composite))

    if composite >= 70:
        recommendation = "GO"
    elif composite >= 40:
        recommendation = "CAUTION"
    else:
        recommendation = "STOP"

    # Build summary text
    session = store.get_session(session_id)
    cycle_count = session["cycle_count"] if session else 0
    total_pnl = sum(t["net_pnl"] for t in real_trades)

    passed_count = sum(1 for c in checks if c["passed"])
    trade_note = f"{real_count} real/paper trades"
    if sim_count:
        trade_note += f" ({sim_count} simulated excluded)"
    summary = (
        f"Analyzed {cycle_count} cycles with {trade_note} (real P&L: ${total_pnl:.2f}). "
        f"{passed_count}/6 checks passed. "
        f"Recommendation: {recommendation}."
    )

    return {
        "score": round(composite, 1),
        "recommendation": recommendation,
        "checks": checks,
        "summary": summary,
        "simulated_only": simulated_only,
        "real_trade_count": real_count,
        "simulated_trade_count": sim_count,
    }


def _check_win_rate(
    trades: list[dict[str, Any]],
    simulated_count: int = 0,
) -> dict[str, Any]:
    """Win rate >= 60% over trades in session."""
    if not trades:
        detail = "No real/paper trades recorded"
        if simulated_count > 0:
            detail += f" ({simulated_count} simulated excluded)"
        return {
            "name": "win_rate",
            "score": 0.0,
            "weight": W_WIN_RATE,
            "detail": detail,
            "passed": False,
        }
    wins = sum(1 for t in trades if t["net_pnl"] > 0)
    rate = wins / len(trades)
    score = min(1.0, rate / 0.60)  # 60% = full score
    passed = rate >= 0.60
    sim_tag = f" ({simulated_count} simulated excluded)" if simulated_count > 0 else ""
    return {
        "name": "win_rate",
        "score": score,
        "weight": W_WIN_RATE,
        "detail": f"{rate:.0%} ({wins}/{len(trades)} trades){sim_tag}",
        "passed": passed,
    }


def _check_safety_pass_rate(store: ReportStore, session_id: int) -> dict[str, Any]:
    """Safety pass rate >= 80%."""
    cycles = store.get_cycles(session_id, limit=9999)
    if not cycles:
        return {
            "name": "safety_pass_rate",
            "score": 0.0,
            "weight": W_SAFETY_PASS,
            "detail": "No cycles recorded",
            "passed": False,
        }
    total_opps = sum(c["opps_found"] for c in cycles)
    total_rejections = len(store.get_safety_rejections(session_id, limit=99999))

    if total_opps == 0:
        return {
            "name": "safety_pass_rate",
            "score": 0.5,
            "weight": W_SAFETY_PASS,
            "detail": "No opportunities found",
            "passed": False,
        }

    pass_rate = 1.0 - (total_rejections / max(total_opps, 1))
    pass_rate = max(0.0, pass_rate)
    score = min(1.0, pass_rate / 0.80)
    passed = pass_rate >= 0.80
    return {
        "name": "safety_pass_rate",
        "score": score,
        "weight": W_SAFETY_PASS,
        "detail": f"{pass_rate:.0%} pass rate ({total_rejections} rejections / {total_opps} opps)",
        "passed": passed,
    }


def _check_circuit_breaker(
    store: ReportStore,
    session_id: int,
    trades: list[dict[str, Any]],
    simulated_count: int = 0,
) -> dict[str, Any]:
    """No circuit breaker trips in session."""
    session = store.get_session(session_id)
    if not session:
        return {
            "name": "circuit_breaker",
            "score": 0.0,
            "weight": W_CIRCUIT_BREAKER,
            "detail": "Session not found",
            "passed": False,
        }
    if not trades:
        detail = "No real/paper trades to evaluate breaker behavior"
        if simulated_count > 0:
            detail += f" ({simulated_count} simulated excluded)"
        return {
            "name": "circuit_breaker",
            "score": 0.5,
            "weight": W_CIRCUIT_BREAKER,
            "detail": detail,
            "passed": False,
        }

    # Check consecutive losses
    max_consecutive_losses = 0
    current_streak = 0
    for t in trades:
        if t["net_pnl"] < 0:
            current_streak += 1
            max_consecutive_losses = max(max_consecutive_losses, current_streak)
        else:
            current_streak = 0

    if max_consecutive_losses >= 5:
        score = 0.2
    elif max_consecutive_losses >= 3:
        score = 0.6
    else:
        score = 1.0

    passed = max_consecutive_losses < 5
    return {
        "name": "circuit_breaker",
        "score": score,
        "weight": W_CIRCUIT_BREAKER,
        "detail": f"Max consecutive losses: {max_consecutive_losses}",
        "passed": passed,
    }


def _check_pnl_trend(
    trades: list[dict[str, Any]],
    simulated_count: int = 0,
) -> dict[str, Any]:
    """Positive cumulative P&L over last 20 real/paper trades."""
    if not trades:
        detail = "No real/paper P&L data"
        if simulated_count > 0:
            detail += f" ({simulated_count} simulated excluded)"
        return {
            "name": "pnl_trend",
            "score": 0.0,
            "weight": W_PNL_TREND,
            "detail": detail,
            "passed": False,
        }

    cumulative = sum(t["net_pnl"] for t in trades)
    recent = trades[-20:] if len(trades) >= 20 else trades
    recent_pnl = sum(t["net_pnl"] for t in recent)

    if cumulative > 0 and recent_pnl > 0:
        score = 1.0
    elif cumulative > 0:
        score = 0.6
    elif recent_pnl > 0:
        score = 0.4
    else:
        score = 0.1

    passed = cumulative > 0
    return {
        "name": "pnl_trend",
        "score": score,
        "weight": W_PNL_TREND,
        "detail": f"Cumulative: ${cumulative:.2f}, Recent: ${recent_pnl:.2f}",
        "passed": passed,
    }


def _check_execution_speed(
    trades: list[dict[str, Any]],
    simulated_count: int = 0,
) -> dict[str, Any]:
    """Average execution time < 500ms."""
    if not trades:
        detail = "No real/paper trades recorded"
        if simulated_count > 0:
            detail += f" ({simulated_count} simulated excluded)"
        return {
            "name": "execution_speed",
            "score": 0.5,
            "weight": W_EXEC_SPEED,
            "detail": detail,
            "passed": False,
        }
    avg_ms = sum(t["execution_time_ms"] for t in trades) / len(trades)
    score = min(1.0, 500.0 / max(avg_ms, 1.0))
    passed = avg_ms < 500.0
    sim_tag = f" ({simulated_count} simulated excluded)" if simulated_count > 0 else ""
    return {
        "name": "execution_speed",
        "score": score,
        "weight": W_EXEC_SPEED,
        "detail": f"Avg: {avg_ms:.0f}ms{sim_tag}",
        "passed": passed,
    }


def _check_opportunity_consistency(store: ReportStore, session_id: int) -> dict[str, Any]:
    """Average >= 5 opportunities per cycle."""
    cycles = store.get_cycles(session_id, limit=9999)
    if not cycles:
        return {
            "name": "opportunity_consistency",
            "score": 0.0,
            "weight": W_OPP_CONSISTENCY,
            "detail": "No cycles recorded",
            "passed": False,
        }
    avg_opps = sum(c["opps_found"] for c in cycles) / len(cycles)
    score = min(1.0, avg_opps / 5.0)
    passed = avg_opps >= 5.0
    return {
        "name": "opportunity_consistency",
        "score": score,
        "weight": W_OPP_CONSISTENCY,
        "detail": f"Avg: {avg_opps:.1f} opps/cycle across {len(cycles)} cycles",
        "passed": passed,
    }
