"""
Report module — pipeline telemetry + interactive dashboard.

Usage:
    from report import create_collector
    collector = create_collector(enabled=True)  # or False for NullCollector
"""

from __future__ import annotations

from report.collector import ReportCollector, NullCollector


def create_collector(
    enabled: bool = False,
    db_path: str | None = None,
) -> ReportCollector | NullCollector:
    """Factory: returns active collector or zero-overhead null."""
    if not enabled:
        return NullCollector()
    from report.store import ReportStore
    store = ReportStore(db_path=db_path)
    return ReportCollector(store)
