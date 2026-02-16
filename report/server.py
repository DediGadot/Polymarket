"""
FastAPI server for the pipeline dashboard. Runs as a daemon thread.

Endpoints serve data from SQLite (read-only). SSE pushes cycle summaries.
"""

from __future__ import annotations

import asyncio
import json
import logging
import threading
from pathlib import Path
from typing import Any

from report.store import ReportStore
from report.readiness import compute_readiness

logger = logging.getLogger(__name__)

STATIC_DIR = Path(__file__).parent / "static"

# SSE subscriber queues (populated by notify_cycle, consumed by /live endpoint)
_sse_queues: list[asyncio.Queue] = []
_sse_lock = threading.Lock()


def notify_cycle(summary: dict[str, Any]) -> None:
    """Push a cycle summary to all SSE subscribers. Thread-safe."""
    with _sse_lock:
        stale: list[asyncio.Queue] = []
        for q in _sse_queues:
            try:
                q.put_nowait(summary)
            except asyncio.QueueFull:
                stale.append(q)
        for q in stale:
            _sse_queues.remove(q)


def create_app(store: ReportStore) -> Any:
    """Build and return the FastAPI application."""
    from fastapi import FastAPI, Query, Request
    from fastapi.responses import HTMLResponse, JSONResponse, StreamingResponse

    app = FastAPI(title="Polymarket Pipeline Dashboard", docs_url="/docs")

    # ── Static ──

    @app.get("/", response_class=HTMLResponse)
    async def index():
        index_path = STATIC_DIR / "index.html"
        if not index_path.exists():
            return HTMLResponse("<h1>Dashboard not found</h1>", status_code=404)
        return HTMLResponse(index_path.read_text())

    # ── Helpers ──

    def _strip_secrets(row: dict) -> dict:
        """Remove config_json from API responses to prevent secret leakage."""
        out = dict(row)
        out.pop("config_json", None)
        return out

    # ── Sessions ──

    @app.get("/api/sessions")
    async def list_sessions():
        return [_strip_secrets(s) for s in store.get_sessions()]

    @app.get("/api/sessions/{session_id}")
    async def get_session(session_id: int):
        row = store.get_session(session_id)
        if not row:
            return JSONResponse({"error": "Session not found"}, status_code=404)
        return _strip_secrets(row)

    # ── Cycles ──

    @app.get("/api/sessions/{session_id}/cycles")
    async def get_cycles(
        session_id: int,
        limit: int = Query(500, ge=1, le=5000),
        offset: int = Query(0, ge=0),
    ):
        return store.get_cycles(session_id, limit=limit, offset=offset)

    # ── Funnel ──

    @app.get("/api/sessions/{session_id}/funnel")
    async def get_funnel(session_id: int, cycle_id: int | None = None):
        if cycle_id is not None:
            return store.get_funnel(cycle_id)
        return store.get_funnel_aggregated(session_id)

    # ── Opportunities ──

    @app.get("/api/sessions/{session_id}/opportunities")
    async def get_opportunities(
        session_id: int,
        type: str | None = None,
        min_score: float = Query(0.0, ge=0.0),
        limit: int = Query(100, ge=1, le=1000),
    ):
        return store.get_opportunities(
            session_id, opp_type=type, min_score=min_score, limit=limit,
        )

    @app.get("/api/sessions/{session_id}/opportunities/unique-actionable")
    async def get_unique_actionable(
        session_id: int,
        limit: int = Query(50, ge=1, le=1000),
        min_score: float = Query(0.0, ge=0.0),
        min_fill: float = Query(0.50, ge=0.0, le=1.0),
        min_persistence: float = Query(0.50, ge=0.0, le=1.0),
    ):
        return store.get_unique_actionable(
            session_id,
            limit=limit,
            min_score=min_score,
            min_fill_score=min_fill,
            min_persistence=min_persistence,
        )

    # ── Safety ──

    @app.get("/api/sessions/{session_id}/safety")
    async def get_safety(
        session_id: int,
        limit: int = Query(200, ge=1, le=5000),
    ):
        return store.get_safety_rejections(session_id, limit=limit)

    # ── Strategy ──

    @app.get("/api/sessions/{session_id}/strategy")
    async def get_strategy(session_id: int):
        return store.get_strategy_timeline(session_id)

    # ── Cross-Platform ──

    @app.get("/api/sessions/{session_id}/cross-platform")
    async def get_cross_platform(session_id: int):
        return store.get_cross_platform_matches(session_id)

    # ── Trades ──

    @app.get("/api/sessions/{session_id}/trades")
    async def get_trades(session_id: int):
        return store.get_trades(session_id)

    # ── P&L ──

    @app.get("/api/sessions/{session_id}/pnl")
    async def get_pnl(session_id: int):
        return store.get_pnl_series(session_id)

    # ── Scanners ──

    @app.get("/api/sessions/{session_id}/scanners")
    async def get_scanners(session_id: int):
        return store.get_scanner_distribution(session_id)

    # ── Readiness ──

    @app.get("/api/sessions/{session_id}/readiness")
    async def get_readiness(session_id: int):
        return compute_readiness(store, session_id)

    # ── Untapped ──

    @app.get("/api/sessions/{session_id}/untapped")
    async def get_untapped(
        session_id: int,
        limit: int = Query(50, ge=1, le=500),
    ):
        return store.get_untapped(session_id, limit=limit)

    # ── SSE ──

    @app.get("/api/live")
    async def live_stream(request: Request):
        q: asyncio.Queue = asyncio.Queue(maxsize=100)
        with _sse_lock:
            _sse_queues.append(q)

        async def event_generator():
            try:
                while True:
                    if await request.is_disconnected():
                        break
                    try:
                        data = await asyncio.wait_for(q.get(), timeout=30.0)
                        yield f"data: {json.dumps(data)}\n\n"
                    except asyncio.TimeoutError:
                        yield ": keepalive\n\n"
            finally:
                with _sse_lock:
                    if q in _sse_queues:
                        _sse_queues.remove(q)

        return StreamingResponse(
            event_generator(),
            media_type="text/event-stream",
            headers={
                "Cache-Control": "no-cache",
                "Connection": "keep-alive",
                "X-Accel-Buffering": "no",
            },
        )

    return app


def start_server(store: ReportStore, host: str = "0.0.0.0", port: int = 8787) -> threading.Thread:
    """Start FastAPI in a daemon thread. Returns the thread."""
    import uvicorn

    app = create_app(store)

    def _run():
        uvicorn.run(
            app,
            host=host,
            port=port,
            log_level="warning",
            access_log=False,
        )

    thread = threading.Thread(target=_run, daemon=True, name="report-server")
    thread.start()
    logger.info("Dashboard server started at http://%s:%d", host, port)
    return thread
