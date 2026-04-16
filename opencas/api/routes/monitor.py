"""Monitor API routes for the OpenCAS dashboard."""

from __future__ import annotations

from typing import Any, Dict, List

from fastapi import APIRouter
from pydantic import BaseModel, Field

from opencas.diagnostics.models import CheckStatus
from opencas.telemetry import EventKind

router = APIRouter(tags=["monitor"])


class HealthResponse(BaseModel):
    overall: str
    failures: int
    warnings: int
    checks: List[Dict[str, Any]]


class BaaStatusResponse(BaseModel):
    queue_size: int
    held_size: int
    active_count: int


class EmbeddingStatusResponse(BaseModel):
    total_records: int
    model_id: str
    latency_ms: float
    healthy: bool


class EventSampleResponse(BaseModel):
    events: List[Dict[str, Any]]


class RuntimeStatusResponse(BaseModel):
    readiness: Dict[str, Any]
    workspace: Dict[str, Any]
    sandbox: Dict[str, Any]
    execution: Dict[str, Any]
    activity: Dict[str, Any] = Field(default_factory=dict)
    consolidation: Dict[str, Any] = Field(default_factory=dict)


def build_monitor_router(runtime: Any) -> APIRouter:
    """Build monitor routes wired to *runtime*."""
    r = APIRouter(prefix="/api/monitor", tags=["monitor"])

    @r.get("/health", response_model=HealthResponse)
    async def get_health() -> HealthResponse:
        report = await runtime.ctx.doctor.run_all()
        return HealthResponse(
            overall=report.overall.value,
            failures=sum(1 for c in report.checks if c.status == CheckStatus.FAIL),
            warnings=sum(1 for c in report.checks if c.status == CheckStatus.WARN),
            checks=[
                {"name": c.name, "status": c.status.value, "message": c.message, "details": c.details}
                for c in report.checks
            ],
        )

    @r.get("/health/history", response_model=EventSampleResponse)
    async def get_health_history(limit: int = 20) -> EventSampleResponse:
        store = runtime.tracer.store
        events = store.query(kinds=[EventKind.DIAGNOSTIC_RUN], limit=limit)
        return EventSampleResponse(
            events=[
                {
                    "timestamp": e.timestamp.isoformat(),
                    "kind": e.kind.value,
                    "message": e.message,
                    "payload": e.payload,
                    "session_id": e.session_id,
                }
                for e in events
            ]
        )

    @r.get("/baa", response_model=BaaStatusResponse)
    async def get_baa_status() -> BaaStatusResponse:
        baa = getattr(runtime.ctx.harness, "baa", None) or getattr(runtime, "baa", None)
        if baa is None:
            return BaaStatusResponse(queue_size=0, held_size=0, active_count=0)
        return BaaStatusResponse(
            queue_size=getattr(baa, "queue_size", 0),
            held_size=getattr(baa, "held_size", 0),
            active_count=getattr(baa, "active_count", 0),
        )

    @r.get("/embeddings", response_model=EmbeddingStatusResponse)
    async def get_embedding_status() -> EmbeddingStatusResponse:
        svc = runtime.ctx.embeddings
        health = await svc.health()
        latency_ms = health.avg_embed_latency_ms_1h or 0.0
        healthy = health.avg_embed_latency_ms_1h is None or latency_ms < 5000
        return EmbeddingStatusResponse(
            total_records=health.total_records,
            model_id=svc.model_id,
            latency_ms=round(latency_ms, 2),
            healthy=healthy,
        )

    @r.get("/events", response_model=EventSampleResponse)
    async def get_events(limit: int = 50) -> EventSampleResponse:
        store = runtime.tracer.store
        events = store.query(limit=limit)
        return EventSampleResponse(
            events=[
                {
                    "timestamp": e.timestamp.isoformat(),
                    "kind": e.kind.value,
                    "message": e.message,
                    "payload": e.payload,
                    "session_id": e.session_id,
                    "span_id": e.span_id,
                }
                for e in events
            ]
        )

    @r.get("/runtime", response_model=RuntimeStatusResponse)
    async def get_runtime_status() -> RuntimeStatusResponse:
        if hasattr(runtime, "control_plane_status"):
            snapshot = runtime.control_plane_status()
        else:
            sandbox = getattr(runtime.ctx, "sandbox", None)
            sandbox_report = sandbox.report_isolation() if sandbox is not None else {}
            snapshot = {
                "readiness": runtime.ctx.readiness.snapshot() if getattr(runtime.ctx, "readiness", None) else {"state": "unknown"},
                "workspace": {
                    "session_id": getattr(runtime.ctx.config, "session_id", None),
                    "state_dir": str(getattr(runtime.ctx.config, "state_dir", "")),
                    "workspace_roots": [],
                    "allowed_roots": [str(root) for root in getattr(sandbox, "allowed_roots", [])],
                },
                "sandbox": sandbox_report,
                "execution": {
                    "processes": {"total_count": 0, "running_count": 0, "completed_count": 0, "scope_count": 0, "entries": []},
                    "pty": {"total_count": 0, "running_count": 0, "completed_count": 0, "scope_count": 0, "entries": []},
                    "browser": {"available": False, "total_count": 0, "scope_count": 0, "entries": []},
                },
                "activity": {},
                "consolidation": {},
            }
        return RuntimeStatusResponse(**snapshot)

    return r
