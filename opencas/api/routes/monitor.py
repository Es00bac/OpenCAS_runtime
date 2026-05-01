"""Monitor API routes for the OpenCAS dashboard."""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Any, Dict, List

from fastapi import APIRouter
from pydantic import BaseModel, Field

from opencas.api.meaningful_loop_observability import build_meaningful_loop_status
from opencas.api.provenance_store import (
    ProvenanceTransitionKind,
    record_provenance_transition,
)
from opencas.bootstrap.task_beacon import (
    build_task_beacon,
    public_task_beacon_payload,
    runtime_task_beacon_fragments,
)
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
    recent_records: List[Dict[str, Any]] = Field(default_factory=list)


class EventSampleResponse(BaseModel):
    events: List[Dict[str, Any]]


class TaskBeaconResponse(BaseModel):
    available: bool = False
    source: str | None = None
    matched_only: bool = False
    error: str | None = None
    headline: str = ""
    counts: Dict[str, int] = Field(default_factory=dict)
    bucket_signature: str = ""
    view_model: Dict[str, Any] = Field(default_factory=dict)
    details: Dict[str, Any] | None = None
    rules: List[str] = Field(default_factory=list)
    model: Dict[str, Any] = Field(default_factory=dict)


class RuntimeStatusResponse(BaseModel):
    readiness: Dict[str, Any]
    workspace: Dict[str, Any]
    sandbox: Dict[str, Any]
    execution: Dict[str, Any]
    activity: Dict[str, Any] = Field(default_factory=dict)
    consolidation: Dict[str, Any] = Field(default_factory=dict)
    web_trust: Dict[str, Any] = Field(default_factory=dict)
    plugin_trust: Dict[str, Any] = Field(default_factory=dict)


class ShadowRegistryResponse(BaseModel):
    available: bool = False
    total_entries: int = 0
    active_clusters: int = 0
    dismissed_clusters: int = 0
    reason_counts: Dict[str, int] = Field(default_factory=dict)
    recent_entries: List[Dict[str, Any]] = Field(default_factory=list)
    top_clusters: List[Dict[str, Any]] = Field(default_factory=list)


class ShadowRegistryClusterResponse(BaseModel):
    available: bool = False
    fingerprint: str | None = None
    count: int = 0
    block_reason: str | None = None
    tool_name: str | None = None
    intent_summary: str | None = None
    latest_captured_at: str | None = None
    triage_status: str = "active"
    annotation: str | None = None
    triaged_at: str | None = None
    dismissed_at: str | None = None
    entries: List[Dict[str, Any]] = Field(default_factory=list)


class ShadowRegistryClusterTriageRequest(BaseModel):
    fingerprint: str
    annotation: str | None = None
    dismissed: bool | None = None


def build_monitor_router(runtime: Any) -> APIRouter:
    """Build monitor routes wired to *runtime*."""
    r = APIRouter(prefix="/api/monitor", tags=["monitor"])

    @r.get("/health", response_model=HealthResponse)
    async def get_health() -> HealthResponse:
        report = await runtime.ctx.doctor.run_all()
        config = getattr(getattr(runtime, "ctx", None), "config", None)
        state_dir = getattr(config, "state_dir", None)
        session_id = str(getattr(config, "session_id", "runtime")) if config is not None else "runtime"
        if state_dir is not None:
            for check in report.checks:
                record_provenance_transition(
                    state_dir=state_dir,
                    kind=ProvenanceTransitionKind.CHECK,
                    session_id=session_id,
                    entity_id=check.name,
                    status="checked",
                    trigger_artifact="monitor|health|runtime",
                    source_artifact="monitor|health|runtime",
                    trigger_action="doctor.run_all",
                    parent_transition_id=str(check.check_id),
                    target_entity=check.name,
                    origin_action_id=str(check.check_id),
                    details={
                        "check_status": check.status.value,
                        "message": check.message,
                    },
                )
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
        recent = await svc.recent_records(limit=6)
        return EmbeddingStatusResponse(
            total_records=health.total_records,
            model_id=svc.model_id,
            latency_ms=round(latency_ms, 2),
            healthy=healthy,
            recent_records=[
                {
                    "embedding_id": str(record.embedding_id),
                    "model_id": record.model_id,
                    "created_at": record.created_at.isoformat(),
                    "updated_at": record.updated_at.isoformat(),
                    "task_type": record.meta.get("task_type"),
                    "source": record.meta.get("source"),
                    "preview": str(record.meta.get("text") or "").strip()[:140],
                    "degraded": bool(record.meta.get("embedding_degraded")),
                }
                for record in recent
            ],
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

    @r.get("/task-beacon", response_model=TaskBeaconResponse, response_model_exclude_none=True)
    async def get_task_beacon() -> TaskBeaconResponse:
        workspace_root = getattr(getattr(runtime.ctx, "config", None), "workspace_root", None)
        return TaskBeaconResponse(
            **public_task_beacon_payload(
                build_task_beacon(
                    workspace_root,
                    limit_per_state=1,
                    live_fragments=runtime_task_beacon_fragments(runtime),
                )
            )
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

    @r.get("/meaningful-loop")
    async def get_meaningful_loop_status(
        task_limit: int = 20,
        pressure_limit: int = 8,
    ) -> Dict[str, Any]:
        return await build_meaningful_loop_status(
            runtime,
            task_limit=task_limit,
            pressure_limit=pressure_limit,
        )

    @r.get("/affective-examinations")
    async def get_affective_examinations(
        limit: int = 20,
        session_id: str | None = None,
        source_type: str | None = None,
        emotion: str | None = None,
        action_pressure: str | None = None,
        consumed_by: str | None = None,
        decay_state: str | None = None,
    ) -> Dict[str, Any]:
        service = getattr(runtime.ctx, "affective_examinations", None)
        loader = getattr(service, "list_recent", None)
        if not callable(loader):
            return {
                "available": False,
                "count": 0,
                "filters": {},
                "items": [],
                "counts": {"active": 0, "expired": 0, "unconsumed": 0},
            }
        bounded_limit = max(1, min(int(limit), 100))
        filters = {
            "session_id": session_id,
            "source_type": source_type,
            "emotion": emotion,
            "action_pressure": action_pressure,
            "consumed_by": consumed_by,
            "decay_state": decay_state,
        }
        try:
            records = await loader(
                limit=bounded_limit,
                session_id=session_id or None,
                source_type=source_type or None,
                primary_emotion=emotion or None,
                action_pressure=action_pressure or None,
                consumed_by=consumed_by or None,
                decay_state=decay_state or None,
            )
        except TypeError:
            records = await loader(limit=bounded_limit, session_id=session_id or None)
        except Exception as exc:
            return {
                "available": False,
                "count": 0,
                "filters": {key: value for key, value in filters.items() if value},
                "items": [],
                "counts": {"active": 0, "expired": 0, "unconsumed": 0},
                "error": str(exc),
            }
        items = [_affective_examination_payload(record) for record in records]
        counts = {
            "active": sum(1 for item in items if item["decay_state"] == "active"),
            "expired": sum(1 for item in items if item["decay_state"] == "expired"),
            "unconsumed": sum(1 for item in items if item["consumed_by"] == "none"),
        }
        pressure_counts: Dict[str, int] = {}
        emotion_counts: Dict[str, int] = {}
        for item in items:
            pressure_counts[item["action_pressure"]] = pressure_counts.get(item["action_pressure"], 0) + 1
            emotion_counts[item["primary_emotion"]] = emotion_counts.get(item["primary_emotion"], 0) + 1
        counts["by_pressure"] = pressure_counts
        counts["by_emotion"] = emotion_counts
        return {
            "available": True,
            "count": len(items),
            "filters": {key: value for key, value in filters.items() if value},
            "counts": counts,
            "items": items,
        }

    @r.get("/shadow-registry", response_model=ShadowRegistryResponse)
    async def get_shadow_registry_status() -> ShadowRegistryResponse:
        shadow_registry = getattr(runtime.ctx, "shadow_registry", None)
        if shadow_registry is None:
            return ShadowRegistryResponse(available=False)
        summary = shadow_registry.summary(limit=8, cluster_limit=5)
        return ShadowRegistryResponse(available=True, **summary)

    @r.get("/shadow-registry/cluster", response_model=ShadowRegistryClusterResponse)
    async def get_shadow_registry_cluster(
        fingerprint: str,
        limit: int = 25,
    ) -> ShadowRegistryClusterResponse:
        shadow_registry = getattr(runtime.ctx, "shadow_registry", None)
        if shadow_registry is None:
            return ShadowRegistryClusterResponse(available=False, fingerprint=fingerprint)
        inspect_cluster = getattr(shadow_registry, "inspect_cluster", None)
        if not callable(inspect_cluster):
            return ShadowRegistryClusterResponse(available=False, fingerprint=fingerprint)
        detail = inspect_cluster(fingerprint, limit=max(1, min(limit, 100)))
        return ShadowRegistryClusterResponse(**detail)

    @r.post("/shadow-registry/cluster/triage", response_model=ShadowRegistryClusterResponse)
    async def post_shadow_registry_cluster_triage(
        request: ShadowRegistryClusterTriageRequest,
    ) -> ShadowRegistryClusterResponse:
        shadow_registry = getattr(runtime.ctx, "shadow_registry", None)
        if shadow_registry is None:
            return ShadowRegistryClusterResponse(available=False, fingerprint=request.fingerprint)
        triage_cluster = getattr(shadow_registry, "triage_cluster", None)
        if not callable(triage_cluster):
            return ShadowRegistryClusterResponse(available=False, fingerprint=request.fingerprint)
        detail = triage_cluster(
            request.fingerprint,
            annotation=request.annotation,
            dismissed=request.dismissed,
        )
        return ShadowRegistryClusterResponse(**detail)

    @r.get("/web-trust")
    async def get_web_trust_status(limit: int = 20) -> Dict[str, Any]:
        service = getattr(runtime.ctx, "web_trust", None)
        if service is None:
            return {"available": False, "entries": []}
        return await service.summary(limit=limit)

    @r.get("/plugin-trust")
    async def get_plugin_trust_status(limit: int = 20) -> Dict[str, Any]:
        service = getattr(runtime.ctx, "plugin_trust", None)
        if service is None:
            return {"available": False, "entries": []}
        return await service.summary(limit=limit)

    return r


def _affective_examination_payload(record: Any) -> Dict[str, Any]:
    affect = getattr(record, "affect", None)
    expires_at = getattr(record, "expires_at", None)
    now = datetime.now(timezone.utc)
    if expires_at is None:
        decay_state = "active"
    else:
        decay_state = "expired" if expires_at <= now else "active"
    meta = getattr(record, "meta", {}) if isinstance(getattr(record, "meta", None), dict) else {}
    return {
        "examination_id": str(getattr(record, "examination_id", "")),
        "created_at": getattr(record, "created_at", None).isoformat()
        if getattr(record, "created_at", None)
        else None,
        "session_id": getattr(record, "session_id", None),
        "source_type": _enum_value(getattr(record, "source_type", "")),
        "source_id": getattr(record, "source_id", ""),
        "source_excerpt": getattr(record, "source_excerpt", ""),
        "target": _enum_value(getattr(record, "target", "")),
        "primary_emotion": _enum_value(getattr(affect, "primary_emotion", "")),
        "intensity": getattr(record, "intensity", None),
        "confidence": getattr(record, "confidence", None),
        "action_pressure": _enum_value(getattr(record, "action_pressure", "")),
        "bounded_reason": getattr(record, "bounded_reason", ""),
        "consumed_by": _enum_value(getattr(record, "consumed_by", "")),
        "expires_at": expires_at.isoformat() if expires_at else None,
        "decay_state": decay_state,
        "already_recognized": bool(meta.get("already_recognized")),
        "meta": meta,
    }


def _enum_value(value: Any) -> str:
    return str(getattr(value, "value", value) or "")
