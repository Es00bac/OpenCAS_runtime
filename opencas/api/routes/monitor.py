"""Monitor API routes for the OpenCAS dashboard."""

from __future__ import annotations

import asyncio
from datetime import datetime, timezone
from pathlib import Path
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
REPO_ROOT = Path(__file__).resolve().parents[3]


class HealthResponse(BaseModel):
    overall: str
    failures: int
    warnings: int
    checks: List[Dict[str, Any]]


class BaaStatusResponse(BaseModel):
    queue_size: int
    held_size: int
    active_count: int
    context_guard_held_count: int = 0
    held_reason_counts: Dict[str, int] = Field(default_factory=dict)
    context_guard_reason_counts: Dict[str, int] = Field(default_factory=dict)
    dependency_reason_counts: Dict[str, int] = Field(default_factory=dict)
    approval_reason_counts: Dict[str, int] = Field(default_factory=dict)


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
    health_cache_ttl_seconds = 10.0
    health_cache: HealthResponse | None = None
    health_cache_expires_at = 0.0
    health_lock = asyncio.Lock()

    async def build_health_response() -> HealthResponse:
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

    async def cached_health_response() -> HealthResponse:
        nonlocal health_cache, health_cache_expires_at
        loop = asyncio.get_running_loop()
        now = loop.time()
        if health_cache is not None and now < health_cache_expires_at:
            return health_cache
        async with health_lock:
            now = loop.time()
            if health_cache is not None and now < health_cache_expires_at:
                return health_cache
            response = await build_health_response()
            health_cache = response
            health_cache_expires_at = loop.time() + health_cache_ttl_seconds
            return response

    @r.get("/health", response_model=HealthResponse)
    async def get_health() -> HealthResponse:
        return await cached_health_response()

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
        held_reason_counts: Dict[str, int] = {}
        context_guard_reason_counts: Dict[str, int] = {}
        dependency_reason_counts: Dict[str, int] = {}
        approval_reason_counts: Dict[str, int] = {}
        context_guard_held_count = 0
        for task in list((getattr(baa, "_held", {}) or {}).values()):
            meta = dict(getattr(task, "meta", {}) or {})
            reason = str(meta.get("context_guard_reason") or meta.get("held_reason") or "unspecified")
            held_reason_counts[reason] = held_reason_counts.get(reason, 0) + 1
            if meta.get("context_guard_status") == "held":
                context_guard_held_count += 1
                context_reason = str(meta.get("context_guard_reason") or "unspecified")
                context_guard_reason_counts[context_reason] = context_guard_reason_counts.get(context_reason, 0) + 1
            held_reason = str(meta.get("held_reason") or "")
            if held_reason == "waiting_for_dependencies":
                dependency_reason_counts[held_reason] = dependency_reason_counts.get(held_reason, 0) + 1
            elif held_reason:
                approval_reason_counts[held_reason] = approval_reason_counts.get(held_reason, 0) + 1
        return BaaStatusResponse(
            queue_size=getattr(baa, "queue_size", 0),
            held_size=getattr(baa, "held_size", 0),
            active_count=getattr(baa, "active_count", 0),
            context_guard_held_count=context_guard_held_count,
            held_reason_counts=held_reason_counts,
            context_guard_reason_counts=context_guard_reason_counts,
            dependency_reason_counts=dependency_reason_counts,
            approval_reason_counts=approval_reason_counts,
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
    @r.get("/runtime-status", response_model=RuntimeStatusResponse)
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

    @r.get("/executive-assistant")
    async def get_executive_assistant_dashboard() -> Dict[str, Any]:
        return await asyncio.to_thread(_executive_assistant_dashboard_payload, runtime)

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

    @r.get("/affective-trends")
    async def get_affective_trends(limit: int = 40) -> Dict[str, Any]:
        bounded_limit = max(2, min(int(limit), 200))
        writer = getattr(runtime.ctx, "affective_registry_writer", None) or getattr(
            runtime,
            "affective_registry_writer",
            None,
        )
        entries: List[Any] = []
        loader = getattr(writer, "get_latest", None)
        registry_available = callable(loader)
        registry_error: str | None = None
        if callable(loader):
            try:
                entries = list(loader(bounded_limit))
            except Exception as exc:
                registry_error = str(exc)
                entries = []
        entries.sort(key=lambda item: getattr(item, "timestamp", datetime.min.replace(tzinfo=timezone.utc)))
        series = [_affective_registry_payload(entry) for entry in entries]
        telemetry_affect = _telemetry_affect_payload(runtime, bounded_limit)
        return {
            "available": bool(series) or bool(telemetry_affect.get("available")),
            "registry": {
                "available": registry_available,
                "count": len(series),
                "limit": bounded_limit,
                "error": registry_error,
            },
            "latest": series[-1] if series else None,
            "trend": _affective_trend_summary(series),
            "series": series,
            "telemetry_affect": telemetry_affect,
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


def _executive_assistant_dashboard_payload(runtime: Any) -> Dict[str, Any]:
    try:
        from tools import executive_assistant_dashboard as dashboard
    except Exception as exc:
        return {
            "available": False,
            "error": f"executive assistant dashboard collector unavailable: {exc}",
            "summary": {},
            "items": [],
            "raw": {},
        }

    config = getattr(getattr(runtime, "ctx", None), "config", None)
    repo_root = _dashboard_repo_root(config)
    state_dir = _dashboard_state_dir(config, repo_root)
    now = datetime.now(timezone.utc)
    try:
        raw = dashboard.collect_metrics(repo_root=repo_root, state_dir=state_dir, now=now)
    except Exception as exc:
        return {
            "available": False,
            "error": str(exc),
            "repo_root": str(repo_root),
            "state_dir": str(state_dir),
            "summary": {},
            "items": [],
            "raw": {},
        }
    summary = _executive_assistant_summary(raw)
    return {
        "available": True,
        "generated_at": raw.get("generated_at"),
        "repo_root": str(repo_root),
        "state_dir": str(state_dir),
        "report_path": _latest_executive_assistant_report(repo_root),
        "summary": summary,
        "items": _executive_assistant_items(raw),
        "raw": raw,
    }


def _dashboard_repo_root(config: Any) -> Path:
    workspace_root = getattr(config, "workspace_root", None)
    if workspace_root:
        return Path(workspace_root)
    workspace_roots = getattr(config, "workspace_roots", None)
    if isinstance(workspace_roots, list) and workspace_roots:
        return Path(workspace_roots[0])
    return REPO_ROOT


def _dashboard_state_dir(config: Any, repo_root: Path) -> Path:
    state_dir = getattr(config, "state_dir", None)
    if state_dir:
        path = Path(state_dir)
        return path if path.is_absolute() else repo_root / path
    return repo_root / ".opencas"


def _latest_executive_assistant_report(repo_root: Path) -> str | None:
    qualification_dir = repo_root / "dev-notes" / "qualification"
    reports = sorted(qualification_dir.glob("executive-assistant-dashboard-*.md"))
    return str(reports[-1]) if reports else None


def _executive_assistant_summary(raw: Dict[str, Any]) -> Dict[str, Any]:
    memory = raw.get("memory_binding") or {}
    proactive = raw.get("proactive") or {}
    loops = raw.get("loop_closures") or {}
    duplicate = memory.get("duplicate_tool_calls") or {}
    channel_audit = proactive.get("channel_audit") or {}
    daydream = proactive.get("daydream") or {}
    observations = proactive.get("observations") or {}
    proof = (loops.get("proof_chain") or {})
    gaps = raw.get("substrate_gaps") or {}
    pa = gaps.get("phenomenological_audit") or {}
    return {
        "duplicate_tool_call_rate": _safe_number(duplicate.get("current_week_rate")),
        "duplicate_tool_call_events": _safe_int(duplicate.get("current_week_duplicates")),
        "proactive_observations_7d": _safe_int(observations.get("last_7_days_count")),
        "audited_turns_7d": _safe_int(channel_audit.get("audited_turns_7d")),
        "channels_with_daily_novel_signal": _safe_int(channel_audit.get("channels_with_daily_novel_signal")),
        "daydream_consecutive_skip_count": _safe_int(daydream.get("consecutive_skip_count")),
        "promise_claim_count": _safe_int(proof.get("promise_claim_count")),
        "promise_receipt_evidence_count": _safe_int(proof.get("promise_claims_with_receipt_evidence")),
        "continuous_present_score": str(gaps.get("continuous_present_score") or "pending"),
        "phenomenological_pa_score": _safe_number(pa.get("pa_score")),
        "duplicate_tool_call_total": _safe_int(duplicate.get("current_week_total")),
    }


def _executive_assistant_items(raw: Dict[str, Any]) -> List[Dict[str, Any]]:
    memory = raw.get("memory_binding") or {}
    proactive = raw.get("proactive") or {}
    loops = raw.get("loop_closures") or {}
    gaps = raw.get("substrate_gaps") or {}
    pa = gaps.get("phenomenological_audit") or {}
    pathology = raw.get("pathology") or {}
    duplicate = memory.get("duplicate_tool_calls") or {}
    handles = memory.get("compaction_continuation_handles") or {}
    recall = memory.get("exact_handle_recall") or {}
    observations = proactive.get("observations") or {}
    flags = proactive.get("usefulness_flags") or {}
    daydream = proactive.get("daydream") or {}
    channel_audit = proactive.get("channel_audit") or {}
    shadow = loops.get("shadow_registry") or {}
    wellbeing = loops.get("wellbeing") or {}
    proof = loops.get("proof_chain") or {}
    return [
        {
            "title": "Memory and Binding",
            "status": _pending_or_status(
                handles.get("current_status") or handles.get("preservation_rate"),
                recall.get("recall_at_5"),
                memory.get("manuscript_authorship_cold_replay"),
            ),
            "duplicate_tool_call_rate": _safe_number(duplicate.get("current_week_rate")),
            "duplicate_tool_call_events": _safe_int(duplicate.get("current_week_duplicates")),
            "continuation_handle_preservation": str(
                handles.get("current_status") or handles.get("preservation_rate") or "unavailable"
            ),
            "continuation_packet_rows": _safe_int(handles.get("current_packet_count")),
            "retained_continuation_handle_preservation": str(
                handles.get("retained_preservation_rate") or handles.get("preservation_rate") or "unavailable"
            ),
            "exact_handle_recall_at_5": str(recall.get("recall_at_5") or "unavailable"),
            "summary": str(handles.get("status") or ""),
        },
        {
            "title": "Proactive Surfacing",
            "status": "pass" if _safe_int(channel_audit.get("audited_turns_7d")) > 0 else "pending",
            "observations_today": _safe_int(observations.get("today_count")),
            "observations_7d": _safe_int(observations.get("last_7_days_count")),
            "operator_useful_rate_7d": _safe_number(flags.get("rolling_7_day_useful_rate")),
            "audited_turns_7d": _safe_int(channel_audit.get("audited_turns_7d")),
            "channels_with_daily_novel_signal": _safe_int(channel_audit.get("channels_with_daily_novel_signal")),
            "daydream_consecutive_skip_count": _safe_int(daydream.get("consecutive_skip_count")),
            "summary": str(channel_audit.get("rendered_novel_summary") or "no channel audit data"),
        },
        {
            "title": "Loop Closures",
            "status": _loop_closure_status(shadow, wellbeing, proof),
            "shadow_active_clusters": _safe_int(shadow.get("active_clusters")),
            "shadow_dismissed_clusters": _safe_int(shadow.get("dismissed_clusters")),
            "addressed_drift_observations": _safe_int(wellbeing.get("addressed_drift_observation_count")),
            "promise_claim_count": _safe_int(proof.get("promise_claim_count")),
            "promise_receipt_evidence_count": _safe_int(proof.get("promise_claims_with_receipt_evidence")),
            "summary": f"wellbeing risk {wellbeing.get('overall_risk')}; coherence {wellbeing.get('coherence')}",
        },
        {
            "title": "Substrate Gaps",
            "status": _gap_status(gaps),
            "continuous_present_score": str(gaps.get("continuous_present_score") or "pending"),
            "phenomenological_audit_dimensions": str(gaps.get("phenomenological_audit_dimensions") or "pending"),
            "phenomenological_pa_score": _safe_number(pa.get("pa_score")),
            "phenomenological_audit_path": str(pa.get("path") or ""),
            "phenomenological_regression_count": len(pa.get("regressions") or []),
            "summary": _gap_summary(gaps),
        },
        {
            "title": "Pathology Recurrence",
            "status": _pathology_status(pathology),
            "recursive_parked_goal_recurrence": _safe_int(pathology.get("recursive_parked_goal_recurrence")),
            "recursive_parked_goal_archived_evidence": _safe_int(
                pathology.get("recursive_parked_goal_archived_evidence")
            ),
            "canned_phrase_regression": str(pathology.get("canned_phrase_regression") or "pending"),
            "summary": "Active recurrence is a bug; archived evidence is retained for audit.",
        },
    ]


def _loop_closure_status(
    shadow: Dict[str, Any],
    wellbeing: Dict[str, Any],
    proof: Dict[str, Any],
) -> str:
    active_clusters = _safe_int(shadow.get("active_clusters"))
    dismissed_clusters = _safe_int(shadow.get("dismissed_clusters"))
    addressed_drift = _safe_int(wellbeing.get("addressed_drift_observation_count"))
    promise_claims = _safe_int(proof.get("promise_claim_count"))
    promise_receipts = _safe_int(proof.get("promise_claims_with_receipt_evidence"))
    if (active_clusters > 0 and dismissed_clusters == 0) or addressed_drift == 0:
        return "warn"
    if promise_claims == 0 or promise_receipts == 0:
        return "pending"
    return "pass"


def _pathology_status(pathology: Dict[str, Any]) -> str:
    if _safe_int(pathology.get("recursive_parked_goal_recurrence")) > 0:
        return "warn"
    return _pending_or_status(pathology.get("canned_phrase_regression"))


def _pending_or_status(*values: Any) -> str:
    text = " ".join(str(value or "").lower() for value in values)
    if "fail" in text:
        return "warn"
    if any(marker in text for marker in ("pending", "unavailable", "not yet run")):
        return "pending"
    return "pass"


def _gap_status(gaps: Dict[str, Any]) -> str:
    text = " ".join(str(value or "").lower() for value in gaps.values())
    pa = gaps.get("phenomenological_audit") if isinstance(gaps.get("phenomenological_audit"), dict) else {}
    current_passes = (
        "1 minute pass" in text
        and _safe_number(pa.get("pa_score")) is not None
        and float(_safe_number(pa.get("pa_score")) or 0.0) >= 0.9
        and len(pa.get("regressions") or []) == 0
    )
    if "fail" in text:
        return "warn"
    if current_passes and "pending" in text:
        return "monitoring"
    if "pending" in text:
        return "pending"
    return "pass"


def _gap_summary(gaps: Dict[str, Any]) -> str:
    status = _gap_status(gaps)
    if status == "monitoring":
        return "Current short-window and PA checks pass; longer continuity windows remain under observation."
    if status == "pending":
        return "Pending values are measurement gaps, not successful closures."
    return "No current substrate gap failure is reported."


def _safe_int(value: Any) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        return 0


def _safe_number(value: Any) -> float | None:
    if value is None:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


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


def _affective_registry_payload(entry: Any) -> Dict[str, Any]:
    affect = getattr(entry, "affective_state", None)
    context = getattr(entry, "execution_context", None)
    metrics = getattr(entry, "system_metrics", None)
    dimensions = {
        "valence": _float_attr(affect, "valence"),
        "arousal": _float_attr(affect, "arousal"),
        "fatigue": _float_attr(affect, "fatigue"),
        "tension": _float_attr(affect, "tension"),
        "focus": _float_attr(affect, "focus"),
        "energy": _float_attr(affect, "energy"),
        "certainty": _float_attr(affect, "certainty"),
        "musubi": _optional_float_attr(affect, "musubi"),
    }
    phase = getattr(entry, "phase", "")
    emotion = getattr(affect, "primary_emotion", "neutral")
    return {
        "entry_id": str(getattr(entry, "entry_id", "")),
        "timestamp": _iso(getattr(entry, "timestamp", None)),
        "phase": _enum_value(phase),
        "session_id": str(getattr(context, "session_id", "") or "runtime"),
        "primary_emotion": _enum_value(emotion),
        "somatic_tag": getattr(affect, "somatic_tag", None),
        "dimensions": dimensions,
        "risk": _affective_risk(dimensions),
        "system": {
            "cpu_percent": _optional_float_attr(metrics, "cpu_percent"),
            "memory_percent": _optional_float_attr(metrics, "memory_percent"),
            "memory_rss_mb": _optional_float_attr(metrics, "memory_rss_mb"),
            "thread_count": getattr(metrics, "thread_count", None),
        },
    }


def _affective_trend_summary(series: List[Dict[str, Any]]) -> Dict[str, Any]:
    if not series:
        return {
            "direction": "no_data",
            "risk_delta": 0.0,
            "sample_count": 0,
            "averages": {},
        }
    first = series[0]
    latest = series[-1]
    risk_delta = round(float(latest.get("risk", 0.0)) - float(first.get("risk", 0.0)), 4)
    if risk_delta >= 0.1:
        direction = "rising_risk"
    elif risk_delta <= -0.1:
        direction = "easing_risk"
    else:
        direction = "stable"
    return {
        "direction": direction,
        "risk_delta": risk_delta,
        "sample_count": len(series),
        "from": first.get("timestamp"),
        "to": latest.get("timestamp"),
        "averages": _dimension_averages(series),
    }


def _dimension_averages(series: List[Dict[str, Any]]) -> Dict[str, float]:
    values: Dict[str, List[float]] = {}
    for item in series:
        for key, value in dict(item.get("dimensions") or {}).items():
            if isinstance(value, (int, float)):
                values.setdefault(key, []).append(float(value))
    return {key: round(sum(items) / len(items), 4) for key, items in values.items() if items}


def _telemetry_affect_payload(runtime: Any, limit: int) -> Dict[str, Any]:
    store = getattr(runtime, "telemetry_affect_store", None)
    if store is None:
        config = getattr(getattr(runtime, "ctx", None), "config", None)
        state_dir = getattr(config, "state_dir", None)
        affect_path = Path(state_dir) / "telemetry_affect" if state_dir else None
        if affect_path is not None and affect_path.exists():
            try:
                from opencas.telemetry.affect_store import AffectStore

                store = AffectStore(affect_path)
            except Exception:
                store = None
    if store is None:
        return {
            "available": False,
            "snapshots": 0,
            "trajectories": 0,
            "quality_signals": 0,
            "alerts": 0,
        }
    try:
        snapshots = sorted(
            list(store.iter_snapshots()),
            key=lambda item: getattr(item, "timestamp", datetime.min.replace(tzinfo=timezone.utc)),
        )[-limit:]
        trajectories = list(store.iter_trajectories())[-limit:]
        quality = list(store.iter_quality_signals())[-limit:]
        alerts = list(store.iter_alerts())[-limit:]
    except Exception as exc:
        return {
            "available": False,
            "snapshots": 0,
            "trajectories": 0,
            "quality_signals": 0,
            "alerts": 0,
            "error": str(exc),
        }
    return {
        "available": True,
        "snapshots": len(snapshots),
        "trajectories": len(trajectories),
        "quality_signals": len(quality),
        "alerts": len(alerts),
        "latest_snapshot": snapshots[-1].to_telemetry_payload() if snapshots else None,
    }


def _affective_risk(dimensions: Dict[str, Any]) -> float:
    tension = float(dimensions.get("tension") or 0.0)
    fatigue = float(dimensions.get("fatigue") or 0.0)
    certainty = float(dimensions.get("certainty") if dimensions.get("certainty") is not None else 0.5)
    valence = float(dimensions.get("valence") or 0.0)
    risk = (
        (tension * 0.4)
        + (fatigue * 0.25)
        + ((1.0 - certainty) * 0.25)
        + (max(0.0, -valence) * 0.1)
    )
    return round(max(0.0, min(1.0, risk)), 4)


def _float_attr(value: Any, name: str) -> float:
    raw = getattr(value, name, 0.0)
    if isinstance(raw, (int, float)):
        return float(raw)
    return 0.0


def _optional_float_attr(value: Any, name: str) -> float | None:
    raw = getattr(value, name, None)
    if isinstance(raw, (int, float)):
        return float(raw)
    return None


def _iso(value: Any) -> str | None:
    if hasattr(value, "isoformat"):
        return value.isoformat()
    return None


def _enum_value(value: Any) -> str:
    return str(getattr(value, "value", value) or "")
