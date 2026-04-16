"""Operations API routes for session inspection, receipts, work management, and qualification state."""

from __future__ import annotations

import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

from fastapi import APIRouter
from opencas.api.operations_monitoring import (
    build_approval_audit_snapshot,
    build_cost_snapshot,
    build_hardening_snapshot,
    build_memory_value_snapshot,
)
from opencas.api.operations_activity import ActivityOperationsService
from opencas.api.operations_browser import BrowserSessionService, register_browser_routes
from opencas.api.operations_qualification import (
    QualificationOperationsService,
    QualificationLabelDetailResponse,
    QualificationRerunDetailResponse,
    QualificationRerunRequest,
    QualificationSummaryResponse,
    ValidationRunDetailResponse,
    ValidationRunListResponse,
    find_rerun_history_by_request_id,
)
from opencas.api.operations_sessions import SessionOperationsService
from opencas.api.operations_tasking import TaskingOperationsService
from opencas.api.operations_models import (
    CommitmentEntry,
    CommitmentListResponse,
    CommitmentUpdateRequest,
    PlanListResponse,
    PlanSummary,
    PlanUpdateRequest,
    ProcessDetailResponse,
    PtyInputRequest,
    ReceiptEntry,
    ReceiptListResponse,
    SessionEntry,
    SessionListResponse,
    SessionScopeEntry,
    TaskEntry,
    TaskListResponse,
    WorkItemEntry,
    WorkListResponse,
    WorkUpdateRequest,
)
from opencas.api.operator_actions import (
    append_operator_action,
    load_recent_operator_actions,
    resolve_operator_actions_path,
    truncate_operator_text,
)
from opencas.api.qualification_models import QualificationArtifactsPaths

REPO_ROOT = Path(__file__).resolve().parents[3]
QUALIFICATION_SUMMARY_PATH = REPO_ROOT / "docs" / "qualification" / "live_validation_summary.json"
QUALIFICATION_REMEDIATION_PATH = REPO_ROOT / "docs" / "qualification" / "qualification_remediation_rollup.json"
VALIDATION_RUNS_DIR = REPO_ROOT / ".opencas_live_test_state"
QUALIFICATION_RERUN_HISTORY_PATH = VALIDATION_RUNS_DIR / "qualification_rerun_history.jsonl"
DEFAULT_OPERATOR_ACTIONS_PATH = VALIDATION_RUNS_DIR / "operator_action_history.jsonl"


def _qualification_paths() -> QualificationArtifactsPaths:
    return QualificationArtifactsPaths(
        repo_root=REPO_ROOT,
        summary_path=QUALIFICATION_SUMMARY_PATH,
        remediation_path=QUALIFICATION_REMEDIATION_PATH,
        validation_runs_dir=VALIDATION_RUNS_DIR,
        rerun_history_path=QUALIFICATION_RERUN_HISTORY_PATH,
    )


def _human_title(text: Optional[str], fallback: str = "Untitled") -> str:
    raw = str(text or "").strip()
    if not raw:
        return fallback
    first_line = raw.splitlines()[0].strip()
    compact = " ".join(first_line.split())
    if len(compact) <= 88:
        return compact
    return compact[:85].rstrip() + "..."


def _task_ui_status(stage: str, status: str) -> str:
    stage_key = str(stage or "").strip().lower()
    status_key = str(status or "").strip().lower()
    if stage_key in {"done"} or status_key in {"completed", "success"}:
        return "completed"
    if stage_key in {"failed"} or status_key in {"failed", "error"}:
        return "failed"
    if stage_key == "needs_approval":
        return "needs approval"
    if stage_key == "needs_clarification":
        return "needs clarification"
    if stage_key in {"queued", "planning", "executing", "verifying", "recovering"}:
        return stage_key
    if status_key:
        return status_key.replace("_", " ")
    return "unknown"


async def _build_memory_value_snapshot(runtime: Any) -> Dict[str, Any]:
    return await build_memory_value_snapshot(runtime)


async def _build_approval_audit_snapshot(runtime: Any, *, window_days: int, limit: int) -> Dict[str, Any]:
    return await build_approval_audit_snapshot(runtime, window_days=window_days, limit=limit)


async def _build_cost_snapshot(runtime: Any, *, window_days: int, bucket_hours: int) -> Dict[str, Any]:
    return await build_cost_snapshot(runtime, window_days=window_days, bucket_hours=bucket_hours)


async def _build_hardening_snapshot(runtime: Any, *, window_days: int, bucket_hours: int, decision_limit: int) -> Dict[str, Any]:
    return await build_hardening_snapshot(runtime, window_days=window_days, bucket_hours=bucket_hours, decision_limit=decision_limit)


def _find_rerun_history_by_request_id(request_id: Optional[str]) -> tuple[Optional[Dict[str, Any]], Optional[Dict[str, Any]]]:
    return find_rerun_history_by_request_id(_qualification_paths, request_id)


def _resolve_operator_actions_path(runtime: Any) -> Path:
    return resolve_operator_actions_path(runtime, DEFAULT_OPERATOR_ACTIONS_PATH)


def _truncate_text(value: Optional[str], limit: int = 160) -> str:
    return truncate_operator_text(value, limit=limit)


def _coerce_mapping_payload(value: Any) -> Dict[str, Any]:
    if isinstance(value, dict):
        return dict(value)
    if hasattr(value, "model_dump"):
        try:
            payload = value.model_dump(mode="json")
        except TypeError:
            payload = value.model_dump()
        if isinstance(payload, dict):
            return dict(payload)
    return {}


def _append_operator_action(runtime: Any, entry: Dict[str, Any]) -> Dict[str, Any]:
    return append_operator_action(runtime, entry, default_path=DEFAULT_OPERATOR_ACTIONS_PATH)


def _load_recent_operator_actions(
    runtime: Any,
    *,
    target_kind: str,
    target_id: str,
    scope_key: Optional[str] = None,
    limit: int = 10,
) -> List[Dict[str, Any]]:
    return load_recent_operator_actions(
        runtime,
        target_kind=target_kind,
        target_id=target_id,
        default_path=DEFAULT_OPERATOR_ACTIONS_PATH,
        scope_key=scope_key,
        limit=limit,
    )


def _browser_sessions(runtime: Any) -> BrowserSessionService:
    return BrowserSessionService(
        runtime,
        append_operator_action=_append_operator_action,
        load_recent_operator_actions=_load_recent_operator_actions,
    )


def _session_operations(runtime: Any) -> SessionOperationsService:
    return SessionOperationsService(
        runtime,
        find_rerun_history_by_request_id=_find_rerun_history_by_request_id,
        append_operator_action=_append_operator_action,
        load_recent_operator_actions=_load_recent_operator_actions,
        truncate_text=_truncate_text,
    )


def _tasking_operations(runtime: Any) -> TaskingOperationsService:
    return TaskingOperationsService(runtime, coerce_mapping_payload=_coerce_mapping_payload)


def _activity_operations(runtime: Any) -> ActivityOperationsService:
    return ActivityOperationsService(
        runtime,
        human_title=_human_title,
        task_ui_status=_task_ui_status,
    )


def _qualification_operations(runtime: Any) -> QualificationOperationsService:
    return QualificationOperationsService(
        runtime,
        paths_provider=_qualification_paths,
        repo_root=REPO_ROOT,
    )


def build_operations_router(runtime: Any) -> APIRouter:
    """Build operations routes for session inspection and work management."""
    r = APIRouter(prefix="/api/operations", tags=["operations"])
    browser_sessions = _browser_sessions(runtime)
    session_operations = _session_operations(runtime)
    tasking_operations = _tasking_operations(runtime)
    activity_operations = _activity_operations(runtime)
    qualification_operations = _qualification_operations(runtime)

    # ── Session Inspection ────────────────────────────────────────────

    @r.get("/qualification", response_model=QualificationSummaryResponse)
    async def get_qualification_summary() -> QualificationSummaryResponse:
        return qualification_operations.get_summary()

    @r.get("/qualification/labels/{label}", response_model=QualificationLabelDetailResponse)
    async def get_qualification_label(label: str) -> QualificationLabelDetailResponse:
        return qualification_operations.get_label_detail(label)

    @r.get("/qualification/reruns/{request_id}", response_model=QualificationRerunDetailResponse)
    async def get_qualification_rerun(request_id: str) -> QualificationRerunDetailResponse:
        return qualification_operations.get_rerun_detail(request_id)

    @r.get("/validation-runs", response_model=ValidationRunListResponse)
    async def list_validation_runs(limit: int = 10, label: Optional[str] = None) -> ValidationRunListResponse:
        return qualification_operations.list_validation_runs(limit=limit, label=label)

    @r.get("/validation-runs/{run_id}", response_model=ValidationRunDetailResponse)
    async def get_validation_run(run_id: str, label: Optional[str] = None) -> ValidationRunDetailResponse:
        return qualification_operations.get_validation_run(run_id, label=label)

    @r.post("/qualification/reruns")
    async def start_qualification_rerun(payload: QualificationRerunRequest) -> Dict[str, Any]:
        return qualification_operations.start_rerun(payload)

    @r.get("/hardening")
    async def get_hardening_summary(
        window_days: int = 7,
        bucket_hours: int = 6,
        decision_limit: int = 12,
    ) -> Dict[str, Any]:
        return await _build_hardening_snapshot(
            runtime,
            window_days=max(1, min(window_days, 30)),
            bucket_hours=max(1, min(bucket_hours, 24)),
            decision_limit=max(1, min(decision_limit, 50)),
        )

    @r.get("/memory-value")
    async def get_memory_value() -> Dict[str, Any]:
        return await _build_memory_value_snapshot(runtime)

    @r.get("/approval-audit")
    async def get_approval_audit(window_days: int = 7, limit: int = 12) -> Dict[str, Any]:
        return await _build_approval_audit_snapshot(
            runtime,
            window_days=max(1, min(window_days, 30)),
            limit=max(1, min(limit, 50)),
        )

    @r.get("/costs")
    async def get_costs(window_days: int = 7, bucket_hours: int = 6) -> Dict[str, Any]:
        return await _build_cost_snapshot(
            runtime,
            window_days=max(1, min(window_days, 30)),
            bucket_hours=max(1, min(bucket_hours, 24)),
        )

    @r.get("/sessions", response_model=SessionListResponse)
    async def list_sessions(scope_key: Optional[str] = None) -> SessionListResponse:
        return session_operations.list_sessions(scope_key=scope_key)

    @r.get("/sessions/process/{process_id}", response_model=ProcessDetailResponse)
    async def get_process_session(
        process_id: str,
        scope_key: str = "default",
        refresh: bool = True,
    ) -> ProcessDetailResponse:
        return session_operations.get_process_session(
            process_id=process_id,
            scope_key=scope_key,
            refresh=refresh,
        )

    @r.delete("/sessions/process/{process_id}")
    async def kill_process_session(
        process_id: str,
        scope_key: str = "default",
    ) -> Dict[str, Any]:
        return session_operations.kill_process_session(process_id=process_id, scope_key=scope_key)

    @r.delete("/sessions/process")
    async def clear_process_sessions(scope_key: str = "default") -> Dict[str, Any]:
        return session_operations.clear_process_sessions(scope_key=scope_key)

    @r.delete("/sessions/pty/{session_id}")
    async def kill_pty_session(
        session_id: str, scope_key: str = "default"
    ) -> Dict[str, Any]:
        return session_operations.kill_pty_session(session_id=session_id, scope_key=scope_key)

    @r.delete("/sessions/pty")
    async def clear_pty_sessions(scope_key: str = "default") -> Dict[str, Any]:
        return session_operations.clear_pty_sessions(scope_key=scope_key)

    @r.get("/sessions/pty/{session_id}")
    async def get_pty_session(
        session_id: str,
        scope_key: str = "default",
        refresh: bool = False,
        idle_seconds: float = 0.25,
        max_wait_seconds: float = 1.5,
    ) -> Dict[str, Any]:
        return session_operations.get_pty_session(
            session_id=session_id,
            scope_key=scope_key,
            refresh=refresh,
            idle_seconds=idle_seconds,
            max_wait_seconds=max_wait_seconds,
        )

    @r.post("/sessions/pty/{session_id}/input")
    async def send_pty_input(
        session_id: str,
        payload: PtyInputRequest,
        scope_key: str = "default",
    ) -> Dict[str, Any]:
        return session_operations.send_pty_input(
            session_id=session_id,
            payload=payload,
            scope_key=scope_key,
        )

    register_browser_routes(
        r,
        runtime=runtime,
        browser_sessions=browser_sessions,
        append_operator_action=_append_operator_action,
        truncate_text=_truncate_text,
    )

    # ── Receipts ──────────────────────────────────────────────────────

    @r.get("/receipts", response_model=ReceiptListResponse)
    async def list_receipts(limit: int = 50) -> ReceiptListResponse:
        return await activity_operations.list_receipts(limit=limit)

    @r.get("/receipts/{receipt_id}")
    async def get_receipt(receipt_id: str) -> Dict[str, Any]:
        return await activity_operations.get_receipt(receipt_id)

    # ── Background Tasks ──────────────────────────────────────────────

    @r.get("/tasks", response_model=TaskListResponse)
    async def list_tasks(limit: int = 50) -> TaskListResponse:
        return await activity_operations.list_tasks(limit=limit)

    @r.get("/tasks/{task_id}")
    async def get_task(task_id: str) -> Dict[str, Any]:
        return await activity_operations.get_task(task_id)

    # ── Work Items ────────────────────────────────────────────────────

    @r.get("/work", response_model=WorkListResponse)
    async def list_work(
        project_id: Optional[str] = None, limit: int = 50
    ) -> WorkListResponse:
        return await tasking_operations.list_work(project_id=project_id, limit=limit)

    @r.get("/work/{work_id}")
    async def get_work_item(work_id: str) -> Dict[str, Any]:
        return await tasking_operations.get_work_item(work_id)

    @r.patch("/work/{work_id}")
    async def update_work_item(work_id: str, payload: WorkUpdateRequest) -> Dict[str, Any]:
        return await tasking_operations.update_work_item(work_id, payload)

    # ── Commitments ───────────────────────────────────────────────────

    @r.get("/commitments", response_model=CommitmentListResponse)
    async def list_commitments(
        status: str = "active", limit: int = 50
    ) -> CommitmentListResponse:
        return await tasking_operations.list_commitments(status=status, limit=limit)

    @r.get("/commitments/{commitment_id}")
    async def get_commitment(commitment_id: str) -> Dict[str, Any]:
        return await tasking_operations.get_commitment(commitment_id)

    @r.patch("/commitments/{commitment_id}")
    async def update_commitment(
        commitment_id: str,
        payload: CommitmentUpdateRequest,
    ) -> Dict[str, Any]:
        return await tasking_operations.update_commitment(commitment_id, payload)

    # ── Plans ─────────────────────────────────────────────────────────

    @r.get("/plans", response_model=PlanListResponse)
    async def list_plans(
        project_id: Optional[str] = None, limit: int = 20
    ) -> PlanListResponse:
        return await tasking_operations.list_plans(project_id=project_id, limit=limit)

    @r.get("/plans/{plan_id}")
    async def get_plan(plan_id: str) -> Dict[str, Any]:
        return await tasking_operations.get_plan(plan_id)

    @r.patch("/plans/{plan_id}")
    async def update_plan(plan_id: str, payload: PlanUpdateRequest) -> Dict[str, Any]:
        return await tasking_operations.update_plan(plan_id, payload)

    return r
