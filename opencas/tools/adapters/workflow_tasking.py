"""Tasking-oriented workflow helpers for commitments, plans, and schedules."""

from __future__ import annotations

import json
import logging
import re
from datetime import datetime, timedelta, timezone
from inspect import isawaitable
from pathlib import Path
from typing import Any, Dict
from uuid import uuid4

from ...api.provenance_store import (
    ProvenanceTransitionKind,
    record_provenance_transition,
)
from ...autonomy.commitment import Commitment, CommitmentStatus, commitment_operator_snapshot
from ...autonomy.commitment_work import settle_linked_work_for_terminal_commitment
from ...autonomy.completion_evidence import completion_evidence_rejection_reason
from ...projects.classifier import (
    PROJECT_TYPE_GENERAL,
    PROJECT_TYPE_WRITING,
    any_marker_in_text,
    classify_project_type,
)
from ...projects.lifecycle import cancel_project as lifecycle_cancel_project
from ...projects.lifecycle import cancel_task as lifecycle_cancel_task
from ...projects.operator_followthrough import (
    copy_safe_operator_project_followthrough_evidence,
    has_operator_project_followthrough_authority,
)
from ...proof_chain import attach_operator_promise_claim
from ...scheduling import ScheduleStatus
from ..models import ToolResult
from .workflow_paths import managed_workspace_root, resolve_managed_output_path

_ACTIVE_RETURN_MAX_DELAY = timedelta(days=14)
_ABSOLUTE_PATH_RE = re.compile(r"/[^\s`'\"<>]+")
_ACTIVE_RETURN_MARKERS = (
    "continue",
    "finish",
    "persist",
    "resume",
    "return",
    "unfinished",
)

logger = logging.getLogger(__name__)


def _record_workflow_provenance(
    runtime: Any,
    *,
    kind: ProvenanceTransitionKind,
    source_artifact: str,
    trigger_action: str,
    parent_transition_id: str | None = None,
    target_entity: str,
    origin_action_id: str,
    status: str,
    details: Dict[str, Any] | None = None,
) -> None:
    config = getattr(getattr(runtime, "ctx", None), "config", None)
    state_dir = getattr(config, "state_dir", None)
    if state_dir is None:
        return
    raw_session_id = getattr(config, "session_id", None) if config is not None else None
    session_id = raw_session_id.strip() if isinstance(raw_session_id, str) and raw_session_id.strip() else target_entity
    record_provenance_transition(
        state_dir=state_dir,
        kind=kind,
        session_id=session_id,
        entity_id=target_entity,
        status=status,
        trigger_artifact=source_artifact,
        source_artifact=source_artifact,
        trigger_action=trigger_action,
        parent_transition_id=parent_transition_id,
        target_entity=target_entity,
        origin_action_id=origin_action_id,
        details=details,
    )


def _schedule_text(args: Dict[str, Any]) -> str:
    values: list[str] = []
    for key in ("title", "description", "objective"):
        value = args.get(key)
        if value:
            values.append(str(value))
    tags = args.get("tags")
    if isinstance(tags, list):
        values.extend(str(tag) for tag in tags)
    return " ".join(values).lower()


def _delay_reason(args: Dict[str, Any]) -> str:
    direct = str(args.get("delay_reason", "") or "").strip()
    if direct:
        return direct
    meta = args.get("meta")
    if isinstance(meta, dict):
        return str(meta.get("delay_reason", "") or "").strip()
    return ""


def _looks_like_active_writing_return(args: Dict[str, Any]) -> bool:
    text = _schedule_text(args)
    return any_marker_in_text(text, _ACTIVE_RETURN_MARKERS) and (
        classify_project_type(current_turn_text=text).project_type == PROJECT_TYPE_WRITING
    )


def _is_active_writing_return(args: Dict[str, Any], action: Any) -> bool:
    if str(getattr(action, "value", action)) != "submit_baa":
        return False
    return _looks_like_active_writing_return(args)


def _find_unmanaged_writing_path(runtime: Any, args: Dict[str, Any]) -> str | None:
    text = " ".join(
        str(args.get(key) or "")
        for key in ("title", "description", "objective")
    )
    if classify_project_type(current_turn_text=text).project_type != PROJECT_TYPE_WRITING:
        return None
    config = getattr(getattr(runtime, "ctx", None), "config", None)
    primary_fn = getattr(config, "primary_workspace_root", None)
    if not callable(primary_fn):
        return None
    primary_root = Path(primary_fn()).expanduser().resolve()
    workspace_root = managed_workspace_root(runtime).expanduser().resolve()
    for match in _ABSOLUTE_PATH_RE.finditer(text):
        raw_path = match.group(0).rstrip(".,;:)]}")
        try:
            candidate = Path(raw_path).expanduser().resolve()
        except (OSError, RuntimeError):
            continue
        if candidate.is_relative_to(primary_root) and not candidate.is_relative_to(workspace_root):
            return raw_path
    return None


def _append_unique_tags(tags: list[Any], additions: list[str]) -> list[Any]:
    seen = {str(tag).strip().lower() for tag in tags if str(tag).strip()}
    merged = list(tags)
    for tag in additions:
        normalized = tag.strip().lower()
        if normalized and normalized not in seen:
            merged.append(tag)
            seen.add(normalized)
    return merged


def _classify_workflow_args(args: Dict[str, Any], *, keys: tuple[str, ...]) -> str:
    text = " ".join(str(args.get(key) or "") for key in keys)
    return classify_project_type(current_turn_text=text).project_type


async def _get_commitment(store: Any, commitment_id: str) -> Any | None:
    getter = getattr(store, "get", None)
    if not callable(getter):
        return None
    result = getter(commitment_id)
    if isawaitable(result):
        result = await result
    return result


def _dt_iso(value: Any) -> str | None:
    return value.isoformat() if hasattr(value, "isoformat") else None


def _enum_value(value: Any) -> Any:
    return getattr(value, "value", value)


def _schedule_payload(item: Any) -> dict[str, Any]:
    return {
        "schedule_id": str(getattr(item, "schedule_id", "")),
        "created_at": _dt_iso(getattr(item, "created_at", None)),
        "updated_at": _dt_iso(getattr(item, "updated_at", None)),
        "kind": _enum_value(getattr(item, "kind", None)),
        "action": _enum_value(getattr(item, "action", None)),
        "status": _enum_value(getattr(item, "status", None)),
        "title": getattr(item, "title", ""),
        "description": getattr(item, "description", ""),
        "objective": getattr(item, "objective", None),
        "start_at": _dt_iso(getattr(item, "start_at", None)),
        "end_at": _dt_iso(getattr(item, "end_at", None)),
        "timezone": getattr(item, "timezone", None),
        "next_run_at": _dt_iso(getattr(item, "next_run_at", None)),
        "last_run_at": _dt_iso(getattr(item, "last_run_at", None)),
        "recurrence": _enum_value(getattr(item, "recurrence", None)),
        "interval_hours": getattr(item, "interval_hours", None),
        "weekdays": list(getattr(item, "weekdays", []) or []),
        "max_occurrences": getattr(item, "max_occurrences", None),
        "occurrence_count": getattr(item, "occurrence_count", 0),
        "priority": getattr(item, "priority", None),
        "tags": list(getattr(item, "tags", []) or []),
        "commitment_id": getattr(item, "commitment_id", None),
        "plan_id": getattr(item, "plan_id", None),
        "meta": dict(getattr(item, "meta", {}) or {}),
    }


def _schedule_run_payload(run: Any) -> dict[str, Any]:
    return {
        "run_id": str(getattr(run, "run_id", "")),
        "schedule_id": str(getattr(run, "schedule_id", "")),
        "scheduled_for": _dt_iso(getattr(run, "scheduled_for", None)),
        "started_at": _dt_iso(getattr(run, "started_at", None)),
        "finished_at": _dt_iso(getattr(run, "finished_at", None)),
        "status": _enum_value(getattr(run, "status", None)),
        "task_id": getattr(run, "task_id", None),
        "error": getattr(run, "error", None),
        "meta": dict(getattr(run, "meta", {}) or {}),
    }


def _plan_payload(plan: Any, *, include_content: bool = True) -> dict[str, Any]:
    payload = {
        "plan_id": getattr(plan, "plan_id", ""),
        "status": getattr(plan, "status", ""),
        "content_preview": str(getattr(plan, "content", "") or "")[:200],
        "created_at": _dt_iso(getattr(plan, "created_at", None)),
        "updated_at": _dt_iso(getattr(plan, "updated_at", None)),
        "project_id": getattr(plan, "project_id", None),
        "task_id": getattr(plan, "task_id", None),
    }
    if include_content:
        payload["content"] = getattr(plan, "content", "") or ""
    return payload


def _plan_action_payload(action: Any) -> dict[str, Any]:
    return {
        "action_id": getattr(action, "action_id", ""),
        "plan_id": getattr(action, "plan_id", ""),
        "tool_name": getattr(action, "tool_name", ""),
        "args": dict(getattr(action, "args", {}) or {}),
        "result_summary": getattr(action, "result_summary", "") or "",
        "success": bool(getattr(action, "success", False)),
        "timestamp": _dt_iso(getattr(action, "timestamp", None)),
    }


def _phase_payload(phase: Any) -> dict[str, Any]:
    if hasattr(phase, "model_dump"):
        return phase.model_dump(mode="json")
    if isinstance(phase, dict):
        return dict(phase)
    return {"value": str(phase)}


def _task_payload(task: Any) -> dict[str, Any]:
    return {
        "task_id": str(getattr(task, "task_id", "")),
        "created_at": _dt_iso(getattr(task, "created_at", None)),
        "updated_at": _dt_iso(getattr(task, "updated_at", None)),
        "objective": getattr(task, "objective", ""),
        "stage": _enum_value(getattr(task, "stage", None)),
        "status": getattr(task, "status", ""),
        "artifacts": list(getattr(task, "artifacts", []) or []),
        "attempt": getattr(task, "attempt", 0),
        "max_attempts": getattr(task, "max_attempts", None),
        "verification_command": getattr(task, "verification_command", None),
        "meta": dict(getattr(task, "meta", {}) or {}),
        "phases": [_phase_payload(phase) for phase in (getattr(task, "phases", []) or [])],
        "scratch_dir": getattr(task, "scratch_dir", None),
        "checkpoint_commit": getattr(task, "checkpoint_commit", None),
        "convergence_hashes": list(getattr(task, "convergence_hashes", []) or []),
        "retry_backoff_seconds": getattr(task, "retry_backoff_seconds", None),
        "depends_on": list(getattr(task, "depends_on", []) or []),
        "project_id": getattr(task, "project_id", None),
        "commitment_id": getattr(task, "commitment_id", None),
    }


def _transition_payload(item: Any) -> dict[str, Any]:
    if not isinstance(item, dict):
        return {}
    return {
        "transition_id": item.get("transition_id"),
        "task_id": item.get("task_id"),
        "from_stage": item.get("from_stage"),
        "to_stage": item.get("to_stage"),
        "reason": item.get("reason"),
        "timestamp": _dt_iso(item.get("timestamp")),
        "context": item.get("context", {}),
    }


def _model_payload(item: Any) -> dict[str, Any]:
    if item is None:
        return {}
    if hasattr(item, "model_dump"):
        return item.model_dump(mode="json")
    if isinstance(item, dict):
        return dict(item)
    return {"value": str(item)}


async def _settle_schedules_for_terminal_commitment(
    runtime: Any,
    *,
    commitment_id: str,
    status: CommitmentStatus,
) -> list[str]:
    """Remove terminal commitment schedules from the upcoming queue."""

    if status not in {CommitmentStatus.COMPLETED, CommitmentStatus.ABANDONED}:
        return []
    store = getattr(getattr(runtime, "ctx", None), "schedule_store", None)
    if store is None:
        service = getattr(runtime, "schedule_service", None)
        store = getattr(service, "store", None)
    if store is None:
        return []
    try:
        items = await store.list_items(status=ScheduleStatus.ACTIVE, limit=1000)
    except TypeError:
        items = await store.list_items(status=ScheduleStatus.ACTIVE)
    target_status = (
        ScheduleStatus.COMPLETED
        if status == CommitmentStatus.COMPLETED
        else ScheduleStatus.CANCELLED
    )
    settled: list[str] = []
    for item in items:
        if str(getattr(item, "commitment_id", "") or "") != commitment_id:
            continue
        item.status = target_status
        item.next_run_at = None
        meta = getattr(item, "meta", None)
        if isinstance(meta, dict):
            meta["settled_by_commitment_status"] = status.value
            meta["settled_at"] = datetime.now(timezone.utc).isoformat()
        await store.save(item)
        settled.append(str(item.schedule_id))
    return settled


async def create_commitment(runtime: Any, args: Dict[str, Any]) -> ToolResult:
    store = runtime.commitment_store
    if store is None:
        return ToolResult(False, "Commitment store not available", {})

    content = str(args.get("content", "")).strip()
    if not content:
        return ToolResult(False, "Missing required argument: content", {})
    if bool(args.get("_audit_only")):
        return ToolResult(
            True,
            json.dumps(
                {
                    "audit_only": True,
                    "would_create": "commitment",
                    "content": content,
                    "priority": float(args.get("priority", 5.0)),
                    "tags": args.get("tags", []),
                }
            ),
            {"audit_only": True, "mutation_suppressed": True},
        )

    priority = float(args.get("priority", 5.0))
    deadline_str = args.get("deadline")
    deadline = datetime.fromisoformat(str(deadline_str)) if deadline_str else None

    tags = args.get("tags", [])
    if isinstance(tags, str):
        tags = [t.strip() for t in tags.split(",") if t.strip()]
    project_type = _classify_workflow_args(args, keys=("content",))
    meta: dict[str, Any] = {}
    meta["source"] = "workflow_create_commitment"
    if project_type != PROJECT_TYPE_GENERAL:
        meta["project_type"] = project_type
        if project_type not in {str(tag).lower() for tag in tags}:
            tags = [*tags, project_type]

    commitment = Commitment(
        content=content,
        priority=priority,
        deadline=deadline,
        tags=tags,
        meta=meta,
    )
    await attach_operator_promise_claim(
        runtime,
        commitment,
        subject="operator_commitment",
        source="workflow_create_commitment",
        evidence_summary="Operator-facing workflow commitment persisted.",
    )
    await store.save(commitment)
    commitment_id = str(commitment.commitment_id)
    _record_workflow_provenance(
        runtime,
        kind=ProvenanceTransitionKind.MUTATION,
        source_artifact=f"workflow|commitment|{commitment_id}",
        trigger_action="workflow_create_commitment",
        target_entity=commitment_id,
        origin_action_id=commitment_id,
        status="mutated",
        details={"content": content, "priority": priority, "tags": tags, "meta": meta},
    )
    return ToolResult(
        True,
        json.dumps(
            {
                "commitment_id": commitment_id,
                "content": content,
                "priority": priority,
                "status": commitment.status.value,
                "tags": tags,
                "meta": meta,
            }
        ),
        {"commitment_id": commitment_id},
    )


async def update_commitment(runtime: Any, args: Dict[str, Any]) -> ToolResult:
    store = runtime.commitment_store
    if store is None:
        return ToolResult(False, "Commitment store not available", {})

    commitment_id = str(args.get("commitment_id", "")).strip()
    if not commitment_id:
        return ToolResult(False, "Missing required argument: commitment_id", {})

    new_status = str(args.get("status", "")).strip().lower()
    status_map = {
        "completed": CommitmentStatus.COMPLETED,
        "complete": CommitmentStatus.COMPLETED,
        "abandoned": CommitmentStatus.ABANDONED,
        "abandon": CommitmentStatus.ABANDONED,
        "blocked": CommitmentStatus.BLOCKED,
        "block": CommitmentStatus.BLOCKED,
        "active": CommitmentStatus.ACTIVE,
        "activate": CommitmentStatus.ACTIVE,
    }
    status = status_map.get(new_status)
    if status is None:
        return ToolResult(
            False,
            f"Invalid status '{new_status}'. Use: completed, abandoned, blocked, active",
            {},
        )
    commitment = await _get_commitment(store, commitment_id)
    completion_evidence = str(args.get("completion_evidence", "") or "").strip()
    if status == CommitmentStatus.COMPLETED and commitment is not None:
        reason = completion_evidence_rejection_reason(commitment, completion_evidence)
        if reason:
            return ToolResult(False, reason, {"commitment_id": commitment_id})

    if commitment is not None:
        commitment.status = status
        commitment.updated_at = datetime.now(timezone.utc)
        if status == CommitmentStatus.COMPLETED and completion_evidence:
            commitment.meta.update(
                {
                    "completion_evidence": completion_evidence,
                    "completion_evidence_recorded_at": commitment.updated_at.isoformat(),
                    "completed_at": commitment.updated_at.isoformat(),
                }
            )
        await store.save(commitment)
        ok = True
    else:
        ok = await store.update_status(commitment_id, status)
        if not ok:
            return ToolResult(False, f"Commitment {commitment_id} not found", {})
    linked_schedule_ids = await _settle_schedules_for_terminal_commitment(
        runtime,
        commitment_id=commitment_id,
        status=status,
    )
    linked_work_ids = (
        await settle_linked_work_for_terminal_commitment(
            runtime,
            commitment,
            status,
            completion_evidence=completion_evidence,
        )
        if commitment is not None
        else []
    )
    _record_workflow_provenance(
        runtime,
        kind=ProvenanceTransitionKind.MUTATION,
        source_artifact=f"workflow|commitment|{commitment_id}",
        trigger_action="workflow_update_commitment",
        target_entity=commitment_id,
        origin_action_id=commitment_id,
        status="mutated",
        details={
            "status": status.value,
            "completion_evidence": completion_evidence or None,
            "linked_schedules_settled": linked_schedule_ids,
            "linked_work_settled": linked_work_ids,
        },
    )

    return ToolResult(
        True,
        json.dumps(
            {
                "commitment_id": commitment_id,
                "status": status.value,
                "linked_schedules_settled": linked_schedule_ids,
                "linked_work_settled": linked_work_ids,
            }
        ),
        {"linked_schedules_settled": linked_schedule_ids, "linked_work_settled": linked_work_ids},
    )


async def list_commitments(runtime: Any, args: Dict[str, Any]) -> ToolResult:
    store = runtime.commitment_store
    if store is None:
        return ToolResult(False, "Commitment store not available", {})

    status_filter = str(args.get("status", "active")).strip().lower()
    limit = int(args.get("limit", 20))
    status_map = {
        "active": CommitmentStatus.ACTIVE,
        "completed": CommitmentStatus.COMPLETED,
        "abandoned": CommitmentStatus.ABANDONED,
        "blocked": CommitmentStatus.BLOCKED,
    }
    status = status_map.get(status_filter, CommitmentStatus.ACTIVE)
    items = await store.list_by_status(status, limit=limit)
    entries = [
        {
            "commitment_id": str(item.commitment_id),
            "content": item.content,
            "priority": item.priority,
            "status": item.status.value,
            "tags": item.tags,
            "deadline": item.deadline.isoformat() if item.deadline else None,
            "created_at": item.created_at.isoformat(),
        }
        for item in items
    ]
    return ToolResult(True, json.dumps({"count": len(entries), "items": entries}), {})


async def get_commitment(runtime: Any, args: Dict[str, Any]) -> ToolResult:
    store = runtime.commitment_store
    if store is None:
        return ToolResult(False, "Commitment store not available", {})

    commitment_id = str(args.get("commitment_id", "") or "").strip()
    if not commitment_id:
        return ToolResult(False, "Missing required argument: commitment_id", {})

    commitment = await _get_commitment(store, commitment_id)
    if commitment is None:
        return ToolResult(
            True,
            json.dumps({"found": False, "commitment_id": commitment_id}),
            {"found": False, "commitment_id": commitment_id},
        )
    payload = commitment_operator_snapshot(commitment, include_meta=True)
    return ToolResult(
        True,
        json.dumps({"found": True, "commitment": payload}),
        {"found": True, "commitment_id": payload["commitment_id"]},
    )


async def create_schedule(runtime: Any, args: Dict[str, Any]) -> ToolResult:
    service = getattr(runtime, "schedule_service", None)
    if service is None:
        return ToolResult(False, "Schedule service not available", {})

    from opencas.scheduling import ScheduleAction, ScheduleKind

    if not str(args.get("title", "")).strip() or not args.get("start_at"):
        return ToolResult(False, "Missing required arguments: title, start_at", {})
    try:
        start_at = datetime.fromisoformat(str(args.get("start_at")))
    except ValueError:
        return ToolResult(False, "Invalid start_at; use ISO-8601 datetime.", {})
    if start_at.tzinfo is None:
        start_at = start_at.replace(tzinfo=timezone.utc)
    start_at_utc = start_at.astimezone(timezone.utc)
    if start_at_utc < datetime.now(timezone.utc) - timedelta(seconds=60):
        return ToolResult(
            False,
            "start_at is in the past for a future schedule; choose a future time or record the event as history.",
            {},
        )

    kind = ScheduleKind(str(args.get("kind", "task")))
    action_arg = args.get("action")
    action = (
        ScheduleAction(str(action_arg))
        if action_arg
        else ScheduleAction.SUBMIT_BAA
        if kind == ScheduleKind.TASK
        else ScheduleAction.REMINDER_ONLY
    )
    if _looks_like_active_writing_return(args) and action != ScheduleAction.SUBMIT_BAA:
        return ToolResult(
            False,
            (
                "Unfinished writing/project return schedules must use action=submit_baa so OpenCAS "
                "resumes the work; use reminder_only only for non-work reminders or completed projects."
            ),
            {},
        )
    if _is_active_writing_return(args, action):
        delay_reason = _delay_reason(args)
        if start_at_utc > datetime.now(timezone.utc) + _ACTIVE_RETURN_MAX_DELAY and not delay_reason:
            return ToolResult(
                False,
                (
                    "start_at is too far in the future for an active unfinished writing/project return; "
                    "choose a nearer time or include delay_reason explaining why waiting that long is intentional."
                ),
                {},
            )
        unmanaged_path = _find_unmanaged_writing_path(runtime, args)
        if unmanaged_path:
            return ToolResult(
                False,
                (
                    f"scheduled writing objective references {unmanaged_path} outside managed workspace root "
                    f"{managed_workspace_root(runtime)}; use the managed workspace path or make broader host access "
                    "explicit."
                ),
                {},
            )
    meta = args.get("meta", {})
    if not isinstance(meta, dict):
        meta = {}
    delay_reason = _delay_reason(args)
    if delay_reason:
        meta = dict(meta)
        meta["delay_reason"] = delay_reason
    project_type = _classify_workflow_args(args, keys=("title", "description", "objective"))
    if project_type != PROJECT_TYPE_GENERAL and not meta.get("project_type"):
        meta = dict(meta)
        meta["project_type"] = project_type
    payload = {
        "kind": kind,
        "action": action,
        "title": str(args.get("title", "")).strip(),
        "description": str(args.get("description", "") or ""),
        "objective": args.get("objective"),
        "start_at": start_at_utc,
        "end_at": datetime.fromisoformat(str(args["end_at"])) if args.get("end_at") else None,
        "timezone": str(args.get("timezone", "America/Denver")),
        "recurrence": str(args.get("recurrence", "none")),
        "interval_hours": args.get("interval_hours"),
        "weekdays": args.get("weekdays", []),
        "max_occurrences": args.get("max_occurrences"),
        "priority": float(args.get("priority", 5.0)),
        "tags": args.get("tags", []),
        "commitment_id": args.get("commitment_id"),
        "plan_id": args.get("plan_id"),
        "meta": meta,
    }
    commitment_id = str(payload.get("commitment_id") or "").strip()
    if commitment_id:
        store = getattr(runtime, "commitment_store", None)
        if store is not None and hasattr(store, "get"):
            commitment = await store.get(commitment_id)
            if commitment is not None:
                status = str(getattr(getattr(commitment, "status", None), "value", getattr(commitment, "status", "")))
                if status in {CommitmentStatus.COMPLETED.value, CommitmentStatus.ABANDONED.value}:
                    return ToolResult(
                        False,
                        (
                            f"Cannot create schedule linked to {status} commitment {commitment_id}; "
                            "create or use an active commitment, or omit commitment_id."
                        ),
                        {},
                    )
                commitment_meta = getattr(commitment, "meta", {}) or {}
                commitment_tags = list(getattr(commitment, "tags", []) or [])
                if has_operator_project_followthrough_authority(commitment_meta, tags=commitment_tags):
                    copied_meta = copy_safe_operator_project_followthrough_evidence(commitment_meta)
                    for key in ("project_key", "project_title", "project_type"):
                        value = commitment_meta.get(key) if isinstance(commitment_meta, dict) else None
                        if value and key not in copied_meta:
                            copied_meta[key] = value
                    payload["meta"] = {**copied_meta, **dict(payload.get("meta") or {})}
                    authority_tags = [
                        tag
                        for tag in ("project_return", "self_directed")
                        if tag in {str(item).strip().lower() for item in commitment_tags}
                    ]
                    payload["tags"] = _append_unique_tags(list(payload.get("tags") or []), authority_tags)
        payload["commitment_id"] = commitment_id
    item = await service.create_schedule(**payload)
    schedule_id = str(item.schedule_id)
    item_meta = getattr(item, "meta", {})
    if not isinstance(item_meta, dict):
        item_meta = {}
    dedupe_payload = {
        key: item_meta[key]
        for key in (
            "dedupe_action",
            "survivor_schedule_id",
            "merged_schedule_ids",
            "duplicate_schedule_ids",
        )
        if key in item_meta
    }
    _record_workflow_provenance(
        runtime,
        kind=ProvenanceTransitionKind.MUTATION,
        source_artifact=f"workflow|schedule|{schedule_id}",
        trigger_action="workflow_create_schedule",
        target_entity=schedule_id,
        origin_action_id=schedule_id,
        status="mutated",
        details={"title": item.title, "kind": item.kind.value, "action": item.action.value},
    )
    return ToolResult(
        True,
        json.dumps(
            {
                "schedule_id": schedule_id,
                "title": item.title,
                "kind": item.kind.value,
                "action": item.action.value,
                "next_run_at": item.next_run_at.isoformat() if item.next_run_at else None,
                **dedupe_payload,
            }
        ),
        {"schedule_id": schedule_id, **dedupe_payload},
    )


async def update_schedule(runtime: Any, args: Dict[str, Any]) -> ToolResult:
    store = getattr(runtime.ctx, "schedule_store", None)
    if store is None:
        return ToolResult(False, "Schedule store not available", {})
    schedule_id = str(args.get("schedule_id", "")).strip()
    if not schedule_id:
        return ToolResult(False, "Missing required argument: schedule_id", {})
    item = await store.get(schedule_id)
    if item is None:
        return ToolResult(False, f"Schedule {schedule_id} not found", {})
    if "status" in args:
        from opencas.scheduling import ScheduleStatus

        item.status = ScheduleStatus(str(args["status"]))
        if item.status in (
            ScheduleStatus.CANCELLED,
            ScheduleStatus.COMPLETED,
            ScheduleStatus.PAUSED,
        ):
            item.next_run_at = None
    for key in ("title", "description", "objective", "priority", "tags"):
        if key in args:
            setattr(item, key, args[key])
    await store.save(item)
    _record_workflow_provenance(
        runtime,
        kind=ProvenanceTransitionKind.MUTATION,
        source_artifact=f"workflow|schedule|{schedule_id}",
        trigger_action="workflow_update_schedule",
        target_entity=schedule_id,
        origin_action_id=schedule_id,
        status="mutated",
        details={"status": item.status.value, "title": item.title},
    )
    return ToolResult(True, json.dumps({"schedule_id": schedule_id, "updated": True}), {})


async def cancel_schedule(runtime: Any, args: Dict[str, Any]) -> ToolResult:
    store = getattr(runtime.ctx, "schedule_store", None)
    if store is None:
        return ToolResult(False, "Schedule store not available", {})
    schedule_id = str(args.get("schedule_id", "")).strip()
    if not schedule_id:
        return ToolResult(False, "Missing required argument: schedule_id", {})
    item = await store.get(schedule_id)
    if item is None:
        return ToolResult(False, f"Schedule {schedule_id} not found", {})

    item.status = ScheduleStatus.CANCELLED
    item.next_run_at = None
    meta = getattr(item, "meta", None)
    if not isinstance(meta, dict):
        meta = {}
    reason = str(args.get("reason", "") or "").strip()
    if reason:
        meta["cancel_reason"] = reason
    meta["cancelled_by_tool"] = "workflow_cancel_schedule"
    item.meta = meta
    await store.save(item)
    _record_workflow_provenance(
        runtime,
        kind=ProvenanceTransitionKind.MUTATION,
        source_artifact=f"workflow|schedule|{schedule_id}",
        trigger_action="workflow_cancel_schedule",
        target_entity=schedule_id,
        origin_action_id=schedule_id,
        status="mutated",
        details={"status": item.status.value, "title": item.title, "reason": reason},
    )
    return ToolResult(
        True,
        json.dumps(
            {
                "schedule_id": schedule_id,
                "cancelled": True,
                "status": item.status.value,
            }
        ),
        {"schedule_id": schedule_id, "cancelled": True, "status": item.status.value},
    )


async def get_schedule(runtime: Any, args: Dict[str, Any]) -> ToolResult:
    store = getattr(runtime.ctx, "schedule_store", None)
    if store is None:
        return ToolResult(False, "Schedule store not available", {})
    schedule_id = str(args.get("schedule_id", "") or "").strip()
    if not schedule_id:
        return ToolResult(False, "Missing required argument: schedule_id", {})
    item = await store.get(schedule_id)
    if item is None:
        return ToolResult(
            True,
            json.dumps({"found": False, "schedule_id": schedule_id}),
            {"found": False, "schedule_id": schedule_id},
        )

    run_limit = int(args.get("run_limit", 10))
    runs = []
    list_runs = getattr(store, "list_runs", None)
    if callable(list_runs) and run_limit > 0:
        runs_result = list_runs(schedule_id=schedule_id, limit=run_limit)
        runs = await runs_result if isawaitable(runs_result) else runs_result
    payload = {
        "found": True,
        "schedule": _schedule_payload(item),
        "recent_runs": [_schedule_run_payload(run) for run in (runs or [])],
    }
    return ToolResult(
        True,
        json.dumps(payload),
        {"found": True, "schedule_id": schedule_id, "run_count": len(payload["recent_runs"])},
    )


async def list_schedules(runtime: Any, args: Dict[str, Any]) -> ToolResult:
    store = getattr(runtime.ctx, "schedule_store", None)
    if store is None:
        return ToolResult(False, "Schedule store not available", {})
    from opencas.scheduling import ScheduleKind, ScheduleStatus

    status = args.get("status", "active")
    kind = args.get("kind")
    items = await store.list_items(
        status=ScheduleStatus(str(status)) if status else None,
        kind=ScheduleKind(str(kind)) if kind else None,
        limit=int(args.get("limit", 20)),
    )
    payload = [
        {
            "schedule_id": str(item.schedule_id),
            "title": item.title,
            "kind": item.kind.value,
            "action": item.action.value,
            "status": item.status.value,
            "next_run_at": item.next_run_at.isoformat() if item.next_run_at else None,
            "recurrence": item.recurrence.value,
        }
        for item in items
    ]
    return ToolResult(True, json.dumps({"count": len(payload), "items": payload}), {})


async def cancel_project(runtime: Any, args: Dict[str, Any]) -> ToolResult:
    result = await lifecycle_cancel_project(
        runtime,
        project_key=str(args.get("project_key", "") or ""),
        project_title=str(args.get("project_title", "") or ""),
        workspace_path=str(args.get("workspace_path", "") or ""),
        reason=str(args.get("reason", "") or ""),
        hard_delete_tasks=bool(args.get("hard_delete_tasks", False)),
    )
    _record_workflow_provenance(
        runtime,
        kind=ProvenanceTransitionKind.MUTATION,
        source_artifact=f"workflow|project|{result.project_key}",
        trigger_action="workflow_cancel_project",
        target_entity=result.project_key,
        origin_action_id=result.receipt_path.as_posix() if result.receipt_path else result.project_key,
        status="mutated",
        details=result.as_dict(),
    )
    return ToolResult(True, json.dumps(result.as_dict()), result.as_dict())


async def cancel_task(runtime: Any, args: Dict[str, Any]) -> ToolResult:
    task_id = str(args.get("task_id", "") or "").strip()
    if not task_id:
        return ToolResult(False, "Missing required argument: task_id", {})
    result = await lifecycle_cancel_task(
        runtime,
        task_id=task_id,
        reason=str(args.get("reason", "") or ""),
        hard_delete=bool(args.get("hard_delete", False)),
    )
    return ToolResult(result.success, json.dumps(result.as_dict()), result.as_dict())


async def list_tasks(runtime: Any, args: Dict[str, Any]) -> ToolResult:
    store = getattr(runtime.ctx, "tasks", None)
    if store is None:
        return ToolResult(False, "Task store not available", {})
    limit = int(args.get("limit", 20))
    items = await store.list_all(limit=limit)
    stage_filter = str(args.get("stage", "") or "").strip()
    status_filter = str(args.get("status", "") or "").strip()
    project_filter = str(args.get("project_id", "") or "").strip()
    commitment_filter = str(args.get("commitment_id", "") or "").strip()
    payload_items = []
    for item in items:
        if stage_filter and str(_enum_value(getattr(item, "stage", ""))) != stage_filter:
            continue
        if status_filter and str(getattr(item, "status", "")) != status_filter:
            continue
        if project_filter and str(getattr(item, "project_id", "") or "") != project_filter:
            continue
        if commitment_filter and str(getattr(item, "commitment_id", "") or "") != commitment_filter:
            continue
        payload_items.append(_task_payload(item))
    return ToolResult(
        True,
        json.dumps({"count": len(payload_items), "items": payload_items}),
        {"count": len(payload_items)},
    )


async def get_task(runtime: Any, args: Dict[str, Any]) -> ToolResult:
    store = getattr(runtime.ctx, "tasks", None)
    if store is None:
        return ToolResult(False, "Task store not available", {})
    task_id = str(args.get("task_id", "") or "").strip()
    if not task_id:
        return ToolResult(False, "Missing required argument: task_id", {})
    task = await store.get(task_id)
    if task is None:
        return ToolResult(
            True,
            json.dumps({"found": False, "task_id": task_id}),
            {"found": False, "task_id": task_id},
        )

    result_payload = None
    get_result = getattr(store, "get_result", None)
    if callable(get_result):
        maybe_result = get_result(task_id)
        result = await maybe_result if isawaitable(maybe_result) else maybe_result
        result_payload = _model_payload(result) if result is not None else None

    transitions = []
    list_transitions = getattr(store, "list_lifecycle_transitions", None)
    if callable(list_transitions):
        maybe_transitions = list_transitions(task_id, limit=int(args.get("transition_limit", 50)))
        transitions = await maybe_transitions if isawaitable(maybe_transitions) else maybe_transitions

    salvage_packets = []
    list_salvage_packets = getattr(store, "list_salvage_packets", None)
    if callable(list_salvage_packets):
        maybe_packets = list_salvage_packets(task_id, limit=int(args.get("salvage_limit", 10)))
        salvage_packets = await maybe_packets if isawaitable(maybe_packets) else maybe_packets

    payload = {
        "found": True,
        "task": _task_payload(task),
        "result": result_payload,
        "transitions": [
            payload for payload in (_transition_payload(item) for item in (transitions or [])) if payload
        ],
        "salvage_packets": [_model_payload(packet) for packet in (salvage_packets or [])],
    }
    return ToolResult(
        True,
        json.dumps(payload),
        {"found": True, "task_id": task_id},
    )


async def create_writing_task(runtime: Any, args: Dict[str, Any]) -> ToolResult:
    store = runtime.commitment_store
    title = str(args.get("title", "")).strip()
    if not title:
        return ToolResult(False, "Missing required argument: title", {})

    description = str(args.get("description", "")).strip()
    output_path = str(args.get("output_path", "")).strip()
    outline = args.get("outline")
    safe_name = title.lower().replace(" ", "_")[:40]
    resolved_output_path = resolve_managed_output_path(
        runtime,
        output_path,
        default_relative_path=Path("notes") / f"{safe_name}.md",
    )

    scaffold = f"# {title}\n\n"
    if description:
        scaffold += f"> {description}\n\n"
    if outline:
        if isinstance(outline, list):
            for section in outline:
                scaffold += f"## {section}\n\n"
        elif isinstance(outline, str):
            scaffold += outline + "\n\n"
    scaffold += "<!-- Created by OpenCAS writing workflow -->\n"

    resolved_output_path.parent.mkdir(parents=True, exist_ok=True)
    resolved_output_path.write_text(scaffold, encoding="utf-8")

    tracking_warnings: list[str] = []
    commitment_id = None
    if store is not None:
        commitment = Commitment(
            content=f"Write: {title}",
            priority=float(args.get("priority", 6.0)),
            tags=["writing"],
        )
        try:
            await store.save(commitment)
            commitment_id = str(commitment.commitment_id)
        except Exception as exc:
            logger.warning(
                "Failed to persist writing-task commitment for %s",
                title,
                exc_info=True,
            )
            tracking_warnings.append(f"commitment: {exc}")

    plan_id = None
    plan_store = getattr(runtime.ctx, "plan_store", None)
    if plan_store is not None and outline:
        outline_text = outline if isinstance(outline, str) else json.dumps(outline)
        candidate_plan_id = f"plan-{uuid4().hex[:8]}"
        plan_created = False
        try:
            await plan_store.create_plan(
                candidate_plan_id,
                content=f"Writing plan for: {title}\n\n{outline_text}",
                project_id=commitment_id,
            )
            plan_created = True
            await plan_store.set_status(candidate_plan_id, "active")
            plan_id = candidate_plan_id
        except Exception as exc:
            if plan_created:
                plan_id = candidate_plan_id
            logger.warning(
                "Failed to persist writing-task plan for %s",
                title,
                exc_info=True,
            )
            tracking_warnings.append(f"plan: {exc}")

    workspace_root = managed_workspace_root(runtime)
    target_entity = resolved_output_path.relative_to(workspace_root).as_posix()
    _record_workflow_provenance(
        runtime,
        kind=ProvenanceTransitionKind.MUTATION,
        source_artifact=f"workflow|writing-task|{title}",
        trigger_action="workflow_create_writing_task",
        parent_transition_id=str(resolved_output_path),
        target_entity=target_entity,
        origin_action_id=str(resolved_output_path),
        status="mutated",
        details={
            "title": title,
            "managed_workspace_root": str(workspace_root),
            "commitment_id": commitment_id,
            "plan_id": plan_id,
        },
    )

    return ToolResult(
        True,
        json.dumps(
            {
                "title": title,
                "output_path": str(resolved_output_path),
                "managed_workspace_root": str(managed_workspace_root(runtime)),
                "commitment_id": commitment_id,
                "plan_id": plan_id,
                "scaffold_written": True,
                "tracking_warnings": tracking_warnings,
            }
        ),
        {"output_path": str(resolved_output_path)},
    )


async def create_plan(runtime: Any, args: Dict[str, Any]) -> ToolResult:
    plan_store = getattr(runtime.ctx, "plan_store", None)
    if plan_store is None:
        return ToolResult(False, "Plan store not available", {})

    content = str(args.get("content", "")).strip()
    if not content:
        return ToolResult(False, "Missing required argument: content", {})

    project_id = args.get("project_id")
    task_id = args.get("task_id")
    plan_id = f"plan-{uuid4().hex[:8]}"
    await plan_store.create_plan(
        plan_id,
        content=content,
        project_id=str(project_id) if project_id else None,
        task_id=str(task_id) if task_id else None,
    )
    await plan_store.set_status(plan_id, "active")
    _record_workflow_provenance(
        runtime,
        kind=ProvenanceTransitionKind.MUTATION,
        source_artifact=f"workflow|plan|{plan_id}",
        trigger_action="workflow_create_plan",
        target_entity=plan_id,
        origin_action_id=plan_id,
        status="mutated",
        details={"content_preview": content[:200], "project_id": project_id, "task_id": task_id},
    )
    return ToolResult(
        True,
        json.dumps(
            {
                "plan_id": plan_id,
                "content_preview": content[:200],
                "status": "active",
            }
        ),
        {"plan_id": plan_id},
    )


async def update_plan(runtime: Any, args: Dict[str, Any]) -> ToolResult:
    plan_store = getattr(runtime.ctx, "plan_store", None)
    if plan_store is None:
        return ToolResult(False, "Plan store not available", {})

    plan_id = str(args.get("plan_id", "")).strip()
    content = str(args.get("content", "")).strip() if "content" in args else ""
    status = str(args.get("status", "")).strip() if "status" in args else ""
    if not plan_id or (not content and not status):
        return ToolResult(False, "Missing required arguments: plan_id and one of content/status", {})

    updated_fields: list[str] = []
    if content:
        ok = await plan_store.update_content(plan_id, content)
        if not ok:
            return ToolResult(False, f"Plan {plan_id} not found", {})
        updated_fields.append("content")
    if status:
        ok = await plan_store.set_status(plan_id, status)
        if not ok:
            return ToolResult(False, f"Plan {plan_id} not found", {})
        updated_fields.append("status")
    if not updated_fields:
        return ToolResult(False, f"Plan {plan_id} not found", {})
    _record_workflow_provenance(
        runtime,
        kind=ProvenanceTransitionKind.MUTATION,
        source_artifact=f"workflow|plan|{plan_id}",
        trigger_action="workflow_update_plan",
        target_entity=plan_id,
        origin_action_id=plan_id,
        status="mutated",
        details={"content_preview": content[:200] if content else None, "status": status or None},
    )
    return ToolResult(True, json.dumps({"plan_id": plan_id, "updated": True, "fields": updated_fields}), {})


async def list_plans(runtime: Any, args: Dict[str, Any]) -> ToolResult:
    plan_store = getattr(runtime.ctx, "plan_store", None)
    if plan_store is None:
        return ToolResult(False, "Plan store not available", {})

    project_id = str(args.get("project_id", "") or "").strip() or None
    task_id = str(args.get("task_id", "") or "").strip() or None
    limit = int(args.get("limit", 20))
    plans = await plan_store.list_active(project_id=project_id, task_id=task_id)
    entries = [_plan_payload(plan, include_content=False) for plan in plans[:limit]]
    return ToolResult(
        True,
        json.dumps({"count": len(entries), "items": entries}),
        {"count": len(entries)},
    )


async def get_plan(runtime: Any, args: Dict[str, Any]) -> ToolResult:
    plan_store = getattr(runtime.ctx, "plan_store", None)
    if plan_store is None:
        return ToolResult(False, "Plan store not available", {})

    plan_id = str(args.get("plan_id", "") or "").strip()
    if not plan_id:
        return ToolResult(False, "Missing required argument: plan_id", {})
    plan = await plan_store.get_plan(plan_id)
    if plan is None:
        return ToolResult(
            True,
            json.dumps({"found": False, "plan_id": plan_id}),
            {"found": False, "plan_id": plan_id},
        )
    action_limit = int(args.get("action_limit", 25))
    actions = await plan_store.get_actions(plan_id, limit=action_limit)
    payload = {
        "found": True,
        "plan": _plan_payload(plan, include_content=True),
        "actions": [_plan_action_payload(action) for action in actions],
    }
    return ToolResult(
        True,
        json.dumps(payload),
        {"found": True, "plan_id": plan_id, "action_count": len(payload["actions"])},
    )


async def repo_triage(runtime: Any, args: Dict[str, Any]) -> ToolResult:
    workspace = str(runtime.ctx.config.primary_workspace_root())
    git_status = await runtime.execute_tool(
        "bash_run_command",
        {"command": "git status --short 2>/dev/null || echo 'not a git repo'", "cwd": workspace},
    )
    git_log = await runtime.execute_tool(
        "bash_run_command",
        {"command": "git log --oneline -10 2>/dev/null || echo 'no git history'", "cwd": workspace},
    )

    work_summary = {"total": 0, "ready": 0, "blocked": 0}
    if getattr(runtime.ctx, "work_store", None) is not None:
        work_summary = await runtime.ctx.work_store.summary_counts()

    commitment_count = 0
    if runtime.commitment_store is not None:
        commitment_count = await runtime.commitment_store.count_by_status(
            CommitmentStatus.ACTIVE
        )

    plan_count = 0
    if getattr(runtime.ctx, "plan_store", None) is not None:
        plan_count = await runtime.ctx.plan_store.count_active()

    return ToolResult(
        True,
        json.dumps(
            {
                "workspace": workspace,
                "git_status": git_status.get("output", ""),
                "recent_commits": git_log.get("output", ""),
                "work_items": work_summary,
                "active_commitments": commitment_count,
                "active_plans": plan_count,
            }
        ),
        {"workspace": workspace},
    )
