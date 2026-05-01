"""Tasking-oriented workflow helpers for commitments, plans, and schedules."""

from __future__ import annotations

import json
import re
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Dict
from uuid import uuid4

from ...api.provenance_store import (
    ProvenanceTransitionKind,
    record_provenance_transition,
)
from ...autonomy.commitment import Commitment, CommitmentStatus
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
_WRITING_RETURN_MARKERS = (
    "chapter",
    "creative",
    "draft",
    "manuscript",
    "prose",
    "revise",
    "revision",
    "write",
    "writing",
)


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
    return any(marker in text for marker in _ACTIVE_RETURN_MARKERS) and any(
        marker in text for marker in _WRITING_RETURN_MARKERS
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
    if not any(marker in text.lower() for marker in _WRITING_RETURN_MARKERS):
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


async def create_commitment(runtime: Any, args: Dict[str, Any]) -> ToolResult:
    store = runtime.commitment_store
    if store is None:
        return ToolResult(False, "Commitment store not available", {})

    content = str(args.get("content", "")).strip()
    if not content:
        return ToolResult(False, "Missing required argument: content", {})

    priority = float(args.get("priority", 5.0))
    deadline_str = args.get("deadline")
    deadline = datetime.fromisoformat(str(deadline_str)) if deadline_str else None

    tags = args.get("tags", [])
    if isinstance(tags, str):
        tags = [t.strip() for t in tags.split(",") if t.strip()]

    commitment = Commitment(
        content=content,
        priority=priority,
        deadline=deadline,
        tags=tags,
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
        details={"content": content, "priority": priority, "tags": tags},
    )
    return ToolResult(
        True,
        json.dumps(
            {
                "commitment_id": commitment_id,
                "content": content,
                "priority": priority,
                "status": commitment.status.value,
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

    ok = await store.update_status(commitment_id, status)
    if not ok:
        return ToolResult(False, f"Commitment {commitment_id} not found", {})
    _record_workflow_provenance(
        runtime,
        kind=ProvenanceTransitionKind.MUTATION,
        source_artifact=f"workflow|commitment|{commitment_id}",
        trigger_action="workflow_update_commitment",
        target_entity=commitment_id,
        origin_action_id=commitment_id,
        status="mutated",
        details={"status": status.value},
    )

    return ToolResult(
        True,
        json.dumps({"commitment_id": commitment_id, "status": status.value}),
        {},
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
                    f"{managed_workspace_root(runtime)}; use the managed workspace path or make broader host access explicit."
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
        payload["commitment_id"] = commitment_id
    item = await service.create_schedule(**payload)
    schedule_id = str(item.schedule_id)
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
            }
        ),
        {"schedule_id": schedule_id},
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

    commitment_id = None
    if store is not None:
        commitment = Commitment(
            content=f"Write: {title}",
            priority=float(args.get("priority", 6.0)),
            tags=["writing"],
        )
        await store.save(commitment)
        commitment_id = str(commitment.commitment_id)

    plan_id = None
    plan_store = getattr(runtime.ctx, "plan_store", None)
    if plan_store is not None and outline:
        outline_text = outline if isinstance(outline, str) else json.dumps(outline)
        plan_id = f"plan-{uuid4().hex[:8]}"
        await plan_store.create_plan(
            plan_id,
            content=f"Writing plan for: {title}\n\n{outline_text}",
            project_id=commitment_id,
        )
        await plan_store.set_status(plan_id, "active")

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
    content = str(args.get("content", "")).strip()
    if not plan_id or not content:
        return ToolResult(False, "Missing required arguments: plan_id, content", {})

    ok = await plan_store.update_content(plan_id, content)
    if not ok:
        return ToolResult(False, f"Plan {plan_id} not found", {})
    _record_workflow_provenance(
        runtime,
        kind=ProvenanceTransitionKind.MUTATION,
        source_artifact=f"workflow|plan|{plan_id}",
        trigger_action="workflow_update_plan",
        target_entity=plan_id,
        origin_action_id=plan_id,
        status="mutated",
        details={"content_preview": content[:200]},
    )
    return ToolResult(True, json.dumps({"plan_id": plan_id, "updated": True}), {})


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
