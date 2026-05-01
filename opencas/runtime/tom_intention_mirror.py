"""Small helpers for mirroring runtime intent into durable ToM state."""

from __future__ import annotations

from typing import Any, Optional

from opencas.tom import BeliefSubject, IntentionStatus


def _normalize_intention(content: Any) -> str:
    return " ".join(str(content or "").strip().lower().split())


def _runtime_tom(runtime: Any) -> Optional[Any]:
    tom = getattr(runtime, "tom", None)
    if tom is not None:
        return tom
    return getattr(getattr(runtime, "ctx", None), "tom", None)


def _trace(runtime: Any, event: str, payload: dict[str, Any]) -> None:
    tracer = getattr(runtime, "_trace", None)
    if callable(tracer):
        tracer(event, payload)


def _has_active_self_intention(tom: Any, normalized_content: str) -> bool:
    lister = getattr(tom, "list_intentions", None)
    if not callable(lister):
        return False
    try:
        intentions = lister(actor=BeliefSubject.SELF, status=IntentionStatus.ACTIVE)
    except TypeError:
        intentions = lister()
    except Exception:
        return False
    return any(
        _normalize_intention(getattr(intention, "content", "")) == normalized_content
        for intention in list(intentions or [])
    )


async def mirror_runtime_intention(
    runtime: Any,
    content: Any,
    *,
    source: str,
    session_id: Optional[str] = None,
    work_id: Optional[str] = None,
    task_id: Optional[str] = None,
    stage: Optional[str] = None,
) -> bool:
    """Record a runtime self-intention in ToM once per active content string."""
    normalized = _normalize_intention(content)
    if not normalized:
        return False
    tom = _runtime_tom(runtime)
    recorder = getattr(tom, "record_intention", None)
    if tom is None or not callable(recorder):
        return False
    if _has_active_self_intention(tom, normalized):
        return False

    meta = {"source": source}
    if session_id:
        meta["session_id"] = session_id
    if work_id:
        meta["work_id"] = work_id
    if task_id:
        meta["task_id"] = task_id
    if stage:
        meta["stage"] = stage

    try:
        await recorder(BeliefSubject.SELF, str(content or "").strip(), meta=meta)
        return True
    except Exception as exc:
        _trace(runtime, "tom_intention_mirror_error", {"source": source, "error": str(exc)})
        return False


async def resolve_runtime_intention(
    runtime: Any,
    content: Any,
    *,
    status: IntentionStatus = IntentionStatus.COMPLETED,
) -> bool:
    """Resolve a mirrored runtime intention if ToM is available."""
    if not _normalize_intention(content):
        return False
    tom = _runtime_tom(runtime)
    resolver = getattr(tom, "resolve_intention", None)
    if tom is None or not callable(resolver):
        return False
    try:
        return bool(await resolver(str(content or "").strip(), status))
    except Exception as exc:
        _trace(runtime, "tom_intention_resolve_error", {"status": status.value, "error": str(exc)})
        return False


async def reconcile_completed_runtime_intentions(runtime: Any, *, limit: int = 500) -> int:
    """Resolve active work-dispatch intentions when task history is already terminal."""
    tom = _runtime_tom(runtime)
    lister = getattr(tom, "list_intentions", None)
    tasks = getattr(getattr(runtime, "ctx", None), "tasks", None)
    task_lister = getattr(tasks, "list_all", None)
    if tom is None or not callable(lister) or not callable(task_lister):
        return 0

    try:
        intentions = list(lister(actor=BeliefSubject.SELF, status=IntentionStatus.ACTIVE) or [])
        task_rows = list(await task_lister(limit=limit))
    except Exception as exc:
        _trace(runtime, "tom_intention_reconcile_load_error", {"error": str(exc)})
        return 0

    terminal_objectives: dict[str, IntentionStatus] = {}
    for task in task_rows:
        terminal_status = _terminal_intention_status_for_task(task)
        if terminal_status is None:
            continue
        normalized_objective = _normalize_intention(getattr(task, "objective", ""))
        if normalized_objective:
            terminal_objectives[normalized_objective] = terminal_status

    resolved = 0
    for intention in intentions:
        meta = dict(getattr(intention, "meta", {}) or {})
        if str(meta.get("source") or "") != "active_work_dispatch":
            continue
        content = getattr(intention, "content", "")
        normalized_content = _normalize_intention(content)
        terminal_status = terminal_objectives.get(normalized_content)
        if terminal_status is None:
            continue
        if await resolve_runtime_intention(runtime, content, status=terminal_status):
            resolved += 1
    if resolved:
        _trace(runtime, "tom_intentions_reconciled", {"resolved_count": resolved})
    return resolved


def _task_is_completed_success(task: Any) -> bool:
    stage = getattr(task, "stage", "")
    stage_value = str(getattr(stage, "value", stage) or "").lower()
    status = str(getattr(task, "status", "") or "").lower()
    return stage_value == "done" and status == "completed"


def _terminal_intention_status_for_task(task: Any) -> Optional[IntentionStatus]:
    if _task_is_completed_success(task):
        return IntentionStatus.COMPLETED
    stage = getattr(task, "stage", "")
    stage_value = str(getattr(stage, "value", stage) or "").lower()
    status = str(getattr(task, "status", "") or "").lower()
    if stage_value in {"failed", "abandoned"} or status in {"failed", "error", "abandoned"}:
        return IntentionStatus.ABANDONED
    return None
