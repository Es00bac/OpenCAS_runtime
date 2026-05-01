"""Meaningful-loop observability helpers for operator surfaces."""

from __future__ import annotations

from collections.abc import Iterable
from collections import Counter
from typing import Any, Dict, List, Optional

_EMPTY_COUNTS = {
    "progress_gate": 0,
    "retry_governor": 0,
    "approval_block": 0,
    "affective_pressure": 0,
    "completion": 0,
}


async def build_meaningful_loop_status(
    runtime: Any,
    *,
    task_limit: int = 20,
    pressure_limit: int = 8,
) -> Dict[str, Any]:
    """Build a dashboard-safe snapshot of recent meaningful-loop outcomes."""

    task_limit = max(1, min(task_limit, 100))
    pressure_limit = max(1, min(pressure_limit, 50))
    counts: Counter[str] = Counter(_EMPTY_COUNTS)
    recent_tasks: list[dict[str, Any]] = []
    latest_artifact: Optional[Dict[str, Any]] = None
    latest_evidence: Optional[Dict[str, Any]] = None
    latest_blocker: Optional[Dict[str, Any]] = None
    latest_stop: Optional[Dict[str, Any]] = None

    task_store = getattr(getattr(runtime, "ctx", None), "tasks", None)
    if task_store is not None and hasattr(task_store, "list_all"):
        tasks = await task_store.list_all(limit=max(task_limit, 50))
        for task in tasks[:task_limit]:
            packet = await latest_salvage_packet(task_store, str(getattr(task, "task_id", "")))
            status = task_meaningful_loop_status(task, packet=packet)
            recent_tasks.append(status)
            cause = status.get("loop_stop_cause")
            if cause in _EMPTY_COUNTS:
                counts[cause] += 1
                latest_stop = _prefer_stop_event(latest_stop, _stop_event_from_task(status))
            if latest_artifact is None and status.get("latest_artifact"):
                latest_artifact = _surface(
                    "task",
                    path=status["latest_artifact"],
                    task_id=status.get("task_id"),
                    signal=status.get("latest_meaningful_signal"),
                )
            if latest_evidence is None and status.get("latest_evidence"):
                latest_evidence = _surface(
                    "task",
                    summary=status["latest_evidence"],
                    task_id=status.get("task_id"),
                    signal=status.get("latest_meaningful_signal"),
                )
            if latest_blocker is None and status.get("latest_blocker"):
                latest_blocker = _surface(
                    "task",
                    cause=cause,
                    reason=status["latest_blocker"],
                    task_id=status.get("task_id"),
                )

    shadow_events = _shadow_stop_events(runtime)
    for event in shadow_events:
        cause = event.get("cause")
        if cause in _EMPTY_COUNTS:
            counts[cause] += 1
            latest_stop = _prefer_stop_event(latest_stop, event)
        if latest_artifact is None and event.get("artifact"):
            latest_artifact = _surface(
                "shadow_registry",
                path=event["artifact"],
                cause=cause,
                task_id=event.get("task_id"),
            )
        if latest_blocker is None and event.get("reason"):
            latest_blocker = _surface(
                "shadow_registry",
                cause=cause,
                reason=event["reason"],
                task_id=event.get("task_id"),
            )

    affective_pressures = await _affective_pressure_events(runtime, limit=pressure_limit)
    for pressure in affective_pressures:
        counts["affective_pressure"] += 1
        latest_stop = _prefer_stop_event(
            latest_stop,
            {
                "cause": "affective_pressure",
                "source": "affective_examination",
                "reason": pressure.get("bounded_reason"),
                "at": pressure.get("created_at"),
                "task_id": None,
                "artifact": None,
            },
        )
        if latest_evidence is None and pressure.get("source_excerpt"):
            latest_evidence = _surface(
                "affective_examination",
                summary=pressure["source_excerpt"],
                cause="affective_pressure",
                signal=pressure.get("action_pressure"),
            )
        if latest_blocker is None and pressure.get("bounded_reason"):
            latest_blocker = _surface(
                "affective_examination",
                cause="affective_pressure",
                reason=pressure["bounded_reason"],
            )

    return {
        "available": bool(recent_tasks or shadow_events or affective_pressures),
        "stop_counts": {key: int(counts.get(key, 0)) for key in _EMPTY_COUNTS},
        "latest_stop": latest_stop,
        "latest_artifact": latest_artifact,
        "latest_evidence": latest_evidence,
        "latest_blocker": latest_blocker,
        "recent_tasks": recent_tasks,
        "shadow_blocks": shadow_events[:8],
        "affective_pressures": affective_pressures,
    }


async def latest_salvage_packet(store: Any, task_id: str) -> Any:
    """Return the newest salvage packet when the task store supports it."""

    loader = getattr(store, "get_latest_salvage_packet", None)
    if not callable(loader):
        return None
    try:
        return await loader(task_id)
    except Exception:
        return None


def task_meaningful_loop_status(task: Any, *, packet: Any = None) -> Dict[str, Any]:
    """Normalize one background task into meaningful-loop status fields."""

    meta = getattr(task, "meta", None) if isinstance(getattr(task, "meta", None), dict) else {}
    retry = meta.get("retry_governor") if isinstance(meta.get("retry_governor"), dict) else {}
    signal = _packet_attr(packet, "meaningful_progress_signal") or None
    artifact = _packet_artifact(packet)
    evidence = _packet_evidence(packet)
    blocker = _packet_blocker(packet)
    cause: Optional[str] = None

    if retry.get("allowed") is False:
        cause = "retry_governor"
        blocker = str(retry.get("reason") or blocker or "retry blocked").strip()
    elif signal in {"no_meaningful_progress", "blocker"}:
        cause = "progress_gate"
        blocker = blocker or "No meaningful progress was recorded."
    elif _task_completed(task) or signal == "completed":
        cause = "completion"

    return {
        "task_id": str(getattr(task, "task_id", "")),
        "title": _human_title(meta.get("title") or getattr(task, "objective", None), fallback="Background task"),
        "status": str(getattr(task, "status", "") or ""),
        "stage": _enum_value(getattr(task, "stage", "")),
        "loop_stop_cause": cause,
        "latest_meaningful_signal": signal,
        "latest_artifact": artifact,
        "latest_evidence": evidence,
        "latest_blocker": blocker,
    }


def _stop_event_from_task(status: Dict[str, Any]) -> Dict[str, Any]:
    return {
        "cause": status.get("loop_stop_cause"),
        "source": "task_store",
        "reason": status.get("latest_blocker") or status.get("latest_evidence") or status.get("status"),
        "task_id": status.get("task_id"),
        "artifact": status.get("latest_artifact"),
        "at": None,
    }


def _prefer_stop_event(
    current: Optional[Dict[str, Any]],
    candidate: Dict[str, Any],
) -> Dict[str, Any]:
    if current is None:
        return candidate
    if current.get("cause") == "completion" and candidate.get("cause") != "completion":
        return candidate
    return current


def _shadow_stop_events(runtime: Any) -> List[Dict[str, Any]]:
    shadow_registry = getattr(getattr(runtime, "ctx", None), "shadow_registry", None)
    summary = getattr(shadow_registry, "summary", None)
    if not callable(summary):
        return []
    try:
        payload = summary(limit=12, cluster_limit=5)
    except Exception:
        return []
    events: list[dict[str, Any]] = []
    for item in payload.get("recent_entries") or []:
        if not isinstance(item, dict):
            continue
        cause = _shadow_cause(item)
        if not cause:
            continue
        events.append(
            {
                "cause": cause,
                "source": "shadow_registry",
                "reason": str(item.get("block_context") or item.get("block_reason") or "").strip(),
                "at": item.get("captured_at"),
                "task_id": item.get("task_id"),
                "artifact": item.get("artifact"),
                "tool_name": item.get("tool_name"),
                "intent_summary": item.get("intent_summary"),
            }
        )
    return events


def _shadow_cause(item: Dict[str, Any]) -> Optional[str]:
    reason = str(item.get("block_reason") or "").lower()
    context = str(item.get("block_context") or "").lower()
    if reason == "retry_blocked":
        return "retry_governor"
    if "approval" in reason or "approval" in context or reason == "approval_denied":
        return "approval_block"
    if reason in {"tool_loop_guard_blocked", "validation_blocked", "hook_blocked", "safety_blocked"}:
        return "progress_gate"
    return None


async def _affective_pressure_events(runtime: Any, *, limit: int) -> List[Dict[str, Any]]:
    store = getattr(getattr(runtime, "ctx", None), "affective_examinations", None)
    loader = getattr(store, "list_unresolved_pressures", None)
    if not callable(loader):
        return []
    try:
        records = await loader(limit=limit)
    except Exception:
        return []
    events: list[dict[str, Any]] = []
    for record in records:
        pressure = _enum_value(getattr(record, "action_pressure", ""))
        if pressure in {"continue", "archive_only", ""}:
            continue
        affect = getattr(record, "affect", None)
        events.append(
            {
                "examination_id": str(getattr(record, "examination_id", "")),
                "created_at": getattr(record, "created_at", None).isoformat()
                if getattr(record, "created_at", None)
                else None,
                "session_id": getattr(record, "session_id", None),
                "source_type": _enum_value(getattr(record, "source_type", "")),
                "source_id": getattr(record, "source_id", ""),
                "source_excerpt": getattr(record, "source_excerpt", ""),
                "source_hash": getattr(record, "source_hash", ""),
                "action_pressure": pressure,
                "primary_emotion": _enum_value(getattr(affect, "primary_emotion", "")),
                "confidence": getattr(record, "confidence", None),
                "bounded_reason": getattr(record, "bounded_reason", ""),
                "already_recognized": bool(getattr(record, "meta", {}).get("already_recognized"))
                if isinstance(getattr(record, "meta", None), dict)
                else False,
            }
        )
    return events


def _packet_artifact(packet: Any) -> Optional[str]:
    if packet is None:
        return None
    canonical = _packet_attr(packet, "canonical_artifact_path")
    if canonical:
        return str(canonical)
    touched = _packet_attr(packet, "artifact_paths_touched") or []
    if isinstance(touched, Iterable) and not isinstance(touched, (str, bytes)):
        for item in touched:
            if item:
                return str(item)
    return None


def _packet_evidence(packet: Any) -> Optional[str]:
    if packet is None:
        return None
    signal = _packet_attr(packet, "meaningful_progress_signal")
    partial = str(_packet_attr(packet, "partial_value") or "").strip()
    if signal in {"evidence", "question", "constraint", "completed"} and partial:
        return partial
    if signal == "artifact" and partial:
        return partial
    return None


def _packet_blocker(packet: Any) -> Optional[str]:
    if packet is None:
        return None
    signal = _packet_attr(packet, "meaningful_progress_signal")
    if signal not in {"no_meaningful_progress", "blocker"}:
        return None
    next_step = str(_packet_attr(packet, "best_next_step") or "").strip()
    if next_step:
        return next_step
    constraints = _packet_attr(packet, "discovered_constraints") or []
    if isinstance(constraints, list) and constraints:
        return str(constraints[0])
    return "No meaningful progress was recorded."


def _packet_attr(packet: Any, name: str) -> Any:
    if packet is None:
        return None
    if isinstance(packet, dict):
        return packet.get(name)
    return getattr(packet, name, None)


def _task_completed(task: Any) -> bool:
    status = str(getattr(task, "status", "") or "").lower()
    stage = _enum_value(getattr(task, "stage", "")).lower()
    return status in {"completed", "success"} or stage == "done"


def _surface(source: str, **values: Any) -> Dict[str, Any]:
    return {"source": source, **{key: value for key, value in values.items() if value is not None}}


def _enum_value(value: Any) -> str:
    return str(getattr(value, "value", value) or "")


def _human_title(text: str | None, fallback: str = "Untitled") -> str:
    raw = str(text or "").strip()
    if not raw:
        return fallback
    first_line = raw.splitlines()[0].strip()
    compact = " ".join(first_line.split())
    return compact if len(compact) <= 88 else compact[:85].rstrip() + "..."
