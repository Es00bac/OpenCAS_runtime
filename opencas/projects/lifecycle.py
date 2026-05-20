"""Project and task lifecycle operations."""

from __future__ import annotations

import json
import re
import shutil
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Optional

from opencas.autonomy.commitment import CommitmentStatus
from opencas.execution.models import ExecutionStage
from opencas.scheduling import ScheduleStatus
from opencas.tools.adapters.workflow_paths import managed_workspace_root

_NON_KEY_RE = re.compile(r"[^a-z0-9]+")


@dataclass(frozen=True)
class ProjectCancelResult:
    """Result of a project cancel/compost operation."""

    success: bool
    project_key: str
    project_title: str
    reason: str
    compost_path: Optional[Path]
    receipt_path: Optional[Path]
    commitments_abandoned: list[str]
    schedules_cancelled: list[str]
    tasks_cancelled: list[str]
    tasks_deleted: list[str]
    salvage_index_path: Optional[Path]
    shadow_registry_id: Optional[str]
    message: str

    def as_dict(self) -> dict[str, Any]:
        return {
            "success": self.success,
            "project_key": self.project_key,
            "project_title": self.project_title,
            "reason": self.reason,
            "compost_path": str(self.compost_path) if self.compost_path else None,
            "receipt_path": str(self.receipt_path) if self.receipt_path else None,
            "commitments_abandoned": self.commitments_abandoned,
            "schedules_cancelled": self.schedules_cancelled,
            "tasks_cancelled": self.tasks_cancelled,
            "tasks_deleted": self.tasks_deleted,
            "salvage_index_path": str(self.salvage_index_path) if self.salvage_index_path else None,
            "shadow_registry_id": self.shadow_registry_id,
            "message": self.message,
        }


@dataclass(frozen=True)
class TaskCancelResult:
    """Result of a task cancel/delete operation."""

    success: bool
    task_id: str
    reason: str
    hard_delete: bool
    message: str

    def as_dict(self) -> dict[str, Any]:
        return {
            "success": self.success,
            "task_id": self.task_id,
            "reason": self.reason,
            "hard_delete": self.hard_delete,
            "message": self.message,
        }


async def cancel_project(
    runtime: Any,
    *,
    project_key: str = "",
    project_title: str = "",
    workspace_path: str = "",
    reason: str = "",
    hard_delete_tasks: bool = False,
) -> ProjectCancelResult:
    """Cancel a project, compost its workspace, and stop linked work."""

    title = project_title.strip() or Path(workspace_path).name or project_key.strip() or "project"
    key = _project_key(project_key or title)
    reason = reason.strip() or "project canceled"
    workspace_root = managed_workspace_root(runtime).resolve()
    project_path = _resolve_workspace_path(workspace_root, workspace_path or title)

    compost_path: Optional[Path] = None
    if project_path is not None and project_path.exists():
        compost_path = _unique_compost_path(workspace_root, key)
        compost_path.parent.mkdir(parents=True, exist_ok=True)
        shutil.move(str(project_path), str(compost_path))

    commitment_ids = await _abandon_matching_commitments(
        runtime,
        key=key,
        title=title,
        reason=reason,
        compost_path=compost_path,
    )
    schedule_ids = await _cancel_matching_schedules(runtime, key=key, title=title, commitment_ids=commitment_ids)
    task_ids, deleted_task_ids = await _cancel_matching_tasks(
        runtime,
        key=key,
        title=title,
        commitment_ids=commitment_ids,
        reason=reason,
        hard_delete=hard_delete_tasks,
    )
    _remove_matching_executive_goals(runtime, key=key, title=title)

    salvage_candidates = _salvage_candidates(compost_path)
    salvage_index_path = _write_salvage_index(
        compost_path=compost_path,
        project_title=title,
        project_key=key,
        reason=reason,
        salvage_candidates=salvage_candidates,
        commitments_abandoned=commitment_ids,
        schedules_cancelled=schedule_ids,
        tasks_cancelled=task_ids,
        tasks_deleted=deleted_task_ids,
    )
    receipt_path = _write_receipt(
        workspace_root=workspace_root,
        compost_path=compost_path,
        payload={
            "event": "project_cancelled",
            "project_key": key,
            "project_title": title,
            "reason": reason,
            "compost_path": str(compost_path) if compost_path else None,
            "commitments_abandoned": commitment_ids,
            "schedules_cancelled": schedule_ids,
            "tasks_cancelled": task_ids,
            "tasks_deleted": deleted_task_ids,
            "salvage_candidates": salvage_candidates,
            "salvage_index_path": str(salvage_index_path) if salvage_index_path else None,
            "created_at": datetime.now(timezone.utc).isoformat(),
        },
    )
    shadow_registry_id = _capture_shadow_compost(
        runtime,
        project_key=key,
        project_title=title,
        reason=reason,
        compost_path=compost_path,
        receipt_path=receipt_path,
        salvage_index_path=salvage_index_path,
        salvage_candidates=salvage_candidates,
        commitments_abandoned=commitment_ids,
        schedules_cancelled=schedule_ids,
        tasks_cancelled=task_ids,
        tasks_deleted=deleted_task_ids,
    )
    if shadow_registry_id:
        _update_receipt(receipt_path, {"shadow_registry_id": shadow_registry_id})
    _trace(
        runtime,
        "project_cancelled",
        {
            "project_key": key,
            "project_title": title,
            "commitments_abandoned": len(commitment_ids),
            "schedules_cancelled": len(schedule_ids),
            "tasks_cancelled": len(task_ids),
            "tasks_deleted": len(deleted_task_ids),
            "compost_path": str(compost_path) if compost_path else None,
            "receipt_path": str(receipt_path) if receipt_path else None,
            "shadow_registry_id": shadow_registry_id,
        },
    )
    return ProjectCancelResult(
        success=True,
        project_key=key,
        project_title=title,
        reason=reason,
        compost_path=compost_path,
        receipt_path=receipt_path,
        commitments_abandoned=commitment_ids,
        schedules_cancelled=schedule_ids,
        tasks_cancelled=task_ids,
        tasks_deleted=deleted_task_ids,
        salvage_index_path=salvage_index_path,
        shadow_registry_id=shadow_registry_id,
        message=f"Cancelled {title}; composted={bool(compost_path)}.",
    )


async def cancel_task(
    runtime: Any,
    *,
    task_id: str,
    reason: str = "",
    hard_delete: bool = False,
) -> TaskCancelResult:
    """Cancel or delete a BAA task."""

    store = _task_store(runtime)
    if store is None:
        return TaskCancelResult(False, task_id, reason, hard_delete, "Task store not available")
    task_id = task_id.strip()
    reason = reason.strip() or "task canceled"
    if hard_delete:
        ok = await store.delete(task_id)
        _drop_live_task(runtime, task_id)
        return TaskCancelResult(ok, task_id, reason, hard_delete, "Task deleted" if ok else "Task not found")
    task = await store.get(task_id)
    if task is None:
        return TaskCancelResult(False, task_id, reason, hard_delete, "Task not found")
    task.stage = ExecutionStage.FAILED
    task.status = "cancelled"
    task.updated_at = datetime.now(timezone.utc)
    task.meta["cancel_reason"] = reason
    task.meta["cancelled_at"] = task.updated_at.isoformat()
    await store.save(task)
    _drop_live_task(runtime, task_id)
    return TaskCancelResult(True, task_id, reason, hard_delete, "Task cancelled")


async def _abandon_matching_commitments(
    runtime: Any,
    *,
    key: str,
    title: str,
    reason: str,
    compost_path: Optional[Path],
) -> list[str]:
    store = getattr(runtime, "commitment_store", None)
    if store is None:
        return []
    matched: list[str] = []
    statuses = (
        CommitmentStatus.ACTIVE,
        CommitmentStatus.BLOCKED,
        CommitmentStatus.COMPLETED,
    )
    for status in statuses:
        try:
            commitments = await store.list_by_status(status, limit=500)
        except Exception:
            continue
        for commitment in commitments:
            if not _matches_text(_commitment_text(commitment), key=key, title=title):
                continue
            commitment.status = CommitmentStatus.ABANDONED
            commitment.updated_at = datetime.now(timezone.utc)
            commitment.meta["cancel_reason"] = reason
            commitment.meta["cancelled_at"] = commitment.updated_at.isoformat()
            if compost_path is not None:
                commitment.meta["compost_path"] = str(compost_path)
            await store.save(commitment)
            matched.append(str(commitment.commitment_id))
    return sorted(set(matched))


async def _cancel_matching_schedules(
    runtime: Any,
    *,
    key: str,
    title: str,
    commitment_ids: Iterable[str],
) -> list[str]:
    store = getattr(getattr(runtime, "ctx", None), "schedule_store", None)
    if store is None:
        service = getattr(runtime, "schedule_service", None)
        store = getattr(service, "store", None)
    if store is None:
        return []
    commitment_set = {str(item) for item in commitment_ids}
    cancelled: list[str] = []
    for item in await store.list_items(status=None, limit=1000):
        schedule_id = str(item.schedule_id)
        if str(item.commitment_id or "") not in commitment_set and not _matches_text(
            _schedule_text(item),
            key=key,
            title=title,
        ):
            continue
        item.status = ScheduleStatus.CANCELLED
        item.next_run_at = None
        item.updated_at = datetime.now(timezone.utc)
        item.meta["cancel_reason"] = "project_cancelled"
        await store.save(item)
        cancelled.append(schedule_id)
    return sorted(set(cancelled))


async def _cancel_matching_tasks(
    runtime: Any,
    *,
    key: str,
    title: str,
    commitment_ids: Iterable[str],
    reason: str,
    hard_delete: bool,
) -> tuple[list[str], list[str]]:
    store = _task_store(runtime)
    if store is None:
        return [], []
    commitment_set = {str(item) for item in commitment_ids}
    cancelled: list[str] = []
    deleted: list[str] = []
    for task in await store.list_all(limit=1000):
        task_id = str(task.task_id)
        if str(task.commitment_id or "") not in commitment_set and not _matches_text(
            _task_text(task),
            key=key,
            title=title,
        ):
            continue
        if hard_delete:
            if await store.delete(task_id):
                deleted.append(task_id)
                _drop_live_task(runtime, task_id)
            continue
        task.stage = ExecutionStage.FAILED
        task.status = "cancelled"
        task.updated_at = datetime.now(timezone.utc)
        task.meta["cancel_reason"] = reason
        task.meta["cancelled_at"] = task.updated_at.isoformat()
        await store.save(task)
        cancelled.append(task_id)
        _drop_live_task(runtime, task_id)
    return sorted(set(cancelled)), sorted(set(deleted))


def _resolve_workspace_path(workspace_root: Path, workspace_path: str) -> Optional[Path]:
    raw = workspace_path.strip()
    if not raw:
        return None
    candidate = Path(raw).expanduser()
    if not candidate.is_absolute():
        candidate = workspace_root / candidate
    candidate = candidate.resolve()
    if not candidate.is_relative_to(workspace_root):
        raise ValueError(f"workspace_path must stay under managed workspace root {workspace_root}")
    return candidate


def _unique_compost_path(workspace_root: Path, key: str) -> Path:
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    base = workspace_root / "_compost" / f"{key}-{stamp}"
    candidate = base
    counter = 2
    while candidate.exists():
        candidate = Path(f"{base}-{counter}")
        counter += 1
    return candidate


def _write_receipt(
    *,
    workspace_root: Path,
    compost_path: Optional[Path],
    payload: dict[str, Any],
) -> Path:
    if compost_path is not None:
        target = compost_path / "COMPOST_RECEIPT.json"
    else:
        receipts = workspace_root / "_compost" / "receipts"
        receipts.mkdir(parents=True, exist_ok=True)
        target = receipts / f"{payload['project_key']}-{datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')}.json"
    target.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return target


def _update_receipt(receipt_path: Optional[Path], updates: dict[str, Any]) -> None:
    if receipt_path is None or not receipt_path.exists():
        return
    try:
        payload = json.loads(receipt_path.read_text(encoding="utf-8"))
    except Exception:
        payload = {}
    payload.update(updates)
    receipt_path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _write_salvage_index(
    *,
    compost_path: Optional[Path],
    project_title: str,
    project_key: str,
    reason: str,
    salvage_candidates: list[str],
    commitments_abandoned: list[str],
    schedules_cancelled: list[str],
    tasks_cancelled: list[str],
    tasks_deleted: list[str],
) -> Optional[Path]:
    if compost_path is None:
        return None
    target = compost_path / "SALVAGE_INDEX.md"
    lines = [
        f"# Compost Salvage Index: {project_title}",
        "",
        f"- project_key: `{project_key}`",
        f"- reason: {reason}",
        f"- commitments_abandoned: {len(commitments_abandoned)}",
        f"- schedules_cancelled: {len(schedules_cancelled)}",
        f"- tasks_cancelled: {len(tasks_cancelled)}",
        f"- tasks_deleted: {len(tasks_deleted)}",
        "",
        "## Salvage Candidates",
        "",
    ]
    if salvage_candidates:
        lines.extend(f"- `{item}`" for item in salvage_candidates)
    else:
        lines.append("- No file candidates were found in the composted workspace.")
    lines.extend(
        [
            "",
            "## Reuse Rule",
            "",
            "Review the receipt and name what failed before reusing material from this composted project.",
        ]
    )
    target.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return target


def _salvage_candidates(compost_path: Optional[Path]) -> list[str]:
    if compost_path is None or not compost_path.exists():
        return []
    candidates: list[str] = []
    useful_suffixes = {
        ".c",
        ".cc",
        ".cpp",
        ".h",
        ".hpp",
        ".md",
        ".py",
        ".qml",
        ".txt",
        ".ui",
    }
    for path in sorted(compost_path.rglob("*")):
        if not path.is_file() or path.name == "COMPOST_RECEIPT.json":
            continue
        if path.suffix.lower() not in useful_suffixes and path.name not in {"CMakeLists.txt", "Makefile"}:
            continue
        try:
            candidates.append(str(path.relative_to(compost_path)))
        except ValueError:
            continue
        if len(candidates) >= 50:
            break
    return candidates


def _capture_shadow_compost(
    runtime: Any,
    *,
    project_key: str,
    project_title: str,
    reason: str,
    compost_path: Optional[Path],
    receipt_path: Optional[Path],
    salvage_index_path: Optional[Path],
    salvage_candidates: list[str],
    commitments_abandoned: list[str],
    schedules_cancelled: list[str],
    tasks_cancelled: list[str],
    tasks_deleted: list[str],
) -> Optional[str]:
    shadow_registry = getattr(getattr(runtime, "ctx", None), "shadow_registry", None) or getattr(
        runtime,
        "shadow_registry",
        None,
    )
    capture = getattr(shadow_registry, "capture_project_compost", None)
    if not callable(capture):
        return None
    try:
        item = capture(
            {
                "project_key": project_key,
                "project_title": project_title,
                "reason": reason,
                "compost_path": str(compost_path) if compost_path else None,
                "receipt_path": str(receipt_path) if receipt_path else None,
                "salvage_index_path": str(salvage_index_path) if salvage_index_path else None,
                "salvage_candidates": salvage_candidates,
                "commitments_abandoned": commitments_abandoned,
                "schedules_cancelled": schedules_cancelled,
                "tasks_cancelled": tasks_cancelled,
                "tasks_deleted": tasks_deleted,
            }
        )
    except Exception:
        return None
    return str(getattr(item, "id", "") or "") or None


def _remove_matching_executive_goals(runtime: Any, *, key: str, title: str) -> None:
    executive = getattr(runtime, "executive", None)
    goals = list(getattr(executive, "_active_goals", []) or [])
    remove_goal = getattr(executive, "remove_goal", None)
    if not callable(remove_goal):
        return
    for goal in goals:
        if _matches_text(str(goal), key=key, title=title):
            remove_goal(goal)


def _drop_live_task(runtime: Any, task_id: str) -> None:
    baa = getattr(runtime, "baa", None)
    live_tasks = getattr(baa, "_live_tasks", None)
    if isinstance(live_tasks, dict):
        live_tasks.pop(task_id, None)
    futures = getattr(baa, "_futures", None)
    if isinstance(futures, dict):
        future = futures.pop(task_id, None)
        cancel = getattr(future, "cancel", None)
        if callable(cancel):
            cancel()


def _task_store(runtime: Any) -> Any:
    return getattr(getattr(runtime, "ctx", None), "task_store", None) or getattr(runtime, "task_store", None)


def _commitment_text(commitment: Any) -> str:
    return " ".join(
        [
            str(getattr(commitment, "content", "") or ""),
            " ".join(str(tag) for tag in getattr(commitment, "tags", []) or []),
            _dict_text(getattr(commitment, "meta", {}) or {}),
        ]
    )


def _schedule_text(item: Any) -> str:
    return " ".join(
        [
            str(getattr(item, "title", "") or ""),
            str(getattr(item, "description", "") or ""),
            str(getattr(item, "objective", "") or ""),
            " ".join(str(tag) for tag in getattr(item, "tags", []) or []),
            _dict_text(getattr(item, "meta", {}) or {}),
        ]
    )


def _task_text(task: Any) -> str:
    return " ".join(
        [
            str(getattr(task, "objective", "") or ""),
            str(getattr(task, "project_id", "") or ""),
            str(getattr(task, "commitment_id", "") or ""),
            _dict_text(getattr(task, "meta", {}) or {}),
        ]
    )


def _dict_text(value: Any) -> str:
    if isinstance(value, dict):
        return " ".join(f"{k} { _dict_text(v) }" for k, v in value.items())
    if isinstance(value, list):
        return " ".join(_dict_text(item) for item in value)
    return str(value or "")


def _matches_text(text: str, *, key: str, title: str) -> bool:
    normalized = _project_key(text)
    needles = {item for item in {key, _project_key(title)} if item}
    return any(item in normalized for item in needles)


def _project_key(value: str) -> str:
    return _NON_KEY_RE.sub("-", str(value or "").lower()).strip("-")


def _trace(runtime: Any, event: str, payload: dict[str, Any]) -> None:
    trace = getattr(runtime, "_trace", None)
    if callable(trace):
        trace(event, payload)
