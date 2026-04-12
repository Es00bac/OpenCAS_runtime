"""Schedule API routes for scheduled tasks, events, and recurring work."""

from __future__ import annotations

from datetime import datetime, timedelta, timezone
from typing import Any, Dict, List, Optional

from fastapi import APIRouter
from pydantic import BaseModel, Field

from opencas.scheduling import (
    ScheduleAction,
    ScheduleItem,
    ScheduleKind,
    ScheduleRecurrence,
    ScheduleRun,
    ScheduleStatus,
)


class ScheduleItemRequest(BaseModel):
    kind: ScheduleKind = ScheduleKind.TASK
    action: Optional[ScheduleAction] = None
    title: str
    description: str = ""
    objective: Optional[str] = None
    start_at: datetime
    end_at: Optional[datetime] = None
    timezone: str = "America/Denver"
    recurrence: ScheduleRecurrence = ScheduleRecurrence.NONE
    interval_hours: Optional[float] = None
    weekdays: List[int] = Field(default_factory=list)
    max_occurrences: Optional[int] = None
    priority: float = 5.0
    tags: List[str] = Field(default_factory=list)
    commitment_id: Optional[str] = None
    plan_id: Optional[str] = None
    meta: Dict[str, Any] = Field(default_factory=dict)


class ScheduleItemUpdateRequest(BaseModel):
    status: Optional[ScheduleStatus] = None
    kind: Optional[ScheduleKind] = None
    action: Optional[ScheduleAction] = None
    title: Optional[str] = None
    description: Optional[str] = None
    objective: Optional[str] = None
    start_at: Optional[datetime] = None
    end_at: Optional[datetime] = None
    timezone: Optional[str] = None
    recurrence: Optional[ScheduleRecurrence] = None
    interval_hours: Optional[float] = None
    weekdays: Optional[List[int]] = None
    max_occurrences: Optional[int] = None
    priority: Optional[float] = None
    tags: Optional[List[str]] = None
    commitment_id: Optional[str] = None
    plan_id: Optional[str] = None
    meta: Optional[Dict[str, Any]] = None


def _item_to_dict(item: ScheduleItem) -> Dict[str, Any]:
    return {
        "schedule_id": str(item.schedule_id),
        "created_at": item.created_at.isoformat(),
        "updated_at": item.updated_at.isoformat(),
        "kind": item.kind.value,
        "action": item.action.value,
        "status": item.status.value,
        "title": item.title,
        "description": item.description,
        "objective": item.objective,
        "start_at": item.start_at.isoformat(),
        "end_at": item.end_at.isoformat() if item.end_at else None,
        "timezone": item.timezone,
        "next_run_at": item.next_run_at.isoformat() if item.next_run_at else None,
        "last_run_at": item.last_run_at.isoformat() if item.last_run_at else None,
        "recurrence": item.recurrence.value,
        "interval_hours": item.interval_hours,
        "weekdays": item.weekdays,
        "max_occurrences": item.max_occurrences,
        "occurrence_count": item.occurrence_count,
        "priority": item.priority,
        "tags": item.tags,
        "commitment_id": item.commitment_id,
        "plan_id": item.plan_id,
        "meta": item.meta,
    }


def _run_to_dict(run: ScheduleRun) -> Dict[str, Any]:
    return {
        "run_id": str(run.run_id),
        "schedule_id": str(run.schedule_id),
        "scheduled_for": run.scheduled_for.isoformat(),
        "started_at": run.started_at.isoformat(),
        "finished_at": run.finished_at.isoformat() if run.finished_at else None,
        "status": run.status.value,
        "task_id": run.task_id,
        "error": run.error,
        "meta": run.meta,
    }


def _parse_dt(value: Optional[str], default: datetime) -> datetime:
    if not value:
        return default
    parsed = datetime.fromisoformat(value)
    if parsed.tzinfo is None:
        return parsed.replace(tzinfo=timezone.utc)
    return parsed.astimezone(timezone.utc)


def build_schedule_router(runtime: Any) -> APIRouter:
    router = APIRouter(prefix="/api/schedule", tags=["schedule"])

    def _service():
        return getattr(runtime, "schedule_service", None) or getattr(runtime.ctx, "schedule_service", None)

    def _store():
        return getattr(runtime.ctx, "schedule_store", None)

    @router.get("/items")
    async def list_items(
        status: Optional[str] = "active",
        kind: Optional[str] = None,
        limit: int = 100,
    ) -> Dict[str, Any]:
        store = _store()
        if store is None:
            return {"count": 0, "items": []}
        status_filter = ScheduleStatus(status) if status else None
        kind_filter = ScheduleKind(kind) if kind else None
        items = await store.list_items(status=status_filter, kind=kind_filter, limit=limit)
        return {"count": len(items), "items": [_item_to_dict(item) for item in items]}

    @router.post("/items")
    async def create_item(payload: ScheduleItemRequest) -> Dict[str, Any]:
        service = _service()
        if service is None:
            return {"created": False, "error": "Schedule service not available"}
        action = payload.action or (
            ScheduleAction.SUBMIT_BAA if payload.kind == ScheduleKind.TASK else ScheduleAction.REMINDER_ONLY
        )
        data = payload.model_dump()
        data["action"] = action
        item = await service.create_schedule(**data)
        return {"created": True, "item": _item_to_dict(item)}

    @router.get("/items/{schedule_id}")
    async def get_item(schedule_id: str) -> Dict[str, Any]:
        store = _store()
        if store is None:
            return {"found": False, "error": "Schedule store not available"}
        item = await store.get(schedule_id)
        if item is None:
            return {"found": False}
        runs = await store.list_runs(schedule_id=schedule_id, limit=20)
        return {"found": True, "item": _item_to_dict(item), "runs": [_run_to_dict(run) for run in runs]}

    @router.patch("/items/{schedule_id}")
    async def update_item(schedule_id: str, payload: ScheduleItemUpdateRequest) -> Dict[str, Any]:
        store = _store()
        if store is None:
            return {"found": False, "error": "Schedule store not available"}
        item = await store.get(schedule_id)
        if item is None:
            return {"found": False}
        updates = payload.model_dump(exclude_unset=True)
        for key, value in updates.items():
            setattr(item, key, value)
        if any(key in updates for key in ("start_at", "recurrence", "interval_hours", "weekdays", "end_at", "max_occurrences")):
            item.next_run_at = item.start_at if item.status == ScheduleStatus.ACTIVE else None
        await store.save(item)
        return {"found": True, "item": _item_to_dict(item)}

    @router.delete("/items/{schedule_id}")
    async def cancel_item(schedule_id: str) -> Dict[str, Any]:
        store = _store()
        if store is None:
            return {"cancelled": False, "error": "Schedule store not available"}
        return {"cancelled": await store.cancel(schedule_id)}

    @router.get("/calendar")
    async def calendar(start: Optional[str] = None, end: Optional[str] = None) -> Dict[str, Any]:
        service = _service()
        if service is None:
            return {"count": 0, "items": []}
        now = datetime.now(timezone.utc)
        start_dt = _parse_dt(start, now - timedelta(days=1))
        end_dt = _parse_dt(end, now + timedelta(days=30))
        items = await service.calendar_range(start=start_dt, end=end_dt)
        return {"count": len(items), "items": items, "start": start_dt.isoformat(), "end": end_dt.isoformat()}

    @router.get("/runs")
    async def list_runs(schedule_id: Optional[str] = None, limit: int = 100) -> Dict[str, Any]:
        store = _store()
        if store is None:
            return {"count": 0, "items": []}
        runs = await store.list_runs(schedule_id=schedule_id, limit=limit)
        return {"count": len(runs), "items": [_run_to_dict(run) for run in runs]}

    @router.post("/items/{schedule_id}/trigger")
    async def trigger(schedule_id: str) -> Dict[str, Any]:
        service = _service()
        if service is None:
            return {"triggered": False, "error": "Schedule service not available"}
        run = await service.trigger(schedule_id, manual=True)
        return {"triggered": True, "run": _run_to_dict(run)}

    return router
