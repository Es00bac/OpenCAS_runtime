"""Scheduling service that advances due items into OpenCAS execution."""

from __future__ import annotations

from datetime import datetime, timedelta, timezone
from typing import Any, Dict, List, Optional
from zoneinfo import ZoneInfo, ZoneInfoNotFoundError

from opencas.execution.models import RepairTask
from opencas.telemetry import EventKind, Tracer

from .models import (
    ScheduleAction,
    ScheduleItem,
    ScheduleKind,
    ScheduleRecurrence,
    ScheduleRun,
    ScheduleRunStatus,
    ScheduleStatus,
)
from .store import ScheduleStore


class ScheduleService:
    """Coordinates due schedule detection, execution, and recurrence advancement."""

    def __init__(
        self,
        store: ScheduleStore,
        runtime: Optional[Any] = None,
        tracer: Optional[Tracer] = None,
        default_timezone: str = "America/Denver",
    ) -> None:
        self.store = store
        self.runtime = runtime
        self.tracer = tracer
        self.default_timezone = default_timezone

    async def create_schedule(self, **kwargs: Any) -> ScheduleItem:
        item = ScheduleItem(**kwargs)
        item.next_run_at = item.next_run_at or item.start_at
        await self.store.save(item)
        self._trace("schedule_created", {"schedule_id": str(item.schedule_id), "kind": item.kind.value})
        return item

    async def process_due(self, now: Optional[datetime] = None, limit: int = 50) -> Dict[str, Any]:
        now = self._as_utc(now or datetime.now(timezone.utc))
        due = await self.store.list_due(now, limit=limit)
        processed = 0
        submitted = 0
        recorded = 0
        skipped = 0
        failed = 0
        for item in due:
            try:
                run = await self.trigger(item, now=now, manual=False)
                processed += 1
                if run.status == ScheduleRunStatus.SUBMITTED:
                    submitted += 1
                elif run.status == ScheduleRunStatus.RECORDED:
                    recorded += 1
                elif run.status == ScheduleRunStatus.SKIPPED:
                    skipped += 1
                elif run.status == ScheduleRunStatus.FAILED:
                    failed += 1
            except Exception as exc:
                failed += 1
                self._trace("schedule_process_error", {"schedule_id": str(item.schedule_id), "error": str(exc)})
        return {
            "processed": processed,
            "submitted": submitted,
            "recorded": recorded,
            "skipped": skipped,
            "failed": failed,
        }

    async def temporal_agenda(
        self,
        now: Optional[datetime] = None,
        *,
        horizon_hours: float = 24.0,
        upcoming_limit: int = 8,
        recent_limit: int = 8,
    ) -> Dict[str, Any]:
        """Return the agent-facing temporal agenda from durable schedule state."""
        now = self._as_utc(now or datetime.now(timezone.utc))
        horizon_hours = max(0.25, float(horizon_hours))
        upcoming_limit = max(1, int(upcoming_limit))
        recent_limit = max(1, int(recent_limit))
        horizon_end = now + timedelta(hours=horizon_hours)

        active_items = await self.store.list_items(status=ScheduleStatus.ACTIVE, limit=1000)
        due_items = [
            item
            for item in active_items
            if item.next_run_at is not None and self._as_utc(item.next_run_at) <= now
        ]
        future_items = [
            item
            for item in active_items
            if item.next_run_at is not None and now < self._as_utc(item.next_run_at) <= horizon_end
        ]
        due_items.sort(key=lambda item: (item.next_run_at or item.start_at, -item.priority))
        future_items.sort(key=lambda item: (item.next_run_at or item.start_at, -item.priority))
        recent_runs = await self.store.list_runs(limit=recent_limit)

        due_payload = [self._agenda_item_payload(item, now) for item in due_items[:upcoming_limit]]
        upcoming_payload = [
            self._agenda_item_payload(item, now) for item in future_items[:upcoming_limit]
        ]
        runs_payload = [self._agenda_run_payload(run) for run in recent_runs]
        next_item = due_payload[0] if due_payload else upcoming_payload[0] if upcoming_payload else None
        counts = {
            "active": len(active_items),
            "due_now": len(due_items),
            "upcoming": len(future_items),
            "recent_runs": len(runs_payload),
            "tasks": sum(1 for item in active_items if item.kind == ScheduleKind.TASK),
            "events": sum(1 for item in active_items if item.kind == ScheduleKind.EVENT),
        }
        return {
            "now": now.isoformat(),
            "horizon_hours": horizon_hours,
            "horizon_end": horizon_end.isoformat(),
            "timezone": self.default_timezone,
            "counts": counts,
            "due_now": due_payload,
            "upcoming": upcoming_payload,
            "recent_runs": runs_payload,
            "next": next_item,
            "summary": self._agenda_summary(counts, next_item),
        }

    async def trigger(
        self,
        item_or_id: ScheduleItem | str,
        now: Optional[datetime] = None,
        manual: bool = True,
    ) -> ScheduleRun:
        now = self._as_utc(now or datetime.now(timezone.utc))
        item = item_or_id if isinstance(item_or_id, ScheduleItem) else await self.store.get(str(item_or_id))
        if item is None:
            raise ValueError("schedule not found")
        scheduled_for = item.next_run_at or item.start_at
        if manual:
            scheduled_for = now
        elif scheduled_for <= now:
            scheduled_for = self.latest_due_occurrence(item, now) or scheduled_for
        run = ScheduleRun(
            schedule_id=item.schedule_id,
            scheduled_for=scheduled_for,
            started_at=now,
            status=ScheduleRunStatus.RECORDED,
            meta={"manual": manual, "action": item.action.value},
        )
        if await self._linked_commitment_finished(item):
            run.status = ScheduleRunStatus.SKIPPED
            run.finished_at = datetime.now(timezone.utc)
            run.meta["skip_reason"] = "linked_commitment_finished"
            item.status = ScheduleStatus.COMPLETED
            item.next_run_at = None
            await self.store.record_run(run)
            await self.store.save(item)
            self._trace(
                "schedule_triggered",
                {
                    "schedule_id": str(item.schedule_id),
                    "run_id": str(run.run_id),
                    "status": run.status.value,
                    "task_id": run.task_id,
                    "skip_reason": run.meta["skip_reason"],
                },
            )
            return run
        try:
            if item.action == ScheduleAction.SUBMIT_BAA:
                task = RepairTask(
                    objective=item.objective or item.title,
                    commitment_id=item.commitment_id,
                    project_id=item.plan_id,
                    meta={
                        **(item.meta or {}),
                        "source": "schedule",
                        "schedule_id": str(item.schedule_id),
                        "schedule_title": item.title,
                        "scheduled_for": scheduled_for.isoformat(),
                        "manual": manual,
                        "recurrence": item.recurrence.value,
                    },
                )
                if self.runtime is None or not getattr(self.runtime, "baa", None):
                    raise RuntimeError("runtime BAA is not available")
                await self.runtime.baa.submit(task)
                run.task_id = str(task.task_id)
                run.status = ScheduleRunStatus.SUBMITTED
            else:
                run.status = ScheduleRunStatus.RECORDED
            run.finished_at = datetime.now(timezone.utc)
        except Exception as exc:
            run.status = ScheduleRunStatus.FAILED
            run.error = str(exc)
            run.finished_at = datetime.now(timezone.utc)
        await self.store.record_run(run)
        if not manual:
            await self._advance_item(item, scheduled_for=scheduled_for, now=now)
        self._trace(
            "schedule_triggered",
            {
                "schedule_id": str(item.schedule_id),
                "run_id": str(run.run_id),
                "status": run.status.value,
                "task_id": run.task_id,
            },
        )
        return run

    async def _linked_commitment_finished(self, item: ScheduleItem) -> bool:
        commitment_id = str(item.commitment_id or "").strip()
        if not commitment_id or self.runtime is None:
            return False
        store = getattr(self.runtime, "commitment_store", None)
        if store is None:
            store = getattr(getattr(self.runtime, "ctx", None), "commitment_store", None)
        if store is None:
            executive = getattr(getattr(self.runtime, "ctx", None), "executive", None)
            store = getattr(executive, "commitment_store", None)
        if store is None:
            return False
        try:
            commitment = await store.get(commitment_id)
        except Exception:
            return False
        if commitment is None:
            return False
        status = str(getattr(getattr(commitment, "status", None), "value", getattr(commitment, "status", "")))
        return status in {"completed", "abandoned"}

    async def _advance_item(self, item: ScheduleItem, scheduled_for: datetime, now: datetime) -> None:
        item.last_run_at = now
        item.occurrence_count += 1
        next_run = self.compute_next_run(item, after=scheduled_for, now=now)
        if next_run is None:
            item.next_run_at = None
            item.status = ScheduleStatus.COMPLETED
        else:
            item.next_run_at = next_run
        await self.store.save(item)

    def compute_next_run(
        self,
        item: ScheduleItem,
        after: Optional[datetime] = None,
        now: Optional[datetime] = None,
    ) -> Optional[datetime]:
        now = self._as_utc(now or datetime.now(timezone.utc))
        after = self._as_utc(after or item.next_run_at or item.start_at)
        if item.max_occurrences is not None and item.occurrence_count >= item.max_occurrences:
            return None
        if item.recurrence == ScheduleRecurrence.NONE:
            return None

        candidate = self._next_candidate(item, after)
        if candidate is None:
            return None
        if item.end_at is not None and candidate > item.end_at:
            return None

        # Catch-up policy: run once if late, then continue with the next future occurrence.
        guard = 0
        while candidate <= now and guard < 1000:
            candidate = self._next_candidate(item, candidate)
            if candidate is None:
                return None
            if item.end_at is not None and candidate > item.end_at:
                return None
            guard += 1
        return candidate

    def occurrences_between(
        self,
        item: ScheduleItem,
        start: datetime,
        end: datetime,
        limit: int = 500,
    ) -> List[datetime]:
        start = self._as_utc(start)
        end = self._as_utc(end)
        if end < start:
            return []
        occurrences: List[datetime] = []
        candidate = item.start_at
        guard = 0
        while candidate and candidate <= end and len(occurrences) < limit and guard < limit * 4:
            if candidate >= start and (item.end_at is None or candidate <= item.end_at):
                occurrences.append(candidate)
            candidate = self._next_candidate(item, candidate)
            if candidate is None:
                break
            guard += 1
        return occurrences

    def latest_due_occurrence(self, item: ScheduleItem, now: datetime) -> Optional[datetime]:
        now = self._as_utc(now)
        candidate = item.next_run_at or item.start_at
        if candidate > now:
            return None
        latest = candidate
        guard = 0
        while guard < 1000:
            next_candidate = self._next_candidate(item, latest)
            if next_candidate is None or next_candidate > now:
                return latest
            if item.end_at is not None and next_candidate > item.end_at:
                return latest
            latest = next_candidate
            guard += 1
        return latest

    def _next_candidate(self, item: ScheduleItem, after: datetime) -> Optional[datetime]:
        after = self._as_utc(after)
        tz = self._zone(item.timezone)
        local_after = after.astimezone(tz)
        if item.recurrence == ScheduleRecurrence.INTERVAL_HOURS:
            return after + timedelta(hours=float(item.interval_hours or 1))
        if item.recurrence == ScheduleRecurrence.DAILY:
            return (local_after + timedelta(days=1)).astimezone(timezone.utc)
        if item.recurrence == ScheduleRecurrence.WEEKDAYS:
            weekdays = [0, 1, 2, 3, 4]
        elif item.recurrence == ScheduleRecurrence.WEEKLY:
            weekdays = item.weekdays or [local_after.weekday()]
        else:
            return None
        for offset in range(1, 8):
            candidate = local_after + timedelta(days=offset)
            if candidate.weekday() in weekdays:
                return candidate.astimezone(timezone.utc)
        return None

    async def calendar_range(
        self,
        start: datetime,
        end: datetime,
        status: ScheduleStatus = ScheduleStatus.ACTIVE,
    ) -> List[Dict[str, Any]]:
        items = await self.store.list_items(status=status, limit=1000)
        entries: List[Dict[str, Any]] = []
        for item in items:
            for occurrence in self.occurrences_between(item, start=start, end=end):
                entries.append(
                    {
                        "schedule_id": str(item.schedule_id),
                        "title": item.title,
                        "kind": item.kind.value,
                        "action": item.action.value,
                        "status": item.status.value,
                        "scheduled_for": occurrence.isoformat(),
                        "recurrence": item.recurrence.value,
                        "tags": item.tags,
                    }
                )
        entries.sort(key=lambda entry: entry["scheduled_for"])
        return entries

    def _agenda_item_payload(self, item: ScheduleItem, now: datetime) -> Dict[str, Any]:
        next_run_at = item.next_run_at or item.start_at
        next_run_utc = self._as_utc(next_run_at)
        return {
            "schedule_id": str(item.schedule_id),
            "title": item.title,
            "kind": item.kind.value,
            "action": item.action.value,
            "status": item.status.value,
            "next_run_at": next_run_utc.isoformat(),
            "seconds_until": round((next_run_utc - now).total_seconds(), 3),
            "is_due": next_run_utc <= now,
            "recurrence": item.recurrence.value,
            "priority": item.priority,
            "tags": item.tags,
            "objective": item.objective,
            "description": item.description,
            "commitment_id": item.commitment_id,
            "plan_id": item.plan_id,
            "meta": item.meta,
        }

    @staticmethod
    def _agenda_run_payload(run: ScheduleRun) -> Dict[str, Any]:
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

    @staticmethod
    def _agenda_summary(counts: Dict[str, int], next_item: Optional[Dict[str, Any]]) -> str:
        due = int(counts.get("due_now", 0))
        upcoming = int(counts.get("upcoming", 0))
        if due:
            return f"{due} schedule item(s) due now; next: {next_item.get('title') if next_item else 'unknown'}."
        if next_item:
            return f"No schedule items due now; next: {next_item.get('title')} at {next_item.get('next_run_at')}."
        if upcoming:
            return f"No schedule items due now; {upcoming} upcoming in the horizon."
        return "No active schedule items due or upcoming in the horizon."

    @staticmethod
    def _as_utc(value: datetime) -> datetime:
        if value.tzinfo is None:
            return value.replace(tzinfo=timezone.utc)
        return value.astimezone(timezone.utc)

    def _zone(self, name: str) -> ZoneInfo:
        try:
            return ZoneInfo(name or self.default_timezone)
        except ZoneInfoNotFoundError:
            return ZoneInfo(self.default_timezone)

    def _trace(self, event: str, payload: Dict[str, Any]) -> None:
        if self.tracer:
            self.tracer.log(EventKind.TOOL_CALL, f"ScheduleService: {event}", payload)
