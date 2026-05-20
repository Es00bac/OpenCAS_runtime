from __future__ import annotations

from dataclasses import asdict, is_dataclass
from datetime import datetime, timezone
from typing import Any

from opencas.autonomy.commitment import CommitmentStatus
from opencas.execution.models import ExecutionStage
from opencas.recovery.models import RecoveryCandidate, RecoveryCandidateKind


class RecoveryScanner:
    def __init__(
        self,
        *,
        commitment_store: Any = None,
        task_store: Any = None,
        harness_store: Any = None,
        schedule_store: Any = None,
        work_store: Any = None,
    ) -> None:
        self.commitment_store = commitment_store
        self.task_store = task_store
        self.harness_store = harness_store
        self.schedule_store = schedule_store
        self.work_store = work_store

    async def scan(self, *, limit_per_store: int = 100) -> list[RecoveryCandidate]:
        candidates: list[RecoveryCandidate] = []
        candidates.extend(await self._scan_commitments(limit_per_store))
        candidates.extend(await self._scan_tasks(limit_per_store))
        candidates.extend(await self._scan_objective_loops(limit_per_store))
        candidates.extend(await self._scan_schedules(limit_per_store))
        candidates.extend(await self._scan_work_objects(limit_per_store))
        return candidates

    async def _scan_commitments(self, limit: int) -> list[RecoveryCandidate]:
        if self.commitment_store is None:
            return []
        candidates: list[RecoveryCandidate] = []
        for commitment in await _call(self.commitment_store, "list_active", limit=limit):
            candidates.append(_commitment_candidate(commitment, RecoveryCandidateKind.ACTIVE_COMMITMENT))
        for commitment in await _call(self.commitment_store, "list_by_status", CommitmentStatus.BLOCKED, limit=limit):
            candidates.append(_commitment_candidate(commitment, RecoveryCandidateKind.BLOCKED_COMMITMENT))
        return candidates

    async def _scan_tasks(self, limit: int) -> list[RecoveryCandidate]:
        if self.task_store is None:
            return []
        tasks = await _call(self.task_store, "list_all", limit=limit)
        candidates: list[RecoveryCandidate] = []
        for task in tasks:
            if _value(getattr(task, "stage", None)) != ExecutionStage.FAILED.value:
                continue
            salvage = await _call(self.task_store, "get_latest_salvage_packet", getattr(task, "task_id"))
            payload = _payload(task)
            if salvage is not None:
                payload["salvage"] = _payload(salvage)
            task_id = str(getattr(task, "task_id"))
            candidates.append(
                RecoveryCandidate(
                    candidate_id=f"task:{task_id}",
                    kind=RecoveryCandidateKind.FAILED_TASK,
                    title=getattr(task, "description", getattr(task, "objective", "Failed repair task")),
                    status="failed",
                    updated_at=getattr(task, "updated_at", None),
                    evidence_refs=[f"task:{task_id}"],
                    payload=payload,
                )
            )
        return candidates

    async def _scan_objective_loops(self, limit: int) -> list[RecoveryCandidate]:
        if self.harness_store is None:
            return []
        loops = await _first_call(self.harness_store, (("list_loops", (), {"limit": limit}), ("list_objective_loops", (), {"limit": limit})))
        candidates: list[RecoveryCandidate] = []
        for loop in loops:
            status = str(_value(getattr(loop, "status", "")))
            if status not in {"paused", "failed"}:
                continue
            loop_id = str(getattr(loop, "loop_id"))
            kind = (
                RecoveryCandidateKind.PAUSED_OBJECTIVE_LOOP
                if status == "paused"
                else RecoveryCandidateKind.FAILED_OBJECTIVE_LOOP
            )
            candidates.append(
                RecoveryCandidate(
                    candidate_id=f"loop:{loop_id}",
                    kind=kind,
                    title=getattr(loop, "title", "Objective loop"),
                    status=status,
                    updated_at=getattr(loop, "updated_at", None),
                    evidence_refs=[f"loop:{loop_id}"],
                    payload=_payload(loop),
                )
            )
        return candidates

    async def _scan_schedules(self, limit: int) -> list[RecoveryCandidate]:
        if self.schedule_store is None:
            return []
        schedules = await _first_call(
            self.schedule_store,
            (
                ("list_items", (), {"limit": limit}),
                ("list_schedules", (), {"limit": limit}),
                ("list_all", (), {"limit": limit}),
                ("list_due", (datetime.now(timezone.utc),), {"limit": limit}),
            ),
        )
        candidates: list[RecoveryCandidate] = []
        for schedule in schedules:
            if not _is_scheduled_or_recurring(schedule):
                continue
            schedule_id = _schedule_id(schedule)
            evidence_ref = f"schedule:{schedule_id}"
            candidates.append(
                RecoveryCandidate(
                    candidate_id=evidence_ref,
                    kind=RecoveryCandidateKind.RECURRING_SCHEDULE,
                    title=_first_present(schedule, "title", "name", "description") or "Recurring schedule",
                    status=str(_value(getattr(schedule, "status", "scheduled"))),
                    updated_at=getattr(schedule, "updated_at", None),
                    evidence_refs=[evidence_ref],
                    payload=_payload(schedule),
                )
            )
        return candidates

    async def _scan_work_objects(self, limit: int) -> list[RecoveryCandidate]:
        if self.work_store is None:
            return []
        works = await _call(self.work_store, "list_blocked", limit=limit)
        return [
            RecoveryCandidate(
                candidate_id=f"work:{getattr(work, 'work_id')}",
                kind=RecoveryCandidateKind.BLOCKED_WORK_OBJECT,
                title=getattr(work, "title", "Blocked work object"),
                status="blocked",
                updated_at=getattr(work, "updated_at", None),
                evidence_refs=[f"work:{getattr(work, 'work_id')}"],
                payload=_payload(work),
            )
            for work in works
        ]


async def _call(obj: Any, method: str, *args: Any, **kwargs: Any) -> Any:
    fn = getattr(obj, method, None)
    if fn is None:
        return []
    return await fn(*args, **kwargs)


async def _first_call(obj: Any, calls: tuple[tuple[str, tuple[Any, ...], dict[str, Any]], ...]) -> Any:
    for method, args, kwargs in calls:
        fn = getattr(obj, method, None)
        if fn is None:
            continue
        return await fn(*args, **kwargs)
    return []


def _commitment_candidate(commitment: Any, kind: RecoveryCandidateKind) -> RecoveryCandidate:
    commitment_id = str(getattr(commitment, "commitment_id"))
    return RecoveryCandidate(
        candidate_id=f"commitment:{commitment_id}",
        kind=kind,
        title=getattr(commitment, "content", "Commitment"),
        status=str(_value(getattr(commitment, "status", "active"))),
        updated_at=getattr(commitment, "updated_at", None),
        evidence_refs=[f"commitment:{commitment_id}"],
        payload=_payload(commitment),
    )


def _payload(value: Any) -> dict[str, Any]:
    if value is None:
        return {}
    if hasattr(value, "model_dump"):
        return value.model_dump(mode="json")
    if is_dataclass(value):
        return asdict(value)
    if isinstance(value, dict):
        return dict(value)
    if hasattr(value, "__dict__"):
        return dict(value.__dict__)
    return {"value": str(value)}


def _is_scheduled_or_recurring(schedule: Any) -> bool:
    recurrence = str(_value(getattr(schedule, "recurrence", "")))
    return bool(getattr(schedule, "next_run_at", None)) or recurrence not in {"", "none", "None"}


def _schedule_id(schedule: Any) -> str:
    return str(_first_present(schedule, "schedule_id", "job_id", "id"))


def _first_present(obj: Any, *names: str) -> Any:
    for name in names:
        value = getattr(obj, name, None)
        if value:
            return value
    return None


def _value(value: Any) -> Any:
    return getattr(value, "value", value)
