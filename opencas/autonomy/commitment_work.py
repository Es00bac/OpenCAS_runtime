"""Helpers for keeping commitments and linked work objects in sync."""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Any

from .commitment import Commitment, CommitmentStatus
from .models import WorkObject, WorkStage

_EXECUTABLE_STAGES = {
    WorkStage.MICRO_TASK,
    WorkStage.PROJECT_SEED,
    WorkStage.PROJECT,
}


async def settle_linked_work_for_terminal_commitment(
    runtime: Any,
    commitment: Commitment,
    status: CommitmentStatus,
    *,
    completion_evidence: str | None = None,
) -> list[str]:
    """Demote executable work objects when their owning commitment is terminal."""

    if status not in {CommitmentStatus.COMPLETED, CommitmentStatus.ABANDONED}:
        return []

    ctx = getattr(runtime, "ctx", None)
    work_store = getattr(ctx, "work_store", None)
    if work_store is None:
        return []

    get_work = getattr(work_store, "get", None)
    save_work = getattr(work_store, "save", None)
    if not callable(get_work) or not callable(save_work):
        return []

    candidates: list[WorkObject] = []
    seen_ids: set[str] = set()
    for work_id in list(getattr(commitment, "linked_work_ids", []) or []):
        work = await get_work(str(work_id))
        if work is None:
            continue
        work_key = str(work.work_id)
        if work_key in seen_ids:
            continue
        seen_ids.add(work_key)
        candidates.append(work)

    list_by_commitment = getattr(work_store, "list_by_commitment", None)
    if callable(list_by_commitment):
        try:
            for work in await list_by_commitment(str(commitment.commitment_id), limit=500):
                work_key = str(work.work_id)
                if work_key in seen_ids:
                    continue
                seen_ids.add(work_key)
                candidates.append(work)
        except (AttributeError, TypeError):
            pass

    settled: list[str] = []
    for work in candidates:
        if not _work_needs_terminal_settlement(work):
            continue
        updated = work.model_copy(deep=True)
        updated.stage = WorkStage.ARTIFACT if status == CommitmentStatus.COMPLETED else WorkStage.NOTE
        updated.blocked_by = []
        updated.updated_at = datetime.now(timezone.utc)
        updated.meta = dict(updated.meta or {})
        updated.meta["terminal_commitment_id"] = str(commitment.commitment_id)
        updated.meta["terminal_commitment_status"] = status.value
        updated.meta["settled_from_commitment_at"] = updated.updated_at.isoformat()
        if completion_evidence:
            updated.meta["completion_evidence"] = str(completion_evidence)[:4000]

        await save_work(updated)
        settled.append(str(updated.work_id))
        _reconcile_live_executive_work(runtime, updated)

    return settled


def _work_needs_terminal_settlement(work: WorkObject) -> bool:
    stage = work.stage if isinstance(work.stage, WorkStage) else WorkStage(str(work.stage))
    return stage in _EXECUTABLE_STAGES or bool(work.blocked_by)


def _reconcile_live_executive_work(runtime: Any, work: WorkObject) -> None:
    executive = getattr(runtime, "executive", None)
    reconcile = getattr(executive, "reconcile_work_update", None)
    if callable(reconcile):
        reconcile(work)
