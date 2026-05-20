"""Forward-only promise-to-proof plumbing."""

from __future__ import annotations

import inspect
from datetime import datetime, timezone
from typing import TYPE_CHECKING, Any

from opencas.autonomy.commitment import (
    Commitment,
    commitment_operator_snapshot,
)
from .models import ProofClaim, ProofClaimType, ProofEvidenceKind

if TYPE_CHECKING:
    from opencas.execution.models import ExecutionReceipt, RepairTask

OPERATOR_PROMISE_SOURCES = frozenset(
    {
        "assistant_response",
        "workflow_create_commitment",
    }
)


def is_operator_facing_promise_commitment(commitment: Commitment) -> bool:
    """Return True only for commitments that came from operator-facing promise paths."""

    meta = dict(commitment.meta or {})
    source = str(meta.get("source") or "").strip()
    return source in OPERATOR_PROMISE_SOURCES


async def attach_operator_promise_claim(
    runtime: Any,
    commitment: Commitment,
    *,
    subject: str,
    source: str,
    evidence_summary: str,
    meta: dict[str, Any] | None = None,
) -> ProofClaim | None:
    """Create and bind a PROMISE claim for a forward operator-facing commitment."""

    commitment.meta = dict(commitment.meta or {})
    commitment.meta.setdefault("source", source)
    if not is_operator_facing_promise_commitment(commitment):
        return None
    existing_claim_id = str(commitment.meta.get("claim_id") or "").strip()
    if existing_claim_id:
        return None

    proof_chain = getattr(runtime, "proof_chain", None)
    record_claim = getattr(proof_chain, "record_claim", None)
    link_evidence = getattr(proof_chain, "link_evidence", None)
    if not callable(record_claim) or not callable(link_evidence):
        return None

    claim_result = record_claim(
        claim_type=ProofClaimType.PROMISE,
        claim=commitment.content,
        subject=subject,
        meta={
            "commitment_id": str(commitment.commitment_id),
            "source": source,
            **dict(meta or {}),
        },
    )
    if not inspect.isawaitable(claim_result):
        return None
    claim = await claim_result
    commitment.meta["claim_id"] = str(claim.claim_id)

    evidence_result = link_evidence(
        claim.claim_id,
        evidence_kind=ProofEvidenceKind.COMMITMENT,
        evidence_id=f"commitment:{commitment.commitment_id}",
        summary=evidence_summary,
        supports_claim=True,
        meta={"source": source, **dict(meta or {})},
    )
    if inspect.isawaitable(evidence_result):
        await evidence_result
    return claim


def receipt_meta_from_task(task: "RepairTask") -> dict[str, Any]:
    """Copy bounded proof-join metadata from a task into its execution receipt."""

    source = dict(task.meta or {})
    meta: dict[str, Any] = {}
    for key in (
        "claim_id",
        "schedule_id",
        "commitment_id",
        "source",
        "schedule_title",
        "scheduled_for",
        "manual",
        "recurrence",
    ):
        value = source.get(key)
        if value not in (None, "", [], {}):
            meta[key] = value
    if task.commitment_id:
        meta.setdefault("commitment_id", str(task.commitment_id))
    if task.project_id:
        meta.setdefault("project_id", str(task.project_id))
    return meta


async def link_receipt_to_promise_claim(runtime: Any, receipt: "ExecutionReceipt") -> None:
    """Attach receipt evidence to a PROMISE claim when a receipt carries claim_id."""

    claim_id = str((receipt.meta or {}).get("claim_id") or "").strip()
    if not claim_id:
        return
    proof_chain = getattr(runtime, "proof_chain", None)
    link_evidence = getattr(proof_chain, "link_evidence", None)
    if not callable(link_evidence):
        return
    evidence_result = link_evidence(
        claim_id,
        evidence_kind=ProofEvidenceKind.RECEIPT,
        evidence_id=f"receipt:{receipt.receipt_id}",
        summary=(
            "Execution receipt completed for a promise-linked task."
            if receipt.success
            else "Execution receipt failed for a promise-linked task."
        ),
        supports_claim=bool(receipt.success),
        meta={
            "task_id": str(receipt.task_id),
            "receipt_id": str(receipt.receipt_id),
            "objective": receipt.objective,
            **dict(receipt.meta or {}),
        },
    )
    if inspect.isawaitable(evidence_result):
        await evidence_result


async def lookup_promise_chains(
    runtime: Any,
    *,
    since: datetime | str | None = None,
    limit: int = 50,
) -> dict[str, Any]:
    """Return joined promise, commitment, schedule, task, receipt, and claim records."""

    parsed_since = _parse_since(since)
    limit = max(1, min(200, int(limit)))
    commitment_store = getattr(runtime, "commitment_store", None)
    if commitment_store is None:
        return {"available": False, "reason": "commitment_store_unavailable", "count": 0, "items": []}
    list_since = getattr(commitment_store, "list_since", None)
    if not callable(list_since):
        return {"available": False, "reason": "commitment_store_missing_list_since", "count": 0, "items": []}

    commitments = await list_since(since=parsed_since, limit=limit * 4)
    commitments = [
        commitment
        for commitment in commitments
        if str((commitment.meta or {}).get("claim_id") or "").strip()
    ][:limit]

    schedule_store = _schedule_store(runtime)
    task_store = _task_store(runtime)
    receipt_store = _receipt_store(runtime)
    proof_store = getattr(runtime, "proof_store", None)
    schedules = await _all_schedules(schedule_store)
    schedule_runs = await _runs_by_schedule(schedule_store, schedules)
    items: list[dict[str, Any]] = []
    for commitment in commitments:
        claim_id = str((commitment.meta or {}).get("claim_id") or "").strip()
        claim = await proof_store.get_claim(claim_id) if proof_store is not None else None
        matching_schedules = [
            schedule
            for schedule in schedules
            if str(schedule.commitment_id or "").strip() == str(commitment.commitment_id)
            or str((schedule.meta or {}).get("claim_id") or "").strip() == claim_id
        ]
        matching_runs = [
            run
            for schedule in matching_schedules
            for run in schedule_runs.get(str(schedule.schedule_id), [])
        ]
        task_ids = [str(run.task_id) for run in matching_runs if str(run.task_id or "").strip()]
        task_ids.extend(str(task_id) for task_id in commitment.linked_task_ids)
        task_ids = list(dict.fromkeys(task_ids))
        tasks = [task for task in [await _get_task(task_store, task_id) for task_id in task_ids] if task is not None]
        receipts = [
            receipt
            for task_id in task_ids
            for receipt in await _receipts_for_task(receipt_store, task_id)
        ]
        items.append(
            {
                "claim": claim.model_dump(mode="json") if claim is not None else {"claim_id": claim_id},
                "commitment": commitment_operator_snapshot(commitment, include_meta=True),
                "schedules": [schedule.model_dump(mode="json") for schedule in matching_schedules],
                "runs": [run.model_dump(mode="json") for run in matching_runs],
                "tasks": [task.model_dump(mode="json") for task in tasks],
                "receipts": [receipt.model_dump(mode="json") for receipt in receipts],
            }
        )

    return {
        "available": True,
        "since": parsed_since.isoformat() if parsed_since else None,
        "count": len(items),
        "items": items,
    }


def _parse_since(value: datetime | str | None) -> datetime | None:
    if value is None:
        return None
    if isinstance(value, datetime):
        parsed = value
    else:
        text = str(value).strip()
        if not text:
            return None
        parsed = datetime.fromisoformat(text.replace("Z", "+00:00"))
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=timezone.utc)
    return parsed.astimezone(timezone.utc)


def _schedule_store(runtime: Any) -> Any:
    return getattr(getattr(runtime, "ctx", None), "schedule_store", None)


def _task_store(runtime: Any) -> Any:
    return getattr(getattr(runtime, "baa", None), "store", None) or getattr(
        getattr(runtime, "ctx", None),
        "tasks",
        None,
    )


def _receipt_store(runtime: Any) -> Any:
    return getattr(getattr(runtime, "ctx", None), "receipt_store", None) or getattr(
        runtime,
        "receipt_store",
        None,
    )


async def _all_schedules(schedule_store: Any) -> list[Any]:
    list_items = getattr(schedule_store, "list_items", None)
    if not callable(list_items):
        return []
    return await list_items(limit=1000)


async def _runs_by_schedule(schedule_store: Any, schedules: list[Any]) -> dict[str, list[Any]]:
    list_runs = getattr(schedule_store, "list_runs", None)
    if not callable(list_runs):
        return {}
    runs_by_schedule: dict[str, list[Any]] = {}
    for schedule in schedules:
        schedule_id = str(schedule.schedule_id)
        runs_by_schedule[schedule_id] = await list_runs(schedule_id=schedule_id, limit=50)
    return runs_by_schedule


async def _get_task(task_store: Any, task_id: str) -> Any:
    get = getattr(task_store, "get", None)
    if not callable(get):
        return None
    return await get(task_id)


async def _receipts_for_task(receipt_store: Any, task_id: str) -> list[ExecutionReceipt]:
    list_by_task = getattr(receipt_store, "list_by_task", None)
    if not callable(list_by_task):
        return []
    return await list_by_task(task_id, limit=50)
