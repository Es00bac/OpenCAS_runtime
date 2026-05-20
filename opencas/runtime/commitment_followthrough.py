"""Cognitive follow-through support for durable commitments."""

from __future__ import annotations

from typing import Any

from opencas.autonomy.commitment import Commitment
from opencas.autonomy.ongoing_support import is_ongoing_support_commitment_shape


async def seed_active_support_commitments(
    runtime: Any,
    *,
    limit: int = 50,
    source: str = "cognitive_maintenance_active_commitment_backfill",
) -> int:
    """Refresh cognitive-state hooks for active operator-support commitments."""

    store = _cognitive_store(runtime)
    commitment_store = getattr(runtime, "commitment_store", None)
    if store is None or commitment_store is None:
        return 0

    try:
        commitments = await commitment_store.list_active(limit=limit)
    except Exception as exc:
        _trace(runtime, "commitment_followthrough_backfill_list_failed", {"error": str(exc)})
        return 0

    count = 0
    for commitment in commitments:
        if await seed_commitment_cognitive_followthrough(
            runtime,
            commitment,
            source=source,
        ):
            count += 1
    return count


async def seed_commitment_cognitive_followthrough(
    runtime: Any,
    commitment: Commitment,
    *,
    source: str = "accepted_ongoing_support_commitment",
) -> bool:
    """Move accepted ongoing-support commitments into attention/prospective state.

    A durable commitment proves the promise exists, but attention/prospective
    memory are what make later reflection and daydream intake notice it.
    """

    if not is_ongoing_support_commitment(commitment):
        return False
    store = _cognitive_store(runtime)
    if store is None:
        return False

    evidence_refs = [f"commitment:{commitment.commitment_id}"]
    priority = _commitment_priority(commitment)
    attention_strength = max(0.84, min(0.95, 0.74 + priority * 0.02))
    working_priority = max(0.78, min(0.95, priority / 10.0 + 0.08))
    label = f"Mission support: {commitment.content}"
    try:
        await store.upsert_attention(
            label,
            strength=attention_strength,
            source=source,
            evidence_refs=evidence_refs,
        )
        await store.upsert_working_memory(
            f"commitment:{str(commitment.commitment_id)[:8]}",
            (
                f"Active ongoing support commitment: {commitment.content}. "
                "Use planning, schedules, tools, research, questions, documents, "
                "and artifacts as needed."
            ),
            priority=working_priority,
            source=source,
            evidence_refs=evidence_refs,
        )
        await store.upsert_prospective_memory(
            f"Proactively follow through on commitment: {commitment.content}",
            condition=(
                "When reflection, planning, scheduling, or daydreaming can advance "
                "the support mission, use available tools or ask the next needed question."
            ),
            proof_ref="",
            confidence=max(0.78, min(0.9, priority / 10.0 + 0.04)),
            evidence_refs=evidence_refs,
        )
        _trace(
            runtime,
            "commitment_cognitive_followthrough_seeded",
            {
                "commitment_id": str(commitment.commitment_id),
                "source": source,
                "priority": commitment.priority,
            },
        )
        return True
    except Exception as exc:
        _trace(
            runtime,
            "commitment_cognitive_followthrough_failed",
            {
                "commitment_id": str(commitment.commitment_id),
                "error": str(exc),
            },
        )
        return False


def is_ongoing_support_commitment(commitment: Commitment) -> bool:
    """Return true when an active commitment should bias follow-through cognition."""

    return is_ongoing_support_commitment_shape(commitment, require_active=True)


def _cognitive_store(runtime: Any) -> Any:
    return getattr(runtime, "cognitive_state_store", None) or getattr(
        getattr(runtime, "ctx", None),
        "cognitive_state_store",
        None,
    )


def _commitment_priority(commitment: Commitment) -> float:
    try:
        return max(1.0, min(10.0, float(commitment.priority)))
    except Exception:
        return 5.0


def _trace(runtime: Any, event: str, payload: dict[str, Any]) -> None:
    trace = getattr(runtime, "_trace", None)
    if callable(trace):
        try:
            trace(event, payload)
        except Exception:
            pass
