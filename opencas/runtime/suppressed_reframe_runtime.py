"""Runtime bridge for suppression metadata rejected during executive boot."""

from __future__ import annotations

import hashlib
import json
from typing import Any

from opencas.telemetry import EventKind
from opencas.thread_registry import BeadSourceKind, ThreadStatus


def _stable_rejection_ref(rejection: dict[str, Any]) -> str:
    metadata = rejection.get("metadata") if isinstance(rejection.get("metadata"), dict) else {}
    duplicate_of = str(metadata.get("duplicate_of_task_id") or "unknown").strip() or "unknown"
    payload = {
        "goal": rejection.get("goal"),
        "reason": rejection.get("reason"),
        "source_artifact": rejection.get("source_artifact"),
        "metadata": metadata,
    }
    digest = hashlib.sha256(
        json.dumps(payload, sort_keys=True, default=str).encode("utf-8")
    ).hexdigest()[:12]
    return f"suppressed_reframe:loaded:{duplicate_of}:{digest}"


def _trace(tracer: Any, message: str, payload: dict[str, Any]) -> None:
    if tracer is None:
        return
    trace = getattr(tracer, "trace", None)
    if callable(trace):
        trace(message, payload)
        return
    log = getattr(tracer, "log", None)
    if callable(log):
        log(EventKind.TOM_EVAL, f"SuppressedReframeRuntime: {message}", payload)


async def record_loaded_suppressed_reframe_thread_beads(
    *,
    executive: Any,
    service: Any,
    tracer: Any | None = None,
) -> dict[str, Any]:
    """Persist loaded executive suppression rejections as audit-only thread beads."""
    consumer = getattr(executive, "consume_suppressed_parked_goal_rejections", None)
    if service is None or not callable(consumer):
        return {
            "available": service is not None,
            "candidate_count": 0,
            "recorded_count": 0,
            "failed_count": 0,
        }

    rejections = [
        rejection
        for rejection in consumer()
        if isinstance(rejection, dict) and str(rejection.get("goal") or "").strip()
    ]
    if not rejections:
        return {
            "available": True,
            "candidate_count": 0,
            "recorded_count": 0,
            "failed_count": 0,
        }

    try:
        anchor = await service.ensure_thread_anchor(
            title="Suppressed recursive reframes",
            kind="suppressed_reframe",
            status=ThreadStatus.PERIPHERAL,
            anchor_id="suppressed-recursive-reframes",
        )
    except Exception as exc:
        _trace(
            tracer,
            "suppressed_reframe_loaded_anchor_failed",
            {"candidate_count": len(rejections), "error": str(exc)},
        )
        return {
            "available": True,
            "candidate_count": len(rejections),
            "recorded_count": 0,
            "failed_count": len(rejections),
        }

    recorded_count = 0
    failed_count = 0
    for rejection in rejections:
        source_ref = _stable_rejection_ref(rejection)
        content = json.dumps(
            {
                "source_ref": source_ref,
                "objective": rejection.get("goal"),
                "canonical_artifact": rejection.get("source_artifact"),
                "details": rejection.get("metadata") if isinstance(rejection.get("metadata"), dict) else {},
                "source": "executive_snapshot_load",
                "consumer": "audit-only thread-registry query",
                "instruction": (
                    "This bead records recursive suppression metadata removed from the "
                    "active parked-goal surface. Do not render it as an active continuity cue."
                ),
            },
            sort_keys=True,
            default=str,
        )
        try:
            await service.create_candidate_bead(
                thread_anchor_id=anchor.anchor_id,
                title="Suppressed recursive reframe",
                summary=(
                    "A loaded executive parked-goal packet was rejected because the source "
                    "artifact repeated the objective and failed framing, so it is preserved "
                    "for audit instead of active prompt continuity."
                ),
                source_kind=BeadSourceKind.SUPPRESSED_REFRAME,
                source_ref=source_ref,
                content=content,
                user_commissioned=False,
            )
            recorded_count += 1
        except Exception as exc:
            failed_count += 1
            _trace(
                tracer,
                "suppressed_reframe_loaded_bead_failed",
                {"source_ref": source_ref, "error": str(exc)},
            )

    _trace(
        tracer,
        "suppressed_reframe_loaded_beads_recorded",
        {
            "candidate_count": len(rejections),
            "recorded_count": recorded_count,
            "failed_count": failed_count,
        },
    )
    return {
        "available": True,
        "candidate_count": len(rejections),
        "recorded_count": recorded_count,
        "failed_count": failed_count,
    }
