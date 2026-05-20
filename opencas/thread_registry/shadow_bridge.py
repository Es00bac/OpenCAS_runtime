"""Thread-registry bridge for mature ShadowRegistry clusters."""

from __future__ import annotations

import json
from typing import Any

from opencas.thread_registry import BeadSourceKind, ThreadStatus

PROMOTABLE_SHADOW_REASONS = {
    "retry_blocked",
    "tool_loop_guard_blocked",
    "project_composted",
}


async def record_shadow_registry_thread_beads(
    runtime: Any,
    *,
    limit: int = 10,
    min_cluster_count: int = 2,
) -> dict[str, Any]:
    """Record mature active blocked-intention clusters as peripheral beads."""

    shadow_registry = getattr(getattr(runtime, "ctx", None), "shadow_registry", None) or getattr(
        runtime,
        "shadow_registry",
        None,
    )
    service = getattr(runtime, "thread_registry_service", None)
    if shadow_registry is None or service is None:
        return {"available": False, "recorded": 0, "skipped": 0, "bead_ids": []}

    summary = getattr(shadow_registry, "summary", None)
    if not callable(summary):
        return {"available": False, "recorded": 0, "skipped": 0, "bead_ids": []}
    try:
        payload = summary(limit=limit, cluster_limit=limit)
    except Exception as exc:
        return {
            "available": False,
            "reason": str(exc),
            "recorded": 0,
            "skipped": 0,
            "bead_ids": [],
        }

    clusters = list(payload.get("top_clusters") or []) if isinstance(payload, dict) else []
    if not clusters:
        return {"available": True, "recorded": 0, "skipped": 0, "bead_ids": []}

    anchor = await service.ensure_thread_anchor(
        title="Mature shadow-registry clusters",
        kind="shadow_intention",
        status=ThreadStatus.PERIPHERAL,
        anchor_id="shadow-registry-mature-clusters",
    )
    recorded: list[str] = []
    skipped = 0
    threshold = max(2, int(min_cluster_count))
    for cluster in clusters[: max(1, int(limit))]:
        fingerprint = _clean_text(cluster.get("fingerprint"))
        block_reason = _clean_text(cluster.get("block_reason"))
        count = int(cluster.get("count") or 0)
        if (
            not fingerprint
            or count < threshold
            or block_reason not in PROMOTABLE_SHADOW_REASONS
            or _clean_text(cluster.get("triage_status")) == "dismissed"
        ):
            skipped += 1
            continue

        tool_name = _clean_text(cluster.get("tool_name"))
        intent_summary = _clean_text(cluster.get("intent_summary"))
        source_ref = f"shadow_registry:{fingerprint}"
        content = json.dumps(
            {
                "source_ref": source_ref,
                "fingerprint": fingerprint,
                "block_reason": block_reason,
                "tool_name": tool_name,
                "intent_summary": intent_summary,
                "consumer": "thread-registry continuity cues",
                "instruction": (
                    "Verify whether the blocker still applies before repeating "
                    "the same intention framing."
                ),
            },
            sort_keys=True,
        )
        bead = await service.create_candidate_bead(
            thread_anchor_id=anchor.anchor_id,
            title=f"Shadow cluster: {_short_title(intent_summary or block_reason)}",
            summary=(
                f"{count}x {block_reason} around {intent_summary or tool_name or fingerprint}. "
                "Use as a blocker-continuity cue, not as an automatic task."
            ),
            source_kind=BeadSourceKind.SHADOW_INTENTION,
            source_ref=source_ref,
            content=content,
            user_commissioned=False,
        )
        recorded.append(bead.bead_id)

    trace = getattr(runtime, "_trace", None)
    if callable(trace) and (recorded or skipped):
        trace(
            "shadow_registry_thread_beads_recorded",
            {"recorded": len(recorded), "skipped": skipped, "limit": limit},
        )
    return {
        "available": True,
        "recorded": len(recorded),
        "skipped": skipped,
        "bead_ids": recorded,
    }


def _clean_text(value: Any) -> str:
    return " ".join(str(value or "").split())


def _short_title(value: str, *, max_chars: int = 72) -> str:
    text = _clean_text(value)
    if len(text) <= max_chars:
        return text
    return text[: max_chars - 3].rstrip() + "..."
