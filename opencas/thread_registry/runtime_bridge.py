"""Runtime hooks for recording daydream insights into the thread registry."""

from __future__ import annotations

import json
from typing import Any

from opencas.daydream import DaydreamReflection, DaydreamThought, DaydreamThoughtRoute
from opencas.thread_registry import BeadSourceKind, ThreadStatus


async def record_runtime_daydream_thread_beads(
    runtime: Any,
    reflection: DaydreamReflection,
) -> dict[str, Any]:
    """Record non-discarded daydream thoughts as peripheral beads."""

    service = getattr(runtime, "thread_registry_service", None)
    if service is None:
        return {"available": False, "recorded": 0}

    recorded: list[str] = []
    thoughts = list(getattr(reflection, "thoughts", []) or [])
    for index, thought in enumerate(thoughts):
        if thought.route == DaydreamThoughtRoute.DISCARD:
            continue
        thread_title = (
            str(getattr(reflection, "fascination_thread", "") or "").strip()
            or _thread_title_for_thought(thought)
        )
        summary = _summary_for_thought(thought, reflection)
        content = _content_for_thought(thought, reflection)
        anchor = await service.ensure_thread_anchor(
            title=thread_title,
            kind=thought.kind.value,
            status=ThreadStatus.PERIPHERAL,
        )
        bead = await service.create_candidate_bead(
            thread_anchor_id=anchor.anchor_id,
            title=_title_for_thought(thought),
            summary=summary,
            source_kind=BeadSourceKind.DAYDREAM_REFLECTION,
            source_ref=f"daydream:{reflection.reflection_id}:{index}",
            content=content,
            user_commissioned=False,
        )
        recorded.append(bead.bead_id)

    trace = getattr(runtime, "_trace", None)
    if callable(trace) and recorded:
        trace(
            "daydream_thread_beads_recorded",
            {
                "reflection_id": str(getattr(reflection, "reflection_id", "")),
                "recorded": len(recorded),
            },
        )
    return {"available": True, "recorded": len(recorded), "bead_ids": recorded}


async def record_failed_tool_call_transit_thread_beads(
    runtime: Any,
    *,
    session_id: str,
    user_input: str,
    tool_call_transits: list[Any] | None,
    max_per_turn: int = 3,
) -> dict[str, Any]:
    """Record failed tool-call transits as bounded peripheral beads."""

    service = getattr(runtime, "thread_registry_service", None)
    if service is None:
        return {"available": False, "recorded": 0, "bead_ids": []}

    recorded: list[str] = []
    failures = [
        transit
        for transit in list(tool_call_transits or [])
        if not bool(getattr(transit, "success", False))
    ]
    if not failures:
        return {"available": True, "recorded": 0, "bead_ids": []}

    anchor = await service.ensure_thread_anchor(
        title="Failed tool-call transits",
        kind="tool_call_transit",
        status=ThreadStatus.PERIPHERAL,
        anchor_id="failed-tool-call-transits",
    )
    for transit in failures[: max(0, int(max_per_turn))]:
        result_shape = getattr(transit, "result_shape", {}) or {}
        if not isinstance(result_shape, dict):
            result_shape = {}
        tool_name = _clean_text(getattr(transit, "tool_name", "")) or "unknown tool"
        entry_intent = _clean_text(getattr(transit, "entry_intent", ""))
        tags = [
            _clean_text(tag)
            for tag in list(result_shape.get("tags") or [])
            if _clean_text(tag)
        ]
        source_ref = ":".join(
            [
                "tool_call_transit",
                _source_ref_part(session_id) or "default",
                _source_ref_part(getattr(transit, "chain_id", "")) or "chain",
                _source_ref_part(getattr(transit, "call_id", "")) or "call",
            ]
        )
        summary = (
            f"Failed tool-call transit for {tool_name}: "
            f"{entry_intent or 'No entry intent recorded.'}"
        )
        if tags:
            summary += f" Result tags: {', '.join(tags[:5])}."
        content = json.dumps(
            {
                "source_ref": source_ref,
                "tool_name": tool_name,
                "entry_intent": entry_intent,
                "trust_context": _clean_text(getattr(transit, "trust_context", "")),
                "result_tags": tags,
                "result_shape": result_shape,
            },
            sort_keys=True,
        )
        bead = await service.create_candidate_bead(
            thread_anchor_id=anchor.anchor_id,
            title=f"Failed tool call: {tool_name[:72]}",
            summary=summary,
            source_kind=BeadSourceKind.TOOL_CALL_TRANSIT,
            source_ref=source_ref,
            content=content,
            user_commissioned=False,
        )
        recorded.append(bead.bead_id)

    trace = getattr(runtime, "_trace", None)
    if callable(trace) and recorded:
        trace(
            "failed_tool_call_transit_thread_beads_recorded",
            {
                "session_id": session_id,
                "recorded": len(recorded),
            },
        )
    return {"available": True, "recorded": len(recorded), "bead_ids": recorded}


def _thread_title_for_thought(thought: DaydreamThought) -> str:
    return "Daydream " + " ".join(thought.kind.value.split("_"))


def _title_for_thought(thought: DaydreamThought) -> str:
    cleaned = " ".join(str(thought.summary or "").split())
    if cleaned:
        return cleaned[:96]
    return "Daydream " + " ".join(thought.kind.value.split("_"))


def _summary_for_thought(
    thought: DaydreamThought,
    reflection: DaydreamReflection,
) -> str:
    parts = [
        thought.summary,
        thought.question,
        thought.hypothesis,
        thought.possible_experiment,
        reflection.synthesis,
    ]
    cleaned = " ".join(part.strip() for part in parts if str(part or "").strip())
    if len(cleaned) >= 20:
        return cleaned
    return "A daydream thought preserved for later peripheral pickup."


def _content_for_thought(
    thought: DaydreamThought,
    reflection: DaydreamReflection,
) -> str:
    payload = {
        "reflection_id": str(reflection.reflection_id),
        "spark_content": reflection.spark_content,
        "synthesis": reflection.synthesis,
        "fascination_thread": reflection.fascination_thread,
        "thought": thought.model_dump(mode="json"),
    }
    return json.dumps(payload, sort_keys=True)


def _clean_text(value: Any) -> str:
    return " ".join(str(value or "").split())


def _source_ref_part(value: Any) -> str:
    cleaned = _clean_text(value)
    return "".join(char if char.isalnum() or char in {"-", "_", "."} else "-" for char in cleaned).strip("-")
