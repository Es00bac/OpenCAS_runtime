"""Runtime hooks for recording conversation self-inspection packets."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, Dict, Optional

from opencas.cognition import (
    build_post_turn_self_inspection_record,
    build_pre_turn_self_inspection_record,
)

if TYPE_CHECKING:
    from .agent_loop import AgentRuntime


async def record_pre_turn_self_inspection(
    runtime: "AgentRuntime",
    *,
    session_id: str,
    user_input: str,
) -> None:
    """Persist assumptions and active promise pressure before generation."""
    store = getattr(runtime, "self_inspection_store", None)
    if store is None:
        return
    try:
        active_commitments = []
        commitment_store = getattr(runtime, "commitment_store", None)
        if commitment_store is not None:
            active_commitments = await commitment_store.list_active(limit=20)
        record = build_pre_turn_self_inspection_record(
            session_id=session_id,
            user_input=user_input,
            active_commitments=active_commitments,
        )
        await store.save(record)
    except Exception as exc:
        runtime._trace(
            "self_inspection_pre_turn_error",
            {"session_id": session_id, "error": str(exc)},
        )


async def record_post_turn_self_inspection(
    runtime: "AgentRuntime",
    *,
    session_id: str,
    user_input: str,
    content: str,
    assistant_meta_extra: Optional[Dict[str, Any]],
    pre_somatic_state: Any,
    captured_commitments: list[Any],
    tool_use_inspections: list[Any],
) -> None:
    """Persist response-shape, valence-source, tool-use, and promise-gap evidence."""
    store = getattr(runtime, "self_inspection_store", None)
    if store is None:
        return
    try:
        integrity_review = {}
        if isinstance(assistant_meta_extra, dict):
            integrity_review = dict(assistant_meta_extra.get("response_integrity") or {})
        recent_records = await store.list_recent(session_id=session_id, limit=8)
        post_somatic_state = runtime.ctx.somatic.state.model_copy()
        record = build_post_turn_self_inspection_record(
            session_id=session_id,
            user_input=user_input,
            assistant_output=content,
            captured_commitments=captured_commitments,
            integrity_review=integrity_review,
            tool_use_inspections=tool_use_inspections,
            pre_somatic_state=pre_somatic_state,
            post_somatic_state=post_somatic_state,
            recent_records=recent_records,
        )
        await store.save(record)
    except Exception as exc:
        runtime._trace(
            "self_inspection_post_turn_error",
            {"session_id": session_id, "error": str(exc)},
        )
