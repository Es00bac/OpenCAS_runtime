"""Runtime housekeeping and event-hook helpers for AgentRuntime."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Optional

from opencas.infra import BaaCompletedEvent
from opencas.somatic import AppraisalEventType
from opencas.telemetry import EventKind

from .lifecycle import shutdown_runtime_resources


async def maybe_compact_runtime_session(runtime: Any, session_id: str, tail_size: int = 10) -> None:
    """Compact old session context and trace the resulting compaction record."""
    record = await runtime.compactor.compact_session(session_id, tail_size=tail_size)
    if record:
        runtime._trace(
            "compaction_triggered",
            {
                "session_id": session_id,
                "removed_count": record.removed_count,
                "compaction_id": str(record.compaction_id),
            },
        )


async def run_runtime_consolidation(runtime: Any) -> Dict[str, Any]:
    """Run the nightly consolidation engine with runtime activity bookkeeping."""
    runtime._set_activity("consolidating")
    try:
        result = await runtime.consolidation.run()
        payload = result.model_dump(mode="json")
        runtime._last_consolidation_result = payload
        return payload
    finally:
        runtime._set_activity("idle")


async def maybe_record_runtime_somatic_snapshot(
    runtime: Any,
    source: str,
    trigger_event_id: Optional[str] = None,
) -> None:
    """Persist a somatic snapshot when the store is enabled."""
    if runtime.ctx.somatic.store is not None:
        await runtime.ctx.somatic.record_snapshot(
            source=source,
            trigger_event_id=trigger_event_id,
        )


def sync_runtime_executive_snapshot(runtime: Any) -> None:
    """Persist the current executive snapshot to the runtime state directory."""
    snapshot_path = runtime.ctx.config.state_dir / "executive.json"
    runtime.executive.save_snapshot(snapshot_path)


async def handle_runtime_baa_completed(runtime: Any, event: BaaCompletedEvent) -> None:
    """Resolve goal outcomes for completed BAA work and persist executive state."""
    if event.success:
        resolved_goals = await runtime.executive.check_goal_resolution(event.output)
        for goal in resolved_goals:
            await runtime.ctx.somatic.emit_appraisal_event(
                AppraisalEventType.GOAL_ACHIEVED,
                source_text=f"Goal achieved: {goal}",
                trigger_event_id=event.task_id,
            )
    else:
        await runtime.ctx.somatic.emit_appraisal_event(
            AppraisalEventType.GOAL_BLOCKED,
            source_text=f"BAA task failed: {event.objective}",
            trigger_event_id=event.task_id,
            meta={"stage": event.stage, "output": event.output},
        )
    sync_runtime_executive_snapshot(runtime)


async def close_runtime_stores(runtime: Any) -> None:
    """Gracefully close the runtime resource stack."""
    await shutdown_runtime_resources(runtime)


def extract_runtime_response_content(response: Dict[str, Any]) -> str:
    """Extract the assistant message content from an OpenAI-style response payload."""
    choices = response.get("choices", [])
    if choices:
        message = choices[0].get("message", {})
        return message.get("content", "")
    return ""


def trace_runtime_event(
    runtime: Any,
    event: str,
    payload: Optional[Dict[str, Any]] = None,
) -> None:
    """Write a runtime trace event when telemetry is available."""
    if runtime.tracer:
        runtime.tracer.log(
            EventKind.TOM_EVAL,
            f"AgentRuntime: {event}",
            payload or {},
        )
