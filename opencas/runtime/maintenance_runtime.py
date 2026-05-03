"""Runtime housekeeping and event-hook helpers for AgentRuntime."""

from __future__ import annotations

import asyncio
import inspect
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Optional

from opencas.infra import BaaCompletedEvent
from opencas.somatic import AppraisalEventType
from opencas.telemetry import EventKind

from .consolidation_state import persist_consolidation_runtime_state
from .consolidation_worker import run_consolidation_in_worker_process
from .continuity_breadcrumbs import current_runtime_focus, record_burst_continuity
from .lifecycle import shutdown_runtime_resources
from .tom_intention_mirror import resolve_runtime_intention


async def maybe_compact_runtime_session(
    runtime: Any,
    session_id: str,
    tail_size: int = 10,
    min_removed_count: int = 1,
) -> Any:
    """Compact old session context and trace the resulting compaction record."""
    record = await runtime.compactor.compact_session(
        session_id,
        tail_size=tail_size,
        min_removed_count=min_removed_count,
    )
    if record:
        runtime._trace(
            "compaction_triggered",
            {
                "session_id": session_id,
                "removed_count": record.removed_count,
                "compaction_id": str(record.compaction_id),
            },
        )
    return record


async def compact_runtime_backlog(
    runtime: Any,
    *,
    max_sessions: int = 8,
    min_session_lag: int = 20,
    tail_size: int = 10,
    max_candidates: int = 1000,
) -> Dict[str, Any]:
    """Compact the largest old uncompacted session backlogs within a small budget."""
    ctx = getattr(runtime, "ctx", None)
    memory = getattr(ctx, "memory", None) or getattr(runtime, "memory", None)
    compactor = getattr(runtime, "compactor", None)
    if memory is None or compactor is None:
        return {
            "available": False,
            "reason": "missing_memory_or_compactor",
            "sessions_compacted": 0,
            "episodes_compacted": 0,
        }

    episodes = await memory.list_non_compacted_episodes(limit=max_candidates)
    total_lag = len(episodes)
    count_func = getattr(memory, "count_non_compacted_episodes", None)
    if callable(count_func):
        try:
            counted = count_func()
            if inspect.isawaitable(counted):
                counted = await counted
            if isinstance(counted, int):
                total_lag = counted
        except Exception:
            pass

    session_counts: Counter[str] = Counter()
    for episode in episodes:
        session_id = getattr(episode, "session_id", None)
        if session_id:
            session_counts[str(session_id)] += 1

    selected = [
        (session_id, count)
        for session_id, count in session_counts.most_common()
        if count >= max(min_session_lag, tail_size + 1)
    ][: max(0, max_sessions)]

    compacted_sessions = []
    episodes_compacted = 0
    for session_id, count in selected:
        record = await compactor.compact_session(
            session_id,
            tail_size=tail_size,
            min_removed_count=1,
        )
        if record is None:
            continue
        removed = int(getattr(record, "removed_count", 0) or 0)
        episodes_compacted += removed
        compacted_sessions.append(
            {
                "session_id": session_id,
                "candidate_lag": count,
                "removed_count": removed,
                "compaction_id": str(getattr(record, "compaction_id", "")),
            }
        )

    result = {
        "available": True,
        "candidate_lag": len(episodes),
        "total_lag": total_lag,
        "sessions_considered": len(session_counts),
        "sessions_compacted": len(compacted_sessions),
        "episodes_compacted": episodes_compacted,
        "remaining_lag": max(0, total_lag - episodes_compacted),
        "compacted_sessions": compacted_sessions,
    }
    trace = getattr(runtime, "_trace", None)
    if callable(trace):
        trace("compaction_backlog_sweep", result)
    return result


def _persist_runtime_consolidation_state(
    runtime: Any,
    payload: Dict[str, Any],
) -> None:
    state_dir = getattr(getattr(runtime, "ctx", None), "config", None)
    runtime_state_dir = getattr(state_dir, "state_dir", None)
    if runtime_state_dir is None:
        return
    timestamp = str(payload.get("timestamp") or datetime.now(timezone.utc).isoformat())
    persist_consolidation_runtime_state(
        Path(runtime_state_dir),
        {
            "last_run_at": timestamp,
            "last_result_id": payload.get("result_id"),
            "budget_exhausted": payload.get("budget_exhausted"),
            "budget_reason": payload.get("budget_reason"),
        },
    )


async def run_runtime_consolidation(
    runtime: Any,
    *,
    budget: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """Run the nightly consolidation engine with runtime activity bookkeeping."""
    runtime._set_activity("consolidating")
    try:
        config = getattr(getattr(runtime, "ctx", None), "config", None)
        worker_enabled = bool(getattr(config, "consolidation_worker_enabled", False))
        if worker_enabled:
            payload = await run_consolidation_in_worker_process(
                runtime,
                budget=dict(budget or {}),
            )
            backlog_result = await compact_runtime_backlog(
                runtime,
                max_sessions=int((budget or {}).get("max_compaction_sessions", 8)),
                min_session_lag=int((budget or {}).get("min_compaction_session_lag", 20)),
                tail_size=int((budget or {}).get("compaction_tail_size", 10)),
                max_candidates=int((budget or {}).get("max_compaction_candidates", 1000)),
            )
            if backlog_result.get("available"):
                payload["compaction_backlog"] = backlog_result
            runtime._last_consolidation_result = payload
            _persist_runtime_consolidation_state(runtime, payload)
            trace = getattr(runtime, "_trace", None)
            if callable(trace):
                trace("consolidation_worker_complete", payload.get("worker") or {})
            return payload

        run_kwargs: Dict[str, Any] = {}
        if budget:
            run_kwargs["budget"] = budget
        run_coro = runtime.consolidation.run(**run_kwargs)
        max_seconds = None
        if budget and budget.get("max_seconds") is not None:
            try:
                max_seconds = float(budget["max_seconds"])
            except (TypeError, ValueError):
                max_seconds = None
        try:
            if max_seconds is not None and max_seconds > 0:
                result = await asyncio.wait_for(run_coro, timeout=max_seconds)
            else:
                result = await run_coro
        except asyncio.TimeoutError:
            timestamp = datetime.now(timezone.utc).isoformat()
            payload = {
                "result_id": f"timeout-{timestamp}",
                "timestamp": timestamp,
                "budget": dict(budget or {}),
                "budget_exhausted": True,
                "budget_reason": "timeout",
            }
            runtime._last_consolidation_result = payload
            _persist_runtime_consolidation_state(runtime, payload)
            trace = getattr(runtime, "_trace", None)
            if callable(trace):
                trace("consolidation_budget_timeout", payload)
            return payload
        payload = result.model_dump(mode="json")
        backlog_result = await compact_runtime_backlog(
            runtime,
            max_sessions=int((budget or {}).get("max_compaction_sessions", 8)),
            min_session_lag=int((budget or {}).get("min_compaction_session_lag", 20)),
            tail_size=int((budget or {}).get("compaction_tail_size", 10)),
            max_candidates=int((budget or {}).get("max_compaction_candidates", 1000)),
        )
        if backlog_result.get("available"):
            payload["compaction_backlog"] = backlog_result
        runtime._last_consolidation_result = payload
        _persist_runtime_consolidation_state(runtime, payload)
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
    try:
        decision = "task completed and goal checks updated"
        next_step = "continue draining queue and monitor executive outcomes"
        if not event.success:
            decision = "task failed and logged as blocked goal"
            next_step = "surface recovery guidance and continue executive recovery"
        await record_burst_continuity(
            runtime,
            trigger="work_burst_completed",
            phase="end",
            intent=f"BAA completion for {event.objective[:120]}",
            focus=current_runtime_focus(runtime, event.objective),
            next_step=next_step,
            note=(
                f"task_id={getattr(event, 'task_id', None)}"
                f";success={event.success}"
                f";decision={decision}"
            ),
            episode_id=str(getattr(event, "task_id", "")) or None,
        )
    except Exception:
        runtime._trace("continuity_breadcrumb_baa_complete_error", {"task_id": getattr(event, "task_id", None)})

    if event.success:
        await resolve_runtime_intention(runtime, event.objective)
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
