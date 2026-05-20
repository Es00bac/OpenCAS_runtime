"""Bootstrap context shutdown helpers."""

from __future__ import annotations

import asyncio
import logging
from typing import Any

logger = logging.getLogger(__name__)
_BACKGROUND_TASK_SHUTDOWN_TIMEOUT_SECONDS = 5.0


async def close_bootstrap_context(context: Any) -> None:
    """Close all owned runtime stores and services once per object."""
    await context.health_monitor.stop()
    context.readiness.shutdown("context_closed")
    context.identity.record_shutdown(session_id=context.config.session_id)
    await context.token_telemetry.flush()
    await _drain_background_tasks(getattr(context, "background_tasks", ()))
    exit_stack = getattr(context, "_exit_stack", None)
    if exit_stack is not None:
        context._exit_stack = None
        await exit_stack.aclose()
        return

    seen: set[int] = set()

    async def _close_once(obj: Any) -> None:
        if obj is None:
            return
        obj_id = id(obj)
        if obj_id in seen:
            return
        seen.add(obj_id)
        action = getattr(obj, "close", None)
        if not callable(action):
            action = getattr(obj, "stop", None)
        if not callable(action):
            return
        result = action()
        if hasattr(result, "__await__"):
            await result

    closables = [
        context.mcp_registry,
        context.embeddings,
        context.memory,
        context.tasks,
        context.receipt_store,
        context.context_store,
        getattr(context, "context_proposal_store", None),
        context.work_store,
        context.relational,
        context.daydream_store,
        getattr(context, "daydream_signal_store", None),
        context.conflict_store,
        context.somatic_store,
        context.curation_store,
        context.web_trust,
        getattr(context, "plugin_trust", None),
        getattr(context.ledger, "store", None),
        getattr(context.harness, "store", None),
        context.commitment_store,
        getattr(context, "self_inspection_store", None),
        getattr(context, "cognitive_state_store", None),
        getattr(context, "wellbeing_store", None),
        getattr(context, "dream_store", None),
        getattr(context, "proof_store", None),
        getattr(context, "thread_registry_store", None),
        context.portfolio_store,
        context.tom_store,
        context.plugin_store,
        context.plan_store,
        context.schedule_store,
        getattr(context, "recovery_ledger", None),
        context.workspace_index,
        getattr(context, "affective_examinations", None),
    ]
    for obj in closables:
        await _close_once(obj)


async def _drain_background_tasks(tasks: Any) -> None:
    tasks = tuple(task for task in (tasks or ()) if task is not None)
    if not tasks:
        return
    task_names = [_task_name(task) for task in tasks]
    try:
        await asyncio.wait_for(
            asyncio.gather(*tasks, return_exceptions=True),
            timeout=_BACKGROUND_TASK_SHUTDOWN_TIMEOUT_SECONDS,
        )
    except TimeoutError:
        logger.warning(
            "Background tasks did not finish before shutdown timeout: %s",
            ", ".join(task_names),
        )
        for task in tasks:
            if not task.done():
                task.cancel()
        await asyncio.gather(*tasks, return_exceptions=True)


def _task_name(task: asyncio.Task[Any]) -> str:
    get_name = getattr(task, "get_name", None)
    if callable(get_name):
        return str(get_name())
    return repr(task)
