"""Bootstrap context shutdown helpers."""

from __future__ import annotations

from typing import Any


async def close_bootstrap_context(context: Any) -> None:
    """Close all owned runtime stores and services once per object."""
    await context.health_monitor.stop()
    context.readiness.shutdown("context_closed")
    context.identity.record_shutdown(session_id=context.config.session_id)
    await context.token_telemetry.flush()
    if context.background_tasks:
        for task in context.background_tasks:
            task.cancel()
        import asyncio

        await asyncio.gather(*context.background_tasks, return_exceptions=True)

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
        context.work_store,
        context.relational,
        context.daydream_store,
        context.conflict_store,
        context.somatic_store,
        context.curation_store,
        getattr(context.ledger, "store", None),
        getattr(context.harness, "store", None),
        context.commitment_store,
        context.portfolio_store,
        context.tom_store,
        context.plugin_store,
        context.plan_store,
        context.schedule_store,
        context.workspace_index,
    ]
    for obj in closables:
        await _close_once(obj)
