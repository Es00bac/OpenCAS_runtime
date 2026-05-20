"""Runtime hooks for autobiographical session anchors."""

from __future__ import annotations

import asyncio
from datetime import datetime, timedelta, timezone
from typing import Any

from opencas.infra.hook_bus import POST_SESSION_LIFECYCLE, HookResult

_ANCHOR_TRANSITIONS = {
    "shutdown",
    "stop",
    "end",
    "complete",
    "commit",
    "interrupt",
    "abort",
    "fail",
    "rollback",
}


def register_autobiography_hooks(runtime: Any) -> None:
    """Register lifecycle hook that closes sessions into autobiography anchors."""
    if getattr(runtime, "_autobiography_hooks_registered", False):
        return
    hook_bus = getattr(getattr(runtime, "ctx", None), "hook_bus", None)
    if hook_bus is None:
        return
    hook_bus.register(
        POST_SESSION_LIFECYCLE,
        lambda hook_name, ctx: _post_session_lifecycle(runtime, hook_name, ctx),
        priority=-200,
    )
    runtime._autobiography_hooks_registered = True


def schedule_autobiography_boot_recovery(
    runtime: Any,
    *,
    since: datetime | None = None,
    limit: int = 80,
) -> None:
    """Backfill missing anchors for recent sessions without blocking startup."""
    ctx = getattr(runtime, "ctx", None)
    memory = getattr(runtime, "memory", None) or getattr(ctx, "memory", None)
    anchor_store = getattr(ctx, "autobiography_anchor_store", None)
    composer = getattr(ctx, "autobiography_composer", None)
    if memory is None or anchor_store is None or composer is None:
        return
    since = since or datetime.now(timezone.utc) - timedelta(days=30)

    async def _recover() -> None:
        try:
            created = await recover_missing_anchors(
                memory,
                anchor_store,
                composer,
                since=since,
                limit=limit,
            )
            _trace(runtime, "autobiography_boot_recovery", {"anchors_created": created})
        except Exception as exc:
            _trace(
                runtime,
                "autobiography_boot_recovery_error",
                {"error": str(exc), "error_type": type(exc).__name__},
            )

    try:
        task = asyncio.get_running_loop().create_task(
            _recover(),
            name="autobiography_boot_recovery",
        )
        _register_background_task(runtime, task)
    except RuntimeError:
        return


async def recover_missing_anchors(
    memory_store: Any,
    anchor_store: Any,
    composer: Any,
    *,
    since: datetime,
    limit: int = 80,
) -> int:
    """Create skeleton anchors for sessions with episodes but no anchor row."""
    db = getattr(memory_store, "_db", None)
    if db is None:
        return 0
    cursor = await db.execute(
        """
        SELECT session_id, MAX(created_at) AS last_seen
        FROM episodes
        WHERE session_id IS NOT NULL
          AND session_id != ''
          AND created_at >= ?
        GROUP BY session_id
        ORDER BY last_seen DESC
        LIMIT ?
        """,
        (since.isoformat(), limit),
    )
    rows = await cursor.fetchall()
    session_ids = [str(row["session_id"]) for row in rows if str(row["session_id"] or "").strip()]
    missing = await anchor_store.list_missing_for_sessions(session_ids)
    created = 0
    for session_id in missing:
        anchor = await composer.compose_skeleton(session_id)
        await anchor_store.upsert(anchor)
        created += 1
    return created


def _post_session_lifecycle(runtime: Any, _hook_name: str, ctx: dict[str, Any]) -> HookResult:
    transition = str(ctx.get("transition", "") or "").strip().lower()
    if transition not in _ANCHOR_TRANSITIONS:
        return HookResult(allowed=True)
    session_id = str(ctx.get("session_id", "") or "").strip()
    if not session_id:
        config = getattr(getattr(runtime, "ctx", None), "config", None)
        session_id = str(getattr(config, "session_id", "") or "").strip()
    if not session_id:
        return HookResult(allowed=True)

    composer = getattr(getattr(runtime, "ctx", None), "autobiography_composer", None)
    anchor_store = getattr(getattr(runtime, "ctx", None), "autobiography_anchor_store", None)
    if composer is None or anchor_store is None:
        return HookResult(allowed=True)

    async def _compose() -> None:
        try:
            anchor = await composer.compose_skeleton(session_id)
            await anchor_store.upsert(anchor)
            _trace(
                runtime,
                "autobiography_session_anchor_written",
                {"session_id": session_id, "transition": transition},
            )
        except Exception as exc:
            _trace(
                runtime,
                "autobiography_session_anchor_error",
                {
                    "session_id": session_id,
                    "transition": transition,
                    "error": str(exc),
                    "error_type": type(exc).__name__,
                },
            )

    try:
        task = asyncio.get_running_loop().create_task(
            _compose(),
            name=f"autobiography_session_anchor:{session_id}",
        )
        _register_background_task(runtime, task)
    except RuntimeError:
        return HookResult(allowed=True)
    return HookResult(allowed=True)


def _register_background_task(runtime: Any, task: asyncio.Task[Any]) -> None:
    ctx = getattr(runtime, "ctx", None)
    if ctx is None:
        return
    tasks = getattr(ctx, "background_tasks", ())
    if isinstance(tasks, tuple):
        ctx.background_tasks = (*tasks, task)
        return
    if isinstance(tasks, list):
        tasks.append(task)
        return
    ctx.background_tasks = (task,)


def _trace(runtime: Any, event: str, payload: dict[str, Any]) -> None:
    tracer = getattr(runtime, "_trace", None)
    if callable(tracer):
        try:
            tracer(event, payload)
        except Exception:
            pass
