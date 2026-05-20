from __future__ import annotations

import asyncio
from datetime import datetime, timezone
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest

from opencas.bootstrap import context_close
from opencas.bootstrap.context_close import close_bootstrap_context
from opencas.runtime.autobiography_hooks import (
    _post_session_lifecycle,
    schedule_autobiography_boot_recovery,
)


def _minimal_close_context(background_tasks=()):
    async def _noop_async():
        return None

    return SimpleNamespace(
        health_monitor=SimpleNamespace(stop=_noop_async),
        readiness=SimpleNamespace(shutdown=lambda _reason: None),
        identity=SimpleNamespace(record_shutdown=lambda session_id=None: None),
        config=SimpleNamespace(session_id="session-1"),
        token_telemetry=SimpleNamespace(flush=_noop_async),
        background_tasks=background_tasks,
        _exit_stack=None,
        mcp_registry=None,
        embeddings=None,
        memory=None,
        tasks=None,
        receipt_store=None,
        context_store=None,
        context_proposal_store=None,
        work_store=None,
        relational=None,
        daydream_store=None,
        daydream_signal_store=None,
        conflict_store=None,
        somatic_store=None,
        curation_store=None,
        web_trust=None,
        plugin_trust=None,
        ledger=SimpleNamespace(store=None),
        harness=SimpleNamespace(store=None),
        commitment_store=None,
        self_inspection_store=None,
        cognitive_state_store=None,
        wellbeing_store=None,
        dream_store=None,
        proof_store=None,
        thread_registry_store=None,
        portfolio_store=None,
        tom_store=None,
        plugin_store=None,
        plan_store=None,
        schedule_store=None,
        workspace_index=None,
        affective_examinations=None,
    )


@pytest.mark.asyncio
async def test_autobiography_lifecycle_task_registers_with_context() -> None:
    composer = SimpleNamespace(compose_skeleton=AsyncMock(return_value="anchor"))
    anchor_store = SimpleNamespace(upsert=AsyncMock())
    runtime = SimpleNamespace(
        ctx=SimpleNamespace(
            autobiography_composer=composer,
            autobiography_anchor_store=anchor_store,
            background_tasks=(),
            config=SimpleNamespace(session_id="session-1"),
        ),
        _trace=lambda *_args, **_kwargs: None,
    )

    result = _post_session_lifecycle(
        runtime,
        "post_session_lifecycle",
        {"transition": "shutdown", "session_id": "session-1"},
    )

    assert result.allowed is True
    assert len(runtime.ctx.background_tasks) == 1
    await runtime.ctx.background_tasks[0]
    anchor_store.upsert.assert_awaited_once_with("anchor")


@pytest.mark.asyncio
async def test_autobiography_recovery_task_registers_with_context() -> None:
    runtime = SimpleNamespace(
        memory=SimpleNamespace(_db=None),
        ctx=SimpleNamespace(
            memory=SimpleNamespace(_db=None),
            autobiography_anchor_store=object(),
            autobiography_composer=object(),
            background_tasks=(),
        ),
        _trace=lambda *_args, **_kwargs: None,
    )

    schedule_autobiography_boot_recovery(
        runtime,
        since=datetime(2026, 5, 1, tzinfo=timezone.utc),
    )

    assert len(runtime.ctx.background_tasks) == 1
    await runtime.ctx.background_tasks[0]


@pytest.mark.asyncio
async def test_context_close_waits_for_background_tasks_before_closing_stores() -> None:
    events: list[str] = []

    async def _background_write() -> None:
        await asyncio.sleep(0.01)
        events.append("background_written")

    class Store:
        async def close(self) -> None:
            events.append("store_closed")

    task = asyncio.create_task(_background_write())
    context = _minimal_close_context(background_tasks=(task,))
    context.memory = Store()

    await close_bootstrap_context(context)

    assert events == ["background_written", "store_closed"]


@pytest.mark.asyncio
async def test_context_close_cancels_background_tasks_after_timeout(
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
) -> None:
    monkeypatch.setattr(context_close, "_BACKGROUND_TASK_SHUTDOWN_TIMEOUT_SECONDS", 0.01)
    caplog.set_level("WARNING", logger="opencas.bootstrap.context_close")
    task = asyncio.create_task(asyncio.sleep(60), name="slow-autobiography")
    context = _minimal_close_context(background_tasks=(task,))

    await close_bootstrap_context(context)

    assert task.cancelled()
    assert "slow-autobiography" in caplog.text
