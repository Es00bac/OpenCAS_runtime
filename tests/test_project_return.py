"""Tests for chat-born project return capture."""

from __future__ import annotations

from datetime import datetime, timedelta, timezone
from types import SimpleNamespace

import pytest
import pytest_asyncio

from opencas.autonomy.commitment_store import CommitmentStore
from opencas.runtime.project_return import capture_project_return_from_turn
from opencas.scheduling import ScheduleAction, ScheduleKind, ScheduleRecurrence, ScheduleService, ScheduleStore


@pytest_asyncio.fixture
async def project_runtime(tmp_path):
    commitment_store = CommitmentStore(tmp_path / "commitments.db")
    await commitment_store.connect()
    schedule_store = ScheduleStore(tmp_path / "schedules.db")
    await schedule_store.connect()
    schedule_service = ScheduleService(schedule_store)
    runtime = SimpleNamespace(
        commitment_store=commitment_store,
        schedule_service=schedule_service,
        ctx=SimpleNamespace(schedule_store=schedule_store),
        _trace=lambda *_args, **_kwargs: None,
    )
    try:
        yield runtime
    finally:
        await schedule_store.close()
        await commitment_store.close()


@pytest.mark.asyncio
async def test_chat_project_context_creates_return_commitment_and_schedule(project_runtime) -> None:
    now = datetime(2026, 4, 30, 1, 0, tzinfo=timezone.utc)
    manifest = SimpleNamespace(
        to_message_list=lambda: [
            {
                "role": "system",
                "content": (
                    "Earlier conversation: the OpenCAS agent was working on a book, Chronicle 4246. "
                    "The user asked her to keep working on it, revise, edit, critique "
                    "her own work, and continue until it feels complete."
                ),
            },
            {
                "role": "user",
                "content": "You should not need my approval to follow through on this creative research task.",
            },
        ]
    )

    capture = await capture_project_return_from_turn(
        project_runtime,
        session_id="telegram:private:1",
        user_input="What feels right to you?",
        assistant_content=(
            "This is the anchor I needed. Onnen is attested as a Brythonic woman's "
            "given name. Next I need to fold this naming decision back into the "
            "manuscript and keep revising Chronicle 4246."
        ),
        manifest=manifest,
        now=now,
    )

    assert capture is not None
    assert "revise and finish Chronicle 4246" in capture.project_intent
    commitments = await project_runtime.commitment_store.list_active()
    assert len(commitments) == 1
    commitment = commitments[0]
    assert commitment.content == "Return to project: Chronicle 4246"
    assert "project_return" in commitment.tags
    assert commitment.meta["project_title"] == "Chronicle 4246"
    assert "fold this naming decision" in commitment.meta["next_step"]
    assert "revise and finish Chronicle 4246" in commitment.meta["project_intent"]

    schedules = await project_runtime.ctx.schedule_store.list_items()
    assert len(schedules) == 1
    schedule = schedules[0]
    assert schedule.kind == ScheduleKind.TASK
    assert schedule.action == ScheduleAction.SUBMIT_BAA
    assert schedule.recurrence == ScheduleRecurrence.NONE
    assert schedule.interval_hours is None
    assert schedule.commitment_id == str(commitment.commitment_id)
    assert schedule.start_at == now + timedelta(minutes=5)
    assert "Book-level intent: revise and finish Chronicle 4246" in schedule.objective
    assert "Immediate next step: fold this naming decision" in schedule.objective
    assert "decide whether to continue, finish, or schedule another return" in schedule.objective
    assert "use your OpenCAS calendar to choose and create the next return time" in schedule.objective


@pytest.mark.asyncio
async def test_project_return_capture_deduplicates_existing_project_schedule(project_runtime) -> None:
    now = datetime(2026, 4, 30, 1, 0, tzinfo=timezone.utc)
    kwargs = {
        "runtime": project_runtime,
        "session_id": "telegram:private:1",
        "user_input": "Keep working on Chronicle 4246 until done.",
        "assistant_content": "I need to continue revising Chronicle 4246 after this naming pass.",
        "manifest": SimpleNamespace(to_message_list=lambda: []),
    }

    first = await capture_project_return_from_turn(now=now, **kwargs)
    second = await capture_project_return_from_turn(now=now + timedelta(minutes=1), **kwargs)

    assert first is not None
    assert second is not None
    commitments = await project_runtime.commitment_store.list_active()
    schedules = await project_runtime.ctx.schedule_store.list_items()
    assert len(commitments) == 1
    assert len(schedules) == 1
    assert first.commitment_id == second.commitment_id
    assert first.schedule_id == second.schedule_id


@pytest.mark.asyncio
async def test_project_return_capture_ignores_non_project_chat(project_runtime) -> None:
    capture = await capture_project_return_from_turn(
        project_runtime,
        session_id="chat",
        user_input="Thanks, that answers my question.",
        assistant_content="You're welcome.",
        manifest=SimpleNamespace(to_message_list=lambda: []),
        now=datetime(2026, 4, 30, 1, 0, tzinfo=timezone.utc),
    )

    assert capture is None
    assert await project_runtime.commitment_store.list_active() == []
    assert await project_runtime.ctx.schedule_store.list_items() == []
