"""Focused tests for runtime housekeeping helpers."""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace
from uuid import uuid4

import pytest

from opencas.infra import BaaCompletedEvent
from opencas.runtime.maintenance_runtime import (
    close_runtime_stores,
    extract_runtime_response_content,
    handle_runtime_baa_completed,
    maybe_compact_runtime_session,
    maybe_record_runtime_somatic_snapshot,
    run_runtime_consolidation,
    sync_runtime_executive_snapshot,
    trace_runtime_event,
)


class _FakeSomatic:
    def __init__(self, store=True) -> None:
        self.store = object() if store else None
        self.snapshots = []
        self.events = []

    async def record_snapshot(self, source: str, trigger_event_id=None) -> None:
        self.snapshots.append((source, trigger_event_id))

    async def emit_appraisal_event(self, event_type, source_text="", trigger_event_id=None, meta=None):
        self.events.append((event_type, source_text, trigger_event_id, meta or {}))


class _FakeExecutive:
    def __init__(self) -> None:
        self.saved_paths = []

    async def check_goal_resolution(self, output: str):
        return ["ship operator layer"] if "ship" in output else []

    def save_snapshot(self, path: Path) -> None:
        self.saved_paths.append(path)


class _FakeTracer:
    def __init__(self) -> None:
        self.calls = []

    def log(self, kind, message, payload):
        self.calls.append((kind, message, payload))


@pytest.mark.asyncio
async def test_maybe_compact_runtime_session_traces_record() -> None:
    traced = []
    runtime = SimpleNamespace(
        compactor=SimpleNamespace(
            compact_session=lambda session_id, tail_size=10: _completed(
                SimpleNamespace(
                    removed_count=4,
                    compaction_id=uuid4(),
                )
            )
        ),
        _trace=lambda event, payload=None: traced.append((event, payload or {})),
    )

    await maybe_compact_runtime_session(runtime, "session-1")

    assert traced
    assert traced[0][0] == "compaction_triggered"
    assert traced[0][1]["session_id"] == "session-1"


@pytest.mark.asyncio
async def test_run_runtime_consolidation_updates_activity_and_payload() -> None:
    activity = []
    runtime = SimpleNamespace(
        consolidation=SimpleNamespace(run=lambda: _completed(SimpleNamespace(model_dump=lambda mode='json': {"clusters": 2}))),
        _set_activity=lambda value: activity.append(value),
        _last_consolidation_result=None,
    )

    payload = await run_runtime_consolidation(runtime)

    assert payload == {"clusters": 2}
    assert runtime._last_consolidation_result == {"clusters": 2}
    assert activity == ["consolidating", "idle"]


@pytest.mark.asyncio
async def test_handle_runtime_baa_completed_records_goal_events_and_snapshot(tmp_path: Path) -> None:
    somatic = _FakeSomatic()
    executive = _FakeExecutive()
    runtime = SimpleNamespace(
        executive=executive,
        ctx=SimpleNamespace(
            somatic=somatic,
            config=SimpleNamespace(state_dir=tmp_path),
        ),
    )

    await handle_runtime_baa_completed(
        runtime,
        BaaCompletedEvent(
            task_id="task-1",
            objective="ship operator layer",
            success=True,
            output="ship operator layer complete",
            stage="micro_task",
        ),
    )

    assert somatic.events
    assert "Goal achieved" in somatic.events[0][1]
    assert executive.saved_paths == [tmp_path / "executive.json"]


@pytest.mark.asyncio
async def test_maybe_record_runtime_somatic_snapshot_respects_store_flag() -> None:
    runtime = SimpleNamespace(ctx=SimpleNamespace(somatic=_FakeSomatic(store=True)))
    await maybe_record_runtime_somatic_snapshot(runtime, "turn_complete", "evt-1")
    assert runtime.ctx.somatic.snapshots == [("turn_complete", "evt-1")]

    runtime_no_store = SimpleNamespace(ctx=SimpleNamespace(somatic=_FakeSomatic(store=False)))
    await maybe_record_runtime_somatic_snapshot(runtime_no_store, "turn_complete", "evt-1")
    assert runtime_no_store.ctx.somatic.snapshots == []


@pytest.mark.asyncio
async def test_close_runtime_stores_delegates_to_lifecycle(monkeypatch: pytest.MonkeyPatch) -> None:
    called = []
    async def _fake_shutdown(runtime):
        called.append(runtime)
    monkeypatch.setattr("opencas.runtime.maintenance_runtime.shutdown_runtime_resources", _fake_shutdown)

    runtime = SimpleNamespace()
    await close_runtime_stores(runtime)

    assert called == [runtime]


def test_extract_runtime_response_content_and_trace_runtime_event() -> None:
    tracer = _FakeTracer()
    runtime = SimpleNamespace(tracer=tracer)

    assert extract_runtime_response_content({"choices": [{"message": {"content": "ok"}}]}) == "ok"
    assert extract_runtime_response_content({}) == ""

    trace_runtime_event(runtime, "tool_finished", {"ok": True})

    assert tracer.calls
    assert tracer.calls[0][1] == "AgentRuntime: tool_finished"


def _completed(value):
    async def _inner():
        return value
    return _inner()
