"""Focused tests for runtime housekeeping helpers."""

from __future__ import annotations

import asyncio
from datetime import datetime, timedelta, timezone
import json
from pathlib import Path
from types import SimpleNamespace
from uuid import uuid4

import pytest

from opencas.infra import BaaCompletedEvent
from opencas.identity import IdentityManager, IdentityStore
from opencas.runtime.continuity_breadcrumbs import build_runtime_burst_breadcrumb
from opencas.runtime.continuity_breadcrumbs import build_burst_breadcrumb
from opencas.runtime.continuity_breadcrumbs import parse_burst_breadcrumb
from opencas.runtime.continuity_breadcrumbs import is_recoverable_burst_breadcrumb
from opencas.runtime.continuity_breadcrumbs import recover_burst_continuity_context
from opencas.runtime.continuity_breadcrumbs import record_burst_continuity
from opencas.relational import MusubiStore, RelationalEngine
from opencas.runtime.consolidation_state import (
    consolidation_delay_until_due,
    load_consolidation_runtime_state,
    persist_consolidation_runtime_state,
)
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
from opencas.runtime.episodic_runtime import run_runtime_continuity_check
from opencas.tom import IntentionStatus


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
        self.intention = None
        self._task_queue = []

    async def check_goal_resolution(self, output: str):
        return ["ship operator layer"] if "ship" in output else []

    def save_snapshot(self, path: Path) -> None:
        self.saved_paths.append(path)

    @property
    def task_queue(self):
        return list(self._task_queue)


class _FakeTom:
    def __init__(self) -> None:
        self.resolved = []

    async def resolve_intention(self, content: str, status=IntentionStatus.COMPLETED) -> bool:
        self.resolved.append((content, status))
        return True


class _FailingIdentity:
    def __init__(self) -> None:
        self.calls = []

    def record_continuity_breadcrumb(self, *args, **kwargs):
        self.calls.append((args, kwargs))
        raise RuntimeError("identity write failed")


class _RecordingIdentity:
    def __init__(self) -> None:
        self.calls = []

    def record_continuity_breadcrumb(self, *args, **kwargs):
        self.calls.append((args, kwargs))
        return "identity breadcrumb stored"


class _RecordingRelational:
    def __init__(self) -> None:
        self.calls = []
        self.records = []
        self.history = []
        self.state = SimpleNamespace(musubi=0.42)

    async def record_burst_event(self, **kwargs):
        self.calls.append(kwargs)
        breadcrumb = kwargs.get("continuity_breadcrumb")
        if breadcrumb:
            self.history.append(breadcrumb)
        self.records.append(SimpleNamespace(**kwargs))

    async def list_recent_continuity_breadcrumbs(self, limit: int = 5):
        return list(reversed(self.history[-limit:]))

    async def list_recent_burst_records(self, limit: int = 5):
        return list(reversed(self.records[-limit:]))


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
            compact_session=lambda session_id, tail_size=10, min_removed_count=1: _completed(
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
async def test_run_runtime_consolidation_persists_last_run_timestamp(tmp_path: Path) -> None:
    activity = []
    runtime = SimpleNamespace(
        consolidation=SimpleNamespace(
            run=lambda: _completed(
                SimpleNamespace(model_dump=lambda mode='json': {"clusters": 2, "timestamp": "2026-04-21T08:00:00Z"})
            )
        ),
        _set_activity=lambda value: activity.append(value),
        _last_consolidation_result=None,
        ctx=SimpleNamespace(config=SimpleNamespace(state_dir=tmp_path)),
    )

    payload = await run_runtime_consolidation(runtime)

    assert payload["clusters"] == 2
    state = load_consolidation_runtime_state(tmp_path)
    assert state["last_run_at"] == "2026-04-21T08:00:00Z"


@pytest.mark.asyncio
async def test_run_runtime_consolidation_marks_budget_timeout() -> None:
    activity = []

    async def slow_run(**kwargs):
        await asyncio.sleep(0.2)

    runtime = SimpleNamespace(
        consolidation=SimpleNamespace(run=slow_run),
        _set_activity=lambda value: activity.append(value),
        _last_consolidation_result=None,
    )

    payload = await run_runtime_consolidation(
        runtime,
        budget={"max_seconds": 0.01, "max_llm_calls": 1},
    )

    assert payload["budget_exhausted"] is True
    assert payload["budget_reason"] == "timeout"
    assert activity == ["consolidating", "idle"]


def test_consolidation_delay_until_due_uses_persisted_last_run(tmp_path: Path) -> None:
    persist_consolidation_runtime_state(tmp_path, {"last_run_at": "2026-04-21T05:00:00+00:00"})
    now = datetime(2026, 4, 21, 20, 0, tzinfo=timezone.utc)

    delay = consolidation_delay_until_due(
        tmp_path,
        consolidation_interval=86400,
        now=now,
    )

    assert delay == pytest.approx(9 * 3600)


def test_consolidation_delay_until_due_ignores_failed_worker_result(tmp_path: Path) -> None:
    persist_consolidation_runtime_state(
        tmp_path,
        {
            "last_run_at": "2026-04-21T05:00:00+00:00",
            "last_result_id": "worker-timeout-run-1",
        },
    )
    now = datetime(2026, 4, 21, 20, 0, tzinfo=timezone.utc)

    delay = consolidation_delay_until_due(
        tmp_path,
        consolidation_interval=86400,
        now=now,
    )

    assert delay == 0.0


@pytest.mark.parametrize(
    ("trigger", "phase", "intent", "focus", "next_step", "context_note"),
    [
        (
            "work_burst_started",
            "start",
            "Dispatching burst for scheduler resume path",
            "scheduler resume path",
            "await BAA completion and goal resolution",
            "work_id=work-1",
        ),
        (
            "work_burst_completed",
            "end",
            "BAA completion for ship operator layer",
            "ship operator layer",
            "continue draining queue and monitor executive outcomes",
            "task_id=task-1;success=True;decision=task completed and goal checks updated",
        ),
        (
            "work_burst_interrupted",
            "interrupt",
            "Interrupted work burst during runtime shutdown",
            "scheduler resume path",
            "recover the interrupted burst before starting new work",
            "shutdown interruption",
        ),
    ],
)
@pytest.mark.asyncio
async def test_record_burst_continuity_writes_the_same_short_note_for_each_burst_boundary(
    trigger: str,
    phase: str,
    intent: str,
    focus: str,
    next_step: str,
    context_note: str,
) -> None:
    identity = _RecordingIdentity()
    relational = _RecordingRelational()
    runtime = SimpleNamespace(
        ctx=SimpleNamespace(
            identity=identity,
            relational=relational,
        ),
        _trace=lambda *args, **kwargs: None,
    )
    fixed_timestamp = datetime(2026, 4, 18, 0, 0, tzinfo=timezone.utc)
    expected_breadcrumb = build_runtime_burst_breadcrumb(
        runtime,
        phase=phase,
        intent=intent,
        focus=focus,
        next_step=next_step,
        note=context_note,
        timestamp=fixed_timestamp,
    )

    await record_burst_continuity(
        runtime,
        trigger=trigger,
        phase=phase,
        intent=intent,
        focus=focus,
        next_step=next_step,
        note=context_note,
        timestamp=fixed_timestamp,
    )

    assert identity.calls
    assert relational.calls
    identity_kwargs = identity.calls[0][1]
    identity_note = identity_kwargs["note"]
    relational_note = relational.calls[0]["note"]
    assert identity_kwargs["decision"] == f"{phase} burst boundary captured"
    assert identity_note == expected_breadcrumb.note
    assert relational_note == expected_breadcrumb.note
    assert f"phase: {phase}" in relational.calls[0]["continuity_breadcrumb"]
    assert len(identity_note) <= 320
    assert len(relational_note) <= 320
    assert relational_note.startswith("timestamp=2026-04-18T00:00:00+00:00;")
    assert "intent=" in relational_note
    assert "branch=" in relational_note
    assert "next_recovery_cue=" in relational_note


def test_build_runtime_burst_breadcrumb_tracks_current_musubi_state() -> None:
    runtime = SimpleNamespace(
        ctx=SimpleNamespace(
            relational=SimpleNamespace(state=SimpleNamespace(musubi=0.18))
        )
    )

    first = build_runtime_burst_breadcrumb(
        runtime,
        phase="start",
        intent="Dispatching burst for scheduler resume path",
        focus="scheduler resume path",
        next_step="await BAA completion and goal resolution",
        note="scheduler resume path",
        timestamp=datetime(2026, 4, 18, 0, 0, tzinfo=timezone.utc),
    )
    runtime.ctx.relational.state.musubi = 0.73
    second = build_runtime_burst_breadcrumb(
        runtime,
        phase="start",
        intent="Dispatching burst for scheduler resume path",
        focus="scheduler resume path",
        next_step="await BAA completion and goal resolution",
        note="scheduler resume path",
        timestamp=datetime(2026, 4, 18, 0, 0, tzinfo=timezone.utc),
    )

    assert first.note.startswith("timestamp=2026-04-18T00:00:00+00:00;")
    assert second.note.startswith("timestamp=2026-04-18T00:00:00+00:00;")
    assert "branch=scheduler resume path" in first.note
    assert "branch=scheduler resume path" in second.note
    assert "next_recovery_cue=await BAA completion and goal resolution" in first.note
    assert "next_recovery_cue=await BAA completion and goal resolution" in second.note
    assert first.note.endswith("musubi=+0.18")
    assert second.note.endswith("musubi=+0.73")
    assert first.note != second.note
    assert len(first.note) <= 240
    assert len(second.note) <= 240


def test_build_runtime_burst_breadcrumb_falls_back_to_somatic_musubi() -> None:
    runtime = SimpleNamespace(
        ctx=SimpleNamespace(
            relational=SimpleNamespace(state=SimpleNamespace(musubi=None)),
            somatic=SimpleNamespace(state=SimpleNamespace(musubi=0.57)),
        )
    )

    breadcrumb = build_runtime_burst_breadcrumb(
        runtime,
        phase="start",
        intent="Dispatching burst for scheduler resume path",
        focus="scheduler resume path",
        next_step="await BAA completion and goal resolution",
    )

    assert breadcrumb.note.endswith("musubi=+0.57")
    assert len(breadcrumb.note) <= 240


@pytest.mark.asyncio
async def test_handle_runtime_baa_completed_records_goal_events_and_snapshot(tmp_path: Path) -> None:
    somatic = _FakeSomatic()
    executive = _FakeExecutive()
    tom = _FakeTom()
    runtime = SimpleNamespace(
        executive=executive,
        tom=tom,
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
    assert tom.resolved == [("ship operator layer", IntentionStatus.COMPLETED)]
    assert executive.saved_paths == [tmp_path / "executive.json"]


@pytest.mark.asyncio
async def test_handle_runtime_baa_completed_falls_back_to_musubi_history_when_identity_write_fails(
    tmp_path: Path,
) -> None:
    somatic = _FakeSomatic()
    executive = _FakeExecutive()
    relational = _RecordingRelational()
    runtime = SimpleNamespace(
        executive=executive,
        ctx=SimpleNamespace(
            somatic=somatic,
            relational=relational,
            config=SimpleNamespace(state_dir=tmp_path),
            identity=SimpleNamespace(record_continuity_breadcrumb=_FailingIdentity().record_continuity_breadcrumb),
        ),
        _trace=lambda *args, **kwargs: None,
    )

    await handle_runtime_baa_completed(
        runtime,
        BaaCompletedEvent(
            task_id="task-2",
            objective="ship operator layer",
            success=False,
            output="blocked on missing review",
            stage="micro_task",
        ),
    )

    assert relational.calls
    assert relational.calls[0]["trigger"] == "work_burst_completed"
    assert "phase: end" in relational.calls[0]["continuity_breadcrumb"]
    assert "last_action: ship operator layer" in relational.calls[0]["continuity_breadcrumb"]
    assert "next_resume_point: surface recovery guidance and continue executive recovery" in relational.calls[0]["continuity_breadcrumb"]
    assert "intent: BAA completion for ship operator layer" in relational.calls[0]["continuity_breadcrumb"]
    assert relational.calls[0]["note"].startswith("timestamp=")
    assert "branch=ship operator layer" in relational.calls[0]["note"]
    assert "intent=BAA completion for ship operator layer" in relational.calls[0]["note"]
    assert "next_recovery_cue=surface recovery guidance" in relational.calls[0]["note"]


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


@pytest.mark.asyncio
async def test_shutdown_runtime_resources_records_an_interruption_breadcrumb(tmp_path: Path) -> None:
    identity = IdentityStore(tmp_path / "identity")
    mgr = IdentityManager(identity)
    mgr.load()
    mgr.record_continuity_breadcrumb(
        intent="dispatch burst for scheduler resume path",
        decision="focus: scheduler resume path; musubi=+0.42",
        next_step="await completion and recover if interrupted",
    )
    mgr.record_shutdown(session_id="sess-1")

    relational = _RecordingRelational()

    async def _close_ctx() -> None:
        return None

    runtime = SimpleNamespace(
        reliability=None,
        process_supervisor=None,
        pty_supervisor=None,
        browser_supervisor=None,
        _telegram=None,
        ctx=SimpleNamespace(
            identity=mgr,
            relational=relational,
            close=_close_ctx,
        ),
        executive=SimpleNamespace(
            intention="scheduler resume path",
            task_queue=[],
        ),
        _trace=lambda *args, **kwargs: None,
    )

    await close_runtime_stores(runtime)

    assert relational.calls
    assert relational.calls[0]["trigger"] == "work_burst_interrupted"
    assert "recover the interrupted burst" in relational.calls[0]["continuity_breadcrumb"]
    assert "last_action: scheduler resume path" in relational.calls[0]["continuity_breadcrumb"]
    assert "next_resume_point: recover the interrupted burst before starting new work" in relational.calls[0]["continuity_breadcrumb"]
    assert relational.calls[0]["note"].startswith("timestamp=")
    assert "branch=scheduler resume path" in relational.calls[0]["note"]
    assert "intent=Interrupted work burst during runtime shutdown" in relational.calls[0]["note"]
    assert "next_recovery_cue=recover the interrupted burst before starting new work" in relational.calls[0]["note"]


@pytest.mark.asyncio
async def test_next_session_recovers_intent_from_latest_breadcrumb(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    store = IdentityStore(tmp_path / "identity")
    mgr = IdentityManager(store)
    mgr.load()
    burst = build_burst_breadcrumb(
        phase="start",
        intent="dispatch burst for scheduler resume path",
        focus="scheduler resume path",
        next_step="await completion and recover if interrupted",
        note="resume checkpoint: preserve scheduler handoff",
        musubi=0.42,
    )
    breadcrumb = mgr.record_continuity_breadcrumb(
        intent=f"{burst.phase} burst: {burst.intent}",
        decision="burst boundary captured",
        note=burst.note,
        next_step=burst.next_step,
    )
    mgr.record_shutdown(session_id="sess-1")
    mgr._continuity.last_shutdown_time = datetime.now(timezone.utc) - timedelta(hours=2)
    mgr.save()

    runtime = SimpleNamespace(
        ctx=SimpleNamespace(
            config=SimpleNamespace(session_id="sess-2", continuous_present_enabled=True),
            identity=mgr,
            relational=SimpleNamespace(state=SimpleNamespace(musubi=0.73)),
            somatic=SimpleNamespace(
                state=SimpleNamespace(fatigue=0.0, tension=0.0),
                emit_appraisal_event=lambda *args, **kwargs: None,
            ),
        ),
        memory=SimpleNamespace(list_episodes=lambda *args, **kwargs: _completed([])),
        _trace=lambda *args, **kwargs: None,
    )

    async def _fake_record_runtime_episode(*args, **kwargs):
        return None

    monkeypatch.setattr("opencas.runtime.episodic_runtime.record_runtime_episode", _fake_record_runtime_episode)

    await run_runtime_continuity_check(runtime)

    assert breadcrumb in mgr.continuity.continuity_breadcrumbs
    assert "Most recent work-burst breadcrumb:" in mgr.continuity.last_continuity_monologue
    assert "current musubi=+0.73" in mgr.continuity.last_continuity_monologue
    assert "intent=dispatch burst for scheduler resume path" in mgr.continuity.last_continuity_monologue
    assert "branch=scheduler resume path" in mgr.continuity.last_continuity_monologue
    assert "next_recovery_cue=await completion and recover if interrupted" in mgr.continuity.last_continuity_monologue
    expected_recovery = recover_burst_continuity_context(
        breadcrumb,
        0.73,
        note=burst.note,
    )
    assert expected_recovery is not None
    assert expected_recovery in mgr.continuity.last_continuity_monologue
    assert "intent: resume context after 2.0 hours" in mgr.continuity.continuity_breadcrumbs[-1]
    assert is_recoverable_burst_breadcrumb(breadcrumb, 0.73, note=burst.note)


@pytest.mark.asyncio
async def test_next_session_recovers_burst_context_from_musubi_history_fallback(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    store = IdentityStore(tmp_path / "identity")
    mgr = IdentityManager(store)
    mgr.load()
    relational = _RecordingRelational()
    seed_runtime = SimpleNamespace(
        ctx=SimpleNamespace(
            identity=_FailingIdentity(),
            relational=relational,
        ),
        _trace=lambda *args, **kwargs: None,
    )
    await record_burst_continuity(
        seed_runtime,
        trigger="work_burst_interrupted",
        phase="interrupt",
        intent="Interrupted work burst during runtime shutdown",
        focus="scheduler resume path",
        next_step="recover the interrupted burst before starting new work",
        note="shutdown interruption",
    )
    burst_record = relational.records[-1]
    assert burst_record.note is not None
    assert len(burst_record.note) <= 240
    relational.state = SimpleNamespace(musubi=0.73)
    mgr.record_shutdown(session_id="sess-2")
    mgr._continuity.last_shutdown_time = datetime.now(timezone.utc) - timedelta(hours=2)
    mgr.save()

    runtime = SimpleNamespace(
        ctx=SimpleNamespace(
            config=SimpleNamespace(session_id="sess-3", continuous_present_enabled=True),
            identity=mgr,
            relational=relational,
            somatic=SimpleNamespace(
                state=SimpleNamespace(fatigue=0.0, tension=0.0),
                emit_appraisal_event=lambda *args, **kwargs: None,
            ),
        ),
        memory=SimpleNamespace(list_episodes=lambda *args, **kwargs: _completed([])),
        _trace=lambda *args, **kwargs: None,
    )

    async def _fake_record_runtime_episode(*args, **kwargs):
        return None

    monkeypatch.setattr("opencas.runtime.episodic_runtime.record_runtime_episode", _fake_record_runtime_episode)

    await run_runtime_continuity_check(runtime)

    assert "Most recent work-burst breadcrumb:" in mgr.continuity.last_continuity_monologue
    assert "branch=scheduler resume path" in mgr.continuity.last_continuity_monologue
    assert burst_record.note.startswith("timestamp=")
    assert "branch=scheduler resume path" in burst_record.note
    assert "current musubi=+0.73" in mgr.continuity.last_continuity_monologue
    assert recover_burst_continuity_context(
        burst_record.continuity_breadcrumb,
        0.73,
        note=burst_record.note,
    ) in mgr.continuity.last_continuity_monologue


@pytest.mark.asyncio
async def test_restart_recovers_burst_intent_from_musubi_state_without_history(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    identity_store = IdentityStore(tmp_path / "identity")
    mgr = IdentityManager(identity_store)
    mgr.load()

    relational_store = MusubiStore(tmp_path / "musubi.db")
    relational = RelationalEngine(relational_store)
    await relational.connect()

    seed_runtime = SimpleNamespace(
        ctx=SimpleNamespace(
            identity=_FailingIdentity(),
            relational=relational,
        ),
        _trace=lambda *args, **kwargs: None,
    )
    await record_burst_continuity(
        seed_runtime,
        trigger="work_burst_started",
        phase="start",
        intent="dispatch burst for scheduler resume path",
        focus="scheduler resume path",
        next_step="await completion and recover if interrupted",
    )

    assert relational.state.continuity_breadcrumb.startswith("timestamp=")
    assert "branch=scheduler resume path" in relational.state.continuity_breadcrumb
    assert "intent=dispatch burst for scheduler resume path" in relational.state.continuity_breadcrumb
    assert "next_recovery_cue=await completion and recover if interrupted" in relational.state.continuity_breadcrumb

    await relational.store._db.execute("DELETE FROM musubi_history")
    await relational.store._db.commit()
    await relational.close()

    restarted_store = MusubiStore(tmp_path / "musubi.db")
    restarted_relational = RelationalEngine(restarted_store)
    await restarted_relational.connect()
    assert restarted_relational.state.continuity_breadcrumb == relational.state.continuity_breadcrumb

    mgr.record_shutdown(session_id="sess-2")
    mgr._continuity.last_shutdown_time = datetime.now(timezone.utc) - timedelta(hours=2)
    mgr.save()

    runtime = SimpleNamespace(
        ctx=SimpleNamespace(
            config=SimpleNamespace(session_id="sess-3", continuous_present_enabled=True),
            identity=mgr,
            relational=restarted_relational,
            somatic=SimpleNamespace(
                state=SimpleNamespace(fatigue=0.0, tension=0.0),
                emit_appraisal_event=lambda *args, **kwargs: None,
            ),
        ),
        memory=SimpleNamespace(list_episodes=lambda *args, **kwargs: _completed([])),
        _trace=lambda *args, **kwargs: None,
    )

    async def _fake_record_runtime_episode(*args, **kwargs):
        return None

    monkeypatch.setattr("opencas.runtime.episodic_runtime.record_runtime_episode", _fake_record_runtime_episode)

    await run_runtime_continuity_check(runtime)

    assert "Most recent work-burst breadcrumb:" in mgr.continuity.last_continuity_monologue
    assert "current musubi=+0.00" in mgr.continuity.last_continuity_monologue
    assert "next_recovery_cue=await completion and recover if interrupted" in mgr.continuity.last_continuity_monologue
    assert "branch=scheduler resume path" in mgr.continuity.last_continuity_monologue

    await restarted_relational.close()


def test_burst_breadcrumb_note_is_recoverable_with_delimiters() -> None:
    runtime = SimpleNamespace(
        ctx=SimpleNamespace(
            relational=SimpleNamespace(state=SimpleNamespace(musubi=0.61))
        )
    )
    breadcrumb = build_runtime_burst_breadcrumb(
        runtime,
        phase="start",
        intent="dispatch burst for scheduler resume path",
        focus="scheduler resume path",
        next_step="recover burst after interruption",
        note="dispatch burst; branch=keep moving | carefully",
    )

    assert "branch=scheduler resume path" in breadcrumb.note
    assert "next_recovery_cue=recover burst after interruption" in breadcrumb.note
    assert breadcrumb.comment == "dispatch burst; branch=keep moving | carefully"
    assert is_recoverable_burst_breadcrumb(breadcrumb.breadcrumb, 0.61, note=breadcrumb.note)


def test_parse_burst_breadcrumb_accepts_legacy_note_shape() -> None:
    legacy_note = (
        "timestamp=2026-04-18T00:00:00+00:00;"
        "intent=dispatch burst for scheduler resume path;"
        "last_action=scheduler resume path;"
        "next_resume_point=await completion and recover if interrupted;"
        "musubi=+0.42"
    )

    parsed = parse_burst_breadcrumb(legacy_note)

    assert parsed is not None
    assert parsed.intent == "dispatch burst for scheduler resume path"
    assert parsed.branch == "scheduler resume path"
    assert parsed.next_step == "await completion and recover if interrupted"


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

def test_build_continuity_note_can_carry_salvage_linkage_without_losing_short_form():
    from opencas.runtime.continuity_breadcrumbs import build_continuity_note
    note = build_continuity_note(
        note="verification failed",
        objective="continue chronicle 4246",
        handoff="resume chapter repair from canonical manuscript",
        musubi=0.42,
        timestamp="2026-04-18T00:00:00+00:00",
        salvage_packet_id="packet-1",
        project_signature="chronicle-4246",
    )

    assert "salvage_packet_id=packet-1" in note
    assert "project_signature=chronicle-4246" in note
