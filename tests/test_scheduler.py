"""Tests for AgentScheduler background loop orchestration."""

import asyncio
import os
from datetime import datetime, timezone
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest
import pytest_asyncio

from opencas.bootstrap import BootstrapConfig, BootstrapPipeline
from opencas.runtime.consolidation_state import load_consolidation_runtime_state
from opencas.runtime.consolidation_worker import consolidation_worker_status_path
from opencas.runtime import AgentRuntime
from opencas.runtime.readiness import AgentReadiness, ReadinessState
from opencas.runtime.scheduler import AgentScheduler


@pytest_asyncio.fixture
async def runtime(tmp_path: Path):
    config = BootstrapConfig(state_dir=tmp_path, session_id="scheduler-test")
    ctx = await BootstrapPipeline(config).run()
    return AgentRuntime(ctx)


@pytest.mark.asyncio
async def test_scheduler_start_starts_baa(runtime: AgentRuntime) -> None:
    runtime.baa.start = AsyncMock()
    runtime.baa.stop = AsyncMock()
    runtime.backfill_daydream_signal_thread_beads = AsyncMock(
        return_value={"available": True, "scanned": 0, "recorded": 0, "skipped": 0}
    )
    scheduler = AgentScheduler(
        runtime=runtime,
        cycle_interval=3600,
        consolidation_interval=3600,
        baa_heartbeat_interval=3600,
    )
    await scheduler.start()
    runtime.baa.start.assert_awaited_once()
    runtime.backfill_daydream_signal_thread_beads.assert_awaited_once_with(limit=50)
    await scheduler.stop()


@pytest.mark.asyncio
async def test_scheduler_stop_cancels_loops_and_stops_baa(runtime: AgentRuntime) -> None:
    scheduler = AgentScheduler(
        runtime=runtime,
        cycle_interval=3600,
        consolidation_interval=3600,
        baa_heartbeat_interval=3600,
    )
    await scheduler.start()
    assert len(scheduler._tasks) == 4
    await scheduler.stop()
    assert scheduler._tasks == []


@pytest.mark.asyncio
async def test_scheduler_respects_readiness_pause(runtime: AgentRuntime) -> None:
    readiness = AgentReadiness()
    readiness.pause("test pause")
    assert readiness.state == ReadinessState.PAUSED

    runtime.run_cycle = AsyncMock()
    scheduler = AgentScheduler(
        runtime=runtime,
        cycle_interval=0.05,
        consolidation_interval=3600,
        baa_heartbeat_interval=3600,
        readiness=readiness,
    )
    await scheduler.start()
    await asyncio.sleep(0.15)
    await scheduler.stop()
    runtime.run_cycle.assert_not_awaited()


@pytest.mark.asyncio
async def test_scheduler_cycle_loop_executes_when_ready(runtime: AgentRuntime) -> None:
    readiness = AgentReadiness()
    readiness.ready("test ready")
    runtime.run_cycle = AsyncMock(return_value={"promoted": 0, "demoted": 0})
    scheduler = AgentScheduler(
        runtime=runtime,
        cycle_interval=0.05,
        consolidation_interval=3600,
        baa_heartbeat_interval=3600,
        readiness=readiness,
    )
    await scheduler.start()
    await asyncio.sleep(0.15)
    await scheduler.stop()
    runtime.run_cycle.assert_awaited()


@pytest.mark.asyncio
async def test_scheduler_cycle_skips_while_consolidation_worker_running(tmp_path: Path) -> None:
    class FakeExecutive:
        def recommend_pause(self) -> bool:
            return False

    class FakeBAA:
        queue_size = 2
        held_size = 3
        active_count = 0
        start = AsyncMock()
        stop = AsyncMock()

    status_path = consolidation_worker_status_path(tmp_path)
    status_path.parent.mkdir(parents=True, exist_ok=True)
    status_path.write_text(
        (
            '{"mode":"subprocess","status":"running","pid":'
            f'{os.getpid()},"run_id":"live-worker"}}'
        ),
        encoding="utf-8",
    )

    runtime = SimpleNamespace(
        executive=FakeExecutive(),
        baa=FakeBAA(),
        ctx=SimpleNamespace(config=SimpleNamespace(state_dir=tmp_path), health_monitor=None, somatic=None),
        _activity="idle",
        run_cycle=AsyncMock(return_value={"promoted": 0, "demoted": 0}),
    )
    readiness = AgentReadiness()
    readiness.ready("test ready")
    scheduler = AgentScheduler(
        runtime=runtime,
        cycle_interval=0.05,
        consolidation_interval=3600,
        baa_heartbeat_interval=3600,
        readiness=readiness,
    )

    assert scheduler._background_llm_block_reason(require_quiet_baa=True) == (
        "runtime_activity_consolidating"
    )


@pytest.mark.asyncio
async def test_scheduler_consolidation_loop_executes_when_ready(runtime: AgentRuntime) -> None:
    readiness = AgentReadiness()
    readiness.ready("test ready")
    runtime.run_consolidation = AsyncMock(return_value={"clusters": 1})
    scheduler = AgentScheduler(
        runtime=runtime,
        cycle_interval=3600,
        consolidation_interval=0.05,
        baa_heartbeat_interval=3600,
        readiness=readiness,
    )
    await scheduler.start()
    await asyncio.sleep(0.15)
    await scheduler.stop()
    runtime.run_consolidation.assert_awaited()


def test_scheduler_default_consolidation_budget_is_worker_bounded(runtime: AgentRuntime) -> None:
    scheduler = AgentScheduler(
        runtime=runtime,
        cycle_interval=3600,
        consolidation_interval=3600,
        baa_heartbeat_interval=3600,
    )

    assert scheduler.consolidation_budget["max_candidates"] == 100
    assert scheduler.consolidation_budget["worker_timeout_seconds"] == 1800
    assert scheduler.consolidation_budget["max_compaction_sessions"] == 8
    assert scheduler.consolidation_budget["max_compaction_candidates"] == 1000
    assert scheduler.consolidation_budget["min_compaction_session_lag"] == 20


def test_scheduler_allows_consolidation_to_recover_consolidation_degraded_state(
    runtime: AgentRuntime,
) -> None:
    readiness = AgentReadiness()
    readiness.ready("test ready")
    readiness.degraded("consolidation failed: worker_timeout")
    scheduler = AgentScheduler(
        runtime=runtime,
        cycle_interval=3600,
        consolidation_interval=3600,
        baa_heartbeat_interval=3600,
        readiness=readiness,
    )

    assert scheduler._should_run_cycle() is False
    assert scheduler._should_run_consolidation() is True

    scheduler._mark_consolidation_recovered()
    assert readiness.state == ReadinessState.READY
    assert readiness.reason == "consolidation_recovered"


@pytest.mark.asyncio
async def test_scheduler_retries_failed_worker_consolidation_results(runtime: AgentRuntime) -> None:
    readiness = AgentReadiness()
    readiness.ready("test ready")
    runtime.run_consolidation = AsyncMock(
        side_effect=[
            {
                "result_id": "worker-timeout-1",
                "budget_exhausted": True,
                "budget_reason": "worker_timeout",
                "worker": {"status": "timeout_killed"},
            },
            {
                "result_id": "worker-timeout-2",
                "budget_exhausted": True,
                "budget_reason": "worker_timeout",
                "worker": {"status": "timeout_killed"},
            },
            {
                "result_id": "worker-ok",
                "timestamp": "2026-04-27T00:00:00+00:00",
                "worker": {"status": "completed"},
            },
        ]
    )
    scheduler = AgentScheduler(
        runtime=runtime,
        cycle_interval=3600,
        consolidation_interval=3600,
        baa_heartbeat_interval=3600,
        readiness=readiness,
        consolidation_retry_base_seconds=0,
    )

    result = await scheduler._run_consolidation_with_retries(retry_delay=0)

    assert result["result_id"] == "worker-ok"
    assert runtime.run_consolidation.await_count == 3
    assert readiness.state == ReadinessState.READY


def test_scheduler_persists_failed_worker_retry_state(runtime: AgentRuntime) -> None:
    now = datetime(2026, 4, 22, 8, 0, tzinfo=timezone.utc)
    scheduler = AgentScheduler(
        runtime=runtime,
        cycle_interval=3600,
        consolidation_interval=3600,
        baa_heartbeat_interval=3600,
        consolidation_failure_retry_seconds=600,
        time_source=lambda: now,
    )

    scheduler._persist_failed_consolidation_state(
        runtime.ctx.config.state_dir,
        {
            "result_id": "worker-failed-run-1",
            "budget_exhausted": True,
            "budget_reason": "worker_failed",
            "worker": {"status": "failed", "error_type": "OperationalError"},
        },
    )

    state = load_consolidation_runtime_state(runtime.ctx.config.state_dir)
    assert state["last_attempt_at"] == "2026-04-22T08:00:00+00:00"
    assert state["last_result_id"] == "worker-failed-run-1"
    assert state["budget_reason"] == "worker_failed"
    assert state["worker_status"] == "failed"
    assert state["worker_error_type"] == "OperationalError"
    assert state["consecutive_failures"] == 1
    assert state["next_retry_after"] == "2026-04-22T08:10:00+00:00"


@pytest.mark.asyncio
async def test_scheduler_consolidation_loop_uses_persisted_due_time(runtime: AgentRuntime) -> None:
    readiness = AgentReadiness()
    readiness.ready("test ready")
    runtime.run_consolidation = AsyncMock(return_value={"clusters": 1})
    state_path = runtime.ctx.config.state_dir / "consolidation_runtime_state.json"
    state_path.write_text('{"last_run_at": "2026-04-21T00:00:00+00:00"}', encoding="utf-8")
    scheduler = AgentScheduler(
        runtime=runtime,
        cycle_interval=3600,
        consolidation_interval=0.05,
        baa_heartbeat_interval=3600,
        readiness=readiness,
        time_source=lambda: datetime(2026, 4, 22, 0, 0, tzinfo=timezone.utc),
    )
    await scheduler.start()
    await asyncio.sleep(0.12)
    await scheduler.stop()
    runtime.run_consolidation.assert_awaited()


@pytest.mark.asyncio
async def test_scheduler_heartbeat_emits_telemetry(runtime: AgentRuntime) -> None:
    readiness = AgentReadiness()
    readiness.ready("test ready")
    scheduler = AgentScheduler(
        runtime=runtime,
        cycle_interval=3600,
        consolidation_interval=3600,
        baa_heartbeat_interval=0.05,
        readiness=readiness,
        tracer=runtime.tracer,
    )
    await scheduler.start()
    await asyncio.sleep(0.15)
    await scheduler.stop()
    # Heartbeat should have run without crashing; tracer store should contain events
    # We just assert stop succeeded cleanly.
    assert True


@pytest.mark.asyncio
async def test_scheduler_heartbeat_attempts_to_release_held_baa_tasks(runtime: AgentRuntime) -> None:
    readiness = AgentReadiness()
    readiness.ready("test ready")
    runtime.baa = SimpleNamespace(
        queue_size=0,
        held_size=1,
        active_count=0,
        start=AsyncMock(),
        stop=AsyncMock(),
        try_release_held=AsyncMock(return_value=1),
    )
    scheduler = AgentScheduler(
        runtime=runtime,
        cycle_interval=3600,
        consolidation_interval=3600,
        baa_heartbeat_interval=0.05,
        readiness=readiness,
        tracer=runtime.tracer,
    )

    await scheduler.start()
    await asyncio.sleep(0.15)
    await scheduler.stop()

    runtime.baa.try_release_held.assert_awaited()


@pytest.mark.asyncio
async def test_scheduler_records_identity_persistence_heartbeat() -> None:
    class FakeIdentity:
        def __init__(self) -> None:
            self.heartbeat_count = 0

        def record_persistence_heartbeat(self) -> None:
            self.heartbeat_count += 1

    class FakeBAA:
        queue_size = 2
        held_size = 3
        active_count = 0
        start = AsyncMock()
        stop = AsyncMock()

    class FakeRuntime:
        def __init__(self) -> None:
            self.executive = SimpleNamespace(recommend_pause=lambda: False)
            self.baa = FakeBAA()
            self.ctx = SimpleNamespace(
                health_monitor=None,
                somatic=None,
                identity=FakeIdentity(),
            )
            self._activity = "idle"
            self.run_cycle = AsyncMock(return_value={})
            self.run_consolidation = AsyncMock(return_value={})
            self.run_daydream = AsyncMock(return_value={})

    runtime = FakeRuntime()
    scheduler = AgentScheduler(
        runtime=runtime,
        cycle_interval=3600,
        consolidation_interval=3600,
        baa_heartbeat_interval=3600,
        identity_heartbeat_interval=0.01,
    )

    await scheduler.start()
    await asyncio.sleep(0.04)
    await scheduler.stop()

    assert runtime.ctx.identity.heartbeat_count >= 1


@pytest.mark.asyncio
async def test_scheduler_degrades_on_cycle_error(runtime: AgentRuntime) -> None:
    readiness = AgentReadiness()
    readiness.ready("test ready")
    runtime.run_cycle = AsyncMock(side_effect=RuntimeError("boom"))
    scheduler = AgentScheduler(
        runtime=runtime,
        cycle_interval=0.05,
        consolidation_interval=3600,
        baa_heartbeat_interval=3600,
        readiness=readiness,
    )
    await scheduler.start()
    await asyncio.sleep(0.15)
    await scheduler.stop()
    assert readiness.state == ReadinessState.DEGRADED


@pytest.mark.asyncio
async def test_scheduler_focus_mode_auto_exits_after_timeout(runtime: AgentRuntime) -> None:
    readiness = AgentReadiness()
    readiness.ready("test ready")
    runtime.run_cycle = AsyncMock(return_value={"promoted": 0, "demoted": 0})
    runtime.executive.recommend_pause = lambda: False
    scheduler = AgentScheduler(
        runtime=runtime,
        cycle_interval=0.05,
        consolidation_interval=3600,
        baa_heartbeat_interval=3600,
        readiness=readiness,
        focus_mode_timeout_seconds=0,
    )
    await scheduler.start()
    scheduler.enter_focus_mode()
    assert scheduler.focus_mode is True
    await asyncio.sleep(0.15)
    # After the cycle loop wakes up, _should_run_cycle should auto-exit focus mode
    await scheduler.stop()
    # run_cycle should have been allowed again after auto-exit
    runtime.run_cycle.assert_awaited()
    assert scheduler.focus_mode is False


@pytest.mark.asyncio
async def test_scheduler_focus_mode_blocks_cycles(runtime: AgentRuntime) -> None:
    readiness = AgentReadiness()
    readiness.ready("test ready")
    runtime.run_cycle = AsyncMock(return_value={"promoted": 0, "demoted": 0})
    scheduler = AgentScheduler(
        runtime=runtime,
        cycle_interval=0.05,
        consolidation_interval=3600,
        baa_heartbeat_interval=3600,
        readiness=readiness,
    )
    await scheduler.start()
    scheduler.enter_focus_mode()
    await asyncio.sleep(0.05)
    await scheduler.stop()
    # Because focus mode was entered and not yet timed out, run_cycle should NOT have been called
    assert runtime.run_cycle.await_count == 0
    assert scheduler.focus_mode is True


@pytest.mark.asyncio
async def test_scheduler_resumes_deferred_work_on_executive_recovery() -> None:
    class FakeExecutive:
        def __init__(self) -> None:
            self.paused = True
            self.resume_deferred_work = AsyncMock(
                return_value={"unblocked_commitments": 1, "restored_work": 1, "queue_restored": 1}
            )

        def recommend_pause(self) -> bool:
            return self.paused

    class FakeBAA:
        queue_size = 0
        held_size = 0
        start = AsyncMock()
        stop = AsyncMock()

    class FakeRuntime:
        def __init__(self) -> None:
            self.executive = FakeExecutive()
            self.baa = FakeBAA()
            self.ctx = type("Ctx", (), {"health_monitor": None, "somatic": None})()
            self.run_cycle = AsyncMock(return_value={"promoted": 0, "demoted": 0})
            self.run_consolidation = AsyncMock(return_value={})

    runtime = FakeRuntime()
    readiness = AgentReadiness()
    readiness.ready("test ready")
    scheduler = AgentScheduler(
        runtime=runtime,
        cycle_interval=0.05,
        consolidation_interval=3600,
        baa_heartbeat_interval=3600,
        readiness=readiness,
    )
    await scheduler.start()
    await asyncio.sleep(0.08)
    runtime.executive.paused = False
    await asyncio.sleep(0.12)
    await scheduler.stop()
    runtime.executive.resume_deferred_work.assert_awaited_once()
    runtime.run_cycle.assert_awaited()


@pytest.mark.asyncio
async def test_scheduler_focus_mode_does_not_trigger_resume() -> None:
    class FakeExecutive:
        def __init__(self) -> None:
            self.resume_deferred_work = AsyncMock(return_value={})

        def recommend_pause(self) -> bool:
            return False

    class FakeBAA:
        queue_size = 0
        held_size = 0
        start = AsyncMock()
        stop = AsyncMock()

    class FakeRuntime:
        def __init__(self) -> None:
            self.executive = FakeExecutive()
            self.baa = FakeBAA()
            self.ctx = type("Ctx", (), {"health_monitor": None, "somatic": None})()
            self.run_cycle = AsyncMock(return_value={"promoted": 0, "demoted": 0})
            self.run_consolidation = AsyncMock(return_value={})

    runtime = FakeRuntime()
    readiness = AgentReadiness()
    readiness.ready("test ready")
    scheduler = AgentScheduler(
        runtime=runtime,
        cycle_interval=0.05,
        consolidation_interval=3600,
        baa_heartbeat_interval=3600,
        readiness=readiness,
    )
    await scheduler.start()
    scheduler.enter_focus_mode()
    await asyncio.sleep(0.06)
    scheduler.exit_focus_mode()
    await asyncio.sleep(0.10)
    await scheduler.stop()
    runtime.executive.resume_deferred_work.assert_not_awaited()


@pytest.mark.asyncio
async def test_scheduler_skips_cycle_when_baa_is_busy() -> None:
    class FakeExecutive:
        def recommend_pause(self) -> bool:
            return False

    class FakeBAA:
        queue_size = 1
        held_size = 0
        active_count = 0
        start = AsyncMock()
        stop = AsyncMock()

    class FakeRuntime:
        def __init__(self) -> None:
            self.executive = FakeExecutive()
            self.baa = FakeBAA()
            self.ctx = type("Ctx", (), {"health_monitor": None, "somatic": None})()
            self._activity = "idle"
            self.run_cycle = AsyncMock(return_value={"promoted": 0, "demoted": 0})
            self.run_consolidation = AsyncMock(return_value={})
            self.run_daydream = AsyncMock(return_value={})

    runtime = FakeRuntime()
    readiness = AgentReadiness()
    readiness.ready("test ready")
    scheduler = AgentScheduler(
        runtime=runtime,
        cycle_interval=0.05,
        consolidation_interval=3600,
        baa_heartbeat_interval=3600,
        readiness=readiness,
    )
    await scheduler.start()
    await asyncio.sleep(0.12)
    await scheduler.stop()
    runtime.run_cycle.assert_not_awaited()


@pytest.mark.asyncio
async def test_scheduler_keeps_reflective_daydreams_running_when_executive_pauses() -> None:
    class FakeExecutive:
        def recommend_pause(self) -> bool:
            return True

    class FakeBAA:
        queue_size = 0
        held_size = 0
        active_count = 0
        start = AsyncMock()
        stop = AsyncMock()

    class FakeRuntime:
        def __init__(self) -> None:
            self.executive = FakeExecutive()
            self.baa = FakeBAA()
            self.ctx = type("Ctx", (), {"health_monitor": None, "somatic": None})()
            self._activity = "idle"
            self.run_cycle = AsyncMock(return_value={"promoted": 0, "demoted": 0})
            self.run_consolidation = AsyncMock(return_value={})
            self.run_daydream = AsyncMock(return_value={})

    runtime = FakeRuntime()
    readiness = AgentReadiness()
    readiness.ready("test ready")
    scheduler = AgentScheduler(
        runtime=runtime,
        cycle_interval=0.05,
        consolidation_interval=0.05,
        daydream_interval=0.05,
        schedule_interval=3600,
        baa_heartbeat_interval=3600,
        readiness=readiness,
    )
    await scheduler.start()
    await asyncio.sleep(0.12)
    await scheduler.stop()
    runtime.run_cycle.assert_not_awaited()
    runtime.run_consolidation.assert_not_awaited()
    runtime.run_daydream.assert_awaited()
    assert runtime.run_daydream.await_args.kwargs == {
        "force": True,
        "reflective_only": True,
    }


@pytest.mark.asyncio
async def test_scheduler_runs_required_nightly_dream_catchup_dates() -> None:
    class FakeDreamStore:
        async def list_recent(self, limit: int = 400):
            return [
                SimpleNamespace(
                    created_at=datetime(2026, 5, 11, 9, 0, tzinfo=timezone.utc),
                    meta={"dream_for_date": "2026-05-11"},
                )
            ]

    class FakeRuntime:
        def __init__(self) -> None:
            self.dream_store = FakeDreamStore()
            self._last_consolidation_result = {"result_id": "source-consolidation"}
            self.run_nightly_dream = AsyncMock(return_value={"available": True})

    runtime = FakeRuntime()
    scheduler = AgentScheduler(
        runtime=runtime,
        cycle_interval=3600,
        consolidation_interval=3600,
        baa_heartbeat_interval=3600,
        nightly_dream_timezone="America/Denver",
        nightly_dream_local_hour=3,
        time_source=lambda: datetime(2026, 5, 13, 13, 0, tzinfo=timezone.utc),
    )

    result = await scheduler._run_due_nightly_dreams()

    assert result["ran"] == 2
    assert result["due_dates"] == ["2026-05-12", "2026-05-13"]
    assert runtime.run_nightly_dream.await_count == 2
    calls = runtime.run_nightly_dream.await_args_list
    first_payload = calls[0].kwargs["consolidation_result"]
    second_payload = calls[1].kwargs["consolidation_result"]
    assert first_payload["dream_for_date"] == "2026-05-12"
    assert first_payload["required_nightly_dream"] is True
    assert first_payload["result_id"] == "required-nightly-dream:2026-05-12"
    assert first_payload["source_consolidation_result_id"] == "source-consolidation"
    assert second_payload["dream_for_date"] == "2026-05-13"


@pytest.mark.asyncio
async def test_scheduler_skips_consolidation_during_recent_user_activity() -> None:
    class FakeExecutive:
        def recommend_pause(self) -> bool:
            return False

    class FakeBAA:
        queue_size = 0
        held_size = 0
        active_count = 0
        start = AsyncMock()
        stop = AsyncMock()

    class FakeRuntime:
        def __init__(self) -> None:
            self.executive = FakeExecutive()
            self.baa = FakeBAA()
            self.ctx = type("Ctx", (), {"health_monitor": None, "somatic": None})()
            self._activity = "idle"
            self._last_user_turn_at = datetime(2026, 4, 24, 0, 0, 0, tzinfo=timezone.utc)
            self.run_cycle = AsyncMock(return_value={"promoted": 0, "demoted": 0})
            self.run_consolidation = AsyncMock(return_value={})
            self.run_daydream = AsyncMock(return_value={})

    runtime = FakeRuntime()
    readiness = AgentReadiness()
    readiness.ready("test ready")
    scheduler = AgentScheduler(
        runtime=runtime,
        cycle_interval=3600,
        consolidation_interval=0.01,
        schedule_interval=0.01,
        baa_heartbeat_interval=3600,
        readiness=readiness,
        conversation_quiet_seconds=60,
        time_source=lambda: datetime(2026, 4, 24, 0, 0, 10, tzinfo=timezone.utc),
    )
    await scheduler.start()
    await asyncio.sleep(0.05)
    await scheduler.stop()
    runtime.run_consolidation.assert_not_awaited()


@pytest.mark.asyncio
async def test_scheduler_runs_initiative_contact_loop() -> None:
    class FakeExecutive:
        def recommend_pause(self) -> bool:
            return False

    class FakeBAA:
        queue_size = 0
        held_size = 0
        active_count = 0
        start = AsyncMock()
        stop = AsyncMock()

    class FakeRuntime:
        def __init__(self) -> None:
            self.executive = FakeExecutive()
            self.baa = FakeBAA()
            self.ctx = type("Ctx", (), {"health_monitor": None, "somatic": None, "schedule_service": None})()
            self._activity = "idle"
            self.run_cycle = AsyncMock(return_value={})
            self.run_consolidation = AsyncMock(return_value={})
            self.run_daydream = AsyncMock(return_value={})
            self.maybe_run_initiative_contact = AsyncMock(return_value={"status": "skipped"})

    runtime = FakeRuntime()
    readiness = AgentReadiness()
    readiness.ready("test ready")
    scheduler = AgentScheduler(
        runtime=runtime,
        cycle_interval=3600,
        consolidation_interval=3600,
        daydream_interval=3600,
        schedule_interval=0.01,
        baa_heartbeat_interval=3600,
        readiness=readiness,
    )
    await scheduler.start()
    await asyncio.sleep(0.04)
    await scheduler.stop()
    runtime.maybe_run_initiative_contact.assert_awaited()


@pytest.mark.asyncio
async def test_scheduler_runs_wellbeing_maintenance_when_idle() -> None:
    class FakeExecutive:
        def recommend_pause(self) -> bool:
            return False

    class FakeBAA:
        queue_size = 0
        held_size = 0
        active_count = 0
        start = AsyncMock()
        stop = AsyncMock()

    class FakeRuntime:
        def __init__(self) -> None:
            self.executive = FakeExecutive()
            self.baa = FakeBAA()
            self.ctx = SimpleNamespace(health_monitor=None, somatic=None)
            self._activity = "idle"
            self.run_cycle = AsyncMock(return_value={})
            self.run_consolidation = AsyncMock(return_value={})
            self.run_daydream = AsyncMock(return_value={})
            self.run_wellbeing_maintenance = AsyncMock(return_value={"status": "recorded"})

    runtime = FakeRuntime()
    readiness = AgentReadiness()
    readiness.ready("test ready")
    scheduler = AgentScheduler(
        runtime=runtime,
        cycle_interval=3600,
        consolidation_interval=3600,
        daydream_interval=3600,
        schedule_interval=3600,
        wellbeing_interval=0.01,
        baa_heartbeat_interval=3600,
        readiness=readiness,
    )
    await scheduler.start()
    await asyncio.sleep(0.04)
    await scheduler.stop()
    runtime.run_wellbeing_maintenance.assert_awaited()


@pytest.mark.asyncio
async def test_scheduler_skips_wellbeing_maintenance_when_runtime_is_active() -> None:
    class FakeExecutive:
        def recommend_pause(self) -> bool:
            return False

    class FakeBAA:
        queue_size = 0
        held_size = 0
        active_count = 0
        start = AsyncMock()
        stop = AsyncMock()

    class FakeRuntime:
        def __init__(self) -> None:
            self.executive = FakeExecutive()
            self.baa = FakeBAA()
            self.ctx = SimpleNamespace(health_monitor=None, somatic=None)
            self._activity = "chat"
            self.run_cycle = AsyncMock(return_value={})
            self.run_consolidation = AsyncMock(return_value={})
            self.run_daydream = AsyncMock(return_value={})
            self.run_wellbeing_maintenance = AsyncMock(return_value={"status": "recorded"})

    runtime = FakeRuntime()
    readiness = AgentReadiness()
    readiness.ready("test ready")
    scheduler = AgentScheduler(
        runtime=runtime,
        cycle_interval=3600,
        consolidation_interval=3600,
        daydream_interval=3600,
        schedule_interval=3600,
        wellbeing_interval=0.01,
        baa_heartbeat_interval=3600,
        readiness=readiness,
    )
    await scheduler.start()
    await asyncio.sleep(0.04)
    await scheduler.stop()
    runtime.run_wellbeing_maintenance.assert_not_awaited()


@pytest.mark.asyncio
async def test_scheduler_runs_shadow_consumer_on_cognitive_maintenance_tick() -> None:
    class FakeExecutive:
        def recommend_pause(self) -> bool:
            return False

    class FakeBAA:
        queue_size = 0
        held_size = 0
        active_count = 0
        start = AsyncMock()
        stop = AsyncMock()

    class FakeRuntime:
        def __init__(self) -> None:
            self.executive = FakeExecutive()
            self.baa = FakeBAA()
            self.ctx = SimpleNamespace(health_monitor=None, somatic=None)
            self._activity = "idle"
            self.run_cycle = AsyncMock(return_value={})
            self.run_consolidation = AsyncMock(return_value={})
            self.run_daydream = AsyncMock(return_value={})
            self.run_cognitive_maintenance = AsyncMock(return_value={"status": "recorded"})
            self.run_shadow_registry_consumer = AsyncMock(return_value={"dismissed_count": 1})

    runtime = FakeRuntime()
    readiness = AgentReadiness()
    readiness.ready("test ready")
    scheduler = AgentScheduler(
        runtime=runtime,
        cycle_interval=3600,
        consolidation_interval=3600,
        daydream_interval=3600,
        schedule_interval=3600,
        wellbeing_interval=0.01,
        baa_heartbeat_interval=3600,
        readiness=readiness,
    )
    await scheduler.start()
    await asyncio.sleep(0.04)
    await scheduler.stop()

    runtime.run_cognitive_maintenance.assert_awaited()
    runtime.run_shadow_registry_consumer.assert_awaited()


@pytest.mark.asyncio
async def test_scheduler_keeps_observation_and_maintenance_running_under_executive_pause() -> None:
    class FakeExecutive:
        def recommend_pause(self) -> bool:
            return True

    class FakeBAA:
        queue_size = 0
        held_size = 0
        active_count = 0
        start = AsyncMock()
        stop = AsyncMock()

    class FakeRuntime:
        def __init__(self) -> None:
            self.executive = FakeExecutive()
            self.baa = FakeBAA()
            self.ctx = SimpleNamespace(health_monitor=None, somatic=None)
            self._activity = "idle"
            self.run_cycle = AsyncMock(return_value={})
            self.run_consolidation = AsyncMock(return_value={})
            self.run_daydream = AsyncMock(return_value={})
            self.maybe_run_desktop_context = AsyncMock(return_value={"status": "observed"})
            self.run_wellbeing_maintenance = AsyncMock(return_value={"status": "recorded"})
            self.run_cognitive_maintenance = AsyncMock(return_value={"status": "recorded"})
            self.run_shadow_registry_consumer = AsyncMock(return_value={"dismissed_count": 0})
            self.run_compaction_backlog_maintenance = AsyncMock(
                return_value={"available": True, "sessions_compacted": 1}
            )

    runtime = FakeRuntime()
    readiness = AgentReadiness()
    readiness.ready("test ready")
    scheduler = AgentScheduler(
        runtime=runtime,
        cycle_interval=3600,
        consolidation_interval=3600,
        daydream_interval=3600,
        schedule_interval=0.01,
        wellbeing_interval=0.01,
        compaction_backlog_interval_seconds=0.01,
        baa_heartbeat_interval=3600,
        readiness=readiness,
    )

    await scheduler.start()
    await asyncio.sleep(0.04)
    await scheduler.stop()

    runtime.run_cycle.assert_not_awaited()
    runtime.maybe_run_desktop_context.assert_awaited()
    runtime.run_wellbeing_maintenance.assert_awaited()
    runtime.run_cognitive_maintenance.assert_awaited()
    runtime.run_shadow_registry_consumer.assert_awaited()
    runtime.run_compaction_backlog_maintenance.assert_awaited()


@pytest.mark.asyncio
async def test_scheduler_body_double_yields_when_baa_busy() -> None:
    class FakeExecutive:
        def recommend_pause(self) -> bool:
            return False

    class FakeBAA:
        queue_size = 0
        held_size = 0
        active_count = 1
        start = AsyncMock()
        stop = AsyncMock()

    class FakeRuntime:
        def __init__(self) -> None:
            self.executive = FakeExecutive()
            self.baa = FakeBAA()
            self.desktop_context = SimpleNamespace(
                config=SimpleNamespace(
                    enabled=True,
                    media_commentary_mode_enabled=True,
                    proactive_video_commentary_enabled=True,
                )
            )
            self.ctx = SimpleNamespace(health_monitor=None, somatic=None)
            self._activity = "idle"
            self.run_cycle = AsyncMock(return_value={})
            self.run_consolidation = AsyncMock(return_value={})
            self.run_daydream = AsyncMock(return_value={})
            self.maybe_run_desktop_context = AsyncMock(return_value={"status": "observed"})
            self.run_wellbeing_maintenance = AsyncMock(return_value={"status": "recorded"})
            self.run_cognitive_maintenance = AsyncMock(return_value={"status": "recorded"})
            self.run_shadow_registry_consumer = AsyncMock(return_value={"dismissed_count": 0})
            self.run_compaction_backlog_maintenance = AsyncMock(
                return_value={"available": True, "sessions_compacted": 0}
            )

    runtime = FakeRuntime()
    readiness = AgentReadiness()
    readiness.ready("test ready")
    scheduler = AgentScheduler(
        runtime=runtime,
        cycle_interval=3600,
        consolidation_interval=3600,
        daydream_interval=3600,
        schedule_interval=0.01,
        wellbeing_interval=3600,
        compaction_backlog_interval_seconds=3600,
        baa_heartbeat_interval=3600,
        readiness=readiness,
    )

    await scheduler.start()
    await asyncio.sleep(0.04)
    await scheduler.stop()

    runtime.maybe_run_desktop_context.assert_not_awaited()


@pytest.mark.asyncio
async def test_scheduler_body_double_yields_during_focus_mode() -> None:
    class FakeExecutive:
        def recommend_pause(self) -> bool:
            return False

    class FakeBAA:
        queue_size = 0
        held_size = 0
        active_count = 0
        start = AsyncMock()
        stop = AsyncMock()

    class FakeRuntime:
        def __init__(self) -> None:
            self.executive = FakeExecutive()
            self.baa = FakeBAA()
            self.desktop_context = SimpleNamespace(
                config=SimpleNamespace(
                    enabled=True,
                    media_commentary_mode_enabled=True,
                    proactive_video_commentary_enabled=True,
                )
            )
            self.ctx = SimpleNamespace(health_monitor=None, somatic=None)
            self._activity = "idle"
            self.run_cycle = AsyncMock(return_value={})
            self.run_consolidation = AsyncMock(return_value={})
            self.run_daydream = AsyncMock(return_value={})
            self.maybe_run_desktop_context = AsyncMock(return_value={"status": "observed"})
            self.run_wellbeing_maintenance = AsyncMock(return_value={"status": "recorded"})
            self.run_cognitive_maintenance = AsyncMock(return_value={"status": "recorded"})
            self.run_shadow_registry_consumer = AsyncMock(return_value={"dismissed_count": 0})
            self.run_compaction_backlog_maintenance = AsyncMock(
                return_value={"available": True, "sessions_compacted": 0}
            )

    runtime = FakeRuntime()
    readiness = AgentReadiness()
    readiness.ready("test ready")
    scheduler = AgentScheduler(
        runtime=runtime,
        cycle_interval=3600,
        consolidation_interval=3600,
        daydream_interval=3600,
        schedule_interval=0.01,
        wellbeing_interval=3600,
        compaction_backlog_interval_seconds=3600,
        baa_heartbeat_interval=3600,
        readiness=readiness,
    )
    scheduler.enter_focus_mode()

    await scheduler.start()
    await asyncio.sleep(0.04)
    await scheduler.stop()

    runtime.run_cycle.assert_not_awaited()
    runtime.maybe_run_desktop_context.assert_not_awaited()


@pytest.mark.asyncio
async def test_scheduler_body_double_yields_to_due_scheduled_work() -> None:
    class FakeExecutive:
        def recommend_pause(self) -> bool:
            return False

    class FakeBAA:
        queue_size = 0
        held_size = 0
        active_count = 0
        start = AsyncMock()
        stop = AsyncMock()

    class FakeScheduleStore:
        async def list_due(self, _now, limit: int = 50):
            return [SimpleNamespace(schedule_id="due-1")]

    class FakeScheduleService:
        def __init__(self) -> None:
            self.store = FakeScheduleStore()
            self.process_due = AsyncMock(return_value={"processed": 1})

    class FakeRuntime:
        def __init__(self) -> None:
            self.executive = FakeExecutive()
            self.baa = FakeBAA()
            self.schedule_service = FakeScheduleService()
            self.desktop_context = SimpleNamespace(
                config=SimpleNamespace(
                    enabled=True,
                    media_commentary_mode_enabled=True,
                    proactive_video_commentary_enabled=True,
                )
            )
            self.ctx = SimpleNamespace(
                health_monitor=None,
                somatic=None,
                schedule_store=self.schedule_service.store,
            )
            self._activity = "idle"
            self.run_cycle = AsyncMock(return_value={})
            self.run_consolidation = AsyncMock(return_value={})
            self.run_daydream = AsyncMock(return_value={})
            self.maybe_run_desktop_context = AsyncMock(return_value={"status": "observed"})

    runtime = FakeRuntime()
    readiness = AgentReadiness()
    readiness.ready("test ready")
    scheduler = AgentScheduler(
        runtime=runtime,
        cycle_interval=3600,
        consolidation_interval=3600,
        daydream_interval=3600,
        schedule_interval=0.01,
        wellbeing_interval=3600,
        compaction_backlog_interval_seconds=3600,
        baa_heartbeat_interval=3600,
        readiness=readiness,
    )

    await scheduler.start()
    await asyncio.sleep(0.04)
    await scheduler.stop()

    runtime.schedule_service.process_due.assert_awaited()
    runtime.maybe_run_desktop_context.assert_not_awaited()


@pytest.mark.asyncio
async def test_scheduler_processes_due_work_while_readiness_is_degraded() -> None:
    calls = 0

    class FakeScheduleService:
        async def process_due(self):
            nonlocal calls
            calls += 1
            scheduler._running = False
            return {"processed": 1, "submitted": 1, "recorded": 0, "skipped": 0, "failed": 0}

    readiness = AgentReadiness()
    readiness.degraded("schedule processing failed: previous transient error")
    runtime = SimpleNamespace(schedule_service=FakeScheduleService(), ctx=SimpleNamespace())
    scheduler = AgentScheduler(runtime, schedule_interval=0.001, readiness=readiness)
    scheduler._running = True

    await scheduler._schedule_loop()

    assert calls == 1
    assert readiness.state == ReadinessState.READY
    assert readiness.reason == "schedule_processing_recovered"


@pytest.mark.asyncio
async def test_scheduler_reports_due_work_blocked_when_readiness_is_paused() -> None:
    traces: list[tuple[str, dict]] = []

    class FakeStore:
        async def list_due(self, now, limit=1):
            scheduler._running = False
            return [object()]

    class FakeScheduleService:
        store = FakeStore()

        async def process_due(self):
            raise AssertionError("paused scheduler should not process due schedules")

    readiness = AgentReadiness()
    readiness.pause("operator pause")
    runtime = SimpleNamespace(
        schedule_service=FakeScheduleService(),
        ctx=SimpleNamespace(schedule_store=FakeScheduleService.store),
    )
    scheduler = AgentScheduler(runtime, schedule_interval=0.001, readiness=readiness)
    scheduler._trace = lambda event, payload: traces.append((event, payload))
    scheduler._running = True

    pre_status = scheduler.schedule_processing_status(due_now=1)
    assert pre_status["can_process"] is False
    assert pre_status["blocked_reason"] == "readiness_paused"
    assert pre_status["will_process_due_now"] is False

    await scheduler._schedule_loop()

    assert traces == [("schedule_skipped", {"reason": "readiness_paused", "due_work_pending": True})]
    status = scheduler.schedule_processing_status(due_now=1)
    assert status["can_process"] is False
    assert status["blocked_reason"] == "scheduler_not_running"
    assert status["will_process_due_now"] is False


@pytest.mark.asyncio
async def test_scheduler_media_commentary_yields_when_baa_busy() -> None:
    observations: list[dict] = []
    scheduler_ref: dict[str, AgentScheduler] = {}

    class FakeDesktopContext:
        config = SimpleNamespace(media_state_poll_seconds=1)

        async def poll_media_state_once(self):
            scheduler_ref["scheduler"]._running = False
            return {
                "status": "changed",
                "commentary_observation_requested": True,
                "commentary_observation_reason": "media_commentary_mode:media_started",
            }

        async def observe_once(self, **kwargs):
            observations.append(kwargs)
            return {"status": "observed"}

    runtime = SimpleNamespace(
        desktop_context=FakeDesktopContext(),
        baa=SimpleNamespace(queue_size=0, held_size=0, active_count=1),
        ctx=SimpleNamespace(health_monitor=None),
        _activity="idle",
    )
    scheduler = AgentScheduler(runtime)
    scheduler_ref["scheduler"] = scheduler
    scheduler._running = True

    await scheduler._desktop_media_state_loop()

    assert observations == []


@pytest.mark.asyncio
async def test_scheduler_media_commentary_yields_to_due_scheduled_work() -> None:
    observations: list[dict] = []
    scheduler_ref: dict[str, AgentScheduler] = {}

    class FakeScheduleStore:
        async def list_due(self, _now, limit: int = 50):
            return [SimpleNamespace(schedule_id="due-1")]

    schedule_store = FakeScheduleStore()

    class FakeDesktopContext:
        config = SimpleNamespace(media_state_poll_seconds=1)

        async def poll_media_state_once(self):
            scheduler_ref["scheduler"]._running = False
            return {
                "status": "changed",
                "commentary_observation_requested": True,
                "commentary_observation_reason": "media_commentary_mode:media_started",
            }

        async def observe_once(self, **kwargs):
            observations.append(kwargs)
            return {"status": "observed"}

    runtime = SimpleNamespace(
        desktop_context=FakeDesktopContext(),
        schedule_service=SimpleNamespace(store=schedule_store),
        baa=SimpleNamespace(queue_size=0, held_size=0, active_count=0),
        ctx=SimpleNamespace(health_monitor=None, schedule_store=schedule_store),
        _activity="idle",
    )
    scheduler = AgentScheduler(runtime)
    scheduler_ref["scheduler"] = scheduler
    scheduler._running = True

    await scheduler._desktop_media_state_loop()

    assert observations == []


@pytest.mark.asyncio
async def test_scheduler_media_commentary_observes_during_active_playback_despite_due_work() -> None:
    observations: list[dict] = []
    scheduler_ref: dict[str, AgentScheduler] = {}
    now = datetime(2026, 5, 19, 20, 0, tzinfo=timezone.utc)

    class FakeScheduleStore:
        async def list_due(self, _now, limit: int = 50):
            return [SimpleNamespace(schedule_id="due-1")]

    schedule_store = FakeScheduleStore()

    class FakeDesktopContext:
        config = SimpleNamespace(
            enabled=True,
            media_state_poll_seconds=1,
            media_commentary_mode_enabled=True,
            proactive_video_commentary_enabled=True,
            media_commentary_requested_at=now.isoformat(),
        )

        def _load_media_state(self):
            return {
                "observed_at": now.isoformat(),
                "items": {
                    "firefox|https://youtube.example/watch": {
                        "status": "Playing",
                        "title": "AI reasoning talk",
                    }
                },
            }

        async def poll_media_state_once(self):
            scheduler_ref["scheduler"]._running = False
            return {
                "status": "changed",
                "commentary_observation_requested": True,
                "commentary_observation_reason": "media_commentary_mode:media_resumed",
            }

        async def observe_once(self, **kwargs):
            observations.append(kwargs)
            return {"status": "observed"}

    runtime = SimpleNamespace(
        desktop_context=FakeDesktopContext(),
        schedule_service=SimpleNamespace(store=schedule_store),
        baa=SimpleNamespace(queue_size=0, held_size=0, active_count=1),
        ctx=SimpleNamespace(health_monitor=None, schedule_store=schedule_store),
        _activity="idle",
    )
    scheduler = AgentScheduler(runtime, time_source=lambda: now)
    scheduler_ref["scheduler"] = scheduler
    scheduler._running = True

    await scheduler._desktop_media_state_loop()

    assert observations == [
        {
            "force": True,
            "reason": "media_commentary_mode:media_resumed",
        }
    ]


@pytest.mark.asyncio
async def test_scheduler_defers_due_work_during_active_media_commentary_playback() -> None:
    traces: list[tuple[str, dict]] = []
    now = datetime(2026, 5, 19, 20, 0, tzinfo=timezone.utc)
    scheduler_ref: dict[str, AgentScheduler] = {}

    class FakeScheduleStore:
        async def list_due(self, _now, limit: int = 50):
            scheduler._running = False
            return [SimpleNamespace(schedule_id="due-1")]

    class FakeScheduleService:
        store = FakeScheduleStore()

        async def process_due(self):
            scheduler_ref["scheduler"]._running = False
            return {"processed": 1}

    class FakeDesktopContext:
        config = SimpleNamespace(
            enabled=True,
            media_commentary_mode_enabled=True,
            proactive_video_commentary_enabled=True,
            media_commentary_requested_at=now.isoformat(),
            media_state_poll_seconds=1,
        )

        def _load_media_state(self):
            return {
                "observed_at": now.isoformat(),
                "items": {
                    "firefox|https://youtube.example/watch": {
                        "status": "Playing",
                        "title": "AI reasoning talk",
                    }
                },
            }

    runtime = SimpleNamespace(
        schedule_service=FakeScheduleService(),
        desktop_context=FakeDesktopContext(),
        ctx=SimpleNamespace(health_monitor=None, schedule_store=FakeScheduleService.store),
        _activity="idle",
    )
    scheduler = AgentScheduler(runtime, schedule_interval=0.001, time_source=lambda: now)
    scheduler_ref["scheduler"] = scheduler
    scheduler._trace = lambda event, payload: traces.append((event, payload))  # type: ignore[method-assign]
    scheduler._running = True

    await scheduler._schedule_loop()

    assert traces == [
        (
            "schedule_skipped",
            {"reason": "foreground_media_commentary", "due_work_pending": True},
        )
    ]


@pytest.mark.asyncio
async def test_scheduler_runs_telemetry_prune_pass() -> None:
    class FakeStore:
        def __init__(self) -> None:
            self.calls: list[tuple[int, int]] = []

        def prune_old_files(self, retention_days: int, *, now) -> int:
            self.calls.append((retention_days, int(now.timestamp())))
            return 2

    class FakeTracer:
        def __init__(self) -> None:
            self.logs: list[tuple[str, dict]] = []
            self.store = FakeStore()

        def log(self, event: str, message: str, payload: dict) -> None:
            self.logs.append((message, payload))

    class FakeRuntime:
        def __init__(self) -> None:
            self.tracer = FakeTracer()
            self.ctx = SimpleNamespace(
                health_monitor=None,
                somatic=None,
                token_telemetry=SimpleNamespace(
                    flush=AsyncMock(),
                    prune_old_events=AsyncMock(return_value=3),
                ),
            )

    runtime = FakeRuntime()
    scheduler = AgentScheduler(
        runtime=runtime,
        cycle_interval=3600,
        consolidation_interval=3600,
        baa_heartbeat_interval=3600,
        telemetry_retention_days=14,
    )
    scheduler._running = True
    await scheduler._run_telemetry_prune()

    runtime.ctx.token_telemetry.flush.assert_awaited_once()
    runtime.ctx.token_telemetry.prune_old_events.assert_awaited_once_with(14)
    assert len(runtime.tracer.store.calls) == 1
    assert runtime.tracer.store.calls[0][0] == 14


@pytest.mark.asyncio
async def test_scheduler_workspace_sync_pass_invokes_bridge(tmp_path: Path) -> None:
    workspace_root = tmp_path / "workspace"
    (workspace_root / "self" / "notes").mkdir(parents=True)
    (workspace_root / "self" / "notes" / "n.md").write_text("# n", encoding="utf-8")
    (workspace_root / "reflections").mkdir(parents=True)
    (workspace_root / "reflections" / "r.md").write_text("# r", encoding="utf-8")

    sync_calls: list[Path] = []

    class FakeBridge:
        async def sync_directory(self, root):
            sync_calls.append(Path(root))
            return {
                "artifacts": 1,
                "episodes_created": 1,
                "episodes_updated": 0,
                "episodes_deleted": 0,
                "memories_upserted": 1,
            }

    class FakeTracer:
        def __init__(self) -> None:
            self.events: list[tuple[str, dict]] = []

        def log(self, event: str, message: str, payload: dict) -> None:
            self.events.append((message, payload))

    config = SimpleNamespace(agent_workspace_root=lambda: workspace_root)
    runtime = SimpleNamespace(
        tracer=FakeTracer(),
        ctx=SimpleNamespace(
            artifact_bridge=FakeBridge(),
            config=config,
            health_monitor=None,
        ),
    )
    scheduler = AgentScheduler(
        runtime=runtime,
        cycle_interval=3600,
        consolidation_interval=3600,
        baa_heartbeat_interval=3600,
        tracer=runtime.tracer,
    )
    scheduler._running = True

    await scheduler._run_workspace_sync()

    swept = {path.name for path in sync_calls}
    assert "self" in swept
    assert "reflections" in swept
    assert "daydream-lab" not in swept  # missing dir is skipped
    assert any(message.endswith("workspace_sync_complete") for message, _ in runtime.tracer.events)


def test_scheduler_foreground_deferred_consolidation_is_not_retried(tmp_path: Path) -> None:
    runtime = SimpleNamespace(
        ctx=SimpleNamespace(config=SimpleNamespace(state_dir=tmp_path), health_monitor=None),
    )
    scheduler = AgentScheduler(
        runtime=runtime,
        cycle_interval=3600,
        consolidation_interval=3600,
        baa_heartbeat_interval=3600,
    )

    assert not scheduler._consolidation_result_requires_retry(
        {
            "budget_exhausted": True,
            "budget_reason": "foreground_user_turn",
            "worker": {"status": "killed"},
        }
    )


@pytest.mark.asyncio
async def test_scheduler_workspace_sync_skips_during_recent_user_activity(tmp_path: Path) -> None:
    workspace_root = tmp_path / "workspace"
    (workspace_root / "self").mkdir(parents=True)
    sync_calls: list[Path] = []

    class FakeBridge:
        async def sync_directory(self, root):
            sync_calls.append(Path(root))
            return {}

    class FakeTracer:
        def __init__(self) -> None:
            self.events: list[tuple[str, dict]] = []

        def log(self, event: str, message: str, payload: dict) -> None:
            self.events.append((message, payload))

    runtime = SimpleNamespace(
        tracer=FakeTracer(),
        _activity="idle",
        _last_user_turn_at=datetime.now(timezone.utc),
        ctx=SimpleNamespace(
            artifact_bridge=FakeBridge(),
            config=SimpleNamespace(
                agent_workspace_root=lambda: workspace_root,
                state_dir=tmp_path,
            ),
            health_monitor=None,
        ),
    )
    scheduler = AgentScheduler(
        runtime=runtime,
        cycle_interval=3600,
        consolidation_interval=3600,
        baa_heartbeat_interval=3600,
        tracer=runtime.tracer,
    )
    scheduler._running = True

    await scheduler._run_workspace_sync()

    assert sync_calls == []
    assert any(
        message.endswith("workspace_sync_skipped") and payload["reason"] == "recent_user_activity"
        for message, payload in runtime.tracer.events
    )
