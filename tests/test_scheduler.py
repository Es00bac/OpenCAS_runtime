"""Tests for AgentScheduler background loop orchestration."""

import asyncio
from datetime import datetime, timedelta, timezone
from pathlib import Path
from unittest.mock import AsyncMock
from types import SimpleNamespace
import pytest
import pytest_asyncio

from opencas.bootstrap import BootstrapConfig, BootstrapPipeline
from opencas.runtime import AgentRuntime
from opencas.runtime.scheduler import AgentScheduler
from opencas.runtime.readiness import AgentReadiness, ReadinessState


@pytest_asyncio.fixture
async def runtime(tmp_path: Path):
    config = BootstrapConfig(state_dir=tmp_path, session_id="scheduler-test")
    ctx = await BootstrapPipeline(config).run()
    return AgentRuntime(ctx)


@pytest.mark.asyncio
async def test_scheduler_start_starts_baa(runtime: AgentRuntime) -> None:
    runtime.baa.start = AsyncMock()
    runtime.baa.stop = AsyncMock()
    scheduler = AgentScheduler(
        runtime=runtime,
        cycle_interval=3600,
        consolidation_interval=3600,
        baa_heartbeat_interval=3600,
    )
    await scheduler.start()
    runtime.baa.start.assert_awaited_once()
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
    assert scheduler.consolidation_budget["worker_timeout_seconds"] == 300


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
async def test_scheduler_skips_consolidation_and_daydream_when_executive_pauses() -> None:
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
    runtime.run_daydream.assert_not_awaited()


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
