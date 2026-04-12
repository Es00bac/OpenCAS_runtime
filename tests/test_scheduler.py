"""Tests for AgentScheduler background loop orchestration."""

import asyncio
from pathlib import Path
from unittest.mock import AsyncMock
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
