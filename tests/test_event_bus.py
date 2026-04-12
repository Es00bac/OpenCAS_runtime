"""Tests for the EventBus and BAA event integration."""

import asyncio
from pathlib import Path

import pytest
import pytest_asyncio

from opencas.infra import BaaCompletedEvent, BaaProgressEvent, EventBus
from opencas.execution import BoundedAssistantAgent, RepairTask
from opencas.tools import ToolRegistry


@pytest.mark.asyncio
async def test_event_bus_subscribe_and_emit() -> None:
    bus = EventBus()
    received = []

    async def handler(event: BaaProgressEvent) -> None:
        received.append(event)

    bus.subscribe(BaaProgressEvent, handler)
    await bus.emit(BaaProgressEvent(task_id="t1", stage="planning", objective="test", attempt=1))
    await asyncio.sleep(0)

    assert len(received) == 1
    assert received[0].task_id == "t1"


@pytest.mark.asyncio
async def test_event_bus_multiple_handlers() -> None:
    bus = EventBus()
    calls = []

    async def h1(event: BaaProgressEvent) -> None:
        calls.append("h1")

    async def h2(event: BaaProgressEvent) -> None:
        calls.append("h2")

    bus.subscribe(BaaProgressEvent, h1)
    bus.subscribe(BaaProgressEvent, h2)
    await bus.emit(BaaProgressEvent(task_id="t1", stage="planning", objective="test", attempt=1))
    await asyncio.sleep(0)

    assert sorted(calls) == ["h1", "h2"]


@pytest.mark.asyncio
async def test_event_bus_unsubscribe() -> None:
    bus = EventBus()
    received = []

    async def handler(event: BaaProgressEvent) -> None:
        received.append(event)

    bus.subscribe(BaaProgressEvent, handler)
    bus.unsubscribe(BaaProgressEvent, handler)
    await bus.emit(BaaProgressEvent(task_id="t1", stage="planning", objective="test", attempt=1))
    await asyncio.sleep(0)

    assert len(received) == 0


@pytest.mark.asyncio
async def test_event_bus_handler_error_does_not_crash_bus() -> None:
    bus = EventBus()
    received = []

    async def bad_handler(event: BaaProgressEvent) -> None:
        raise RuntimeError("oops")

    async def good_handler(event: BaaProgressEvent) -> None:
        received.append(event)

    bus.subscribe(BaaProgressEvent, bad_handler)
    bus.subscribe(BaaProgressEvent, good_handler)
    await bus.emit(BaaProgressEvent(task_id="t1", stage="planning", objective="test", attempt=1))
    await asyncio.sleep(0)

    assert len(received) == 1


@pytest.mark.asyncio
async def test_event_bus_typed_isolation() -> None:
    bus = EventBus()
    progress_received = []
    completed_received = []

    async def on_progress(event: BaaProgressEvent) -> None:
        progress_received.append(event)

    async def on_completed(event: BaaCompletedEvent) -> None:
        completed_received.append(event)

    bus.subscribe(BaaProgressEvent, on_progress)
    bus.subscribe(BaaCompletedEvent, on_completed)

    await bus.emit(BaaProgressEvent(task_id="t1", stage="planning", objective="test", attempt=1))
    await bus.emit(BaaCompletedEvent(task_id="t1", success=True, stage="done", objective="test"))
    await asyncio.sleep(0)

    assert len(progress_received) == 1
    assert len(completed_received) == 1


@pytest.mark.asyncio
async def test_baa_emits_progress_and_completed_events(tmp_path: Path) -> None:
    bus = EventBus()
    progress_events = []
    completed_events = []

    async def on_progress(event: BaaProgressEvent) -> None:
        progress_events.append(event)

    async def on_completed(event: BaaCompletedEvent) -> None:
        completed_events.append(event)

    bus.subscribe(BaaProgressEvent, on_progress)
    bus.subscribe(BaaCompletedEvent, on_completed)

    tools = ToolRegistry()
    workspace = str(tmp_path)
    from opencas.autonomy.models import ActionRiskTier
    from opencas.tools import ShellToolAdapter

    shell = ShellToolAdapter(cwd=workspace, timeout=30.0)
    tools.register("bash_run_command", "Run command", shell, ActionRiskTier.SHELL_LOCAL)

    baa = BoundedAssistantAgent(tools=tools, event_bus=bus)
    task = RepairTask(
        objective="event test",
        verification_command="echo ok",
    )
    future = await baa.submit(task)
    await baa.start()
    result = await asyncio.wait_for(future, timeout=5.0)
    await baa.stop()

    assert result.success is True
    assert len(progress_events) >= 1
    assert any(e.task_id == str(task.task_id) for e in progress_events)
    assert len(completed_events) == 1
    assert completed_events[0].task_id == str(task.task_id)
    assert completed_events[0].success is True
