"""Tests for ReliabilityCoordinator."""

import pytest
from opencas.infra import EventBus, BaaCompletedEvent, BaaPauseEvent
from opencas.execution import ReliabilityCoordinator


@pytest.mark.asyncio
async def test_no_pause_when_failure_rate_low():
    bus = EventBus()
    coordinator = ReliabilityCoordinator(bus, window_size=5, failure_threshold=0.7)
    emitted: list[BaaPauseEvent] = []
    bus.subscribe(BaaPauseEvent, lambda e: emitted.append(e))

    for _ in range(4):
        await bus.emit(BaaCompletedEvent(task_id="t1", success=True, stage="done", objective="test"))

    assert len(emitted) == 0
    coordinator.stop()


@pytest.mark.asyncio
async def test_pause_when_failure_rate_spikes():
    bus = EventBus()
    coordinator = ReliabilityCoordinator(bus, window_size=5, failure_threshold=0.7)
    emitted: list[BaaPauseEvent] = []
    bus.subscribe(BaaPauseEvent, lambda e: emitted.append(e))

    for _ in range(5):
        await bus.emit(BaaCompletedEvent(task_id="t1", success=False, stage="failed", objective="test"))

    assert len(emitted) == 1
    assert emitted[0].failure_rate == pytest.approx(1.0)
    assert emitted[0].window_size == 5
    coordinator.stop()


@pytest.mark.asyncio
async def test_cooldown_prevents_duplicate_pauses():
    bus = EventBus()
    coordinator = ReliabilityCoordinator(bus, window_size=5, failure_threshold=0.7, cooldown_seconds=60)
    emitted: list[BaaPauseEvent] = []
    bus.subscribe(BaaPauseEvent, lambda e: emitted.append(e))

    for _ in range(5):
        await bus.emit(BaaCompletedEvent(task_id="t1", success=False, stage="failed", objective="test"))
    assert len(emitted) == 1

    # Another spike within cooldown should not emit again
    for _ in range(5):
        await bus.emit(BaaCompletedEvent(task_id="t1", success=False, stage="failed", objective="test"))
    assert len(emitted) == 1

    coordinator.stop()


@pytest.mark.asyncio
async def test_stats_reflect_history():
    bus = EventBus()
    coordinator = ReliabilityCoordinator(bus, window_size=5, failure_threshold=0.7)

    await bus.emit(BaaCompletedEvent(task_id="t1", success=False, stage="failed", objective="test"))
    await bus.emit(BaaCompletedEvent(task_id="t2", success=True, stage="done", objective="test"))

    stats = coordinator.get_stats()
    assert stats["failure_rate"] == pytest.approx(0.5)
    assert stats["window_size"] == 2
    coordinator.stop()
