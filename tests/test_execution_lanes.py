"""Tests for lane-based command queues."""

import asyncio

import pytest

from opencas.execution.lanes import CommandLane, LaneConfig, LaneManager


class TestLaneManager:
    @pytest.fixture
    def manager(self):
        return LaneManager(
            configs={
                CommandLane.CHAT: LaneConfig(max_concurrent=1),
                CommandLane.BAA: LaneConfig(max_concurrent=2),
            }
        )

    @pytest.mark.asyncio
    async def test_submit_and_get(self, manager):
        manager.submit(CommandLane.CHAT, "msg1")
        manager.submit(CommandLane.BAA, "task1")
        assert manager.qsize(CommandLane.CHAT) == 1
        assert manager.qsize(CommandLane.BAA) == 1

    @pytest.mark.asyncio
    async def test_worker_factory(self, manager):
        processed = []

        async def worker(lane):
            while True:
                item = await manager.get(lane)
                if item is None:
                    manager.task_done(lane)
                    break
                processed.append((lane.value, item))
                manager.task_done(lane)

        manager.start(worker_factory=worker)
        manager.submit(CommandLane.CHAT, "a")
        manager.submit(CommandLane.BAA, "b")
        manager.submit(CommandLane.BAA, "c")
        await asyncio.sleep(0.05)
        await manager.stop()

        assert len(processed) == 3
        chat_items = [p for p in processed if p[0] == CommandLane.CHAT.value]
        baa_items = [p for p in processed if p[0] == CommandLane.BAA.value]
        assert len(chat_items) == 1
        assert len(baa_items) == 2

    @pytest.mark.asyncio
    async def test_drain(self, manager):
        processed = []

        async def worker(lane):
            while True:
                item = await manager.get(lane)
                if item is None:
                    manager.task_done(lane)
                    break
                processed.append(item)
                manager.task_done(lane)

        manager.start(worker_factory=worker)
        manager.submit(CommandLane.BAA, "x")
        manager.submit(CommandLane.BAA, "y")
        await manager.drain(CommandLane.BAA)
        await manager.stop()
        assert len(processed) == 2

    @pytest.mark.asyncio
    async def test_stop_sends_sentinels(self, manager):
        processed = []

        async def worker(lane):
            while True:
                item = await manager.get(lane)
                if item is None:
                    manager.task_done(lane)
                    break
                processed.append(item)
                manager.task_done(lane)

        manager.start(worker_factory=worker)
        manager.submit(CommandLane.CHAT, "a")
        await asyncio.sleep(0.05)
        await manager.stop()
        assert "a" in processed
        assert not manager.running

    @pytest.mark.asyncio
    async def test_reset_increments_generation(self, manager):
        gen1 = manager.generation
        gen2 = manager.reset()
        assert gen2 == gen1 + 1
