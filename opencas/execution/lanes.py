"""Lane-based command queues for OpenCAS execution."""

from __future__ import annotations

import asyncio
from dataclasses import dataclass, field
from enum import StrEnum
from typing import Any, Dict, List, Optional


class CommandLane(StrEnum):
    """Execution lanes isolating different workload types."""

    CHAT = "chat"
    BAA = "baa"
    CONSOLIDATION = "consolidation"
    CRON = "cron"


@dataclass
class LaneConfig:
    """Configuration for a single lane."""

    max_concurrent: int = 2


@dataclass
class LaneState:
    """Runtime state for a single lane."""

    queue: asyncio.Queue[Any] = field(default_factory=asyncio.Queue)
    workers: List[asyncio.Task[None]] = field(default_factory=list)
    running: bool = False


class LaneManager:
    """Manages named lanes with independent queues and worker pools."""

    def __init__(
        self,
        configs: Optional[Dict[CommandLane, LaneConfig]] = None,
    ) -> None:
        defaults: Dict[CommandLane, LaneConfig] = {
            CommandLane.CHAT: LaneConfig(max_concurrent=1),
            CommandLane.BAA: LaneConfig(max_concurrent=2),
            CommandLane.CONSOLIDATION: LaneConfig(max_concurrent=1),
            CommandLane.CRON: LaneConfig(max_concurrent=1),
        }
        if configs:
            defaults.update(configs)
        self._configs = defaults
        self._lanes: Dict[CommandLane, LaneState] = {
            lane: LaneState() for lane in CommandLane
        }
        self._running = False
        self._generation = 0

    @property
    def running(self) -> bool:
        return self._running

    def submit(self, lane: CommandLane, item: Any) -> None:
        """Enqueue *item* into the specified *lane*."""
        if lane not in self._lanes:
            raise ValueError(f"Unknown lane: {lane}")
        self._lanes[lane].queue.put_nowait(item)

    async def get(self, lane: CommandLane) -> Any:
        """Blocking dequeue from *lane*."""
        return await self._lanes[lane].queue.get()

    def task_done(self, lane: CommandLane) -> None:
        """Mark a lane task as done."""
        self._lanes[lane].queue.task_done()

    def qsize(self, lane: CommandLane) -> int:
        """Return approximate queue depth for *lane*."""
        return self._lanes[lane].queue.qsize()

    def start(
        self,
        worker_factory,
    ) -> None:
        """Spawn worker tasks for every configured lane.

        *worker_factory* is a callable that takes a *lane* argument and
        returns a coroutine suitable for ``asyncio.create_task``.
        """
        if self._running:
            return
        self._running = True
        self._generation += 1
        for lane, config in self._configs.items():
            state = self._lanes[lane]
            state.running = True
            state.workers = [
                asyncio.create_task(worker_factory(lane))
                for _ in range(config.max_concurrent)
            ]

    async def stop(self) -> None:
        """Signal all lanes to drain and shut down."""
        if not self._running:
            return
        self._running = False
        for lane, state in self._lanes.items():
            state.running = False
            config = self._configs[lane]
            for _ in range(config.max_concurrent):
                state.queue.put_nowait(None)
        all_workers: List[asyncio.Task[None]] = []
        for state in self._lanes.values():
            all_workers.extend(state.workers)
        if all_workers:
            await asyncio.gather(*all_workers, return_exceptions=True)
        for state in self._lanes.values():
            state.workers.clear()

    async def drain(self, lane: Optional[CommandLane] = None) -> None:
        """Wait for the queue(s) to empty.

        If *lane* is omitted, drain all lanes.
        """
        targets = [lane] if lane else list(CommandLane)
        for target in targets:
            await self._lanes[target].queue.join()

    def reset(self) -> int:
        """Increment generation counter and return the new generation."""
        self._generation += 1
        return self._generation

    @property
    def generation(self) -> int:
        return self._generation
