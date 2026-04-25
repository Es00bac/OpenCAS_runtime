"""Bounded Assistant Agent (BAA) for background execution."""

from __future__ import annotations

import asyncio
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

from opencas.api import LLMClient
from opencas.embeddings import EmbeddingService
from opencas.infra import BaaCompletedEvent, BaaProgressEvent, EventBus
from opencas.memory import EpisodeKind, MemoryStore
from opencas.telemetry import EventKind, Tracer
from opencas.tools import ToolRegistry

from .executor import RepairExecutor
from .lanes import CommandLane, LaneConfig, LaneManager
from .lifecycle import LifecycleStage, TaskLifecycleMachine
from .models import ExecutionStage, RepairResult, RepairTask
from .receipt_store import ExecutionReceiptStore
from .store import TaskStore


def _sentinel_task() -> RepairTask:
    """Queue sentinel used to signal worker shutdown."""
    return RepairTask(objective="__sentinel__")


def _resolve_lane(task: RepairTask) -> CommandLane:
    """Map a task's lane string to a CommandLane enum value."""
    if task.lane:
        try:
            return CommandLane(task.lane)
        except ValueError:
            pass
    return CommandLane.BAA


class BoundedAssistantAgent:
    """Background repair executor with bounded concurrency per lane.

    Submitted tasks are queued into named lanes and executed by worker pools
    capped per lane. Results are stored in-memory and can be awaited via
    futures returned at submission time. When a *store* is provided, tasks are
    persisted durably and pending tasks are auto-resumed on start.
    """

    def __init__(
        self,
        tools: ToolRegistry,
        llm: Optional[LLMClient] = None,
        tracer: Optional[Tracer] = None,
        max_concurrent: int = 2,
        store: Optional[TaskStore] = None,
        event_bus: Optional[EventBus] = None,
        receipt_store: Optional[ExecutionReceiptStore] = None,
        runtime: Optional[Any] = None,
        memory: Optional[MemoryStore] = None,
        embeddings: Optional[EmbeddingService] = None,
    ) -> None:
        self.executor = RepairExecutor(tools=tools, llm=llm, tracer=tracer, runtime=runtime)
        self.tracer = tracer
        self.store = store
        self.event_bus = event_bus
        self.receipt_store = receipt_store
        self.max_concurrent = max_concurrent
        self.memory = memory
        self.embeddings = embeddings
        self.runtime = runtime
        self._lanes = LaneManager(
            configs={
                CommandLane.CHAT: LaneConfig(max_concurrent=1),
                CommandLane.BAA: LaneConfig(max_concurrent=max(1, max_concurrent)),
                CommandLane.CONSOLIDATION: LaneConfig(max_concurrent=1),
                CommandLane.CRON: LaneConfig(max_concurrent=1),
            }
        )
        self._results: Dict[str, RepairResult] = {}
        self._futures: Dict[str, asyncio.Future[RepairResult]] = {}
        self._held: Dict[str, RepairTask] = {}
        self._running = False

    async def submit(
        self,
        task: RepairTask,
        lane: Optional[CommandLane] = None,
    ) -> asyncio.Future[RepairResult]:
        """Enqueue *task* into *lane* and return a future for its result."""
        future: asyncio.Future[RepairResult] = asyncio.get_running_loop().create_future()
        task_id = str(task.task_id)
        self._futures[task_id] = future

        resolved_lane = lane or _resolve_lane(task)
        if lane is not None and task.lane is None:
            task.lane = resolved_lane.value

        if task.depends_on:
            ready = await self._dependencies_ready(task.depends_on)
            if not ready:
                self._held[task_id] = task
                if self.store:
                    await self.store.save(task)
                self._trace("task_held", {"task_id": task_id, "depends_on": task.depends_on, "lane": resolved_lane.value})
                return future

        if self.store:
            await self.store.save(task)
        self._lanes.submit(resolved_lane, task)
        self._trace("task_submitted", {"task_id": task_id, "objective": task.objective, "lane": resolved_lane.value})
        return future

    async def _dependencies_ready(self, deps: List[str]) -> bool:
        """Check if all dependency tasks or work objects have completed."""
        for dep_id in deps:
            if dep_id in self._results:
                continue
            if self.store:
                result = await self.store.get_result(dep_id)
                if result is not None:
                    continue
            if await self._work_dependency_ready(dep_id):
                continue
            return False
        return True

    async def _work_dependency_ready(self, dep_id: str) -> bool:
        """Return True when *dep_id* points at a completed work object."""
        runtime = self.runtime or getattr(self.executor, "runtime", None)
        ctx = getattr(runtime, "ctx", None)
        work_store = getattr(ctx, "work_store", None)
        if work_store is None:
            return False

        work = await work_store.get(dep_id)
        if work is None:
            return False

        stage = work.stage.value if hasattr(work.stage, "value") else str(work.stage)
        if stage not in {"artifact", "durable_work_stream"}:
            return False
        blocked_by = getattr(work, "blocked_by", None) or []
        return len(blocked_by) == 0

    async def _try_release_held(self) -> None:
        """Move held tasks whose dependencies are now ready into their lanes."""
        released: List[str] = []
        for task_id, task in list(self._held.items()):
            if await self._dependencies_ready(task.depends_on):
                self._lanes.submit(_resolve_lane(task), task)
                released.append(task_id)
                self._trace("task_released", {"task_id": task_id, "objective": task.objective, "lane": task.lane or CommandLane.BAA.value})
        for task_id in released:
            del self._held[task_id]

    async def start(self) -> None:
        """Start background workers for all lanes if not already running."""
        if self._running:
            return
        self._running = True
        if self.store:
            pending = await self.store.list_pending()
            for task in pending:
                task_id = str(task.task_id)
                if task_id not in self._futures:
                    future = asyncio.get_running_loop().create_future()
                    self._futures[task_id] = future
                task.stage = ExecutionStage.QUEUED
                task.status = "queued"
                if task.depends_on:
                    ready = await self._dependencies_ready(task.depends_on)
                    if not ready:
                        self._held[task_id] = task
                        await self.store.save(task)
                        continue
                await self.store.save(task)
                self._lanes.submit(_resolve_lane(task), task)
        self._lanes.start(worker_factory=self._worker_loop)

    async def resolve_hold(self, task_id: str) -> bool:
        """Release a held task back into its lane for execution."""
        task = self._held.pop(task_id, None)
        if task is None:
            return False
        await self._transition_task(
            task, LifecycleStage.QUEUED, f"hold resolved for {task_id}"
        )
        self._lanes.submit(_resolve_lane(task), task)
        self._trace("hold_resolved", {"task_id": task_id, "lane": task.lane or CommandLane.BAA.value})
        return True

    async def stop(self) -> None:
        """Signal workers to stop and drain the queues."""
        if not self._running:
            return
        self._running = False
        for lane in CommandLane:
            for _ in range(self._lanes._configs[lane].max_concurrent):
                self._lanes.submit(lane, _sentinel_task())
        await self._lanes.stop()

    def _worker_loop(self, lane: CommandLane) -> asyncio.Task[None]:
        """Return a coroutine that processes tasks from *lane*."""
        # This method returns a coroutine, which asyncio.create_task will wrap
        return self._worker_coro(lane)

    async def _worker_coro(self, lane: CommandLane) -> None:
        """Worker coroutine draining tasks from a specific lane."""
        while self._running:
            task = await self._lanes.get(lane)
            if task is None or task.objective == "__sentinel__":
                self._lanes.task_done(lane)
                break
            try:
                await self._run_bounded(task)
            except Exception as exc:
                self._trace("worker_exception", {"error": str(exc), "lane": lane.value})
            finally:
                pass  # task_done is called inside _run_bounded

    async def _transition_task(
        self,
        task: RepairTask,
        to_stage: LifecycleStage,
        reason: Optional[str] = None,
    ) -> None:
        """Validate and record a lifecycle stage transition for *task*."""
        stage_map: Dict[ExecutionStage, LifecycleStage] = {
            ExecutionStage.QUEUED: LifecycleStage.QUEUED,
            ExecutionStage.PLANNING: LifecycleStage.PLANNING,
            ExecutionStage.EXECUTING: LifecycleStage.EXECUTING,
            ExecutionStage.VERIFYING: LifecycleStage.VERIFYING,
            ExecutionStage.NEEDS_APPROVAL: LifecycleStage.NEEDS_APPROVAL,
            ExecutionStage.NEEDS_CLARIFICATION: LifecycleStage.NEEDS_CLARIFICATION,
            ExecutionStage.DONE: LifecycleStage.DONE,
            ExecutionStage.FAILED: LifecycleStage.FAILED,
        }
        from_stage = stage_map.get(task.stage)
        if from_stage is None and task.stage == ExecutionStage.RECOVERING:
            from_stage = LifecycleStage.EXECUTING
        if from_stage is None:
            from_stage = LifecycleStage.QUEUED
        if from_stage == to_stage:
            # No-op transition; just update execution stage mapping and save
            execution_stage_map: Dict[LifecycleStage, ExecutionStage] = {
                LifecycleStage.QUEUED: ExecutionStage.QUEUED,
                LifecycleStage.PLANNING: ExecutionStage.PLANNING,
                LifecycleStage.EXECUTING: ExecutionStage.EXECUTING,
                LifecycleStage.VERIFYING: ExecutionStage.VERIFYING,
                LifecycleStage.NEEDS_APPROVAL: ExecutionStage.NEEDS_APPROVAL,
                LifecycleStage.NEEDS_CLARIFICATION: ExecutionStage.NEEDS_CLARIFICATION,
                LifecycleStage.DONE: ExecutionStage.DONE,
                LifecycleStage.FAILED: ExecutionStage.FAILED,
            }
            task.stage = execution_stage_map.get(to_stage, task.stage)
            if self.store:
                await self.store.save(task)
            return
        transition = TaskLifecycleMachine.transition(
            task_id=str(task.task_id),
            from_stage=from_stage,
            to_stage=to_stage,
            reason=reason,
        )
        if self.store:
            await self.store.record_lifecycle_transition(
                transition_id=str(transition.transition_id),
                task_id=transition.task_id,
                from_stage=transition.from_stage.value,
                to_stage=transition.to_stage.value,
                reason=transition.reason,
                context=transition.context,
            )
        execution_stage_map: Dict[LifecycleStage, ExecutionStage] = {
            LifecycleStage.QUEUED: ExecutionStage.QUEUED,
            LifecycleStage.PLANNING: ExecutionStage.PLANNING,
            LifecycleStage.EXECUTING: ExecutionStage.EXECUTING,
            LifecycleStage.VERIFYING: ExecutionStage.VERIFYING,
            LifecycleStage.NEEDS_APPROVAL: ExecutionStage.NEEDS_APPROVAL,
            LifecycleStage.NEEDS_CLARIFICATION: ExecutionStage.NEEDS_CLARIFICATION,
            LifecycleStage.DONE: ExecutionStage.DONE,
            LifecycleStage.FAILED: ExecutionStage.FAILED,
        }
        task.stage = execution_stage_map.get(to_stage, task.stage)
        if self.store:
            await self.store.save(task)
        # Auto-pause tasks that require operator input
        if to_stage in (LifecycleStage.NEEDS_APPROVAL, LifecycleStage.NEEDS_CLARIFICATION):
            task_id = str(task.task_id)
            self._held[task_id] = task
            self._trace(
                "task_held_for_approval",
                {"task_id": task_id, "stage": task.stage.value, "reason": reason},
            )

    async def _run_bounded(self, task: RepairTask) -> None:
        lane = _resolve_lane(task)
        await self._transition_task(task, LifecycleStage.PLANNING, "worker started")
        task_id = str(task.task_id)
        if task_id in self._held:
            self._lanes.task_done(lane)
            return
        if self.event_bus:
            await self.event_bus.emit(
                BaaProgressEvent(
                    task_id=task_id,
                    stage=task.stage.value,
                    objective=task.objective,
                    attempt=task.attempt,
                )
            )
        pre_phases = len(task.phases)
        await self._transition_task(task, LifecycleStage.EXECUTING, "begin execution")
        if task_id in self._held:
            self._lanes.task_done(lane)
            return
        result = await self.executor.run(task)
        if self.event_bus:
            for phase_record in task.phases[pre_phases:]:
                await self.event_bus.emit(
                    BaaProgressEvent(
                        task_id=task_id,
                        stage=phase_record.phase.value,
                        objective=task.objective,
                        attempt=task.attempt,
                    )
                )
        terminal_stages = (ExecutionStage.DONE, ExecutionStage.FAILED)
        if result.stage == ExecutionStage.RECOVERING:
            recovery_count = task.meta.get("recovery_count", 0) + 1
            task.meta["recovery_count"] = recovery_count
            if self.store:
                await self.store.record_transition(
                    task_id=task_id,
                    from_stage=ExecutionStage.EXECUTING,
                    to_stage=ExecutionStage.RECOVERING,
                    reason=f"attempt {task.attempt} failed, recovery_count={recovery_count}",
                )
            if recovery_count >= 10:
                await self._transition_task(
                    task, LifecycleStage.FAILED,
                    f"Recovery cap exceeded after {recovery_count} retries.",
                )
                task.status = "failed"
                result = RepairResult(
                    task_id=task.task_id,
                    success=False,
                    stage=ExecutionStage.FAILED,
                    output=f"Recovery cap exceeded after {recovery_count} retries.",
                    artifacts=task.artifacts,
                )
            else:
                await self._transition_task(
                    task, LifecycleStage.QUEUED,
                    f"retry scheduled, recovery_count={recovery_count}",
                )
                task.status = "queued"
                self._lanes.submit(lane, task)
                self._trace(
                    "task_requeued",
                    {"task_id": task_id, "recovery_count": recovery_count, "lane": lane.value},
                )
                self._lanes.task_done(lane)
                await self._try_release_held()
                return

        if result.stage in terminal_stages:
            if result.stage == ExecutionStage.DONE:
                await self._transition_task(task, LifecycleStage.DONE, "execution completed successfully")
                await self._extract_procedural_memory(task, result)
            elif result.stage == ExecutionStage.FAILED and task.stage != ExecutionStage.FAILED:
                await self._transition_task(task, LifecycleStage.FAILED, result.output or "execution failed")
            if self.store:
                await self.store.save_result(result)
            if self.receipt_store:
                await self.receipt_store.save(task, result)
            if self.event_bus:
                await self.event_bus.emit(
                    BaaCompletedEvent(
                        task_id=task_id,
                        success=result.success,
                        stage=result.stage.value,
                        objective=task.objective,
                        output=result.output,
                    )
                )
            self._results[task_id] = result
            future = self._futures.pop(task_id, None)
            if future and not future.done():
                future.set_result(result)
        if task_id in self._held:
            self._lanes.task_done(lane)
            await self._try_release_held()
            return
        await self._try_release_held()
        self._trace(
            "task_finished",
            {"task_id": task_id, "success": result.success, "stage": result.stage.value, "lane": lane.value},
        )
        self._lanes.task_done(lane)

    def list_results(self) -> List[RepairResult]:
        """Return all completed results in submission order (best effort)."""
        return list(self._results.values())

    @property
    def queue_size(self) -> int:
        """Total approximate queue depth across all lanes."""
        return sum(self._lanes.qsize(lane) for lane in CommandLane)

    @property
    def held_size(self) -> int:
        """Number of tasks held for dependencies."""
        return len(self._held)

    @property
    def active_count(self) -> int:
        """Approximate number of in-flight tasks currently being worked."""
        unresolved = len(self._futures)
        return max(0, unresolved - self.held_size - self.queue_size)

    def lane_snapshot(self) -> Dict[str, Any]:
        """Per-lane queue depths and concurrency limits for operator visibility."""
        return {
            lane.value: {
                "queue_depth": self._lanes.qsize(lane),
                "max_concurrent": self._lanes._configs[lane].max_concurrent,
            }
            for lane in CommandLane
        }

    async def _extract_procedural_memory(self, task: RepairTask, result: RepairResult) -> None:
        """Summarize a successful task's tool sequence into a procedural episode."""
        if self.memory is None:
            return
        from datetime import datetime, timezone
        from opencas.memory import Episode, EpisodeKind

        task_id = str(task.task_id)
        episodes = await self.memory.list_episodes(session_id=task_id, limit=200)
        tool_eps = [
            ep for ep in episodes
            if ep.kind in (EpisodeKind.ACTION, EpisodeKind.OBSERVATION)
        ]
        if not tool_eps:
            return
        tool_eps.sort(key=lambda ep: ep.created_at)
        tool_lines = [f"- {ep.content}" for ep in tool_eps]
        summary = (
            f"Objective: {task.objective}\n"
            f"Tool sequence:\n" + "\n".join(tool_lines) + "\n"
            f"Outcome: {result.output}"
        )
        embed_id: Optional[str] = None
        if self.embeddings is not None:
            try:
                embed_record = await self.embeddings.embed(
                    summary, task_type="retrieval_document"
                )
                embed_id = embed_record.source_hash
            except Exception:
                pass
        procedural_episode = Episode(
            kind=EpisodeKind.PROCEDURAL,
            session_id=task_id,
            content=summary,
            embedding_id=embed_id,
            salience=2.0,
        )
        await self.memory.save_episode(procedural_episode)
        self._trace("procedural_memory_saved", {"task_id": task_id, "episode_id": str(procedural_episode.episode_id)})

    def _trace(self, event: str, payload: Dict[str, Any]) -> None:
        if self.tracer:
            self.tracer.log(
                EventKind.TOOL_CALL,
                f"BoundedAssistantAgent: {event}",
                payload,
            )
