"""Bounded Assistant Agent (BAA) for background execution."""

from __future__ import annotations

import asyncio
import inspect
import json
from datetime import datetime, timedelta, timezone
from typing import Any, Dict, List, Optional, Tuple

from opencas.api import LLMClient
from opencas.api.provenance_store import (
    ProvenanceTransitionKind,
    record_provenance_transition,
)
from opencas.autonomy.commitment import CommitmentStatus
from opencas.autonomy.goal_hygiene import is_self_referential_suppression_metadata
from opencas.embeddings import EmbeddingService
from opencas.infra import BaaCompletedEvent, BaaProgressEvent, EventBus
from opencas.memory import EpisodeKind, MemoryStore
from opencas.proof_chain import link_receipt_to_promise_claim
from opencas.provenance_events_adapter import ProvenanceEventType, emit_provenance_event
from opencas.projects.operator_followthrough import (
    copy_safe_operator_project_followthrough_evidence,
    has_operator_project_followthrough_authority,
)
from opencas.runtime.system_message_awareness import record_agent_visible_system_message
from opencas.scheduling.models import (
    ScheduleAction,
    ScheduleKind,
    ScheduleRecurrence,
    ScheduleStatus,
)
from opencas.telemetry import EventKind, Tracer
from opencas.thread_registry import BeadSourceKind, ThreadStatus
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


_NON_LIVE_REFRAME_ORIGINS = frozenset(
    {
        "cognitive_curiosity",
        "cognitive_maintenance",
        "daydream",
        "daydream_keeper",
        "daydream_signal",
        "self_inspection_gap",
    }
)
_RECOVERABLE_FOLLOWTHROUGH_MARKERS = (
    "tool loop guard",
    "meaningful progress contract",
    "new-project contract",
    "project contract",
    "project_contract_blocked",
    "no meaningful progress",
    "retry blocked",
    "reframe narrowly",
    "preserve partial value",
    "provider backoff",
    "rate limit",
    "429",
    "readtimeout",
    "read timeout",
    "timeout",
    "dns_resolution_failed",
    "temporarily unavailable",
    "circuit open",
    "server error",
    "500",
    "upstream",
    "sibling-project materialization",
    "sibling project artifacts",
    "materializing sibling",
)
_EXTERNAL_BLOCKER_MARKERS = (
    "auth required",
    "authentication required",
    "approval required",
    "credential required",
    "credentials required",
    "external input",
    "inaccessible workspace",
    "missing input",
    "missing required user input",
    "needs clarification",
    "permission denied",
    "permission required",
    "policy refusal",
    "privacy boundary",
    "refusal",
    "safety policy",
    "workspace inaccessible",
    "workspace missing",
)
_RETRY_EXHAUSTION_MARKERS = (
    "after retries",
    "after 3 attempts",
    "attempts exhausted",
    "execution failed after",
    "exhausted",
    "failed after 3 attempts",
    "recovery cap exceeded",
    "retry exhaustion",
    "retries exhausted",
)
_RECOVERABLE_LEGACY_FAILURE_MARKERS = (
    "planning failed",
    "verification failed",
    "tool-loop timeout",
    "tool loop timeout",
)
_FOLLOWTHROUGH_SCHEDULE_SOURCES = frozenset(
    {
        "project_followthrough_recovery",
        "project_followthrough_success_continuation",
        "project_followthrough_legacy_recovery",
    }
)
_SCHEDULE_FAILURE_RECOVERY_SOURCES = frozenset({"schedule_failure_recovery"})
_ARTIFACT_PROGRESS_RECOVERY_CAP = 50
_SCHEDULE_FAILURE_RECOVERY_CAP = 3


class BoundedAssistantAgent:
    """Background repair executor with bounded concurrency per lane.

    Submitted tasks are queued into named lanes and executed by worker pools
    capped per lane. Results are stored in-memory and can be awaited via
    futures returned at submission time. When a *store* is provided, tasks are
    persisted durably and pending tasks are auto-resumed on start.
    """

    _DUPLICATE_SUCCESS_COOLDOWN_SECONDS = 6 * 60 * 60
    _DUPLICATE_FAILURE_COOLDOWN_SECONDS = 45 * 60
    _AWARENESS_DEDUP_STATUSES = frozenset({"held"})

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
        self.executor = RepairExecutor(
            tools=tools,
            llm=llm,
            tracer=tracer,
            runtime=runtime,
            store=store,
        )
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
        self._live_tasks: Dict[str, RepairTask] = {}
        self._duplicate_aliases: Dict[str, List[asyncio.Future[RepairResult]]] = {}
        self._recent_terminal: Dict[Tuple[str, str, str], Tuple[RepairTask, RepairResult]] = {}
        self._running = False

    async def submit(
        self,
        task: RepairTask,
        lane: Optional[CommandLane] = None,
    ) -> asyncio.Future[RepairResult]:
        """Enqueue *task* into *lane* and return a future for its result."""
        duplicate_future = await self._duplicate_future(task)
        if duplicate_future is not None:
            return duplicate_future

        future: asyncio.Future[RepairResult] = asyncio.get_running_loop().create_future()
        task_id = str(task.task_id)
        self._futures[task_id] = future

        resolved_lane = lane or _resolve_lane(task)
        if lane is not None and task.lane is None:
            task.lane = resolved_lane.value

        context_hold_reason = await self._context_guard_hold_reason(task)
        if context_hold_reason:
            task.meta["context_guard_status"] = "held"
            task.meta["context_guard_reason"] = context_hold_reason
            self._held[task_id] = task
            self._live_tasks[task_id] = task
            if self.store:
                await self.store.save(task)
            await self._record_baa_awareness(
                task,
                status="held",
                reason=context_hold_reason,
                lane=resolved_lane,
            )
            self._trace(
                "task_held_context_guard",
                {
                    "task_id": task_id,
                    "reason": context_hold_reason,
                    "lane": resolved_lane.value,
                    "origin_context_lane": task.meta.get("origin_context_lane"),
                },
            )
            return future

        executive_pause_reason = self._executive_pause_hold_reason(task)
        if executive_pause_reason:
            await self._hold_task_for_executive_pause(
                task,
                lane=resolved_lane,
                reason=executive_pause_reason,
            )
            return future

        self.executor.record_task_boundary(
            task,
            boundary="task_accepted",
            workflow_phase="start",
            artifact="repair-task|default|accepted",
            why=f"task accepted into {resolved_lane.value}",
            action="COMMIT",
            risk="LOW",
            source_trace={
                "lane": resolved_lane.value,
                "depends_on": list(task.depends_on),
            },
        )

        if task.depends_on:
            ready = await self._dependencies_ready(task.depends_on)
            if not ready:
                task.meta["held_reason"] = "waiting_for_dependencies"
                self._held[task_id] = task
                self._live_tasks[task_id] = task
                if self.store:
                    await self.store.save(task)
                await self._record_baa_awareness(
                    task,
                    status="held",
                    reason="waiting for dependencies",
                    lane=resolved_lane,
                )
                self._trace("task_held", {"task_id": task_id, "depends_on": task.depends_on, "lane": resolved_lane.value})
                return future

        if self.store:
            await self.store.save(task)
        self._live_tasks[task_id] = task
        self._lanes.submit(resolved_lane, task)
        await self._record_baa_awareness(
            task,
            status="submitted",
            reason=f"task queued in {resolved_lane.value}",
            lane=resolved_lane,
        )
        self._trace("task_submitted", {"task_id": task_id, "objective": task.objective, "lane": resolved_lane.value})
        return future

    async def _context_guard_hold_reason(self, task: RepairTask) -> Optional[str]:
        """Prevent reflective context from becoming executable without arbitration."""

        meta = task.meta or {}
        if str(meta.get("origin_context_lane") or "").strip().lower() != "reflective":
            return None
        if (
            str(meta.get("authority") or "").strip().lower() == "executive_committed"
            and str(meta.get("proposal_validation_status") or "").strip().lower() == "accepted"
            and str(meta.get("arbiter_decision_id") or "").strip()
        ):
            return None
        accepted_ids = self._accepted_proposal_ids_from_meta(meta)
        if accepted_ids and await self._accepted_proposals_are_committed(accepted_ids):
            meta["authority"] = "executive_committed"
            meta["proposal_validation_status"] = "accepted"
            meta.setdefault("accepted_proposal_ids", accepted_ids)
            if not str(meta.get("arbiter_decision_id") or "").strip():
                meta["arbiter_decision_id"] = f"accepted_proposals:{','.join(accepted_ids)}"
            task.meta = meta
            return None
        return "reflective_proposal_not_committed"

    def _executive_pause_hold_reason(self, task: RepairTask) -> Optional[str]:
        meta = task.meta if isinstance(task.meta, dict) else {}
        if meta.get("manual") is True:
            return None
        runtime = self.runtime or getattr(self.executor, "runtime", None)
        executive = getattr(runtime, "executive", None) if runtime is not None else None
        if executive is None:
            return None
        try:
            paused = bool(executive.recommend_pause())
        except Exception:
            return None
        if not paused:
            return None
        reason = "executive_pause"
        pause_reason = getattr(executive, "pause_reason", None)
        if callable(pause_reason):
            try:
                reason = str(pause_reason() or reason)
            except Exception:
                reason = "executive_pause"
        if reason == "overload" and self._has_operator_project_return_evidence(task):
            return None
        return reason

    @staticmethod
    def _has_operator_project_return_evidence(task: RepairTask) -> bool:
        return has_operator_project_followthrough_authority(
            task.meta if isinstance(task.meta, dict) else {},
            allow_schedule_authority=True,
        )

    async def _hold_task_for_executive_pause(
        self,
        task: RepairTask,
        *,
        lane: CommandLane,
        reason: str,
    ) -> None:
        task_id = str(task.task_id)
        task.meta["held_reason"] = "executive_recommended_pause"
        task.meta["executive_pause_reason"] = reason
        task.status = "held"
        task.updated_at = datetime.now(timezone.utc)
        self._held[task_id] = task
        self._live_tasks[task_id] = task
        if self.store:
            await self.store.save(task)
        await self._record_baa_awareness(
            task,
            status="held",
            reason=f"executive pause: {reason}",
            lane=lane,
        )
        self._trace(
            "task_held_executive_pause",
            {"task_id": task_id, "reason": reason, "lane": lane.value},
        )

    @staticmethod
    def _accepted_proposal_ids_from_meta(meta: Dict[str, Any]) -> List[str]:
        raw_values = []
        explicit = meta.get("accepted_proposal_ids")
        if isinstance(explicit, (list, tuple, set)):
            raw_values.extend(explicit)
        elif explicit:
            raw_values.append(explicit)
        for key in ("context_proposal_id", "proposal_id"):
            value = meta.get(key)
            if value:
                raw_values.append(value)
        proposal_ids: List[str] = []
        for value in raw_values:
            text = str(value or "").strip()
            if text and text not in proposal_ids:
                proposal_ids.append(text)
        return proposal_ids

    def _context_proposal_store(self) -> Any:
        runtime = self.runtime or getattr(self.executor, "runtime", None)
        return (
            getattr(runtime, "context_proposals", None)
            or getattr(getattr(runtime, "ctx", None), "context_proposal_store", None)
        )

    async def _accepted_proposals_are_committed(self, proposal_ids: List[str]) -> bool:
        store = self._context_proposal_store()
        get_proposal = getattr(store, "get", None)
        if not callable(get_proposal):
            return False
        for proposal_id in proposal_ids:
            try:
                proposal = get_proposal(proposal_id)
                if inspect.isawaitable(proposal):
                    proposal = await proposal
            except Exception:
                return False
            if proposal is None:
                return False
            status = str(
                getattr(getattr(proposal, "status", ""), "value", getattr(proposal, "status", ""))
            ).strip().lower()
            authority = str(
                getattr(getattr(proposal, "authority", ""), "value", getattr(proposal, "authority", ""))
            ).strip().lower()
            validation = dict(getattr(proposal, "validation", {}) or {})
            decision_id = str(validation.get("arbiter_decision_id") or "").strip()
            if status != "accepted" or authority != "executive_committed":
                return False
            if not decision_id:
                return False
        return True

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
            executive_pause_reason = self._executive_pause_hold_reason(task)
            if executive_pause_reason:
                task.meta["held_reason"] = "executive_recommended_pause"
                task.meta["executive_pause_reason"] = executive_pause_reason
                if self.store:
                    await self.store.save(task)
                self._trace(
                    "task_still_held_executive_pause",
                    {
                        "task_id": task_id,
                        "reason": executive_pause_reason,
                        "lane": task.lane or CommandLane.BAA.value,
                    },
                )
                continue
            context_hold_reason = await self._context_guard_hold_reason(task)
            if context_hold_reason:
                task.meta["context_guard_status"] = "held"
                task.meta["context_guard_reason"] = context_hold_reason
                if self.store:
                    await self.store.save(task)
                self._trace(
                    "task_still_held_context_guard",
                    {
                        "task_id": task_id,
                        "reason": context_hold_reason,
                        "lane": task.lane or CommandLane.BAA.value,
                    },
                )
                continue
            if await self._dependencies_ready(task.depends_on):
                if task.meta.get("context_guard_status") == "held":
                    task.meta["context_guard_status"] = "released"
                task.meta.pop("context_guard_reason", None)
                task.meta.pop("held_reason", None)
                task.meta.pop("executive_pause_reason", None)
                self._record_session_resume(
                    task,
                    reason=f"dependencies resolved for {task_id}",
                    source_trace={"task_id": task_id, "source": "dependency"},
                )
                if self.store:
                    await self.store.save(task)
                self._lanes.submit(_resolve_lane(task), task)
                await self._record_baa_awareness(
                    task,
                    status="released",
                    reason=f"dependencies resolved for {task_id}",
                    lane=_resolve_lane(task),
                )
                released.append(task_id)
                self._trace("task_released", {"task_id": task_id, "objective": task.objective, "lane": task.lane or CommandLane.BAA.value})
        for task_id in released:
            del self._held[task_id]

    async def try_release_held(self) -> int:
        """Public heartbeat hook for rechecking held tasks."""
        await self.reconcile_legacy_followthrough_commitments()
        before = len(self._held)
        await self._try_release_held()
        return max(0, before - len(self._held))

    async def start(self) -> None:
        """Start background workers for all lanes if not already running."""
        if self._running:
            return
        self._running = True
        await self._hydrate_duplicate_state_from_store()
        await self.reconcile_legacy_followthrough_commitments()
        if self.store:
            pending = await self.store.list_pending()
            for task in pending:
                task_id = str(task.task_id)
                if task_id in self._futures or task_id in self._results:
                    continue
                if task_id not in self._futures:
                    future = asyncio.get_running_loop().create_future()
                    self._futures[task_id] = future
                task.stage = ExecutionStage.QUEUED
                if task.status != "held":
                    task.status = "queued"
                context_hold_reason = await self._context_guard_hold_reason(task)
                if context_hold_reason:
                    task.meta["context_guard_status"] = "held"
                    task.meta["context_guard_reason"] = context_hold_reason
                    self._held[task_id] = task
                    self._live_tasks[task_id] = task
                    await self.store.save(task)
                    await self._record_baa_awareness(
                        task,
                        status="held",
                        reason=f"restored task held by context guard: {context_hold_reason}",
                        lane=_resolve_lane(task),
                    )
                    continue
                executive_pause_reason = self._executive_pause_hold_reason(task)
                if executive_pause_reason:
                    await self._hold_task_for_executive_pause(
                        task,
                        lane=_resolve_lane(task),
                        reason=executive_pause_reason,
                    )
                    continue
                if task.depends_on:
                    ready = await self._dependencies_ready(task.depends_on)
                    if not ready:
                        task.meta["held_reason"] = "waiting_for_dependencies"
                        self._held[task_id] = task
                        self._live_tasks[task_id] = task
                        await self.store.save(task)
                        await self._record_baa_awareness(
                            task,
                            status="held",
                            reason="restored task waiting for dependencies",
                            lane=_resolve_lane(task),
                        )
                        continue
                if task.status == "held":
                    task.status = "queued"
                    if task.meta.get("held_reason") == "executive_recommended_pause":
                        task.meta.pop("held_reason", None)
                        task.meta.pop("executive_pause_reason", None)
                await self.store.save(task)
                self._live_tasks[task_id] = task
                self._record_session_resume(
                    task,
                    reason="pending task restored on start",
                    source_trace={"lane": _resolve_lane(task).value, "source": "store"},
                )
                if self.store:
                    await self.store.save(task)
                self._lanes.submit(_resolve_lane(task), task)
                await self._record_baa_awareness(
                    task,
                    status="restored",
                    reason="pending task restored on start",
                    lane=_resolve_lane(task),
                )
        self._lanes.start(worker_factory=self._worker_loop)

    async def resolve_hold(self, task_id: str) -> bool:
        """Release a held task back into its lane for execution."""
        task = self._held.get(task_id)
        if task is None:
            return False
        context_hold_reason = await self._context_guard_hold_reason(task)
        if context_hold_reason:
            task.meta["context_guard_status"] = "held"
            task.meta["context_guard_reason"] = context_hold_reason
            if self.store:
                await self.store.save(task)
            self._trace(
                "hold_resolution_blocked_context_guard",
                {
                    "task_id": task_id,
                    "reason": context_hold_reason,
                    "lane": task.lane or CommandLane.BAA.value,
                },
            )
            return False
        self._held.pop(task_id, None)
        if task.meta.get("context_guard_status") == "held":
            task.meta["context_guard_status"] = "released"
        task.meta.pop("context_guard_reason", None)
        task.meta.pop("held_reason", None)
        task.meta.pop("executive_pause_reason", None)
        await self._transition_task(
            task, LifecycleStage.QUEUED, f"hold resolved for {task_id}"
        )
        self._record_session_resume(
            task,
            reason=f"hold resolved for {task_id}",
            source_trace={"task_id": task_id, "source": "operator"},
        )
        if self.store:
            await self.store.save(task)
        self._live_tasks[str(task.task_id)] = task
        self._lanes.submit(_resolve_lane(task), task)
        await self._record_baa_awareness(
            task,
            status="released",
            reason=f"hold resolved for {task_id}",
            lane=_resolve_lane(task),
        )
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
                await self._handle_worker_exception(task, lane=lane, exc=exc)
            finally:
                self._lanes.task_done(lane)

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
            self.executor.record_task_boundary(
                task,
                boundary="operator_input_requested",
                workflow_phase="pause",
                artifact="repair-task|default|operator-input",
                why=reason or f"operator input requested at {to_stage.value}",
                action="UPDATE",
                risk="MEDIUM",
                source_trace={
                    "from_stage": from_stage.value,
                    "to_stage": to_stage.value,
                    "reason": reason,
                },
            )
            self._record_waiting_provenance(task=task, reason=reason)
            if self.store:
                await self.store.save(task)
            task_id = str(task.task_id)
            task.meta["held_reason"] = to_stage.value
            self._held[task_id] = task
            self._trace(
                "task_held_for_approval",
                {"task_id": task_id, "stage": task.stage.value, "reason": reason},
            )
            await self._notify_owner_task_attention(
                task,
                status="needs_attention",
                reason=reason or f"task requires {to_stage.value}",
                urgency="high",
            )
            await self._record_baa_awareness(
                task,
                status="needs_attention",
                reason=reason or f"task requires {to_stage.value}",
                stage=to_stage,
                lane=_resolve_lane(task),
            )
            if self.store:
                await self.store.save(task)

    async def _run_bounded(self, task: RepairTask) -> None:
        lane = _resolve_lane(task)
        executive_pause_reason = self._executive_pause_hold_reason(task)
        if executive_pause_reason:
            await self._hold_task_for_executive_pause(
                task,
                lane=lane,
                reason=executive_pause_reason,
            )
            return
        await self._transition_task(task, LifecycleStage.PLANNING, "worker started")
        task_id = str(task.task_id)
        if task_id in self._held:
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
            return
        await self._record_baa_awareness(
            task,
            status="started",
            reason="begin execution",
            lane=lane,
        )
        result = await self.executor.run(task)
        if self.store:
            # Persist any provenance or other task meta mutations produced by execution wrappers.
            await self.store.save(task)
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
        future_to_resolve: Optional[asyncio.Future[RepairResult]] = None
        aliases_to_resolve: List[asyncio.Future[RepairResult]] = []
        if result.stage == ExecutionStage.RECOVERING:
            artifact_progress_boundary = self._artifact_progress_boundary_recovery(task, result)
            if artifact_progress_boundary:
                recovery_count = int(task.meta.get("recovery_count", 0) or 0)
                progress_boundary_count = int(task.meta.get("progress_boundary_recovery_count", 0) or 0) + 1
                task.meta["progress_boundary_recovery_count"] = progress_boundary_count
            else:
                recovery_count = int(task.meta.get("recovery_count", 0) or 0) + 1
                progress_boundary_count = int(task.meta.get("progress_boundary_recovery_count", 0) or 0)
                task.meta["recovery_count"] = recovery_count
            if self.store:
                await self.store.record_transition(
                    task_id=task_id,
                    from_stage=ExecutionStage.EXECUTING,
                    to_stage=ExecutionStage.RECOVERING,
                    reason=(
                        "artifact progress boundary; continuing without burning failure cap, "
                        f"progress_boundary_recovery_count={progress_boundary_count}"
                        if artifact_progress_boundary
                        else f"attempt {task.attempt} failed, recovery_count={recovery_count}"
                    ),
                )
            if (
                (not artifact_progress_boundary and recovery_count >= task.max_attempts)
                or (
                    artifact_progress_boundary
                    and progress_boundary_count >= _ARTIFACT_PROGRESS_RECOVERY_CAP
                )
            ):
                await self._transition_task(
                    task, LifecycleStage.FAILED,
                    (
                        f"Artifact progress boundary cap exceeded after {progress_boundary_count} continuations."
                        if artifact_progress_boundary
                        else f"Recovery cap exceeded after {recovery_count} retries."
                    ),
                )
                task.status = "failed"
                result = RepairResult(
                    task_id=task.task_id,
                    success=False,
                    stage=ExecutionStage.FAILED,
                    output=(
                        f"Artifact progress boundary cap exceeded after {progress_boundary_count} continuations."
                        if artifact_progress_boundary
                        else f"Recovery cap exceeded after {recovery_count} retries."
                    ),
                    artifacts=task.artifacts,
                )
            else:
                await self._transition_task(
                    task, LifecycleStage.QUEUED,
                    (
                        f"artifact progress continuation scheduled, progress_boundary_recovery_count={progress_boundary_count}"
                        if artifact_progress_boundary
                        else f"retry scheduled, recovery_count={recovery_count}"
                    ),
                )
                task.status = "queued"
                executive_pause_reason = self._executive_pause_hold_reason(task)
                if executive_pause_reason:
                    await self._hold_task_for_executive_pause(
                        task,
                        lane=lane,
                        reason=executive_pause_reason,
                    )
                    return
                self._lanes.submit(lane, task)
                self._trace(
                    "task_requeued",
                    {
                        "task_id": task_id,
                        "recovery_count": recovery_count,
                        "progress_boundary_recovery_count": progress_boundary_count,
                        "artifact_progress_boundary": artifact_progress_boundary,
                        "lane": lane.value,
                    },
                )
                await self._try_release_held()
                return

        if result.stage in terminal_stages:
            self._live_tasks.pop(task_id, None)
            if result.stage == ExecutionStage.DONE:
                await self._transition_task(task, LifecycleStage.DONE, "execution completed successfully")
                await self._extract_procedural_memory(task, result)
                await self._ensure_success_followthrough_continuation(task, result)
            elif result.stage == ExecutionStage.FAILED:
                if task.stage != ExecutionStage.FAILED:
                    await self._transition_task(task, LifecycleStage.FAILED, result.output or "execution failed")
                continued = await self._ensure_recoverable_followthrough_continuation(
                    task,
                    result,
                    reason=result.output or "execution failed",
                )
                if not continued:
                    continued = await self._ensure_schedule_failure_recovery(
                        task,
                        result,
                        reason=result.output or "execution failed",
                    )
                if not continued:
                    await self._block_linked_commitment_on_failure(task, result.output or "execution failed")
                    await self._notify_owner_task_attention(
                        task,
                        status="failed",
                        reason=result.output or "execution failed",
                        urgency="high",
                    )
            if self.store:
                await self.store.save_result(result)
            receipt = None
            if self.receipt_store:
                receipt = await self.receipt_store.save(task, result)
                await self._link_promise_receipt(receipt)
            await self._record_baa_awareness(
                task,
                status="completed" if result.success else "failed",
                reason=result.output or result.stage.value,
                result=result,
                receipt=receipt,
                lane=lane,
            )
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
            future_to_resolve = self._futures.pop(task_id, None)
            aliases_to_resolve = self._duplicate_aliases.pop(task_id, [])
            self._remember_terminal_duplicate(task, result)
        if task_id in self._held:
            await self._try_release_held()
            if future_to_resolve and not future_to_resolve.done():
                future_to_resolve.set_result(result)
            for alias in aliases_to_resolve:
                if not alias.done():
                    alias.set_result(result)
            return
        await self._try_release_held()
        self._trace(
            "task_finished",
            {"task_id": task_id, "success": result.success, "stage": result.stage.value, "lane": lane.value},
        )
        if future_to_resolve and not future_to_resolve.done():
            future_to_resolve.set_result(result)
        for alias in aliases_to_resolve:
            if not alias.done():
                alias.set_result(result)

    async def _handle_worker_exception(self, task: RepairTask, *, lane: CommandLane, exc: Exception) -> None:
        """Turn unexpected worker crashes into durable failed task state."""
        task_id = str(task.task_id)
        message = f"worker exception: {exc}"
        try:
            await self._transition_task(task, LifecycleStage.FAILED, message)
        except Exception as transition_exc:
            task.stage = ExecutionStage.FAILED
            task.status = "failed"
            self._trace(
                "task_failure_transition_error",
                {"task_id": task_id, "error": str(transition_exc), "original_error": str(exc)},
            )
            if self.store:
                await self.store.save(task)

        result = RepairResult(
            task_id=task.task_id,
            success=False,
            stage=ExecutionStage.FAILED,
            output=message,
            artifacts=task.artifacts,
        )
        self._live_tasks.pop(task_id, None)
        continued = await self._ensure_schedule_failure_recovery(task, result, reason=message)
        if not continued:
            await self._block_linked_commitment_on_failure(task, message)
            await self._notify_owner_task_attention(
                task,
                status="failed",
                reason=message,
                urgency="high",
            )
        if self.store:
            await self.store.save_result(result)
        receipt = None
        if self.receipt_store:
            receipt = await self.receipt_store.save(task, result)
            await self._link_promise_receipt(receipt)
        await self._record_baa_awareness(
            task,
            status="failed",
            reason=message,
            result=result,
            receipt=receipt,
            lane=lane,
        )
        if self.event_bus:
            await self.event_bus.emit(
                BaaCompletedEvent(
                    task_id=task_id,
                    success=False,
                    stage=ExecutionStage.FAILED.value,
                    objective=task.objective,
                    output=message,
                )
            )
        self._results[task_id] = result
        future = self._futures.pop(task_id, None)
        if future and not future.done():
            future.set_result(result)
        for alias in self._duplicate_aliases.pop(task_id, []):
            if not alias.done():
                alias.set_result(result)
        self._remember_terminal_duplicate(task, result)
        self._trace(
            "task_finished",
            {"task_id": task_id, "success": False, "stage": ExecutionStage.FAILED.value, "lane": lane.value},
        )

    @staticmethod
    def _artifact_progress_boundary_recovery(task: RepairTask, result: RepairResult) -> bool:
        """Return true when RECOVERING means a bounded progress slice finished."""
        if result.stage != ExecutionStage.RECOVERING:
            return False
        meta = task.meta if isinstance(task.meta, dict) else {}
        loop_meta = meta.get("last_tool_loop") if isinstance(meta.get("last_tool_loop"), dict) else {}
        try:
            iterations = int(loop_meta.get("iterations") or 0)
        except (TypeError, ValueError):
            iterations = 0
        loop_boundary = iterations >= 32 and not bool(loop_meta.get("guard_fired"))
        text = " ".join(
            [
                str(result.output or ""),
                " ".join(str(item) for item in task.artifacts[-5:]),
            ]
        ).lower()
        if "artifact progress boundary" in text or "reached maximum number of tool-use iterations" in text:
            loop_boundary = True
        timeout_progress = meta.get("timeout_artifact_progress")
        timeout_paths = (
            timeout_progress.get("paths")
            if isinstance(timeout_progress, dict) and isinstance(timeout_progress.get("paths"), list)
            else []
        )
        if timeout_paths and (
            "artifact progress boundary" in text or "tool loop timed out" in text
        ):
            return True
        if not loop_boundary:
            return False
        calls = meta.get("last_tool_calls") if isinstance(meta.get("last_tool_calls"), list) else []
        write_tools = {"fs_write_file", "edit_file", "write_file", "fs_edit_file"}
        return any(isinstance(call, dict) and call.get("name") in write_tools for call in calls)

    async def _record_baa_awareness(
        self,
        task: RepairTask,
        *,
        status: str,
        reason: str = "",
        result: Optional[RepairResult] = None,
        receipt: Optional[Any] = None,
        lane: Optional[CommandLane] = None,
        stage: Optional[LifecycleStage] = None,
    ) -> None:
        """Mirror BAA subsystem activity into the main agent's own awareness."""
        runtime = self.runtime or getattr(self.executor, "runtime", None)
        if runtime is None:
            return
        task_id = str(task.task_id)
        resolved_lane = lane or _resolve_lane(task)
        task_stage = stage.value if stage is not None else task.stage.value
        signature = self._baa_awareness_signature(
            task=task,
            status=status,
            reason=reason,
            lane=resolved_lane,
            task_stage=task_stage,
        )
        if status in self._AWARENESS_DEDUP_STATUSES:
            task.status = status
            task.updated_at = datetime.now(timezone.utc)
            if str(task.meta.get("last_baa_awareness_signature") or "") == signature:
                if self.store:
                    await self.store.save(task)
                self._trace(
                    "baa_awareness_duplicate_suppressed",
                    {"task_id": task_id, "status": status, "lane": resolved_lane.value},
                )
                return
            task.meta["last_baa_awareness_signature"] = signature
            task.meta["last_baa_awareness_at"] = task.updated_at.isoformat()
        output = str(getattr(result, "output", "") or "").strip()
        artifacts = list(getattr(result, "artifacts", None) or task.artifacts or [])
        content = self._baa_awareness_content(
            task=task,
            status=status,
            reason=reason,
            output=output,
            artifacts=artifacts,
            receipt=receipt,
            lane=resolved_lane,
            task_stage=task_stage,
        )
        payload = {
            "task_id": task_id,
            "objective": task.objective,
            "status": status,
            "stage": task_stage,
            "lane": resolved_lane.value,
            "attempt": task.attempt,
            "artifacts": artifacts[:20],
            "receipt_id": str(getattr(receipt, "receipt_id", "") or ""),
            "project_id": str(task.project_id or ""),
            "commitment_id": str(task.commitment_id or task.meta.get("commitment_id") or ""),
            "output_preview": output[:500],
            "meta_source": str(task.meta.get("source") or ""),
            "project_title": str(task.meta.get("project_title") or ""),
            "workspace": str(task.meta.get("workspace_rel_path") or task.meta.get("workspace_abs_path") or ""),
        }
        source_session_id = str(task.meta.get("source_session_id") or "").strip()
        awareness_session_id = (
            source_session_id
            if source_session_id and source_session_id != "system:automated"
            else None
        )
        urgency = "high" if status in {"failed", "needs_attention"} else "normal"
        salience = 7.5 if status in {"failed", "completed", "needs_attention"} else 4.5
        try:
            await record_agent_visible_system_message(
                runtime,
                content=content,
                event_kind="baa_task_activity",
                status=status,
                reason=reason or status,
                source="baa",
                source_id=task_id,
                channel="internal",
                urgency=urgency,
                session_id=awareness_session_id,
                salience=salience,
                payload=payload,
            )
        except Exception as exc:
            self._trace(
                "baa_awareness_record_failed",
                {"task_id": task_id, "status": status, "error": str(exc)},
            )
        if status in self._AWARENESS_DEDUP_STATUSES and self.store:
            await self.store.save(task)

    @staticmethod
    def _baa_awareness_signature(
        *,
        task: RepairTask,
        status: str,
        reason: str,
        lane: CommandLane,
        task_stage: str,
    ) -> str:
        return json.dumps(
            {
                "task_id": str(task.task_id),
                "status": status,
                "reason": reason,
                "lane": lane.value,
                "stage": task_stage,
                "attempt": task.attempt,
            },
            sort_keys=True,
        )

    @staticmethod
    def _baa_awareness_content(
        *,
        task: RepairTask,
        status: str,
        reason: str,
        output: str,
        artifacts: List[str],
        receipt: Optional[Any],
        lane: CommandLane,
        task_stage: str,
    ) -> str:
        meta = task.meta if isinstance(task.meta, dict) else {}
        project_title = str(meta.get("project_title") or "").strip()
        project_type = str(meta.get("project_type") or "").strip()
        workspace = str(meta.get("workspace_rel_path") or meta.get("workspace_abs_path") or "").strip()
        if project_title or workspace:
            if status == "completed":
                work_state = "completed a background work task on"
            elif status == "failed":
                work_state = "hit a failed background work task for"
            elif status == "needs_attention":
                work_state = "marked a background work task as needing attention for"
            else:
                work_state = f"recorded {status} background work state for"
            project_label = project_title or workspace
            project_kind = f"{project_type} project" if project_type else "project"
            project_memory_line = (
                f"Recent project work memory: I {work_state} the {project_kind} "
                f"{project_label}."
            )
            if workspace:
                project_memory_line += f" Workspace: {workspace}."
        else:
            project_memory_line = ""
        lines = [
            "BAA subsystem activity recorded.",
            f"Progress update: {BoundedAssistantAgent._baa_operator_progress_update(status=status, reason=reason, artifacts=artifacts)}",
            f"Task ID: {task.task_id}",
            f"Task: {task.objective}",
            f"Status: {status}",
            f"Stage: {task_stage}",
            f"Lane: {lane.value}",
        ]
        if project_memory_line:
            lines.append(project_memory_line)
        if project_type == "writing":
            lines.append(
                "Writing project recall cues: book, novel, manuscript, draft, chapter, revision, research."
            )
        if reason:
            lines.append(f"Reason: {reason}")
        receipt_id = str(getattr(receipt, "receipt_id", "") or "")
        if receipt_id:
            lines.append(f"Receipt: {receipt_id}")
        if artifacts:
            lines.append("Artifacts: " + ", ".join(str(item) for item in artifacts[:8]))
        if output:
            lines.extend(["Output:", output[:1200]])
        return "\n".join(lines)

    @staticmethod
    def _baa_operator_progress_update(*, status: str, reason: str, artifacts: List[str]) -> str:
        normalized = str(status or "").strip().lower()
        reason_text = str(reason or "").strip()
        artifact_count = len(artifacts or [])
        artifact_note = (
            f" I have {artifact_count} artifact{'s' if artifact_count != 1 else ''} recorded."
            if artifact_count
            else ""
        )
        if normalized == "submitted":
            return "I queued this background task and am moving it into execution without waiting for another prompt."
        if normalized == "started":
            return "I started this background task and am continuing the work."
        if normalized == "completed":
            return f"I completed this background task.{artifact_note}"
        if normalized == "failed":
            detail = f" Reason: {reason_text}" if reason_text else ""
            return f"I hit a real blocker on this background task and recorded it instead of pretending it completed.{detail}"
        if normalized == "needs_attention":
            detail = f" Reason: {reason_text}" if reason_text else ""
            return f"I need operator attention for this task and recorded the blocker.{detail}"
        if normalized == "held":
            detail = f" Reason: {reason_text}" if reason_text else ""
            return f"I held this task because it cannot proceed yet; this is a recorded pause, not silent waiting.{detail}"
        if normalized in {"queued", "recovering"}:
            detail = f" Reason: {reason_text}" if reason_text else ""
            return f"I saved progress and queued the next bounded attempt so the work can continue.{detail}"
        detail = f" Reason: {reason_text}" if reason_text else ""
        return f"I recorded background task state `{normalized or 'unknown'}` and kept the work state durable.{detail}"

    async def _link_promise_receipt(self, receipt: Any) -> None:
        if self.runtime is None:
            return
        try:
            await link_receipt_to_promise_claim(self.runtime, receipt)
        except Exception as exc:
            self._trace(
                "promise_receipt_link_failed",
                {
                    "receipt_id": str(getattr(receipt, "receipt_id", "")),
                    "task_id": str(getattr(receipt, "task_id", "")),
                    "error": str(exc),
                },
            )

    async def _ensure_success_followthrough_continuation(
        self,
        task: RepairTask,
        result: RepairResult,
    ) -> bool:
        commitment = await self._active_followthrough_commitment(task)
        if commitment is None:
            return False
        commitment.meta.update(
            {
                "resume_policy": "auto_until_completion_evidence",
                "last_successful_continuation_task_id": str(task.task_id),
                "last_successful_continuation_at": datetime.now(timezone.utc).isoformat(),
                "last_successful_continuation_output": str(result.output or "")[:500],
            }
        )
        await self._save_commitment(commitment)
        return await self._create_followthrough_schedule(
            task=task,
            commitment=commitment,
            result=result,
            source="project_followthrough_success_continuation",
            delay_seconds=60,
        )

    async def _ensure_recoverable_followthrough_continuation(
        self,
        task: RepairTask,
        result: RepairResult,
        *,
        reason: str,
    ) -> bool:
        commitment = await self._active_followthrough_commitment(task)
        if commitment is None:
            return False
        if not self._is_recoverable_followthrough_failure(reason, task=task, commitment=commitment):
            return False
        now = datetime.now(timezone.utc).isoformat()
        commitment.meta.update(
            {
                "resume_policy": "auto_on_recoverable_task_blocker",
                "last_recoverable_task_failure": str(reason or "task failed")[:500],
                "last_recoverable_task_id": str(task.task_id),
                "last_recoverable_task_at": now,
                "last_recoverable_task_artifacts": list(result.artifacts or task.artifacts or [])[:20],
            }
        )
        commitment.meta.pop("blocked_reason", None)
        commitment.meta.pop("blocked_task_id", None)
        commitment.meta.pop("blocked_at", None)
        commitment.updated_at = datetime.now(timezone.utc)
        await self._save_commitment(commitment)
        return await self._create_followthrough_schedule(
            task=task,
            commitment=commitment,
            result=result,
            source="project_followthrough_recovery",
            delay_seconds=self._followthrough_recovery_delay_seconds(reason),
            reason=reason,
        )

    async def _ensure_schedule_failure_recovery(
        self,
        task: RepairTask,
        result: RepairResult,
        *,
        reason: str,
    ) -> bool:
        if not self._is_schedule_origin_task(task):
            return False
        if self._schedule_failure_requires_operator_attention(reason):
            task.meta["schedule_recovery_blocked_reason"] = str(reason or "scheduled task failed")[:500]
            task.meta["schedule_recovery_blocked_class"] = "operator_attention_required"
            if self.store:
                await self.store.save(task)
            self._trace(
                "schedule_failure_recovery_blocked",
                {
                    "task_id": str(task.task_id),
                    "reason": str(reason)[:500],
                    "blocker_class": "operator_attention_required",
                },
            )
            return False
        depth = self._schedule_recovery_depth(task)
        if depth >= _SCHEDULE_FAILURE_RECOVERY_CAP:
            task.meta["schedule_recovery_cap_reached"] = True
            task.meta["schedule_recovery_cap_reason"] = str(reason or "scheduled task failed")[:500]
            if self.store:
                await self.store.save(task)
            self._trace(
                "schedule_failure_recovery_cap_reached",
                {"task_id": str(task.task_id), "depth": depth, "reason": str(reason)[:500]},
            )
            return False

        schedule_service = getattr(self.runtime, "schedule_service", None) if self.runtime is not None else None
        create_schedule = getattr(schedule_service, "create_schedule", None)
        if not callable(create_schedule):
            self._trace(
                "schedule_failure_recovery_unavailable",
                {"task_id": str(task.task_id), "reason": "schedule service unavailable"},
            )
            return False
        if await self._active_schedule_failure_recovery_exists(schedule_service, str(task.task_id)):
            return True

        meta = task.meta if isinstance(task.meta, dict) else {}
        schedule_id = str(meta.get("schedule_id") or meta.get("recovery_from_schedule_id") or "").strip()
        schedule_title = str(
            meta.get("schedule_title")
            or meta.get("original_schedule_title")
            or meta.get("schedule_name")
            or ""
        ).strip()
        display_title = schedule_title or self._short_commitment_title(task.objective, max_chars=80)
        now = datetime.now(timezone.utc)
        recovery_meta = self._schedule_failure_recovery_meta(
            task=task,
            result=result,
            reason=reason,
            schedule_id=schedule_id,
            depth=depth,
        )
        try:
            schedule = await create_schedule(
                kind=ScheduleKind.TASK,
                action=ScheduleAction.SUBMIT_BAA,
                title=f"Recover scheduled task: {display_title}",
                description="Automatic recovery for a failed scheduled task.",
                objective=self._build_schedule_failure_recovery_objective(
                    task=task,
                    result=result,
                    reason=reason,
                    schedule_title=display_title,
                ),
                start_at=now + timedelta(seconds=self._schedule_failure_recovery_delay_seconds(reason)),
                recurrence=ScheduleRecurrence.NONE,
                priority=8.0,
                tags=["schedule_recovery", "auto_recovery", "self_directed"],
                commitment_id=task.commitment_id or str(meta.get("commitment_id") or "").strip() or None,
                meta=recovery_meta,
            )
        except Exception as exc:
            self._trace(
                "schedule_failure_recovery_error",
                {"task_id": str(task.task_id), "schedule_id": schedule_id, "error": str(exc)},
            )
            return False

        task.meta["last_schedule_recovery_id"] = str(getattr(schedule, "schedule_id", "") or "")
        task.meta["last_schedule_recovery_at"] = now.isoformat()
        task.meta["last_schedule_recovery_depth"] = depth + 1
        if self.store:
            await self.store.save(task)
        self._trace(
            "schedule_failure_recovery_created",
            {
                "task_id": str(task.task_id),
                "schedule_id": schedule_id,
                "recovery_schedule_id": str(getattr(schedule, "schedule_id", "") or ""),
                "depth": depth + 1,
            },
        )
        return True

    @staticmethod
    def _is_schedule_origin_task(task: RepairTask) -> bool:
        meta = task.meta if isinstance(task.meta, dict) else {}
        source = str(meta.get("source") or "").strip().lower()
        if source == "schedule" or source in _SCHEDULE_FAILURE_RECOVERY_SOURCES:
            return True
        authority = str(meta.get("authority") or "").strip().lower()
        return bool(meta.get("schedule_id") and authority == "schedule_due")

    @staticmethod
    def _schedule_recovery_depth(task: RepairTask) -> int:
        meta = task.meta if isinstance(task.meta, dict) else {}
        try:
            return max(0, int(meta.get("schedule_recovery_depth") or 0))
        except (TypeError, ValueError):
            return 0

    async def _active_schedule_failure_recovery_exists(self, schedule_service: Any, task_id: str) -> bool:
        store = getattr(schedule_service, "store", None)
        list_items = getattr(store, "list_items", None) or getattr(schedule_service, "list_items", None)
        if not callable(list_items):
            return False
        try:
            items = list_items(status=ScheduleStatus.ACTIVE, limit=1000)
            if inspect.isawaitable(items):
                items = await items
        except TypeError:
            try:
                items = list_items()
                if inspect.isawaitable(items):
                    items = await items
            except Exception:
                return False
        except Exception:
            return False
        for item in items or []:
            meta = getattr(item, "meta", {}) or {}
            if not isinstance(meta, dict):
                continue
            source = str(meta.get("source") or "").strip().lower()
            if source not in _SCHEDULE_FAILURE_RECOVERY_SOURCES:
                continue
            if str(meta.get("recovery_from_task_id") or "").strip() == task_id:
                return True
        return False

    @staticmethod
    def _schedule_failure_recovery_delay_seconds(reason: str) -> int:
        lowered = str(reason or "").lower()
        if any(
            marker in lowered
            for marker in ("429", "rate limit", "provider backoff", "circuit open", "server error", "500", "upstream")
        ):
            return 15 * 60
        if any(marker in lowered for marker in ("timeout", "dns_resolution_failed", "temporarily unavailable")):
            return 5 * 60
        return 60

    @staticmethod
    def _schedule_recovery_strategy(reason: str) -> str:
        lowered = str(reason or "").lower()
        if any(marker in lowered for marker in ("429", "rate limit", "provider backoff", "circuit open")):
            return "provider_backoff_then_retry_alternative_route"
        if any(marker in lowered for marker in ("timeout", "dns_resolution_failed", "temporarily unavailable")):
            return "retry_after_transient_failure_with_narrower_scope"
        if any(marker in lowered for marker in _EXTERNAL_BLOCKER_MARKERS):
            return "diagnose_external_blocker_and_record_next_check"
        if any(marker in lowered for marker in _RETRY_EXHAUSTION_MARKERS):
            return "reframe_after_retry_exhaustion"
        return "diagnose_and_work_around"

    @staticmethod
    def _schedule_failure_requires_operator_attention(reason: str) -> bool:
        lowered = str(reason or "").lower()
        return any(marker in lowered for marker in _EXTERNAL_BLOCKER_MARKERS)

    def _schedule_failure_recovery_meta(
        self,
        *,
        task: RepairTask,
        result: RepairResult,
        reason: str,
        schedule_id: str,
        depth: int,
    ) -> Dict[str, Any]:
        meta = task.meta if isinstance(task.meta, dict) else {}
        recovery_meta: Dict[str, Any] = {
            "source": "schedule_failure_recovery",
            "authority": "schedule_due",
            "recovery_strategy": self._schedule_recovery_strategy(reason),
            "recovery_from_task_id": str(task.task_id),
            "recovery_from_schedule_id": schedule_id,
            "schedule_recovery_depth": depth + 1,
            "previous_task_stage": result.stage.value,
            "previous_task_success": bool(result.success),
            "previous_task_output": str(result.output or reason or "scheduled task failed")[:1000],
            "previous_task_artifacts": list(result.artifacts or task.artifacts or [])[:20],
            "original_objective": str(meta.get("original_objective") or task.objective)[:1000],
            "original_schedule_title": str(
                meta.get("original_schedule_title") or meta.get("schedule_title") or ""
            )[:300],
            "previous_scheduled_for": str(meta.get("scheduled_for") or ""),
        }
        for key in (
            "accepted_proposal_ids",
            "commitment_id",
            "context_truth_snapshot_id",
            "project_id",
            "project_key",
            "project_title",
            "project_type",
            "source_session_id",
            "source_snapshot_id",
            "workspace_abs_path",
            "workspace_rel_path",
            "workspace_project_confidence",
        ):
            value = meta.get(key)
            if value not in (None, "", []):
                recovery_meta[key] = value
        return recovery_meta

    @staticmethod
    def _build_schedule_failure_recovery_objective(
        *,
        task: RepairTask,
        result: RepairResult,
        reason: str,
        schedule_title: str,
    ) -> str:
        output = str(result.output or reason or "scheduled task failed").strip()
        artifacts = list(result.artifacts or task.artifacts or [])
        lines = [
            f"Recover the failed scheduled task: {schedule_title}.",
            f"Original objective: {task.objective}",
            "Diagnose why the scheduled task failed before trying new work.",
            "Inspect prior task metadata, artifacts, logs, and any linked schedule or commitment evidence.",
            "Try a materially different, narrower path or a safe workaround; do not repeat the same failing attempt.",
        ]
        if output:
            lines.append(f"Previous failure: {output[:700]}")
        if artifacts:
            lines.append("Previous artifacts: " + ", ".join(str(item) for item in artifacts[:8]))
        lines.extend(
            [
                "If the blocker is transient, retry with a smaller scope or alternate tool/provider route.",
                "If the blocker is credentials, missing input, safety, or an external dependency, record the exact blocker and the next autonomous check instead of silently waiting.",
                "Finish with explicit evidence of what recovered, what still failed, and what the next scheduled action should be.",
            ]
        )
        return " ".join(line for line in lines if line)

    async def reconcile_legacy_followthrough_commitments(self, *, limit: int = 1000) -> int:
        """Repair legacy blocked project-return commitments after restart.

        Older live processes could block authorized project-return commitments on
        retry exhaustion before the followthrough recovery schedule existed. This
        pass is intentionally narrow and only creates a bounded recovery schedule
        when the persisted blocker is recoverable without operator attention.
        """

        store = self._commitment_store()
        list_by_status = getattr(store, "list_by_status", None)
        if not callable(list_by_status):
            return 0
        try:
            commitments = list_by_status(CommitmentStatus.BLOCKED, limit=limit)
            if inspect.isawaitable(commitments):
                commitments = await commitments
        except Exception as exc:
            self._trace("legacy_followthrough_reconciliation_unavailable", {"error": str(exc)})
            return 0

        repaired = 0
        for commitment in commitments or []:
            if await self._repair_legacy_followthrough_commitment(commitment):
                repaired += 1
        if repaired:
            self._trace("legacy_followthrough_reconciliation_completed", {"repaired": repaired})
        return repaired

    async def _repair_legacy_followthrough_commitment(self, commitment: Any) -> bool:
        if getattr(commitment, "status", None) != CommitmentStatus.BLOCKED:
            return False
        meta = dict(getattr(commitment, "meta", {}) or {})
        tags = getattr(commitment, "tags", []) or ()
        if not has_operator_project_followthrough_authority(
            meta,
            tags=tags,
            allow_schedule_authority=True,
        ):
            return False

        reason, blocked_task, blocked_result = await self._legacy_followthrough_blocker(commitment)
        if not self._is_recoverable_followthrough_failure(reason, task=blocked_task, commitment=commitment):
            return False

        commitment_id = str(getattr(commitment, "commitment_id", "") or "").strip()
        schedule_service = getattr(self.runtime, "schedule_service", None) if self.runtime is not None else None
        project_key = str(meta.get("project_key") or "").strip()
        if schedule_service is not None and await self._active_followthrough_schedule_exists(
            schedule_service,
            commitment_id,
            project_key=project_key,
        ):
            return False

        task = blocked_task or RepairTask(
            objective=str(getattr(commitment, "content", "Continue the linked project commitment.") or "").strip()
            or "Continue the linked project commitment.",
            commitment_id=commitment_id,
            meta={
                "source": "schedule",
                "authority": "schedule_due",
                "commitment_id": commitment_id,
                "project_key": str(meta.get("project_key") or ""),
                "project_title": str(meta.get("project_title") or ""),
                "project_type": str(meta.get("project_type") or ""),
                **copy_safe_operator_project_followthrough_evidence(meta),
            },
        )
        task.commitment_id = task.commitment_id or commitment_id
        task.meta.setdefault("source", "schedule")
        task.meta.setdefault("authority", "schedule_due")
        task.meta.setdefault("commitment_id", commitment_id)

        result = blocked_result or RepairResult(
            task_id=task.task_id,
            success=False,
            stage=ExecutionStage.FAILED,
            output=reason,
            artifacts=list(getattr(task, "artifacts", []) or meta.get("last_recoverable_task_artifacts") or []),
        )
        now = datetime.now(timezone.utc).isoformat()
        previous_blocked_reason = str(meta.get("blocked_reason") or reason or "").strip()
        commitment.meta.update(
            {
                "resume_policy": "auto_on_recoverable_task_blocker",
                "legacy_followthrough_repaired_at": now,
                "legacy_followthrough_repair_source": "project_followthrough_legacy_recovery",
                "last_recoverable_task_failure": str(reason or "legacy blocked task failed")[:500],
                "last_recoverable_task_id": str(task.task_id),
                "last_recoverable_task_at": now,
                "last_recoverable_task_artifacts": list(result.artifacts or task.artifacts or [])[:20],
            }
        )
        if previous_blocked_reason:
            commitment.meta["previous_blocked_reason"] = previous_blocked_reason[:500]
        commitment.meta.pop("blocked_reason", None)
        commitment.meta.pop("blocked_task_id", None)
        commitment.meta.pop("blocked_at", None)
        commitment.updated_at = datetime.now(timezone.utc)

        created = await self._create_followthrough_schedule(
            task=task,
            commitment=commitment,
            result=result,
            source="project_followthrough_legacy_recovery",
            delay_seconds=self._followthrough_recovery_delay_seconds(reason),
            reason=reason,
        )
        if not created:
            return False
        commitment.status = CommitmentStatus.ACTIVE
        commitment.updated_at = datetime.now(timezone.utc)
        await self._save_commitment(commitment)
        return True

    async def _legacy_followthrough_blocker(self, commitment: Any) -> tuple[str, RepairTask | None, RepairResult | None]:
        meta = dict(getattr(commitment, "meta", {}) or {})
        reason = str(meta.get("blocked_reason") or meta.get("last_recoverable_task_failure") or "").strip()
        task = None
        result = None
        task_id = str(meta.get("blocked_task_id") or meta.get("last_recoverable_task_id") or "").strip()
        linked_task_ids = list(getattr(commitment, "linked_task_ids", []) or [])
        if not task_id and linked_task_ids:
            task_id = str(linked_task_ids[-1] or "").strip()
        if task_id and self.store is not None:
            try:
                task = await self.store.get(task_id)
            except Exception:
                task = None
            try:
                result = await self.store.get_result(task_id)
            except Exception:
                result = None
        if result is not None and str(result.output or "").strip():
            reason = f"{reason} {result.output}".strip()
        if not reason and task is not None:
            reason = str(task.meta.get("blocked_reason") or task.status or task.stage.value or "").strip()
        return reason, task, result

    async def _active_followthrough_commitment(self, task: RepairTask) -> Any | None:
        commitment_id = str(task.commitment_id or task.meta.get("commitment_id") or "").strip()
        if not commitment_id:
            return None
        store = self._commitment_store()
        if store is None:
            return None
        try:
            commitment = await store.get(commitment_id)
        except Exception:
            return None
        if commitment is None or commitment.status != CommitmentStatus.ACTIVE:
            return None
        if not self._is_followthrough_commitment(commitment, task):
            return None
        return commitment

    def _commitment_store(self) -> Any | None:
        if self.runtime is None:
            return None
        store = getattr(self.runtime, "commitment_store", None)
        if store is None:
            store = getattr(getattr(self.runtime, "ctx", None), "commitment_store", None)
        return store

    async def _save_commitment(self, commitment: Any) -> None:
        store = self._commitment_store()
        save = getattr(store, "save", None)
        if callable(save):
            await save(commitment)

    @staticmethod
    def _is_followthrough_commitment(commitment: Any, task: RepairTask) -> bool:
        meta = dict(getattr(commitment, "meta", {}) or {})
        tags = {
            str(tag).strip().lower()
            for tag in (getattr(commitment, "tags", []) or [])
            if str(tag).strip()
        }
        task_meta = task.meta if isinstance(task.meta, dict) else {}
        source = str(meta.get("source") or task_meta.get("source") or "").strip()
        if "project_return" in tags or source == "project_return_capture":
            return True
        if "self_directed" in tags and (meta.get("project_key") or task_meta.get("project_key")):
            return True
        return bool(task_meta.get("project_key") or task_meta.get("project_title"))

    @staticmethod
    def _is_recoverable_followthrough_failure(
        reason: str,
        *,
        task: RepairTask | None = None,
        commitment: Any | None = None,
    ) -> bool:
        lowered = str(reason or "").lower()
        if any(marker in lowered for marker in _EXTERNAL_BLOCKER_MARKERS):
            return False
        if any(marker in lowered for marker in _RECOVERABLE_FOLLOWTHROUGH_MARKERS):
            return True
        if any(marker in lowered for marker in _RECOVERABLE_LEGACY_FAILURE_MARKERS):
            return True
        if not any(marker in lowered for marker in _RETRY_EXHAUSTION_MARKERS):
            return False

        task_meta = task.meta if task is not None and isinstance(task.meta, dict) else {}
        commitment_meta = dict(getattr(commitment, "meta", {}) or {}) if commitment is not None else {}
        commitment_tags = getattr(commitment, "tags", []) or () if commitment is not None else ()
        evidence = {
            **copy_safe_operator_project_followthrough_evidence(commitment_meta, task_meta),
            "source": str(commitment_meta.get("source") or task_meta.get("source") or ""),
            "authority": str(task_meta.get("authority") or commitment_meta.get("authority") or ""),
        }
        return has_operator_project_followthrough_authority(
            evidence,
            tags=commitment_tags,
            allow_schedule_authority=True,
        )

    @staticmethod
    def _followthrough_recovery_delay_seconds(reason: str) -> int:
        lowered = str(reason or "").lower()
        if any(
            marker in lowered
            for marker in ("429", "rate limit", "provider backoff", "circuit open", "server error", "500", "upstream")
        ):
            return 15 * 60
        if any(marker in lowered for marker in ("timeout", "dns_resolution_failed", "temporarily unavailable")):
            return 5 * 60
        return 60

    async def _create_followthrough_schedule(
        self,
        *,
        task: RepairTask,
        commitment: Any,
        result: RepairResult,
        source: str,
        delay_seconds: int,
        reason: str = "",
    ) -> bool:
        schedule_service = getattr(self.runtime, "schedule_service", None) if self.runtime is not None else None
        create_schedule = getattr(schedule_service, "create_schedule", None)
        if not callable(create_schedule):
            self._trace(
                "project_followthrough_schedule_unavailable",
                {
                    "task_id": str(task.task_id),
                    "commitment_id": str(getattr(commitment, "commitment_id", "")),
                    "source": source,
                },
            )
            return False
        commitment_id = str(getattr(commitment, "commitment_id", "") or "").strip()
        meta = dict(getattr(commitment, "meta", {}) or {})
        task_meta = task.meta if isinstance(task.meta, dict) else {}
        project_key = str(meta.get("project_key") or task_meta.get("project_key") or "").strip()
        if await self._active_followthrough_schedule_exists(
            schedule_service,
            commitment_id,
            project_key=project_key,
        ):
            return True

        project_title = str(meta.get("project_title") or task_meta.get("project_title") or "").strip()
        display_title = project_title or self._short_commitment_title(str(getattr(commitment, "content", "project")))
        now = datetime.now(timezone.utc)
        evidence = copy_safe_operator_project_followthrough_evidence(meta, task_meta)
        schedule_meta = {
            "source": source,
            "commitment_id": commitment_id,
            "project_key": str(meta.get("project_key") or task_meta.get("project_key") or "").strip(),
            "project_title": project_title,
            "project_type": str(meta.get("project_type") or task_meta.get("project_type") or "").strip(),
            "previous_task_id": str(task.task_id),
            "previous_task_stage": result.stage.value,
            "previous_task_success": bool(result.success),
            "previous_task_output": str(result.output or "")[:500],
            "previous_task_artifacts": list(result.artifacts or task.artifacts or [])[:20],
            **evidence,
        }
        tags = ["project_followthrough", "self_directed", "auto_continue"]
        if self._schedule_should_keep_project_return_tag(schedule_meta, commitment=commitment):
            tags.append("project_return")
        if reason:
            schedule_meta["recovery_from_task_id"] = str(task.task_id)
            schedule_meta["recoverable_blocker"] = str(reason)[:500]
        try:
            schedule = await create_schedule(
                kind=ScheduleKind.TASK,
                action=ScheduleAction.SUBMIT_BAA,
                title=f"Continue {display_title}",
                description="Automatic continuation for unfinished linked project work.",
                objective=self._build_followthrough_objective(
                    task=task,
                    commitment=commitment,
                    result=result,
                    source=source,
                    reason=reason,
                ),
                start_at=now + timedelta(seconds=max(0, int(delay_seconds))),
                recurrence=ScheduleRecurrence.NONE,
                priority=8.5,
                tags=tags,
                commitment_id=commitment_id,
                meta=schedule_meta,
            )
        except Exception as exc:
            self._trace(
                "project_followthrough_schedule_error",
                {"task_id": str(task.task_id), "commitment_id": commitment_id, "error": str(exc)},
            )
            return False

        commitment.meta["last_followthrough_schedule_id"] = str(getattr(schedule, "schedule_id", "") or "")
        commitment.meta["last_followthrough_schedule_source"] = source
        commitment.meta["last_followthrough_scheduled_at"] = now.isoformat()
        await self._save_commitment(commitment)
        self._trace(
            "project_followthrough_schedule_created",
            {
                "task_id": str(task.task_id),
                "commitment_id": commitment_id,
                "schedule_id": str(getattr(schedule, "schedule_id", "") or ""),
                "source": source,
            },
        )
        return True

    @staticmethod
    def _schedule_should_keep_project_return_tag(schedule_meta: Dict[str, Any], *, commitment: Any) -> bool:
        tags = getattr(commitment, "tags", []) or ()
        meta = dict(getattr(commitment, "meta", {}) or {})
        authority_meta = {
            **schedule_meta,
            "source": str(meta.get("source") or schedule_meta.get("source") or ""),
        }
        return has_operator_project_followthrough_authority(authority_meta, tags=tags)

    async def _active_followthrough_schedule_exists(
        self,
        schedule_service: Any,
        commitment_id: str,
        *,
        project_key: str = "",
    ) -> bool:
        store = getattr(schedule_service, "store", None)
        list_items = getattr(store, "list_items", None) or getattr(schedule_service, "list_items", None)
        if not callable(list_items):
            return False
        try:
            items = list_items(status=ScheduleStatus.ACTIVE, limit=1000)
            if inspect.isawaitable(items):
                items = await items
        except TypeError:
            try:
                items = list_items()
                if inspect.isawaitable(items):
                    items = await items
            except Exception:
                return False
        except Exception:
            return False
        for item in items or []:
            item_commitment_id = str(getattr(item, "commitment_id", "") or "").strip()
            meta = getattr(item, "meta", {}) or {}
            item_project_key = str(meta.get("project_key") or "").strip() if isinstance(meta, dict) else ""
            same_commitment = bool(commitment_id and item_commitment_id == commitment_id)
            same_project = bool(project_key and item_project_key == project_key)
            if not (same_commitment or same_project):
                continue
            action = getattr(getattr(item, "action", None), "value", getattr(item, "action", ""))
            if str(action) != ScheduleAction.SUBMIT_BAA.value:
                continue
            tags = {str(tag).strip().lower() for tag in (getattr(item, "tags", []) or []) if str(tag).strip()}
            if "project_followthrough" in tags or "project_return" in tags:
                return True
            if isinstance(meta, dict) and str(meta.get("source") or "") in _FOLLOWTHROUGH_SCHEDULE_SOURCES:
                return True
        return False

    @staticmethod
    def _build_followthrough_objective(
        *,
        task: RepairTask,
        commitment: Any,
        result: RepairResult,
        source: str,
        reason: str,
    ) -> str:
        commitment_content = str(getattr(commitment, "content", "") or "the linked commitment").strip()
        meta = dict(getattr(commitment, "meta", {}) or {})
        project_intent = str(meta.get("project_intent") or "").strip()
        next_step = str(meta.get("next_step") or "").strip()
        lines = [
            f"Continue the linked project commitment: {commitment_content}.",
            "Use the latest persisted artifacts, task evidence, and workspace state before doing new work.",
        ]
        if project_intent:
            lines.append(f"Project intent: {project_intent}")
        if next_step:
            lines.append(f"Known next step: {next_step}")
        if source in {"project_followthrough_recovery", "project_followthrough_legacy_recovery"}:
            lines.extend(
                [
                    f"Previous attempt hit a recoverable blocker: {reason[:500]}",
                    "Do a narrower artifact-producing step instead of repeating the same broad attempt.",
                    "Preserve partial value from the previous attempt and verify the next concrete result.",
                ]
            )
        else:
            lines.append("The previous task made progress, but the linked commitment is still active.")
        output = str(result.output or "").strip()
        if output:
            lines.append(f"Previous task output summary: {output[:700]}")
        artifacts = list(result.artifacts or task.artifacts or [])
        if artifacts:
            lines.append("Previous artifacts: " + ", ".join(str(item) for item in artifacts[:8]))
        lines.extend(
            [
                "Only stop when explicit completion evidence satisfies the linked commitment.",
                "If the commitment is still unfinished at the end of this run, leave it active and create the next continuation schedule.",
                "If an external, safety, credential, or missing-input blocker prevents progress, record that blocker explicitly with reconsideration metadata instead of silently waiting for the owner to say resume.",
            ]
        )
        return " ".join(line for line in lines if line)

    @staticmethod
    def _short_commitment_title(value: str, *, max_chars: int = 90) -> str:
        text = " ".join(str(value or "").split()).strip()
        if text.lower().startswith("return to project:"):
            text = text.split(":", 1)[1].strip()
        if len(text) <= max_chars:
            return text or "project"
        return text[: max_chars - 3].rstrip() + "..."

    async def _block_linked_commitment_on_failure(self, task: RepairTask, reason: str) -> None:
        commitment_id = str(task.commitment_id or task.meta.get("commitment_id") or "").strip()
        if not commitment_id or self.runtime is None:
            return
        store = self._commitment_store()
        if store is None:
            return
        try:
            commitment = await store.get(commitment_id)
        except Exception:
            return
        if commitment is None or commitment.status != CommitmentStatus.ACTIVE:
            return
        now = datetime.now(timezone.utc).isoformat()
        commitment.status = CommitmentStatus.BLOCKED
        commitment.updated_at = datetime.now(timezone.utc)
        commitment.meta.update(
            {
                "blocked_reason": str(reason or "task failed")[:500],
                "blocked_task_id": str(task.task_id),
                "blocked_at": now,
                "resume_policy": "owner_attention_or_new_evidence",
            }
        )
        try:
            await store.save(commitment)
            self._trace(
                "commitment_blocked_after_task_failure",
                {"commitment_id": commitment_id, "task_id": str(task.task_id), "reason": reason},
            )
        except Exception as exc:
            self._trace(
                "commitment_block_after_task_failure_error",
                {"commitment_id": commitment_id, "task_id": str(task.task_id), "error": str(exc)},
            )

    async def _notify_owner_task_attention(
        self,
        task: RepairTask,
        *,
        status: str,
        reason: str,
        urgency: str,
    ) -> None:
        if self.runtime is None or not self._should_notify_owner_for_task(task, status=status):
            return
        meta = task.meta if isinstance(task.meta, dict) else {}
        contact_key = f"{status}:{task.stage.value}:{str(reason or '')[:120]}"
        sent_keys = list(meta.get("owner_contact_keys") or [])
        if contact_key in sent_keys:
            return
        project_title = str(meta.get("project_title") or task.project_id or "").strip()
        workspace_path = str(meta.get("workspace_rel_path") or meta.get("workspace_abs_path") or "").strip()
        lines = ["OpenCAS needs your attention on active work."]
        if project_title:
            lines.append(f"Project: {project_title}")
        lines.append(f"Task: {task.objective}")
        lines.append(f"Status: {status}")
        if reason:
            lines.append(f"Reason: {reason}")
            provider_hint = self._provider_failure_hint(reason)
            if provider_hint:
                lines.append(provider_hint)
        if workspace_path:
            lines.append(f"Workspace: {workspace_path}")
        message = "\n".join(lines)

        result: Any = None
        try:
            contact_owner = getattr(self.runtime, "initiative_contact_owner", None)
            if callable(contact_owner):
                result = contact_owner(
                    message=message,
                    reason=f"baa_task_{status}",
                    urgency=urgency,
                    channel="telegram",
                )
                if inspect.isawaitable(result):
                    result = await result
            else:
                service = getattr(self.runtime, "initiative_contact", None)
                request_contact = getattr(service, "request_contact", None)
                if callable(request_contact):
                    result = request_contact(
                        message=message,
                        reason=f"baa_task_{status}",
                        urgency=urgency,
                        source="baa",
                        channel="telegram",
                    )
                    if inspect.isawaitable(result):
                        result = await result
                else:
                    return
        except Exception as exc:
            result = {"status": "failed", "error": str(exc), "channel": "telegram"}

        sent_keys.append(contact_key)
        meta["owner_contact_keys"] = sent_keys[-20:]
        meta.setdefault("owner_contact_results", []).append(
            {
                "status": status,
                "reason": str(reason or "")[:500],
                "channel": "telegram",
                "result": result if isinstance(result, dict) else str(result),
                "created_at": datetime.now(timezone.utc).isoformat(),
            }
        )
        task.meta = meta
        self._trace(
            "task_owner_attention_contact",
            {
                "task_id": str(task.task_id),
                "status": status,
                "channel": "telegram",
                "result": result if isinstance(result, dict) else str(result),
            },
        )

    @staticmethod
    def _should_notify_owner_for_task(task: RepairTask, *, status: str) -> bool:
        if status == "needs_attention":
            return True
        meta = task.meta if isinstance(task.meta, dict) else {}
        if task.commitment_id or task.project_id:
            return True
        if meta.get("project_key") or meta.get("project_title"):
            return True
        source = str(meta.get("source") or "").strip()
        return source in {"project_return_capture", "schedule"}

    @staticmethod
    def _provider_failure_hint(reason: str) -> str:
        text = str(reason or "")
        lower = text.lower()
        if "429" in text or "rate limit" in lower or "ratelimit" in lower:
            return "Provider rate limit: this should back off instead of retrying immediately."
        if "circuit open" in lower or "providercircuitopen" in lower:
            return "Provider circuit open: upstream failures are being throttled before retry."
        if "500" in text or "server error" in lower or "upstream" in lower:
            return "Provider server error: retry later or use a healthy provider route."
        return ""

    async def _hydrate_duplicate_state_from_store(self) -> None:
        """Warm duplicate suppression state from persisted task history."""
        if self.store is None:
            return
        tasks = await self.store.list_all(limit=200)
        for task in tasks:
            task_id = str(task.task_id)
            if task.stage in (ExecutionStage.DONE, ExecutionStage.FAILED):
                result = await self.store.get_result(task_id)
                if result is not None:
                    self._remember_terminal_duplicate(task, result)
                continue
            self._live_tasks[task_id] = task

    @staticmethod
    def _normalize_objective(value: str) -> str:
        return " ".join(str(value or "").split()).strip().lower()

    def _duplicate_key(self, task: RepairTask) -> Tuple[str, str, str]:
        return (
            self._normalize_objective(task.objective),
            str(task.project_id or "").strip(),
            str(task.commitment_id or "").strip(),
        )

    def _remember_terminal_duplicate(
        self,
        task: RepairTask,
        result: RepairResult,
    ) -> None:
        key = self._duplicate_key(task)
        current = self._recent_terminal.get(key)
        if current is None or current[1].timestamp <= result.timestamp:
            self._recent_terminal[key] = (task, result)

    def _recent_terminal_duplicate(
        self,
        task: RepairTask,
    ) -> Optional[Tuple[RepairTask, RepairResult]]:
        key = self._duplicate_key(task)
        entry = self._recent_terminal.get(key)
        if entry is None:
            return None
        existing_task, result = entry
        cooldown = (
            self._DUPLICATE_SUCCESS_COOLDOWN_SECONDS
            if result.success
            else self._DUPLICATE_FAILURE_COOLDOWN_SECONDS
        )
        age_seconds = (datetime.now(timezone.utc) - result.timestamp).total_seconds()
        if age_seconds > cooldown:
            self._recent_terminal.pop(key, None)
            return None
        if not result.success and not self._failed_duplicate_has_reframe_evidence(existing_task):
            return None
        return existing_task, result

    @staticmethod
    def _failed_duplicate_has_reframe_evidence(task: RepairTask) -> bool:
        """Return True when a failed duplicate has real retry-governor evidence."""
        meta = task.meta if isinstance(task.meta, dict) else {}
        retry_governor = meta.get("retry_governor")
        if not isinstance(retry_governor, dict):
            return False
        return retry_governor.get("allowed") is False

    async def _duplicate_future(
        self,
        task: RepairTask,
    ) -> Optional[asyncio.Future[RepairResult]]:
        requested_task_id = str(task.task_id)
        key = self._duplicate_key(task)
        if not key[0]:
            return None

        for existing_id, existing_task in list(self._live_tasks.items()):
            if existing_id == requested_task_id:
                continue
            if self._duplicate_key(existing_task) != key:
                continue
            task.task_id = existing_task.task_id
            future = self._futures.get(existing_id)
            if future is not None:
                self._trace(
                    "task_duplicate_suppressed",
                    {
                        "requested_task_id": requested_task_id,
                        "duplicate_of": existing_id,
                        "objective": task.objective,
                        "reason": "live_duplicate",
                    },
                )
                return future
            alias: asyncio.Future[RepairResult] = asyncio.get_running_loop().create_future()
            self._duplicate_aliases.setdefault(existing_id, []).append(alias)
            self._trace(
                "task_duplicate_suppressed",
                {
                    "requested_task_id": requested_task_id,
                    "duplicate_of": existing_id,
                    "objective": task.objective,
                    "reason": "live_duplicate",
                },
            )
            return alias

        recent_duplicate = self._recent_terminal_duplicate(task)
        if recent_duplicate is None:
            return None
        existing_task, result = recent_duplicate
        task.task_id = existing_task.task_id
        await self._record_duplicate_reframe_state(
            task,
            existing_task=existing_task,
            result=result,
            requested_task_id=requested_task_id,
        )
        future = asyncio.get_running_loop().create_future()
        future.set_result(result)
        self._trace(
            "task_duplicate_suppressed",
            {
                "requested_task_id": requested_task_id,
                "duplicate_of": str(existing_task.task_id),
                "objective": task.objective,
                "reason": "recent_terminal_duplicate",
                "result_stage": result.stage.value,
            },
        )
        return future

    async def _record_duplicate_reframe_state(
        self,
        task: RepairTask,
        *,
        existing_task: RepairTask,
        result: RepairResult,
        requested_task_id: str,
    ) -> None:
        if result.success:
            return

        canonical_artifact = (
            RepairExecutor._canonical_artifact_path(existing_task)
            or RepairExecutor._canonical_artifact_path(task)
        )
        reframe_hint = self._duplicate_reframe_hint(
            task,
            existing_task=existing_task,
            canonical_artifact=canonical_artifact,
        )
        failed_framings = list(
            dict.fromkeys(
                [
                    str(existing_task.objective or "").strip(),
                    str(task.objective or "").strip(),
                ]
            )
        )
        reason = (
            "Duplicate low-divergence objective suppressed after an unsuccessful recent result. "
            "Do not reopen this line without fresh evidence or a materially different framing."
        )
        retry_governor = self._duplicate_retry_governor_payload(
            existing_task,
            result=result,
        )
        resume_project = self._duplicate_resume_project_payload(
            existing_task,
            canonical_artifact=canonical_artifact,
            reframe_hint=reframe_hint,
        )
        shadow_registry = self._shadow_registry()
        capture = getattr(shadow_registry, "capture_retry_blocked", None)
        if callable(capture):
            capture(
                {
                    "task_id": str(existing_task.task_id),
                    "target_id": str(existing_task.task_id),
                    "target_kind": "repair_task",
                    "objective": existing_task.objective,
                    "attempt": existing_task.attempt,
                    "artifact": canonical_artifact,
                    "canonical_artifact_path": canonical_artifact,
                    "retry_mode": "duplicate_suppressed",
                    "governor_mode": "reframe_required",
                    "retry_governor": retry_governor,
                    "resume_project": resume_project,
                    "reason": reason,
                    "best_next_step": reframe_hint,
                    "reframe_hint": reframe_hint,
                    "failed_framings": failed_framings,
                    "suppression_reason": "recent_terminal_duplicate",
                    "capture_source": "baa_duplicate_suppression",
                    "duplicate_context": {
                        "duplicate_of_task_id": str(existing_task.task_id),
                        "requested_task_id": requested_task_id,
                        "result_stage": result.stage.value,
                    },
                }
            )

        details = {
            "blocked_reason": reason,
            "reframe_hint": reframe_hint,
            "failed_framings": failed_framings,
            "duplicate_of_task_id": str(existing_task.task_id),
            "last_result_stage": result.stage.value,
            "last_requested_task_id": requested_task_id,
            "reframe_rule": (
                "Do not retry this line with cosmetic rewording. "
                "Wake it only with fresh evidence or a materially different plan."
            ),
        }
        is_self_referential = is_self_referential_suppression_metadata(
            existing_task.objective,
            reason="low_divergence_reframe",
            source_artifact=canonical_artifact,
            details=details,
        )
        if is_self_referential:
            await self._record_suppressed_reframe_thread_bead(
                existing_task=existing_task,
                requested_task_id=requested_task_id,
                canonical_artifact=canonical_artifact,
                details=details,
            )

        executive = self._executive_state()
        if (
            executive is not None
            and not is_self_referential
            and not self._is_non_live_duplicate_reframe(task, existing_task)
        ):
            executive.park_goal(
                existing_task.objective,
                reason="low_divergence_reframe",
                wake_trigger=(
                    "fresh evidence, relevant artifact change, materially different framing, "
                    "or direct user request"
                ),
                source_artifact=canonical_artifact,
                details=details,
            )
        elif executive is not None and is_self_referential:
            self._trace(
                "task_duplicate_reframe_not_parked",
                {
                    "task_id": str(existing_task.task_id),
                    "reason": "self_referential_suppression_metadata",
                    "artifact": canonical_artifact,
                },
            )
        elif executive is not None:
            self._trace(
                "task_duplicate_reframe_not_parked",
                {
                    "task_id": str(existing_task.task_id),
                    "reason": "non_live_generated_origin",
                    "origins": self._task_origins(task, existing_task),
                },
            )
        self._trace(
            "task_duplicate_reframe_recorded",
            {
                "task_id": str(existing_task.task_id),
                "requested_task_id": requested_task_id,
                "result_stage": result.stage.value,
                "artifact": canonical_artifact,
            },
        )

    async def _record_suppressed_reframe_thread_bead(
        self,
        *,
        existing_task: RepairTask,
        requested_task_id: str,
        canonical_artifact: Optional[str],
        details: Dict[str, Any],
    ) -> None:
        service = self._thread_registry_service()
        if service is None:
            return

        source_ref = f"suppressed_reframe:{existing_task.task_id}:{requested_task_id}"
        content = json.dumps(
            {
                "source_ref": source_ref,
                "objective": existing_task.objective,
                "canonical_artifact": canonical_artifact,
                "details": details,
                "consumer": "audit-only thread-registry query",
                "instruction": (
                    "This bead records rejected recursive suppression metadata. "
                    "Do not render it as an active continuity cue."
                ),
            },
            sort_keys=True,
        )
        try:
            anchor = await service.ensure_thread_anchor(
                title="Suppressed recursive reframes",
                kind="suppressed_reframe",
                status=ThreadStatus.PERIPHERAL,
                anchor_id="suppressed-recursive-reframes",
            )
            bead = await service.create_candidate_bead(
                thread_anchor_id=anchor.anchor_id,
                title="Suppressed recursive reframe",
                summary=(
                    "A duplicate-suppression packet was rejected because the source artifact "
                    "repeated the objective and the failed framing, so it is preserved for audit "
                    "instead of active prompt continuity."
                ),
                source_kind=BeadSourceKind.SUPPRESSED_REFRAME,
                source_ref=source_ref,
                content=content,
                user_commissioned=False,
            )
        except Exception as exc:
            self._trace(
                "suppressed_reframe_thread_bead_failed",
                {"source_ref": source_ref, "error": str(exc)},
            )
            return

        self._trace(
            "suppressed_reframe_thread_bead_recorded",
            {
                "source_ref": source_ref,
                "bead_id": getattr(bead, "bead_id", None),
            },
        )

    @classmethod
    def _is_non_live_duplicate_reframe(cls, *tasks: RepairTask) -> bool:
        return any(origin in _NON_LIVE_REFRAME_ORIGINS for origin in cls._task_origins(*tasks))

    @staticmethod
    def _task_origins(*tasks: RepairTask) -> List[str]:
        origins: List[str] = []
        for task in tasks:
            meta = dict(getattr(task, "meta", {}) or {})
            for key in ("origin", "source", "capture_source", "producer"):
                value = str(meta.get(key) or "").strip()
                if value:
                    origins.append(value)
            if meta.get("signal_id"):
                origins.append("daydream_signal")
            elif meta.get("source_reflection_id"):
                origins.append("daydream")
        return list(dict.fromkeys(origins))

    @staticmethod
    def _duplicate_reframe_hint(
        task: RepairTask,
        *,
        existing_task: RepairTask,
        canonical_artifact: Optional[str],
    ) -> str:
        existing_meta = existing_task.meta if isinstance(existing_task.meta, dict) else {}
        resume_project = existing_meta.get("resume_project")
        if isinstance(resume_project, dict):
            best_next_step = str(resume_project.get("best_next_step", "") or "").strip()
            if best_next_step:
                return best_next_step
        retry_governor = existing_meta.get("retry_governor")
        if isinstance(retry_governor, dict):
            mode = str(retry_governor.get("mode", "") or "").strip().lower()
            if mode == "resume_existing_artifact" and canonical_artifact:
                return f"Resume from {canonical_artifact} with one narrow edit, then rerun verification."
        if canonical_artifact:
            return f"Inspect {canonical_artifact}, repair the smallest verified gap, then rerun verification."
        objective = str(task.objective or existing_task.objective or "").strip() or "this line"
        return f"Gather fresh evidence or choose a materially different framing before retrying {objective}."

    @staticmethod
    def _duplicate_retry_governor_payload(
        existing_task: RepairTask,
        *,
        result: RepairResult,
    ) -> Dict[str, Any]:
        task_meta = existing_task.meta if isinstance(existing_task.meta, dict) else {}
        prior = task_meta.get("retry_governor")
        payload = dict(prior) if isinstance(prior, dict) else {}
        payload.update(
            {
                "allowed": False,
                "reason": "recent failed duplicate requires reframe",
                "mode": "reframe_required",
                "result_stage": result.stage.value,
            }
        )
        return payload

    @staticmethod
    def _duplicate_resume_project_payload(
        existing_task: RepairTask,
        *,
        canonical_artifact: Optional[str],
        reframe_hint: str,
    ) -> Dict[str, Any]:
        task_meta = existing_task.meta if isinstance(existing_task.meta, dict) else {}
        resume_project = task_meta.get("resume_project")
        payload = dict(resume_project) if isinstance(resume_project, dict) else {}
        if canonical_artifact and not payload.get("canonical_artifact_path"):
            payload["canonical_artifact_path"] = canonical_artifact
        if reframe_hint and not payload.get("best_next_step"):
            payload["best_next_step"] = reframe_hint
        return payload

    def _shadow_registry(self):
        return getattr(getattr(self.runtime, "ctx", None), "shadow_registry", None)

    def _executive_state(self):
        return getattr(getattr(self.runtime, "ctx", None), "executive", None)

    def _thread_registry_service(self):
        return getattr(self.runtime, "thread_registry_service", None) or getattr(
            getattr(self.runtime, "ctx", None),
            "thread_registry_service",
            None,
        )

    def _record_session_resume(
        self,
        task: RepairTask,
        *,
        reason: str,
        source_trace: Optional[Dict[str, Any]] = None,
    ) -> None:
        """Record a durable boundary when a held or persisted task resumes."""
        self.executor.record_task_boundary(
            task,
            boundary="session_resumed",
            workflow_phase="resume",
            artifact="repair-task|default|resumed",
            why=reason,
            action="COMMIT",
            risk="LOW",
            source_trace=source_trace,
        )

    def _record_waiting_provenance(self, *, task: RepairTask, reason: Optional[str]) -> None:
        """Persist provenance for tasks paused on operator input."""
        runtime = self.runtime
        config = getattr(getattr(runtime, "ctx", None), "config", None) if runtime is not None else None
        state_dir = getattr(config, "state_dir", None) if config is not None else None
        if state_dir is None:
            return

        task_id = str(task.task_id)
        raw_session_id = getattr(config, "session_id", None) if config is not None else None
        session_id = raw_session_id.strip() if isinstance(raw_session_id, str) and raw_session_id.strip() else task_id
        record_provenance_transition(
            state_dir=state_dir,
            kind=ProvenanceTransitionKind.WAITING,
            session_id=session_id,
            entity_id=task_id,
            status="blocked",
            trigger_artifact=f"repair|baa|{task_id}",
            source_artifact=f"repair|baa|{task_id}",
            trigger_action="baa.transition_task",
            target_entity=task_id,
            origin_action_id=task_id,
            details={
                "reason": reason,
                "stage": task.stage.value,
            },
        )
        meta = task.meta if isinstance(getattr(task, "meta", None), dict) else {}
        emit_provenance_event(
            meta,
            event_type=ProvenanceEventType.BLOCKED,
            triggering_artifact=f"repair-task|default|{task_id}",
            triggering_action="WAIT",
            parent_link_id=task_id,
            linked_link_ids=[task_id, session_id],
            details={
                "reason": reason,
                "stage": task.stage.value,
                "session_id": session_id,
            },
        )
        if isinstance(getattr(task, "meta", None), dict):
            task.meta = meta

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
        from opencas.memory import Episode

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
