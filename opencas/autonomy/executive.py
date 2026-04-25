"""Executive state tracker for OpenCAS.

Tracks current intention, active goals, task queue, and capacity.
Integrates with identity self_model and somatic state.
"""

from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field

from opencas.identity import IdentityManager
from opencas.somatic import SomaticManager
from opencas.telemetry import EventKind, Tracer

from .commitment import CommitmentStatus
from .commitment_store import CommitmentStore
from .models import WorkObject, WorkStage
from .work_store import WorkStore


class ExecutiveSnapshot(BaseModel):
    """Fast-path JSON snapshot of executive state."""

    updated_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    intention: Optional[str] = None
    active_goals: List[str] = Field(default_factory=list)
    queue_metadata: List[Dict[str, Any]] = Field(default_factory=list)


class ExecutiveState:
    """Live executive state of the agent."""

    def __init__(
        self,
        identity: IdentityManager,
        somatic: Optional[SomaticManager] = None,
        tracer: Optional[Tracer] = None,
        work_store: Optional[WorkStore] = None,
        commitment_store: Optional[CommitmentStore] = None,
    ) -> None:
        self.identity = identity
        self.somatic = somatic
        self.tracer = tracer
        self.work_store = work_store
        self.commitment_store = commitment_store
        self._intention: Optional[str] = None
        self._active_goals: List[str] = []
        self._task_queue: List[WorkObject] = []
        self._max_capacity: int = 5
        self._snapshot_path: Optional[Path] = None
        self._background_tasks: set[asyncio.Task[None]] = set()

    @property
    def intention(self) -> Optional[str]:
        return self._intention

    def set_intention(self, intention: Optional[str]) -> None:
        self._intention = intention
        self._trace("intention_set", {"intention": intention})
        self._auto_save()

    def set_intention_from_work(self, work: WorkObject) -> None:
        """Track the current intention from a dequeued work object."""
        self.set_intention(work.content[:200])

    @property
    def active_goals(self) -> List[str]:
        return list(self._active_goals)

    def add_goal(self, goal: str) -> None:
        if goal not in self._active_goals:
            self._active_goals.append(goal)
            self._sync_identity_goals()
            self._trace("goal_added", {"goal": goal})
            self._auto_save()

    def remove_goal(self, goal: str) -> None:
        if goal in self._active_goals:
            self._active_goals.remove(goal)
            self._sync_identity_goals()
            self._trace("goal_removed", {"goal": goal})
            self._auto_save()

    def clear_goals(self) -> None:
        self._active_goals.clear()
        self._sync_identity_goals()
        self._trace("goals_cleared")
        self._auto_save()

    @property
    def task_queue(self) -> List[WorkObject]:
        return list(self._task_queue)

    def enqueue(
        self,
        work: WorkObject,
        commitment_id: Optional[str] = None,
    ) -> bool:
        """Add work to the queue if capacity allows.

        If *commitment_id* is provided it is set on *work*. Callers that have
        an async context should validate the commitment status before calling
        enqueue(); this method performs the synchronous enqueue only.
        """
        if commitment_id is not None:
            work.commitment_id = commitment_id
        if len(self._task_queue) >= self._max_capacity:
            self._trace("enqueue_rejected", {"work_id": str(work.work_id), "reason": "capacity"})
            return False
        self._task_queue.append(work)
        self._sync_work(work)
        self._trace("enqueue_accepted", {"work_id": str(work.work_id), "stage": work.stage.value})
        self._auto_save()
        return True

    def dequeue(self) -> Optional[WorkObject]:
        """Pop the highest-priority work item."""
        if not self._task_queue:
            return None
        # Sort by promise/user-facing bias first, then creative score.
        self._task_queue.sort(key=self._dequeue_sort_key)
        work = self._task_queue.pop(0)
        self._trace("dequeue", {"work_id": str(work.work_id)})
        self._auto_save()
        return work

    def remove_work(self, work_id: str) -> bool:
        for idx, w in enumerate(self._task_queue):
            if str(w.work_id) == work_id:
                self._task_queue.pop(idx)
                self._delete_work(work_id)
                self._trace("work_removed", {"work_id": work_id})
                self._auto_save()
                return True
        return False

    async def restore_queue(self, limit: int = 100) -> int:
        """Load ready work objects from the store into the task queue.

        Returns the number of items restored.
        """
        if not self.work_store:
            return 0
        work_items = await self.work_store.list_ready(limit=limit)
        restored = 0
        for work in work_items:
            if work.stage in (WorkStage.MICRO_TASK, WorkStage.PROJECT_SEED, WorkStage.PROJECT):
                if work.commitment_id and self.commitment_store:
                    commitment = await self.commitment_store.get(work.commitment_id)
                    if commitment and commitment.status != CommitmentStatus.ACTIVE:
                        continue
                    if commitment:
                        self.apply_commitment_execution_bias(work, commitment)
                if len(self._task_queue) < self._max_capacity:
                    # Avoid duplicating items already in queue
                    if not any(str(w.work_id) == str(work.work_id) for w in self._task_queue):
                        self._task_queue.append(work)
                        restored += 1
                else:
                    break
        self._trace("queue_restored", {"restored": restored})
        self._auto_save()
        return restored

    def restore_goals_from_identity(self) -> int:
        """Hydrate active goals from the identity self-model."""
        identity_goals = list(getattr(self.identity.self_model, "current_goals", []) or [])
        added = 0
        for g in identity_goals:
            if g and g not in self._active_goals:
                self._active_goals.append(g)
                added += 1
        if added:
            self._trace("goals_restored_from_identity", {"count": added})
            self._auto_save()
        return added

    def load_snapshot(self, path: Path | str) -> None:
        """Load executive state from a JSON snapshot file."""
        self._snapshot_path = Path(path)
        if not self._snapshot_path.exists():
            return
        try:
            snap = ExecutiveSnapshot.model_validate_json(
                self._snapshot_path.read_text(encoding="utf-8")
            )
            self._intention = snap.intention
            self._active_goals = list(snap.active_goals)
            # queue_metadata is informational; queue is restored from work_store
            self._sync_identity_goals()
            self._trace("snapshot_loaded", {"goals_count": len(self._active_goals)})
        except Exception:
            pass

    def save_snapshot(self, path: Optional[Path | str] = None) -> None:
        """Persist executive state to a JSON snapshot file."""
        target = Path(path) if path else self._snapshot_path
        if not target:
            return
        target.parent.mkdir(parents=True, exist_ok=True)
        snap = ExecutiveSnapshot(
            updated_at=datetime.now(timezone.utc),
            intention=self._intention,
            active_goals=list(self._active_goals),
            queue_metadata=[
                {"work_id": str(w.work_id), "stage": w.stage.value}
                for w in self._task_queue
            ],
        )
        temp = target.with_suffix(".tmp")
        temp.write_text(snap.model_dump_json(indent=2), encoding="utf-8")
        temp.replace(target)

    def _auto_save(self) -> None:
        if self._snapshot_path:
            self.save_snapshot()

    async def check_goal_resolution(self, completed_text: str) -> List[str]:
        """Return goals/commitments satisfied by the completed work and remove them."""
        resolved: List[str] = []
        for goal in self._active_goals:
            if self._goal_satisfied_by(goal, completed_text):
                resolved.append(goal)
        for g in resolved:
            self.remove_goal(g)

        if self.commitment_store:
            commitments = await self.commitment_store.list_active()
            for commitment in commitments:
                if self._goal_satisfied_by(commitment.content, completed_text):
                    resolved.append(commitment.content)
                    await self.commitment_store.update_status(
                        str(commitment.commitment_id), CommitmentStatus.COMPLETED
                    )
        return resolved

    async def resume_deferred_work(self) -> Dict[str, Any]:
        """Unblock commitments and restore queue when agent recovers from pause."""
        unblocked_commitments = 0
        restored_work = 0

        if self.commitment_store:
            blocked = await self.commitment_store.list_by_status(
                CommitmentStatus.BLOCKED, limit=100
            )
            for commitment in blocked:
                if not self._is_resume_eligible_commitment(commitment):
                    continue
                blocked_reason = str(commitment.meta.get("blocked_reason", "")).strip()
                commitment.status = CommitmentStatus.ACTIVE
                commitment.updated_at = datetime.now(timezone.utc)
                commitment.meta["resume_reason"] = "executive_recovery"
                commitment.meta["resumed_at"] = commitment.updated_at.isoformat()
                if blocked_reason:
                    commitment.meta["previous_blocked_reason"] = blocked_reason
                commitment.meta.pop("blocked_reason", None)
                await self.commitment_store.save(commitment)
                unblocked_commitments += 1
                if self.work_store and commitment.linked_work_ids:
                    for work_id in commitment.linked_work_ids:
                        work = await self.work_store.get(work_id)
                        if work and work.stage in (
                            WorkStage.MICRO_TASK,
                            WorkStage.PROJECT_SEED,
                            WorkStage.PROJECT,
                        ):
                            if self.enqueue(work):
                                restored_work += 1

        queue_restored = await self.restore_queue(limit=100)
        self._trace(
            "deferred_work_resumed",
            {
                "unblocked_commitments": unblocked_commitments,
                "restored_work": restored_work,
                "queue_restored": queue_restored,
            },
        )
        return {
            "unblocked_commitments": unblocked_commitments,
            "restored_work": restored_work,
            "queue_restored": queue_restored,
        }

    @staticmethod
    def _goal_satisfied_by(goal: str, completed_text: str) -> bool:
        """Simple keyword overlap heuristic. Future: LLM-based."""
        goal_tokens = set(goal.lower().split())
        text_tokens = set(completed_text.lower().split())
        overlap = len(goal_tokens & text_tokens)
        return overlap >= max(2, len(goal_tokens) * 0.3)

    def _sync_work(self, work: WorkObject) -> None:
        if not self.work_store:
            return
        try:
            import asyncio
            loop = asyncio.get_running_loop()
            def _sync():
                if self.work_store:
                    task = asyncio.create_task(self.work_store.save(work))
                    self._background_tasks.add(task)
                    task.add_done_callback(self._background_tasks.discard)
            loop.call_soon(_sync)
        except RuntimeError:
            pass

    def _delete_work(self, work_id: str) -> None:
        if not self.work_store:
            return
        try:
            import asyncio
            loop = asyncio.get_running_loop()
            def _delete():
                if self.work_store:
                    task = asyncio.create_task(self.work_store.delete(work_id))
                    self._background_tasks.add(task)
                    task.add_done_callback(self._background_tasks.discard)
            loop.call_soon(_delete)
        except RuntimeError:
            pass

    @property
    def capacity_remaining(self) -> int:
        return max(0, self._max_capacity - len(self._task_queue))

    @property
    def is_overloaded(self) -> bool:
        return len(self._task_queue) >= self._max_capacity

    def pause_reason(self) -> Optional[str]:
        """Return the executive pause reason, if any."""
        if self.is_overloaded:
            return "overload"
        if self.somatic and self.somatic.state.fatigue > 0.7:
            return "fatigue"
        return None

    def recommend_pause(self) -> bool:
        """Recommend pausing new work based on somatic fatigue or overload."""
        return self.pause_reason() is not None

    def snapshot(self) -> Dict[str, Any]:
        """Return a JSON-serializable snapshot of executive state."""
        return {
            "intention": self._intention,
            "active_goals": self._active_goals,
            "capacity_remaining": self.capacity_remaining,
            "queue_size": len(self._task_queue),
            "queue_stages": [w.stage.value for w in self._task_queue],
            "recommend_pause": self.recommend_pause(),
            "timestamp": datetime.now(timezone.utc).isoformat(),
        }

    def _sync_identity_goals(self) -> None:
        """Mirror active goals back to the identity self-model."""
        self.identity.self_model.current_goals = list(self._active_goals)
        self.identity.save()

    @staticmethod
    def _is_resume_eligible_commitment(commitment: Any) -> bool:
        """Return True when a blocked commitment should auto-resume after recovery."""
        reason = str(commitment.meta.get("blocked_reason", "")).strip().lower()
        if reason in {"executive_pause", "executive_fatigue", "executive_overload"}:
            return True
        if commitment.meta.get("resume_policy") == "auto_on_executive_recovery":
            return True
        # Backward compatibility for the in-flight Claude self-commitment rollout
        return (
            not reason
            and commitment.meta.get("source") == "assistant_response"
            and "self_commitment" in commitment.tags
        )

    @staticmethod
    def apply_commitment_execution_bias(work: WorkObject, commitment: Any) -> None:
        """Annotate work with lightweight commitment priority hints for queueing and focus."""
        work.meta = dict(work.meta or {})
        priority_bias = max(
            float(work.meta.get("commitment_priority_bias", 0.0)),
            float(commitment.priority) / 10.0,
        )
        user_facing = ExecutiveState._is_user_facing_commitment(commitment)
        work.meta["commitment_priority_bias"] = round(priority_bias, 4)
        work.meta["user_facing_commitment"] = user_facing
        if user_facing:
            work.meta["commitment_attention_bias"] = max(
                float(work.meta.get("commitment_attention_bias", 0.0)),
                0.2,
            )

    @staticmethod
    def _is_user_facing_commitment(commitment: Any) -> bool:
        source = str((getattr(commitment, "meta", {}) or {}).get("source", "")).lower()
        return source in {"assistant_response", "nightly_consolidation"}

    @staticmethod
    def _dequeue_sort_key(work: WorkObject) -> tuple:
        meta = work.meta or {}
        return (
            -float(meta.get("commitment_priority_bias", 0.0)),
            -float(meta.get("commitment_attention_bias", 0.0)),
            -(1 if meta.get("user_facing_commitment") else 0),
            -work.promotion_score,
            work.created_at,
        )

    def _trace(self, event: str, payload: Optional[Dict[str, Any]] = None) -> None:
        if self.tracer:
            self.tracer.log(
                EventKind.TOM_EVAL,  # reuse ToM eval as executive introspection
                f"Executive: {event}",
                payload or {},
            )
