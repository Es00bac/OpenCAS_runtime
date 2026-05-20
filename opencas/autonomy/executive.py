"""Executive state tracker for OpenCAS.

Tracks current intention, active goals, task queue, and capacity.
Integrates with identity self_model and somatic state.
"""

from __future__ import annotations

import asyncio
import math
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field

from opencas.autonomy.completion_evidence import requires_explicit_completion_evidence
from opencas.identity import IdentityManager
from opencas.identity.text_hygiene import is_bootstrap_placeholder_intention
from opencas.somatic import SomaticManager
from opencas.telemetry import EventKind, Tracer

from .commitment import CommitmentStatus
from .commitment_store import CommitmentStore
from .goal_hygiene import (
    hard_reject_goal_reason,
    is_hard_reject_goal_reason,
    is_self_referential_suppression_metadata,
    split_live_and_parked_goals,
)
from .executive_pause_reason import (
    is_executive_pause_reason,
    is_known_executive_pause_reason,
)
from .models import WorkObject, WorkStage
from .work_store import WorkStore

_ARCHIVABLE_PARK_REASONS = frozenset(
    {
        "abstract_theme_goal",
        "generic_verb_without_binding",
        "machine_fragment_goal",
        "numbered_fragment_goal",
        "path_fragment_goal",
        "stale_promoted_work_fragment",
    }
)
_PARKED_RESIDUE_ARCHIVE_THRESHOLD = 6
_LOW_DIVERGENCE_REFRAME_LIVE_LIMIT = 1


class ExecutiveSnapshot(BaseModel):
    """Fast-path JSON snapshot of executive state."""

    updated_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    intention: Optional[str] = None
    intention_source: Optional[str] = None
    active_goals: List[str] = Field(default_factory=list)
    parked_goals: List[str] = Field(default_factory=list)
    parked_goal_reasons: Dict[str, str] = Field(default_factory=dict)
    parked_goal_metadata: Dict[str, Dict[str, Any]] = Field(default_factory=dict)
    archived_parked_goals: List[str] = Field(default_factory=list)
    archived_parked_goal_metadata: Dict[str, Dict[str, Any]] = Field(default_factory=dict)
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
        self._intention_source: Optional[str] = None
        self._active_goals: List[str] = []
        self._parked_goals: List[str] = []
        self._parked_goal_reasons: Dict[str, str] = {}
        self._parked_goal_metadata: Dict[str, Dict[str, Any]] = {}
        self._archived_parked_goals: List[str] = []
        self._archived_parked_goal_metadata: Dict[str, Dict[str, Any]] = {}
        self._suppressed_parked_goal_rejections: List[Dict[str, Any]] = []
        self._task_queue: List[WorkObject] = []
        self._max_capacity: int = 5
        self._max_queue_depth: Optional[int] = None
        self._snapshot_path: Optional[Path] = None
        self._background_tasks: set[asyncio.Task[None]] = set()

    @property
    def intention(self) -> Optional[str]:
        return self._intention

    @property
    def intention_source(self) -> Optional[str]:
        return self._intention_source

    def set_intention(self, intention: Optional[str], *, source: str = "explicit") -> None:
        previous_intention = self._intention
        self._intention = intention
        self._intention_source = source if intention else None
        self._sync_identity_intention(
            intention,
            previous_intention=previous_intention,
            force=source == "tasklist_live_objective",
        )
        self._trace("intention_set", {"intention": intention, "source": self._intention_source})
        self._auto_save()

    def set_intention_from_work(self, work: WorkObject) -> None:
        """Track active work without clobbering an anchored live objective."""
        candidate = work.content[:200]
        current = str(self._intention or "").strip()
        if (
            not current
            or is_bootstrap_placeholder_intention(current)
            or self._intention_source == "active_work"
        ):
            self.set_intention(candidate, source="active_work")
            return
        self._trace(
            "intention_preserved_for_active_work",
            {"intention": current, "active_work": candidate},
        )

    @property
    def active_goals(self) -> List[str]:
        return list(self._active_goals)

    @property
    def parked_goals(self) -> List[str]:
        return list(self._parked_goals)

    @property
    def parked_goal_metadata(self) -> Dict[str, Dict[str, Any]]:
        return dict(self._parked_goal_metadata)

    @property
    def archived_parked_goals(self) -> List[str]:
        return list(self._archived_parked_goals)

    @property
    def archived_parked_goal_metadata(self) -> Dict[str, Dict[str, Any]]:
        return dict(self._archived_parked_goal_metadata)

    def consume_suppressed_parked_goal_rejections(self) -> List[Dict[str, Any]]:
        """Return and clear suppression metadata rejected while loading snapshots."""
        rejections = list(self._suppressed_parked_goal_rejections)
        self._suppressed_parked_goal_rejections.clear()
        return rejections

    def add_goal(self, goal: str) -> None:
        normalized_goal = str(goal or "").strip()
        if not normalized_goal:
            return
        rejection_reason = hard_reject_goal_reason(normalized_goal)
        if rejection_reason:
            self._trace(
                "goal_rejected",
                {
                    "goal": normalized_goal,
                    "reason": rejection_reason,
                    "source": "add_goal",
                },
            )
            return
        previous_identity_goals = list(getattr(self.identity.self_model, "current_goals", []) or [])
        goal_surface = split_live_and_parked_goals([normalized_goal])
        changed = False
        for live_goal in goal_surface.active_goals:
            if live_goal not in self._active_goals:
                self._active_goals.append(live_goal)
                self._trace("goal_added", {"goal": live_goal})
                changed = True
        if goal_surface.parked_goals:
            if self._merge_parked_goals(goal_surface.parked_goals, goal_surface.parked_reasons):
                self._trace(
                    "goal_parked",
                    {
                        "goal": goal_surface.parked_goals[0],
                        "reason": goal_surface.parked_reasons.get(goal_surface.parked_goals[0]),
                    },
                )
                changed = True
        if changed:
            self._compact_parked_goal_residue()
            self._sync_identity_goals(previous_goals=previous_identity_goals)
            self._sync_structural_load()
            self._auto_save()

    def remove_goal(self, goal: str) -> None:
        removed = False
        if goal in self._active_goals:
            self._active_goals.remove(goal)
            removed = True
        if goal in self._parked_goals:
            self._parked_goals.remove(goal)
            self._parked_goal_reasons.pop(goal, None)
            self._parked_goal_metadata.pop(goal, None)
            removed = True
        if removed:
            previous_identity_goals = list(getattr(self.identity.self_model, "current_goals", []) or [])
            self._sync_identity_goals(previous_goals=previous_identity_goals)
            self._sync_structural_load()
            self._trace("goal_removed", {"goal": goal})
            self._auto_save()

    def park_goal(
        self,
        goal: str,
        *,
        reason: str = "evidence_deferred",
        wake_trigger: Optional[str] = None,
        source_artifact: Optional[str] = None,
        details: Optional[Dict[str, Any]] = None,
    ) -> bool:
        normalized_goal = str(goal or "").strip()
        if not normalized_goal:
            return False
        if is_self_referential_suppression_metadata(
            normalized_goal,
            reason=reason,
            source_artifact=source_artifact,
            details=details,
        ):
            self._trace(
                "parked_goal_suppression_metadata_rejected",
                {
                    "goal": normalized_goal,
                    "reason": reason,
                    "source_artifact": source_artifact,
                },
            )
            return False

        changed = False
        if normalized_goal in self._active_goals:
            self._active_goals.remove(normalized_goal)
            changed = True
        if normalized_goal not in self._parked_goals:
            self._parked_goals.append(normalized_goal)
            changed = True

        previous_reason = self._parked_goal_reasons.get(normalized_goal)
        if previous_reason != reason:
            self._parked_goal_reasons[normalized_goal] = reason
            changed = True

        existing = self._parked_goal_metadata.get(normalized_goal) or {}
        enriched = self._build_parked_goal_metadata(normalized_goal, reason=reason, existing=existing)
        if wake_trigger:
            enriched["wake_trigger"] = wake_trigger
        if source_artifact is not None:
            enriched["source_artifact"] = source_artifact
        if details:
            for key, value in details.items():
                if value is None:
                    continue
                enriched[key] = value
        if enriched != existing:
            self._parked_goal_metadata[normalized_goal] = enriched
            changed = True

        if changed:
            self._compact_parked_goal_residue()
            previous_identity_goals = list(getattr(self.identity.self_model, "current_goals", []) or [])
            self._sync_identity_goals(previous_goals=previous_identity_goals)
            self._sync_structural_load()
            self._trace(
                "goal_parked_explicitly",
                {
                    "goal": normalized_goal,
                    "reason": reason,
                    "wake_trigger": enriched.get("wake_trigger"),
                },
            )
            self._auto_save()
        return changed

    def clear_goals(self) -> None:
        self._active_goals.clear()
        previous_identity_goals = list(getattr(self.identity.self_model, "current_goals", []) or [])
        self._sync_identity_goals(previous_goals=previous_identity_goals)
        self._sync_structural_load()
        self._trace("goals_cleared")
        self._auto_save()

    @property
    def task_queue(self) -> List[WorkObject]:
        return list(self._task_queue)

    @staticmethod
    def _queue_state_for_index(index: int) -> str:
        """Return the queue state for the item at *index* in sorted order."""
        return "active" if index == 0 else "held"

    @staticmethod
    def _queue_state_label(state: str) -> str:
        """Return the human-readable label for a queue state."""
        return "Active" if state == "active" else "Held"

    @staticmethod
    def _queue_bearing_for_work(work: WorkObject, *, state: str) -> str:
        """Return the finer-grained bearing label for a queue item."""
        if state == "active":
            return "ready"
        stage = work.stage if isinstance(work.stage, WorkStage) else None
        if stage == WorkStage.PROJECT:
            return "waiting"
        if stage == WorkStage.PROJECT_SEED:
            return "queued"
        return "ready"

    @staticmethod
    def _queue_bearing_label(bearing: str) -> str:
        """Return the human-readable label for a queue bearing."""
        return {
            "ready": "Ready",
            "queued": "Queued",
            "waiting": "Waiting",
        }.get(bearing, "Ready")

    def _queue_contract_for_work(self, work: WorkObject, *, index: int) -> Dict[str, Any]:
        """Build the canonical queue contract for one work item."""
        state = self._queue_state_for_index(index)
        bearing = self._queue_bearing_for_work(work, state=state)
        return {
            "position": index,
            "work_id": str(work.work_id),
            "title": self._work_preview(work),
            "stage": work.stage.value if hasattr(work.stage, "value") else str(work.stage),
            "state": state,
            "state_label": self._queue_state_label(state),
            "role": state,
            "role_label": self._queue_state_label(state),
            "bearing": bearing,
            "bearing_label": self._queue_bearing_label(bearing),
            "is_active": state == "active",
            "project_id": work.project_id,
            "commitment_id": work.commitment_id,
            "promotion_score": work.promotion_score,
        }

    def _normalize_queue_state(self) -> None:
        """Keep the queue in canonical priority order with one active item."""
        self._task_queue.sort(key=self._dequeue_sort_key)

    @staticmethod
    def _work_preview(work: WorkObject, limit: int = 120) -> str:
        """Return a compact display label for queue snapshots."""
        title = str((work.meta or {}).get("title", "")).strip()
        raw = title or str(getattr(work, "content", "") or "").strip() or "Untitled work"
        compact = " ".join(raw.split())
        if len(compact) <= limit:
            return compact
        return compact[: max(0, limit - 3)].rstrip() + "..."

    def _generated_goal_for_work(self, work: WorkObject) -> str:
        """Return the default active-goal text generated from a work object."""

        return f"Advance promoted work: {self._work_preview(work, limit=96)}"

    def _work_goal_rejection_reason(self, work: WorkObject) -> Optional[str]:
        """Return a hygiene rejection reason for work that should not enter live focus."""

        if work.stage not in (WorkStage.MICRO_TASK, WorkStage.PROJECT_SEED, WorkStage.PROJECT):
            return None
        explicit = str((work.meta or {}).get("active_goal") or "").strip()
        return hard_reject_goal_reason(explicit or self._generated_goal_for_work(work))

    def _block_rejected_work(self, work: WorkObject, *, reason: str, source: str) -> None:
        """Move residue work out of the ready queue without deleting its provenance."""

        marker = f"goal_hygiene:{reason}"
        if marker not in work.blocked_by:
            work.blocked_by.append(marker)
        work.updated_at = datetime.now(timezone.utc)
        work.meta = dict(work.meta or {})
        work.meta["goal_hygiene_rejected_at"] = work.updated_at.isoformat()
        work.meta["goal_hygiene_rejected_reason"] = reason
        work.meta["goal_hygiene_rejected_source"] = source
        self._sync_work(work)

    def _ensure_active_goal_from_work(self, work: WorkObject, *, source: str) -> bool:
        """Create a live goal when promoted work enters the executive queue."""
        if work.stage not in (WorkStage.MICRO_TASK, WorkStage.PROJECT_SEED, WorkStage.PROJECT):
            return False
        explicit = str((work.meta or {}).get("active_goal") or "").strip()
        goal = explicit or self._generated_goal_for_work(work)
        rejection_reason = hard_reject_goal_reason(goal)
        if rejection_reason:
            self._trace(
                "goal_rejected",
                {
                    "goal": goal,
                    "reason": rejection_reason,
                    "source": source,
                    "work_id": str(work.work_id),
                },
            )
            return False
        if goal in self._active_goals:
            return False
        self._active_goals.append(goal)
        self._trace(
            "goal_added_from_work",
            {"goal": goal, "source": source, "work_id": str(work.work_id)},
        )
        return True

    def queue_metadata(self) -> List[Dict[str, Any]]:
        """Return queue items with a stable active/held contract."""
        ordered_queue = sorted(self._task_queue, key=self._dequeue_sort_key)
        return [
            self._queue_contract_for_work(work, index=index)
            for index, work in enumerate(ordered_queue)
        ]

    def queue_summary(self) -> Dict[str, Any]:
        """Return canonical queue metadata plus derived counts."""
        queue_metadata = self.queue_metadata()
        return {
            "counts": {
                "total": len(queue_metadata),
                "active": sum(1 for item in queue_metadata if item.get("state") == "active"),
                "held": sum(1 for item in queue_metadata if item.get("state") == "held"),
                "ready": sum(1 for item in queue_metadata if item.get("bearing") == "ready"),
                "queued": sum(1 for item in queue_metadata if item.get("bearing") == "queued"),
                "waiting": sum(1 for item in queue_metadata if item.get("bearing") == "waiting"),
            },
            "items": queue_metadata,
        }

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
        rejection_reason = self._work_goal_rejection_reason(work)
        if rejection_reason:
            self._block_rejected_work(work, reason=rejection_reason, source="enqueue")
            self._trace(
                "enqueue_rejected",
                {
                    "work_id": str(work.work_id),
                    "reason": rejection_reason,
                    "content": self._work_preview(work, limit=160),
                },
            )
            return False
        if len(self._task_queue) >= self.queue_hard_cap:
            self._trace("enqueue_rejected", {"work_id": str(work.work_id), "reason": "capacity"})
            return False
        previous_identity_goals = list(getattr(self.identity.self_model, "current_goals", []) or [])
        self._task_queue.append(work)
        self._normalize_queue_state()
        goal_changed = self._ensure_active_goal_from_work(work, source="enqueue")
        self._sync_work(work)
        if goal_changed:
            self._sync_identity_goals(previous_goals=previous_identity_goals)
        self._sync_structural_load()
        self._trace("enqueue_accepted", {"work_id": str(work.work_id), "stage": work.stage.value})
        self._auto_save()
        return True

    def dequeue(self) -> Optional[WorkObject]:
        """Pop the highest-priority work item."""
        if not self._task_queue:
            return None
        self._normalize_queue_state()
        work = self._task_queue.pop(0)
        self._sync_structural_load()
        self._trace("dequeue", {"work_id": str(work.work_id)})
        self._auto_save()
        return work

    def remove_work(self, work_id: str) -> bool:
        for idx, w in enumerate(self._task_queue):
            if str(w.work_id) == work_id:
                self._task_queue.pop(idx)
                self._normalize_queue_state()
                self._delete_work(work_id)
                self._sync_structural_load()
                self._trace("work_removed", {"work_id": work_id})
                self._auto_save()
                return True
        return False

    def reconcile_work_update(self, work: WorkObject) -> bool:
        """Reflect external work-store edits in the live queue and goals.

        Operations routes can demote or block work without going through
        ``enqueue``/``dequeue``. When that happens, the work should remain in the
        durable store, but it must stop occupying live executive focus.
        """

        work_id = str(work.work_id)
        executable = work.stage in (
            WorkStage.MICRO_TASK,
            WorkStage.PROJECT_SEED,
            WorkStage.PROJECT,
        ) and not work.blocked_by
        if executable:
            return False

        previous_identity_goals = list(getattr(self.identity.self_model, "current_goals", []) or [])
        changed = False
        before = len(self._task_queue)
        self._task_queue = [item for item in self._task_queue if str(item.work_id) != work_id]
        if len(self._task_queue) != before:
            changed = True
            self._normalize_queue_state()

        explicit = str((work.meta or {}).get("active_goal") or "").strip()
        goal = explicit or self._generated_goal_for_work(work)
        if goal in self._active_goals:
            still_queued = any(
                (str(item.work_id) != work_id)
                and (str((item.meta or {}).get("active_goal") or "").strip() or self._generated_goal_for_work(item)) == goal
                for item in self._task_queue
            )
            if not still_queued:
                self._active_goals.remove(goal)
                changed = True

        if changed:
            self._sync_identity_goals(previous_goals=previous_identity_goals)
            self._sync_structural_load()
            self._trace(
                "work_update_reconciled",
                {
                    "work_id": work_id,
                    "stage": work.stage.value if hasattr(work.stage, "value") else str(work.stage),
                    "blocked_by": list(work.blocked_by),
                },
            )
            self._auto_save()
        return changed

    async def restore_queue(self, limit: int = 100) -> int:
        """Load ready work objects from the store into the task queue.

        Returns the number of items restored.
        """
        if not self.work_store:
            return 0
        work_items = await self.work_store.list_ready(limit=limit)
        restored = 0
        previous_identity_goals = list(getattr(self.identity.self_model, "current_goals", []) or [])
        goal_changed = False
        for work in work_items:
            if work.stage in (WorkStage.MICRO_TASK, WorkStage.PROJECT_SEED, WorkStage.PROJECT):
                rejection_reason = self._work_goal_rejection_reason(work)
                if rejection_reason:
                    self._block_rejected_work(work, reason=rejection_reason, source="restore_queue")
                    if self.work_store:
                        await self.work_store.save(work)
                    self._trace(
                        "restore_queue_skipped",
                        {
                            "work_id": str(work.work_id),
                            "reason": rejection_reason,
                            "content": self._work_preview(work, limit=160),
                        },
                    )
                    continue
                if work.commitment_id and self.commitment_store:
                    commitment = await self.commitment_store.get(work.commitment_id)
                    if commitment and commitment.status != CommitmentStatus.ACTIVE:
                        continue
                    if commitment:
                        self.apply_commitment_execution_bias(work, commitment)
                if len(self._task_queue) < self.queue_hard_cap:
                    # Avoid duplicating items already in queue
                    if not any(str(w.work_id) == str(work.work_id) for w in self._task_queue):
                        self._task_queue.append(work)
                        self._normalize_queue_state()
                        goal_changed = (
                            self._ensure_active_goal_from_work(work, source="restore_queue")
                            or goal_changed
                        )
                        restored += 1
                else:
                    break
        if goal_changed:
            self._sync_identity_goals(previous_goals=previous_identity_goals)
        self._sync_structural_load()
        self._trace("queue_restored", {"restored": restored})
        self._auto_save()
        return restored

    def restore_goals_from_identity(self) -> int:
        """Hydrate active goals from the identity self-model."""
        identity_goals = list(getattr(self.identity.self_model, "current_goals", []) or [])
        filtered_identity_goals, rejected = self._filter_hard_rejected_goals(
            identity_goals,
            source="identity.current_goals",
        )
        goal_surface = split_live_and_parked_goals(filtered_identity_goals)
        added = 0
        for g in goal_surface.active_goals:
            if g and g not in self._active_goals:
                self._active_goals.append(g)
                added += 1
        parked_changed = self._merge_parked_goals(
            goal_surface.parked_goals,
            goal_surface.parked_reasons,
        )
        compacted = self._compact_parked_goal_residue()
        if added or parked_changed or compacted or rejected:
            self._sync_identity_goals(previous_goals=identity_goals)
            self._sync_structural_load()
            self._trace(
                "goals_restored_from_identity",
                {"count": added, "rejected_count": len(rejected)},
            )
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
            self._intention_source = snap.intention_source
            self._active_goals, rejected_active = self._filter_hard_rejected_goals(
                snap.active_goals,
                source="snapshot.active_goals",
            )
            self._parked_goals = list(snap.parked_goals)
            self._parked_goal_reasons = dict(snap.parked_goal_reasons)
            self._parked_goal_metadata = dict(snap.parked_goal_metadata)
            self._archived_parked_goals = list(snap.archived_parked_goals)
            self._archived_parked_goal_metadata = dict(snap.archived_parked_goal_metadata)
            rejected_parked = self._drop_hard_rejected_parked_goals(source="snapshot.parked_goals")
            self._merge_parked_goals(self._parked_goals, self._parked_goal_reasons)
            reconciled = self._reconcile_suppressed_parked_goal_metadata()
            goal_surface = split_live_and_parked_goals(self._active_goals)
            self._active_goals = list(goal_surface.active_goals)
            self._merge_parked_goals(goal_surface.parked_goals, goal_surface.parked_reasons)
            compacted = self._compact_parked_goal_residue()
            # queue_metadata is informational; queue is restored from work_store
            self._sync_identity_intention(
                self._intention,
                previous_intention=None,
            )
            self._sync_identity_goals(previous_goals=snap.active_goals)
            self._sync_structural_load()
            self._trace(
                "snapshot_loaded",
                {
                    "goals_count": len(self._active_goals),
                    "parked_goal_count": len(self._parked_goals),
                    "archived_parked_goal_count": len(self._archived_parked_goals),
                    "rejected_goal_count": len(rejected_active) + len(rejected_parked),
                },
            )
            if compacted or rejected_active or rejected_parked:
                self._auto_save()
            elif reconciled:
                self._trace(
                    "snapshot_suppression_metadata_reconciled",
                    {"rejected_count": len(self._suppressed_parked_goal_rejections)},
                )
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
            intention_source=self._intention_source,
            active_goals=list(self._active_goals),
            parked_goals=list(self._parked_goals),
            parked_goal_reasons=dict(self._parked_goal_reasons),
            parked_goal_metadata=dict(self._parked_goal_metadata),
            archived_parked_goals=list(self._archived_parked_goals),
            archived_parked_goal_metadata=dict(self._archived_parked_goal_metadata),
            queue_metadata=self.queue_metadata(),
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
                if self._requires_explicit_completion_evidence(commitment):
                    continue
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

    @staticmethod
    def _requires_explicit_completion_evidence(commitment: Any) -> bool:
        """Evidence-bearing commitments need proof, not keyword overlap."""
        return requires_explicit_completion_evidence(commitment)

    def _sync_identity_intention(
        self,
        intention: Optional[str],
        *,
        previous_intention: Optional[str],
        force: bool = False,
    ) -> None:
        """Synchronize stale placeholder self-intentions with live executive intent."""
        candidate = str(intention or "").strip() or None
        if candidate is None:
            return

        current = str(self.identity.self_model.current_intention or "").strip()
        if current == candidate:
            return
        if current and not force and not (
            is_bootstrap_placeholder_intention(current)
            or (previous_intention is not None and current == previous_intention)
        ):
            return

        self.identity.self_model.current_intention = candidate
        self.identity.save()

    def _track_background_task(self, task: "asyncio.Task[None]") -> None:
        """Keep fire-and-forget persistence tasks alive until completion."""
        self._background_tasks.add(task)
        task.add_done_callback(self._background_tasks.discard)

    def _sync_work(self, work: WorkObject) -> None:
        if not self.work_store:
            return
        try:
            import asyncio
            loop = asyncio.get_running_loop()
            if self.work_store:
                self._track_background_task(loop.create_task(self.work_store.save(work)))
        except RuntimeError:
            pass

    def _delete_work(self, work_id: str) -> None:
        if not self.work_store:
            return
        try:
            import asyncio
            loop = asyncio.get_running_loop()
            if self.work_store:
                self._track_background_task(loop.create_task(self.work_store.delete(work_id)))
        except RuntimeError:
            pass

    @property
    def capacity_remaining(self) -> int:
        weighted_remaining = math.floor(self._max_capacity - self.weighted_queue_load())
        queue_slots_remaining = self.queue_hard_cap - len(self._task_queue)
        return max(0, min(weighted_remaining, queue_slots_remaining))

    @property
    def is_overloaded(self) -> bool:
        return (
            len(self._task_queue) >= self.queue_hard_cap
            or self.weighted_queue_load() >= float(self._max_capacity)
        )

    @property
    def queue_hard_cap(self) -> int:
        configured = self._max_queue_depth
        if configured is None:
            return max(1, int(self._max_capacity))
        return max(1, configured)

    def weighted_queue_load(self) -> float:
        """Return a weighted executive load where held items count as partial pressure."""
        load = 0.0
        for item in self.queue_metadata():
            load += 1.0 if item.get("state") == "active" else 0.35
        return round(load, 3)

    def pause_reason(self) -> Optional[str]:
        """Return the executive pause reason, if any."""
        if self.is_overloaded:
            return "overload"
        if self.somatic:
            state = self.somatic.state
            rest_until = getattr(state, "rest_until", None)
            if rest_until is not None:
                if getattr(rest_until, "tzinfo", None) is None:
                    rest_until = rest_until.replace(tzinfo=timezone.utc)
                if rest_until > datetime.now(timezone.utc):
                    return "operator_rest"
            if state.fatigue > 0.7:
                return "fatigue"
        return None

    def recommend_pause(self) -> bool:
        """Recommend pausing new work based on somatic fatigue or overload."""
        return self.pause_reason() is not None

    def snapshot(self) -> Dict[str, Any]:
        """Return a JSON-serializable snapshot of executive state."""
        queue_summary = self.queue_summary()
        return {
            "intention": self._intention,
            "intention_source": self._intention_source,
            "active_goals": self._active_goals,
            "parked_goals": self._parked_goals,
            "parked_goal_count": len(self._parked_goals),
            "parked_goal_metadata": self._parked_goal_metadata,
            "archived_parked_goals": self._archived_parked_goals,
            "archived_parked_goal_count": len(self._archived_parked_goals),
            "weighted_load": self.weighted_queue_load(),
            "capacity_remaining": self.capacity_remaining,
            "queue_size": len(self._task_queue),
            "queue_stages": [item["stage"] for item in queue_summary["items"]],
            "queue_metadata": queue_summary["items"],
            "recommend_pause": self.recommend_pause(),
            "timestamp": datetime.now(timezone.utc).isoformat(),
        }

    def _sync_identity_goals(self, *, previous_goals: Optional[List[str]] = None) -> None:
        """Mirror active goals back to the identity self-model."""
        current_identity_goals = list(getattr(self.identity.self_model, "current_goals", []) or [])
        if (
            not current_identity_goals
            or current_identity_goals == list(previous_goals or [])
            or current_identity_goals == list(self._active_goals)
        ):
            self.identity.self_model.current_goals = list(self._active_goals)
            self.identity.save()

    def _merge_parked_goals(
        self,
        goals: List[str],
        reasons: Optional[Dict[str, str]] = None,
    ) -> bool:
        changed = False
        reason_map = reasons or {}
        for goal in goals:
            reason = reason_map.get(goal)
            rejection_reason = hard_reject_goal_reason(goal)
            if rejection_reason or is_hard_reject_goal_reason(reason):
                self._trace(
                    "goal_rejected",
                    {
                        "goal": goal,
                        "reason": rejection_reason or reason,
                        "source": "parked_goal_merge",
                    },
                )
                continue
            if goal not in self._parked_goals:
                self._parked_goals.append(goal)
                changed = True
            if reason and self._parked_goal_reasons.get(goal) != reason:
                self._parked_goal_reasons[goal] = reason
                changed = True
            metadata = self._parked_goal_metadata.get(goal) or {}
            enriched = self._build_parked_goal_metadata(goal, reason=reason, existing=metadata)
            if enriched != metadata:
                self._parked_goal_metadata[goal] = enriched
                changed = True
        return changed

    def _filter_hard_rejected_goals(
        self,
        goals: List[str],
        *,
        source: str,
    ) -> tuple[List[str], List[Dict[str, str]]]:
        kept: List[str] = []
        rejected: List[Dict[str, str]] = []
        for raw_goal in goals:
            goal = str(raw_goal or "").strip()
            if not goal:
                continue
            reason = hard_reject_goal_reason(goal)
            if reason:
                record = {"goal": goal, "reason": reason, "source": source}
                rejected.append(record)
                self._trace("goal_rejected", record)
                continue
            kept.append(goal)
        return kept, rejected

    def _drop_hard_rejected_parked_goals(self, *, source: str) -> List[Dict[str, str]]:
        rejected: List[Dict[str, str]] = []
        for goal in list(self._parked_goals):
            reason = (
                hard_reject_goal_reason(goal)
                or self._parked_goal_reasons.get(goal)
                or (self._parked_goal_metadata.get(goal) or {}).get("reason")
            )
            if not is_hard_reject_goal_reason(reason):
                continue
            if goal in self._parked_goals:
                self._parked_goals.remove(goal)
            self._parked_goal_reasons.pop(goal, None)
            self._parked_goal_metadata.pop(goal, None)
            record = {"goal": goal, "reason": str(reason), "source": source}
            rejected.append(record)
            self._trace("goal_rejected", record)
        return rejected

    def _reconcile_suppressed_parked_goal_metadata(self) -> bool:
        """Remove loaded suppression packets from the active parked-goal surface."""
        changed = False
        for goal in list(self._parked_goals):
            metadata = dict(self._parked_goal_metadata.get(goal) or {})
            reason = str(
                metadata.get("reason")
                or self._parked_goal_reasons.get(goal)
                or ""
            ).strip()
            source_artifact = metadata.get("source_artifact")
            if not is_self_referential_suppression_metadata(
                goal,
                reason=reason,
                source_artifact=str(source_artifact) if source_artifact is not None else None,
                details=metadata,
            ):
                continue

            if goal in self._parked_goals:
                self._parked_goals.remove(goal)
                changed = True
            self._parked_goal_reasons.pop(goal, None)
            self._parked_goal_metadata.pop(goal, None)
            rejection = {
                "goal": goal,
                "reason": reason,
                "source_artifact": source_artifact,
                "metadata": metadata,
            }
            self._suppressed_parked_goal_rejections.append(rejection)
            self._trace(
                "parked_goal_suppression_metadata_loaded_rejected",
                {
                    "goal": goal,
                    "reason": reason,
                    "source_artifact": source_artifact,
                },
            )
        return changed

    def _compact_parked_goal_residue(self) -> bool:
        routine_residue = [
            goal
            for goal in self._parked_goals
            if self._should_archive_parked_goal(goal)
        ]
        candidates: List[str] = []
        if len(routine_residue) >= _PARKED_RESIDUE_ARCHIVE_THRESHOLD:
            candidates.extend(routine_residue)

        low_divergence_reframes = [
            goal
            for goal in self._parked_goals
            if self._is_low_divergence_reframe_goal(goal)
        ]
        overflow_count = max(0, len(low_divergence_reframes) - _LOW_DIVERGENCE_REFRAME_LIVE_LIMIT)
        if overflow_count:
            candidates.extend(low_divergence_reframes[:overflow_count])

        candidates = list(dict.fromkeys(candidates))
        if not candidates:
            return False

        archived_at = datetime.now(timezone.utc).isoformat()
        reason_counts: Dict[str, int] = {}
        changed = False
        for goal in list(candidates):
            metadata = dict(self._parked_goal_metadata.get(goal) or {})
            archive_reason = (
                "low_divergence_reframe_limit"
                if self._is_low_divergence_reframe_goal(goal)
                else "residue_compaction"
            )
            reason = str(
                metadata.get("reason")
                or self._parked_goal_reasons.get(goal)
                or archive_reason
            ).strip() or archive_reason
            reason_counts[reason] = reason_counts.get(reason, 0) + 1
            metadata["reason"] = reason
            metadata["archived_at"] = archived_at
            metadata["archive_reason"] = archive_reason
            metadata.setdefault(
                "wake_trigger",
                "fresh failing artifact, materially new evidence, or direct user request",
            )

            if goal in self._parked_goals:
                self._parked_goals.remove(goal)
                changed = True
            self._parked_goal_reasons.pop(goal, None)
            self._parked_goal_metadata.pop(goal, None)

            if goal not in self._archived_parked_goals:
                self._archived_parked_goals.append(goal)
                changed = True
            previous_archived = self._archived_parked_goal_metadata.get(goal) or {}
            archived_metadata = dict(previous_archived)
            archived_metadata.update(metadata)
            if archived_metadata != previous_archived:
                self._archived_parked_goal_metadata[goal] = archived_metadata
                changed = True

        if changed:
            self._trace(
                "parked_goal_residue_compacted",
                {
                    "archived_count": len(candidates),
                    "remaining_live_count": len(self._parked_goals),
                    "reason_counts": reason_counts,
                },
            )
        return changed

    def _should_archive_parked_goal(self, goal: str) -> bool:
        metadata = self._parked_goal_metadata.get(goal) or {}
        reason = str(
            metadata.get("reason")
            or self._parked_goal_reasons.get(goal)
            or ""
        ).strip()
        if reason not in _ARCHIVABLE_PARK_REASONS:
            return False
        if metadata.get("reframe_hint"):
            return False
        return True

    def _is_low_divergence_reframe_goal(self, goal: str) -> bool:
        metadata = self._parked_goal_metadata.get(goal) or {}
        reason = str(
            metadata.get("reason")
            or self._parked_goal_reasons.get(goal)
            or ""
        ).strip()
        return reason == "low_divergence_reframe"

    def _sync_structural_load(self) -> None:
        if not self.somatic:
            return
        try:
            self.somatic.reflect_structural_load(
                weighted_queue_load=self.weighted_queue_load(),
                queue_depth=len(self._task_queue),
                active_goal_count=len(self._active_goals),
                parked_goal_count=len(self._parked_goals),
            )
        except Exception:
            pass

    def refresh_structural_load(self) -> None:
        """Re-apply structural-load signals after wiring new somatic state."""
        self._sync_structural_load()

    @staticmethod
    def _wake_trigger_for_park_reason(reason: Optional[str]) -> str:
        if reason == "evidence_deferred":
            return "fresh failure, relevant file/config change, objective dependency, or direct user request"
        if reason == "low_divergence_reframe":
            return "fresh evidence, relevant artifact change, materially different framing, or direct user request"
        if reason in {"machine_fragment_goal", "path_fragment_goal", "numbered_fragment_goal"}:
            return "fresh failing artifact or direct user request"
        if reason == "abstract_theme_goal":
            return "direct user request or an explicit live objective"
        return "fresh failing check, dependency change, or direct user request"

    def _build_parked_goal_metadata(
        self,
        goal: str,
        *,
        reason: Optional[str],
        existing: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        metadata = dict(existing or {})
        metadata.setdefault("parked_at", datetime.now(timezone.utc).isoformat())
        metadata["reason"] = reason or metadata.get("reason") or "deferred"
        metadata["wake_trigger"] = self._wake_trigger_for_park_reason(reason or metadata.get("reason"))
        metadata.setdefault(
            "source_artifact",
            goal if "/" in goal or "." in goal or ":" in goal else None,
        )
        return metadata

    def _is_resume_eligible_commitment(self, commitment: Any) -> bool:
        """Return True when a blocked commitment should auto-resume after recovery."""
        reason = str(commitment.meta.get("blocked_reason", "")).strip().lower()
        if is_executive_pause_reason(reason):
            if not is_known_executive_pause_reason(reason):
                self._trace(
                    "resume_with_unknown_executive_block_reason",
                    {
                        "blocked_reason": reason,
                        "commitment_id": str(commitment.commitment_id),
                    },
                )
                if self.tracer:
                    self.tracer.log(
                        EventKind.WARNING,
                        message="Unknown executive pause reason encountered during deferred resume.",
                        payload={
                            "commitment_id": str(commitment.commitment_id),
                            "blocked_reason": reason,
                        },
                    )
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
