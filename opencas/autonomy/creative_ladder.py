"""Creative ladder for OpenCAS.

Manages promotion and demotion of WorkObjects through stages based on
learned experience, semantic similarity, relevance, and capacity.
"""

from __future__ import annotations

import asyncio
import math
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

from opencas.embeddings import EmbeddingService
from opencas.execution.lifecycle import LifecycleStage, TaskLifecycleMachine
from opencas.relational import RelationalEngine
from opencas.telemetry import EventKind, Tracer

from .executive import ExecutiveState
from .models import WorkObject, WorkStage
from .work_store import WorkStore


_PROMOTION_THRESHOLDS: dict[WorkStage, float] = {
    WorkStage.SPARK: 0.30,
    WorkStage.NOTE: 0.40,
    WorkStage.ARTIFACT: 0.50,
    WorkStage.MICRO_TASK: 0.60,
    WorkStage.PROJECT_SEED: 0.70,
    WorkStage.PROJECT: 0.80,
    WorkStage.DURABLE_WORK_STREAM: 1.00,  # terminal
}

_STAGE_ORDER = list(_PROMOTION_THRESHOLDS.keys())


class CreativeLadder:
    """Evaluates and promotes work objects along the creative ladder."""

    def __init__(
        self,
        executive: ExecutiveState,
        embeddings: Optional[EmbeddingService] = None,
        tracer: Optional[Tracer] = None,
        work_store: Optional[WorkStore] = None,
        relational: Optional[RelationalEngine] = None,
        task_store=None,
    ) -> None:
        self.executive = executive
        self.embeddings = embeddings
        self.tracer = tracer
        self.work_store = work_store
        self.relational = relational
        self.task_store = task_store
        self._ladder: List[WorkObject] = []
        self._background_tasks: set[asyncio.Task[None]] = set()
        self._successful_project_hashes: set[str] = set()

    def add(self, work: WorkObject) -> None:
        """Introduce a new work object to the ladder."""
        self._ladder.append(work)
        self._sync_work(work)
        self._trace(
            EventKind.CREATIVE_PROMOTION,
            "added",
            {
                "work_id": str(work.work_id),
                "stage": work.stage.value,
                "ladder_count": len(self._ladder),
            },
        )

    def evaluate(self, work: WorkObject) -> float:
        """Compute promotion score for *work*."""
        score = 0.15
        reasons: List[str] = ["base=0.150"]

        # 1. Semantic similarity to prior successful work
        semantic_score = self._semantic_score(work)
        score += semantic_score
        reasons.append(f"semantic={semantic_score:.3f}")

        # 2. Relevance to current goals / intention
        relevance = self._relevance_boost(work)
        score += relevance
        reasons.append(f"relevance={relevance:.3f}")

        # 3. Access & engagement
        engagement = min(0.10, work.access_count * 0.02)
        score += engagement
        reasons.append(f"engagement={engagement:.3f}")

        # 4. Musubi boost for work aligned with shared goals
        musubi_boost = 0.0
        if self.relational:
            aligns = False
            text = work.content.lower()
            for goal in self.executive.active_goals:
                if goal.lower() in text:
                    aligns = True
                    break
            musubi_boost = self.relational.to_creative_boost(aligns_with_shared_goals=aligns)
            if musubi_boost:
                reasons.append(f"musubi_boost={musubi_boost:.3f}")
        score += musubi_boost

        # 5. Daydream alignment bonus
        daydream_bonus = 0.0
        if work.meta.get("origin") == "daydream":
            alignment_score = work.meta.get("alignment_score", 0.0)
            if alignment_score >= 0.6:
                daydream_bonus = 0.05
            elif alignment_score < 0.2:
                daydream_bonus = -0.05
            if daydream_bonus:
                reasons.append(f"daydream_bonus={daydream_bonus:.3f}")
        score += daydream_bonus

        # 6. Capacity & Somatic penalty for higher-stage promotion
        capacity_penalty = 0.0
        if self.executive.is_overloaded:
            capacity_penalty += 0.15
            reasons.append("overload_penalty=0.15")
        elif self.executive.capacity_remaining < 2:
            capacity_penalty += 0.05
            reasons.append("low_capacity_penalty=0.05")
            
        somatic_penalty = 0.0
        if getattr(self.executive, "somatic", None) and getattr(self.executive.somatic, "state", None):
            fatigue = self.executive.somatic.state.fatigue
            if fatigue > 0.8:
                somatic_penalty += 0.25
                reasons.append(f"severe_fatigue_penalty={somatic_penalty:.2f}")
            elif fatigue > 0.5:
                somatic_penalty += 0.10
                reasons.append(f"fatigue_penalty={somatic_penalty:.2f}")

        score -= (capacity_penalty + somatic_penalty)
        score = max(0.0, min(1.0, score))
        work.promotion_score = round(score, 3)
        work.updated_at = datetime.now(timezone.utc)
        work.meta["score_reasons"] = reasons
        return score

    def try_promote(self, work: WorkObject) -> bool:
        """Attempt to promote *work* if its score exceeds the next threshold."""
        self.evaluate(work)
        current_idx = _STAGE_ORDER.index(work.stage)
        if current_idx >= len(_STAGE_ORDER) - 1:
            return False  # terminal stage

        current_threshold = _PROMOTION_THRESHOLDS[work.stage]
        next_stage = _STAGE_ORDER[current_idx + 1]

        if work.promotion_score >= current_threshold:
            old_stage = work.stage
            work.stage = next_stage
            work.updated_at = datetime.now(timezone.utc)
            self._record_lifecycle_transition(
                work, old_stage, next_stage,
                f"promoted from {old_stage.value} to {next_stage.value}"
            )
            self._trace(
                EventKind.CREATIVE_PROMOTION,
                "promoted",
                {
                    "work_id": str(work.work_id),
                    "from": old_stage.value,
                    "to": next_stage.value,
                    "score": work.promotion_score,
                    "score_reasons": work.meta.get("score_reasons", []),
                    "ladder_count": len(self._ladder),
                },
            )
            self._sync_work(work)
            return True
        return False

    def try_demote(self, work: WorkObject) -> bool:
        """Demote stale work if score has fallen well below its stage threshold."""
        self.evaluate(work)
        current_idx = _STAGE_ORDER.index(work.stage)
        if current_idx == 0:
            return False

        prev_stage = _STAGE_ORDER[current_idx - 1]
        prev_threshold = _PROMOTION_THRESHOLDS[prev_stage]
        # Demote if score is significantly below previous threshold
        if work.promotion_score < prev_threshold - 0.15:
            old_stage = work.stage
            work.stage = prev_stage
            work.updated_at = datetime.now(timezone.utc)
            self._trace(
                EventKind.CREATIVE_PROMOTION,
                "demoted",
                {
                    "work_id": str(work.work_id),
                    "from": old_stage.value,
                    "to": prev_stage.value,
                    "score": work.promotion_score,
                    "score_reasons": work.meta.get("score_reasons", []),
                    "ladder_count": len(self._ladder),
                },
            )
            self._sync_work(work)
            return True
        return False

    def run_cycle(self) -> Dict[str, int]:
        """Evaluate all work objects and apply promotions/demotions."""
        promoted = 0
        demoted = 0
        for work in self._ladder:
            if self.try_promote(work):
                promoted += 1
            elif self.try_demote(work):
                demoted += 1
        return {"promoted": promoted, "demoted": demoted}

    def list_by_stage(self, stage: WorkStage) -> List[WorkObject]:
        return [w for w in self._ladder if w.stage == stage]

    def remove(self, work_id: str) -> bool:
        for idx, w in enumerate(self._ladder):
            if str(w.work_id) == work_id:
                self._ladder.pop(idx)
                self._delete_work(work_id)
                return True
        return False

    def record_success(self, work: WorkObject) -> None:
        """Mark a work object as successfully completed, storing its hash for future similarity."""
        if work.embedding_id:
            self._successful_project_hashes.add(work.embedding_id)
        self._trace(
            EventKind.CREATIVE_PROMOTION,
            "success_recorded",
            {
                "work_id": str(work.work_id),
                "stage": work.stage.value,
                "successful_project_hashes": len(self._successful_project_hashes),
            },
        )

    def _record_lifecycle_transition(
        self,
        work: WorkObject,
        from_stage: WorkStage,
        to_stage: WorkStage,
        reason: str,
    ) -> None:
        """Validate and persist a lifecycle transition for a promoted WorkObject."""
        stage_map = {
            WorkStage.SPARK: LifecycleStage.SPARK,
            WorkStage.NOTE: LifecycleStage.NOTE,
            WorkStage.ARTIFACT: LifecycleStage.ARTIFACT,
            WorkStage.MICRO_TASK: LifecycleStage.QUEUED,
            WorkStage.PROJECT_SEED: LifecycleStage.QUEUED,
            WorkStage.PROJECT: LifecycleStage.QUEUED,
            WorkStage.DURABLE_WORK_STREAM: LifecycleStage.DONE,
        }
        from_lifecycle = stage_map.get(from_stage)
        to_lifecycle = stage_map.get(to_stage)
        if from_lifecycle is None or to_lifecycle is None:
            return
        try:
            transition = TaskLifecycleMachine.transition(
                task_id=str(work.work_id),
                from_stage=from_lifecycle,
                to_stage=to_lifecycle,
                reason=reason,
            )
        except ValueError:
            return
        if self.task_store:
            try:
                import asyncio
                loop = asyncio.get_running_loop()

                async def _persist():
                    await self.task_store.record_lifecycle_transition(
                        transition_id=str(transition.transition_id),
                        task_id=transition.task_id,
                        from_stage=transition.from_stage.value,
                        to_stage=transition.to_stage.value,
                        reason=transition.reason,
                        context=transition.context,
                    )

                task = asyncio.create_task(_persist())
                self._background_tasks.add(task)
                task.add_done_callback(self._background_tasks.discard)
            except RuntimeError:
                pass

    def _semantic_score(self, work: WorkObject) -> float:
        """Return a similarity bonus if the work resembles prior successful projects."""
        if not work.embedding_id:
            return 0.0
        if work.embedding_id in self._successful_project_hashes:
            return 0.15
        if not self.embeddings:
            return 0.0
        # Future: compute actual cosine similarity against stored success vectors
        return 0.0

    def _relevance_boost(self, work: WorkObject) -> float:
        """Boost score when work aligns with current goals or intention."""
        boost = 0.0
        text = work.content.lower()
        for goal in self.executive.active_goals:
            if goal.lower() in text:
                boost += 0.25
        if self.executive.intention and self.executive.intention.lower() in text:
            boost += 0.25
        return min(0.50, boost)

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
            pass  # no event loop (sync test context); skip async save

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

    def _trace(self, kind: EventKind, event: str, payload: Dict[str, Any]) -> None:
        if self.tracer:
            self.tracer.log(
                kind,
                f"CreativeLadder: {event}",
                payload,
            )
