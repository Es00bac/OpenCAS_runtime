from __future__ import annotations

from datetime import datetime, timedelta, timezone
from typing import Any
from uuid import uuid4

from opencas.execution.models import RepairTask
from opencas.recovery.models import RecoveryDecision, RecoveryPlan, RecoveryStrategy


class RecoveryExecutorAdapter:
    def __init__(self, *, baa: Any = None, harness: Any = None) -> None:
        self.baa = baa
        self.harness = harness

    async def execute(self, plan: RecoveryPlan) -> RecoveryDecision:
        decision_id = f"recovery:{uuid4()}"
        if plan.strategy == RecoveryStrategy.LEAVE_BLOCKED_WITH_RECONSIDERATION:
            return RecoveryDecision(
                decision_id=decision_id,
                candidate_id=plan.candidate_id,
                classification="unsafe_or_external_blocker",
                strategy=plan.strategy.value,
                result="blocked",
                evidence_refs=plan.evidence_refs,
                next_reconsideration_at=datetime.now(timezone.utc) + timedelta(hours=6),
            )

        if plan.strategy == RecoveryStrategy.CONFIRM_SCHEDULED_RECURRENCE:
            return RecoveryDecision(
                decision_id=decision_id,
                candidate_id=plan.candidate_id,
                classification="scheduled_or_recurring",
                strategy=plan.strategy.value,
                result="scheduled",
                evidence_refs=plan.evidence_refs,
                next_reconsideration_at=datetime.now(timezone.utc) + timedelta(hours=6),
            )

        if plan.strategy == RecoveryStrategy.CREATE_CONTINUITY_THEN_RESUME and self.harness is not None:
            loop = await self.harness.create_objective_loop(
                title=f"Recovery: {plan.objective}",
                description=plan.objective,
                completion_criteria=plan.continuity_packet.completion_criteria if plan.continuity_packet else [],
                meta={
                    "recovery_candidate_id": plan.candidate_id,
                    "recovery_strategy": plan.strategy.value,
                    "evidence_refs": plan.evidence_refs,
                },
            )
            return RecoveryDecision(
                decision_id=decision_id,
                candidate_id=plan.candidate_id,
                classification="needs_artifact_continuity",
                strategy=plan.strategy.value,
                result="submitted",
                created_loop_id=str(loop.loop_id),
                continuity_packet_id=plan.continuity_packet.packet_id if plan.continuity_packet else None,
                evidence_refs=plan.evidence_refs,
            )

        if self.baa is None:
            return RecoveryDecision(
                decision_id=decision_id,
                candidate_id=plan.candidate_id,
                classification="resume_now",
                strategy=plan.strategy.value,
                result="no_executor_available",
                evidence_refs=plan.evidence_refs,
                next_reconsideration_at=datetime.now(timezone.utc) + timedelta(hours=1),
            )

        metadata = {
            "recovery_candidate_id": plan.candidate_id,
            "recovery_strategy": plan.strategy.value,
            "evidence_refs": plan.evidence_refs,
        }
        task = RepairTask(
            task_id=uuid4(),
            objective=plan.objective,
            meta=metadata,
        )
        object.__setattr__(task, "description", task.objective)
        object.__setattr__(task, "metadata", task.meta)
        await self.baa.submit(task)
        return RecoveryDecision(
            decision_id=decision_id,
            candidate_id=plan.candidate_id,
            classification="resume_now",
            strategy=plan.strategy.value,
            result="submitted",
            created_task_id=str(task.task_id),
            evidence_refs=plan.evidence_refs,
        )
