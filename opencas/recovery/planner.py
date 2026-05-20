from __future__ import annotations

from opencas.recovery.models import ContinuityPacket, RecoveryCandidate, RecoveryClassification, RecoveryPlan, RecoveryStrategy

_SALVAGE_STRATEGY_MAP = {
    "resume_existing_artifact": RecoveryStrategy.RESUME_EXISTING_ARTIFACT,
    "narrow_edit": RecoveryStrategy.NARROW_REPAIR_TASK,
    "deterministic_review": RecoveryStrategy.DETERMINISTIC_REVIEW,
}


class RecoveryPlanner:
    def plan(
        self,
        candidate: RecoveryCandidate,
        classification: RecoveryClassification,
        *,
        continuity_packet: ContinuityPacket | None = None,
    ) -> RecoveryPlan:
        if classification.category == "unsafe_or_external_blocker":
            return RecoveryPlan(
                candidate_id=candidate.candidate_id,
                strategy=RecoveryStrategy.LEAVE_BLOCKED_WITH_RECONSIDERATION,
                objective=classification.reason,
                evidence_refs=classification.evidence_refs,
            )

        if classification.category == "scheduled_or_recurring":
            return RecoveryPlan(
                candidate_id=candidate.candidate_id,
                strategy=RecoveryStrategy.CONFIRM_SCHEDULED_RECURRENCE,
                objective="Confirm recurring work remains scheduled.",
                evidence_refs=classification.evidence_refs,
            )

        if classification.category == "needs_artifact_continuity" and continuity_packet is not None:
            return RecoveryPlan(
                candidate_id=candidate.candidate_id,
                strategy=RecoveryStrategy.CREATE_CONTINUITY_THEN_RESUME,
                objective=continuity_packet.next_concrete_action,
                continuity_packet=continuity_packet,
                evidence_refs=[*classification.evidence_refs, continuity_packet.packet_id],
            )

        salvage = candidate.payload.get("salvage") or {}
        strategy = _SALVAGE_STRATEGY_MAP.get(str(salvage.get("recommended_mode")), RecoveryStrategy.CONVERT_CAPTURE_TO_OBJECTIVE)
        objective = str(salvage.get("best_next_step") or candidate.payload.get("next_concrete_action") or candidate.title)
        return RecoveryPlan(
            candidate_id=candidate.candidate_id,
            strategy=strategy,
            objective=objective,
            evidence_refs=classification.evidence_refs,
        )
