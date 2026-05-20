from __future__ import annotations

from datetime import datetime, timedelta, timezone
from typing import Any

from opencas.recovery.models import BlockerType, RecoveryCandidate, RecoveryCandidateKind, RecoveryClassification

_PROJECT_TYPES_REQUIRING_CONTINUITY = {"creative_writing", "software", "research", "self_directed"}
_STABLE_RECONSIDERATION_BASE = datetime(1970, 1, 1, tzinfo=timezone.utc)
_EXTERNAL_BLOCKERS = {
    "safety_policy": BlockerType.SAFETY_POLICY,
    "permission_required": BlockerType.PERMISSION_REQUIRED,
    "credential_required": BlockerType.CREDENTIAL_REQUIRED,
    "privacy_boundary": BlockerType.PRIVACY_BOUNDARY,
    "missing_input": BlockerType.MISSING_INPUT,
    "external_dependency": BlockerType.EXTERNAL_DEPENDENCY,
}


class RecoveryClassifier:
    def classify(self, candidate: RecoveryCandidate) -> RecoveryClassification:
        blocker = _blocker_type(candidate.payload)
        if blocker is not None:
            return RecoveryClassification.blocked(
                candidate_id=candidate.candidate_id,
                blocker_type=blocker,
                reason=_blocked_reason(candidate.payload),
                evidence_refs=candidate.evidence_refs,
                reconsider_after=_reconsider_after(candidate),
            )

        if candidate.kind == RecoveryCandidateKind.RECURRING_SCHEDULE and candidate.payload.get("next_run_at"):
            return RecoveryClassification(
                candidate_id=candidate.candidate_id,
                category="scheduled_or_recurring",
                reason="Recurring work has a next run time.",
                evidence_refs=candidate.evidence_refs,
            )

        if candidate.kind == RecoveryCandidateKind.FAILED_TASK and candidate.payload.get("salvage"):
            return RecoveryClassification(
                candidate_id=candidate.candidate_id,
                category="resume_now",
                reason="Failed task has salvage evidence for an alternate route.",
                evidence_refs=candidate.evidence_refs,
            )

        if _needs_continuity(candidate.payload):
            return RecoveryClassification(
                candidate_id=candidate.candidate_id,
                category="needs_artifact_continuity",
                reason="Artifact-backed project work needs a continuity packet before recovery.",
                evidence_refs=candidate.evidence_refs,
            )

        if candidate.kind in {
            RecoveryCandidateKind.ACTIVE_COMMITMENT,
            RecoveryCandidateKind.BLOCKED_COMMITMENT,
            RecoveryCandidateKind.PAUSED_OBJECTIVE_LOOP,
            RecoveryCandidateKind.FAILED_OBJECTIVE_LOOP,
            RecoveryCandidateKind.BLOCKED_WORK_OBJECT,
        }:
            return RecoveryClassification(
                candidate_id=candidate.candidate_id,
                category="resume_now",
                reason="Stopped work has no terminal completion evidence.",
                evidence_refs=candidate.evidence_refs,
            )

        return RecoveryClassification(
            candidate_id=candidate.candidate_id,
            category="needs_operator_only_if_unresolvable",
            reason="Candidate needs more evidence before safe recovery.",
            evidence_refs=candidate.evidence_refs,
        )


def _reconsider_after(candidate: RecoveryCandidate) -> datetime:
    return (candidate.updated_at or _STABLE_RECONSIDERATION_BASE) + timedelta(hours=6)


def _needs_continuity(payload: dict[str, Any]) -> bool:
    project_type = payload.get("project_type") or payload.get("meta", {}).get("project_type")
    if project_type not in _PROJECT_TYPES_REQUIRING_CONTINUITY:
        return False
    has_packet = bool(payload.get("continuity_packet_id") or payload.get("continuity_packet"))
    return not has_packet


def _blocker_type(payload: dict[str, Any]) -> BlockerType | None:
    raw = payload.get("blocker_type") or payload.get("meta", {}).get("blocker_type")
    if raw is None:
        return None
    return _EXTERNAL_BLOCKERS.get(str(raw))


def _blocked_reason(payload: dict[str, Any]) -> str:
    return str(
        payload.get("blocked_reason")
        or payload.get("meta", {}).get("blocked_reason")
        or "Recovery is blocked by an explicit external or safety boundary."
    )
