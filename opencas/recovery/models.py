from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any


class RecoveryCandidateKind(str, Enum):
    ACTIVE_COMMITMENT = "active_commitment"
    BLOCKED_COMMITMENT = "blocked_commitment"
    FAILED_TASK = "failed_task"
    PAUSED_OBJECTIVE_LOOP = "paused_objective_loop"
    FAILED_OBJECTIVE_LOOP = "failed_objective_loop"
    BLOCKED_WORK_OBJECT = "blocked_work_object"
    RECURRING_SCHEDULE = "recurring_schedule"


class RecoveryStrategy(str, Enum):
    RESUME_EXISTING_ARTIFACT = "resume_existing_artifact"
    NARROW_REPAIR_TASK = "narrow_repair_task"
    DETERMINISTIC_REVIEW = "deterministic_review"
    CREATE_CONTINUITY_THEN_RESUME = "create_continuity_then_resume"
    CONVERT_CAPTURE_TO_OBJECTIVE = "convert_capture_to_objective"
    CONFIRM_SCHEDULED_RECURRENCE = "confirm_scheduled_recurrence"
    LEAVE_BLOCKED_WITH_RECONSIDERATION = "leave_blocked_with_reconsideration"


class BlockerType(str, Enum):
    SAFETY_POLICY = "safety_policy"
    PERMISSION_REQUIRED = "permission_required"
    CREDENTIAL_REQUIRED = "credential_required"
    PRIVACY_BOUNDARY = "privacy_boundary"
    MISSING_INPUT = "missing_input"
    EXTERNAL_DEPENDENCY = "external_dependency"


@dataclass(frozen=True)
class RecoveryCandidate:
    candidate_id: str
    kind: RecoveryCandidateKind
    title: str
    status: str
    updated_at: datetime | None
    evidence_refs: list[str] = field(default_factory=list)
    payload: dict[str, Any] = field(default_factory=dict)

    @property
    def evidence_key(self) -> str:
        return f"{self.kind.value}:{self.candidate_id}"

    def has_evidence_ref(self, ref: str) -> bool:
        return ref in self.evidence_refs


@dataclass(frozen=True)
class RecoveryClassification:
    candidate_id: str
    category: str
    reason: str
    blocker_type: BlockerType | None = None
    evidence_refs: list[str] = field(default_factory=list)
    reconsider_after: datetime | None = None

    @classmethod
    def blocked(
        cls,
        *,
        candidate_id: str,
        blocker_type: BlockerType,
        reason: str,
        evidence_refs: list[str],
        reconsider_after: datetime,
    ) -> "RecoveryClassification":
        return cls(
            candidate_id=candidate_id,
            category="unsafe_or_external_blocker",
            reason=reason,
            blocker_type=blocker_type,
            evidence_refs=evidence_refs,
            reconsider_after=reconsider_after,
        )


@dataclass(frozen=True)
class ContinuityPacket:
    packet_id: str
    project_key: str
    title: str
    project_type: str
    canonical_artifact_paths: list[str]
    latest_completed_unit: str
    current_status: str
    continuity_facts: list[str]
    unresolved_threads: list[str]
    next_concrete_action: str
    completion_criteria: list[str]
    evidence_refs: list[str]


@dataclass(frozen=True)
class RecoveryPlan:
    candidate_id: str
    strategy: RecoveryStrategy
    objective: str
    continuity_packet: ContinuityPacket | None = None
    evidence_refs: list[str] = field(default_factory=list)


@dataclass(frozen=True)
class RecoveryDecision:
    decision_id: str
    candidate_id: str
    classification: str
    strategy: str
    result: str
    created_task_id: str | None = None
    created_loop_id: str | None = None
    continuity_packet_id: str | None = None
    evidence_refs: list[str] = field(default_factory=list)
    next_reconsideration_at: datetime | None = None


@dataclass(frozen=True)
class RecoverySummary:
    scanned: int = 0
    resume_now: int = 0
    scheduled_or_recurring: int = 0
    duplicate_or_superseded: int = 0
    needs_artifact_continuity: int = 0
    unsafe_or_external_blocker: int = 0
    submitted: int = 0
    skipped_by_ledger: int = 0
