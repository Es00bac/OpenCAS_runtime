from __future__ import annotations

from datetime import datetime, timezone

from opencas.recovery.models import ContinuityPacket, RecoveryCandidate, RecoveryCandidateKind, RecoveryClassification, RecoveryStrategy
from opencas.recovery.planner import RecoveryPlanner


def _candidate(payload: dict) -> RecoveryCandidate:
    return RecoveryCandidate(
        candidate_id="task:abc",
        kind=RecoveryCandidateKind.FAILED_TASK,
        title="Finish app",
        status="failed",
        updated_at=datetime.now(timezone.utc),
        evidence_refs=["task:abc"],
        payload=payload,
    )


def test_planner_uses_salvage_recommended_alternate_strategy() -> None:
    planner = RecoveryPlanner()
    plan = planner.plan(
        _candidate({"salvage": {"recommended_mode": "deterministic_review", "best_next_step": "Review artifact first."}}),
        RecoveryClassification("task:abc", "resume_now", "salvage available", evidence_refs=["task:abc"]),
    )

    assert plan.strategy == RecoveryStrategy.DETERMINISTIC_REVIEW
    assert plan.objective == "Review artifact first."


def test_planner_creates_continuity_then_resume_plan() -> None:
    planner = RecoveryPlanner()
    packet = ContinuityPacket(
        packet_id="continuity:abc",
        project_key="story",
        title="Story",
        project_type="creative_writing",
        canonical_artifact_paths=["workspace/story.md"],
        latest_completed_unit="chapter 2",
        current_status="chapter 3 missing",
        continuity_facts=["A left town."],
        unresolved_threads=[],
        next_concrete_action="Draft chapter 3.",
        completion_criteria=["chapter 3 draft exists"],
        evidence_refs=["artifact:workspace/story.md"],
    )

    plan = planner.plan(
        _candidate({"project_type": "creative_writing"}),
        RecoveryClassification("task:abc", "needs_artifact_continuity", "needs packet", evidence_refs=["task:abc"]),
        continuity_packet=packet,
    )

    assert plan.strategy == RecoveryStrategy.CREATE_CONTINUITY_THEN_RESUME
    assert plan.objective == "Draft chapter 3."
    assert plan.continuity_packet == packet


def test_planner_leaves_external_blocker_blocked() -> None:
    planner = RecoveryPlanner()
    plan = planner.plan(
        _candidate({"meta": {"blocker_type": "credential_required"}}),
        RecoveryClassification("task:abc", "unsafe_or_external_blocker", "OAuth expired", evidence_refs=["task:abc"]),
    )

    assert plan.strategy == RecoveryStrategy.LEAVE_BLOCKED_WITH_RECONSIDERATION
    assert plan.objective == "OAuth expired"
