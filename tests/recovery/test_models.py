from __future__ import annotations

from datetime import datetime, timedelta, timezone

from opencas.recovery.models import (
    BlockerType,
    ContinuityPacket,
    RecoveryCandidate,
    RecoveryCandidateKind,
    RecoveryClassification,
    RecoveryDecision,
    RecoveryPlan,
    RecoveryStrategy,
)


def test_recovery_candidate_evidence_key_is_stable() -> None:
    candidate = RecoveryCandidate(
        candidate_id="task:abc",
        kind=RecoveryCandidateKind.FAILED_TASK,
        title="Write the app",
        status="failed",
        updated_at=datetime(2026, 5, 16, tzinfo=timezone.utc),
        evidence_refs=["task:abc", "salvage:def"],
        payload={"objective": "Write the app"},
    )

    assert candidate.evidence_key == "failed_task:task:abc"
    assert candidate.has_evidence_ref("salvage:def")


def test_blocked_classification_requires_blocker_evidence_and_reconsideration() -> None:
    reconsider_at = datetime.now(timezone.utc) + timedelta(hours=6)

    classification = RecoveryClassification.blocked(
        candidate_id="commitment:abc",
        blocker_type=BlockerType.MISSING_INPUT,
        reason="The next scene target is not inferable from artifacts.",
        evidence_refs=["commitment:abc"],
        reconsider_after=reconsider_at,
    )

    assert classification.category == "unsafe_or_external_blocker"
    assert classification.blocker_type == BlockerType.MISSING_INPUT
    assert classification.evidence_refs == ["commitment:abc"]
    assert classification.reconsider_after == reconsider_at


def test_recovery_plan_links_candidate_packet_and_strategy() -> None:
    packet = ContinuityPacket(
        packet_id="continuity:story:1",
        project_key="story-1",
        title="Creative writing project",
        project_type="creative_writing",
        canonical_artifact_paths=["workspace/writing/story.md"],
        latest_completed_unit="chapter 2",
        current_status="chapter 3 not started",
        continuity_facts=["Character A left town in chapter 2."],
        unresolved_threads=["Character B has not learned the truth."],
        next_concrete_action="Draft chapter 3 opening scene.",
        completion_criteria=["Chapter 3 draft exists", "Continuity facts are preserved"],
        evidence_refs=["artifact:workspace/writing/story.md"],
    )

    plan = RecoveryPlan(
        candidate_id="loop:abc",
        strategy=RecoveryStrategy.CREATE_CONTINUITY_THEN_RESUME,
        objective="Draft chapter 3 opening scene.",
        continuity_packet=packet,
        evidence_refs=["loop:abc", packet.packet_id],
    )

    assert plan.strategy == RecoveryStrategy.CREATE_CONTINUITY_THEN_RESUME
    assert plan.continuity_packet is packet
    assert plan.evidence_refs == ["loop:abc", "continuity:story:1"]


def test_recovery_decision_records_no_progress_reconsideration() -> None:
    reconsider_at = datetime.now(timezone.utc) + timedelta(hours=12)

    decision = RecoveryDecision(
        decision_id="decision:1",
        candidate_id="task:abc",
        classification="resume_now",
        strategy="deterministic_review",
        result="submitted",
        created_task_id="task:new",
        evidence_refs=["task:abc"],
        next_reconsideration_at=reconsider_at,
    )

    assert decision.next_reconsideration_at == reconsider_at
    assert decision.created_task_id == "task:new"
