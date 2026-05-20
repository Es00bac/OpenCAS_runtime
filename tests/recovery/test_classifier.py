from __future__ import annotations

from datetime import datetime, timedelta, timezone

from opencas.recovery.classifier import RecoveryClassifier
from opencas.recovery.models import RecoveryCandidate, RecoveryCandidateKind


def _candidate(kind: RecoveryCandidateKind, payload: dict) -> RecoveryCandidate:
    return RecoveryCandidate(
        candidate_id="candidate:1",
        kind=kind,
        title="Candidate",
        status="stopped",
        updated_at=datetime.now(timezone.utc),
        evidence_refs=["candidate:1"],
        payload=payload,
    )


def test_classifier_treats_scheduled_recurring_work_as_not_failed() -> None:
    classifier = RecoveryClassifier()
    classification = classifier.classify(
        _candidate(
            RecoveryCandidateKind.RECURRING_SCHEDULE,
            {"next_run_at": "2026-05-17T08:00:00+00:00", "recent_success": True},
        )
    )

    assert classification.category == "scheduled_or_recurring"


def test_classifier_requires_continuity_for_creative_work_before_resume() -> None:
    classifier = RecoveryClassifier()
    classification = classifier.classify(
        _candidate(
            RecoveryCandidateKind.PAUSED_OBJECTIVE_LOOP,
            {"meta": {"project_type": "creative_writing"}, "canonical_artifact_paths": ["workspace/writing/story.md"]},
        )
    )

    assert classification.category == "needs_artifact_continuity"


def test_classifier_requires_continuity_for_software_work_before_resume() -> None:
    classifier = RecoveryClassifier()
    classification = classifier.classify(
        _candidate(
            RecoveryCandidateKind.FAILED_TASK,
            {"project_type": "software", "artifact_paths_touched": ["opencas/app.py"]},
        )
    )

    assert classification.category == "needs_artifact_continuity"


def test_classifier_resumes_failed_task_with_salvage_packet() -> None:
    classifier = RecoveryClassifier()
    classification = classifier.classify(
        _candidate(
            RecoveryCandidateKind.FAILED_TASK,
            {"salvage": {"best_next_step": "Run deterministic review.", "recommended_mode": "deterministic_review"}},
        )
    )

    assert classification.category == "resume_now"
    assert "salvage" in classification.reason


def test_classifier_salvage_takes_precedence_over_software_continuity() -> None:
    classifier = RecoveryClassifier()
    classification = classifier.classify(
        _candidate(
            RecoveryCandidateKind.FAILED_TASK,
            {
                "project_type": "software",
                "artifact_paths_touched": ["opencas/app.py"],
                "salvage": {"best_next_step": "Run deterministic review.", "recommended_mode": "deterministic_review"},
            },
        )
    )

    assert classification.category == "resume_now"
    assert "salvage" in classification.reason


def test_classifier_keeps_real_external_blocker_blocked() -> None:
    classifier = RecoveryClassifier()
    candidate = _candidate(
        RecoveryCandidateKind.BLOCKED_COMMITMENT,
        {"meta": {"blocker_type": "credential_required", "blocked_reason": "OAuth token expired"}},
    )
    classification = classifier.classify(candidate)

    assert classification.category == "unsafe_or_external_blocker"
    assert classification.blocker_type.value == "credential_required"
    assert classification.reconsider_after == candidate.updated_at + timedelta(hours=6)


def test_classifier_returns_equal_classifications_for_same_blocker_candidate() -> None:
    classifier = RecoveryClassifier()
    candidate = RecoveryCandidate(
        candidate_id="candidate:blocked:no-updated-at",
        kind=RecoveryCandidateKind.BLOCKED_COMMITMENT,
        title="Candidate",
        status="blocked",
        updated_at=None,
        evidence_refs=["candidate:blocked:no-updated-at"],
        payload={"meta": {"blocker_type": "credential_required", "blocked_reason": "OAuth token expired"}},
    )

    assert classifier.classify(candidate) == classifier.classify(candidate)
