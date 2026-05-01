"""Tests for salvage-first retry governor decisions."""

from datetime import datetime, timezone
from uuid import UUID, uuid4

from opencas.execution.models import AttemptOutcome, AttemptSalvagePacket, RetryMode
from opencas.execution.retry_governor import RetryGovernor


def _packet(
    *,
    divergence_signature: str,
    recommended_mode: RetryMode,
    canonical_artifact_path: str | None,
    attempt: int = 1,
    verification_digest: str | None = "verify-a",
    discovered_constraints: list[str] | None = None,
    unresolved_questions: list[str] | None = None,
) -> AttemptSalvagePacket:
    return AttemptSalvagePacket(
        packet_id=uuid4(),
        task_id=uuid4(),
        attempt=attempt,
        project_signature="chronicle-4246",
        project_id="loop-4246",
        objective="Continue Chronicle 4246 from the existing manuscript.",
        canonical_artifact_path=canonical_artifact_path,
        artifact_paths_touched=[canonical_artifact_path] if canonical_artifact_path else [],
        plan_digest="plan-a",
        execution_digest="exec-a",
        verification_digest=verification_digest,
        tool_signature="tool-a",
        divergence_signature=divergence_signature,
        outcome=AttemptOutcome.VERIFY_FAILED,
        partial_value="bridge scene drafted",
        discovered_constraints=discovered_constraints or ["still missing fallout scene"],
        unresolved_questions=unresolved_questions or [],
        best_next_step="Repair the remaining gap in the manuscript and rerun verification.",
        recommended_mode=recommended_mode,
        llm_spend_class="broad",
        created_at=datetime(2026, 4, 18, 12, 0, tzinfo=timezone.utc),
    )


def test_retry_governor_blocks_second_low_divergence_broad_retry() -> None:
    governor = RetryGovernor()
    task_id = uuid4()
    prior = _packet(
        divergence_signature="same-signature",
        recommended_mode=RetryMode.RESUME_EXISTING_ARTIFACT,
        canonical_artifact_path="workspace/Chronicles/4246/chronicle_4246.md",
        attempt=1,
    ).model_copy(update={"task_id": task_id, "packet_id": UUID("11111111-1111-1111-1111-111111111111")})
    current = _packet(
        divergence_signature="same-signature",
        recommended_mode=RetryMode.RESUME_EXISTING_ARTIFACT,
        canonical_artifact_path="workspace/Chronicles/4246/chronicle_4246.md",
        attempt=2,
    ).model_copy(update={"task_id": task_id, "packet_id": UUID("22222222-2222-2222-2222-222222222222")})

    decision = governor.decide(
        candidate=current,
        prior_packets=[prior],
        has_new_evidence=False,
        broad_attempt=True,
    )

    assert decision.allowed is False
    assert decision.mode == RetryMode.RESUME_EXISTING_ARTIFACT
    assert decision.reuse_packet_id == prior.packet_id


def test_retry_governor_allows_retry_when_new_evidence_exists() -> None:
    governor = RetryGovernor()
    task_id = uuid4()
    prior = _packet(
        divergence_signature="same-signature",
        recommended_mode=RetryMode.RESUME_EXISTING_ARTIFACT,
        canonical_artifact_path="workspace/Chronicles/4246/chronicle_4246.md",
        attempt=1,
        verification_digest="verify-a",
    ).model_copy(update={"task_id": task_id})
    current = _packet(
        divergence_signature="same-signature",
        recommended_mode=RetryMode.RESUME_EXISTING_ARTIFACT,
        canonical_artifact_path="workspace/Chronicles/4246/chronicle_4246.md",
        attempt=2,
        verification_digest="verify-b",
        discovered_constraints=["new continuity defect surfaced"],
    ).model_copy(update={"task_id": task_id})

    decision = governor.decide(
        candidate=current,
        prior_packets=[prior],
        has_new_evidence=True,
        broad_attempt=True,
    )

    assert decision.allowed is True
    assert decision.mode == RetryMode.CONTINUE_RETRY


def test_retry_governor_blocks_broad_no_progress_retry_even_with_changed_digest() -> None:
    governor = RetryGovernor()
    task_id = uuid4()
    prior = _packet(
        divergence_signature="prior-signature",
        recommended_mode=RetryMode.DETERMINISTIC_REVIEW,
        canonical_artifact_path=None,
        attempt=1,
    ).model_copy(
        update={
            "task_id": task_id,
            "meaningful_progress_signal": "no_meaningful_progress",
            "best_next_step": "Stop broad retry and perform deterministic review.",
        }
    )
    current = _packet(
        divergence_signature="changed-but-still-no-progress",
        recommended_mode=RetryMode.DETERMINISTIC_REVIEW,
        canonical_artifact_path=None,
        attempt=2,
        verification_digest="verify-b",
    ).model_copy(
        update={
            "task_id": task_id,
            "meaningful_progress_signal": "no_meaningful_progress",
            "best_next_step": "Stop broad retry and perform deterministic review.",
        }
    )

    decision = governor.decide(
        candidate=current,
        prior_packets=[prior],
        has_new_evidence=False,
        broad_attempt=True,
    )

    assert decision.allowed is False
    assert decision.mode == RetryMode.DETERMINISTIC_REVIEW
    assert decision.reuse_packet_id == prior.packet_id
    assert "no meaningful progress" in decision.reason
