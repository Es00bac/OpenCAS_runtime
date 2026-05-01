"""Policy for salvage-first retry decisions."""

from __future__ import annotations

from .models import AttemptSalvagePacket, RetryDecision, RetryMode


class RetryGovernor:
    """Block low-divergence broad retries when no new evidence appeared."""

    def decide(
        self,
        *,
        candidate: AttemptSalvagePacket,
        prior_packets: list[AttemptSalvagePacket],
        has_new_evidence: bool,
        broad_attempt: bool,
    ) -> RetryDecision:
        latest = prior_packets[-1] if prior_packets else None
        if latest is None:
            return RetryDecision(
                allowed=True,
                reason="first attempt for this task",
                mode=RetryMode.CONTINUE_RETRY,
            )

        same_divergence = latest.divergence_signature == candidate.divergence_signature
        same_artifact = latest.canonical_artifact_path == candidate.canonical_artifact_path

        if (
            broad_attempt
            and not has_new_evidence
            and candidate.meaningful_progress_signal == "no_meaningful_progress"
        ):
            return RetryDecision(
                allowed=False,
                reason="no meaningful progress broad retry without new evidence",
                mode=_blocked_mode(candidate),
                reuse_packet_id=latest.packet_id,
            )

        if broad_attempt and same_divergence and same_artifact and not has_new_evidence:
            return RetryDecision(
                allowed=False,
                reason="low-divergence broad retry without new evidence",
                mode=_blocked_mode(candidate),
                reuse_packet_id=latest.packet_id,
            )

        return RetryDecision(
            allowed=True,
            reason="retry meaningfully diverged or introduced new evidence",
            mode=RetryMode.CONTINUE_RETRY,
            reuse_packet_id=latest.packet_id,
        )


def _blocked_mode(packet: AttemptSalvagePacket) -> RetryMode:
    if packet.canonical_artifact_path:
        return RetryMode.RESUME_EXISTING_ARTIFACT
    return RetryMode.DETERMINISTIC_REVIEW
