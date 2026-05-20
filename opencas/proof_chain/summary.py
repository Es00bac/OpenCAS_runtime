"""Summary helpers for proof-chain diagnostics."""

from __future__ import annotations

from typing import Any

from .models import ProofClaim, ProofClaimStatus, ProofEvidenceKind

STRONG_EVIDENCE_KINDS = {
    ProofEvidenceKind.RECEIPT.value,
    ProofEvidenceKind.TOOL_RESULT.value,
    ProofEvidenceKind.ARTIFACT.value,
    ProofEvidenceKind.MAINTENANCE_OUTCOME.value,
    ProofEvidenceKind.DREAM_RECORD.value,
}


def summarize_claims(claims: list[ProofClaim]) -> dict[str, Any]:
    """Summarize proof claims without confusing status with evidence presence."""

    status_counts: dict[str, int] = {}
    claims_with_evidence = 0
    claims_with_supporting_evidence = 0
    claims_with_strong_evidence = 0
    verified_with_evidence = 0
    verified_without_evidence = 0
    contradicted_with_evidence = 0

    for claim in claims:
        status = claim.status.value
        status_counts[status] = status_counts.get(status, 0) + 1
        links = list(claim.evidence_links or [])
        if links:
            claims_with_evidence += 1
        if any(link.supports_claim for link in links):
            claims_with_supporting_evidence += 1
        if any(link.evidence_kind.value in STRONG_EVIDENCE_KINDS for link in links):
            claims_with_strong_evidence += 1
        if claim.status == ProofClaimStatus.VERIFIED:
            if links:
                verified_with_evidence += 1
            else:
                verified_without_evidence += 1
        if claim.status == ProofClaimStatus.CONTRADICTED and links:
            contradicted_with_evidence += 1

    return {
        "recent_count": len(claims),
        "status_counts": status_counts,
        "unverified_count": status_counts.get(ProofClaimStatus.UNVERIFIED.value, 0),
        # Historical field name, now correctly meaning actual linked evidence.
        "evidence_linked_count": claims_with_evidence,
        "status_evidence_linked_count": status_counts.get(
            ProofClaimStatus.EVIDENCE_LINKED.value,
            0,
        ),
        "verified_count": status_counts.get(ProofClaimStatus.VERIFIED.value, 0),
        "contradicted_count": status_counts.get(ProofClaimStatus.CONTRADICTED.value, 0),
        "claims_with_evidence_count": claims_with_evidence,
        "claims_with_supporting_evidence_count": claims_with_supporting_evidence,
        "claims_with_strong_evidence_count": claims_with_strong_evidence,
        "verified_with_evidence_count": verified_with_evidence,
        "verified_without_evidence_count": verified_without_evidence,
        "contradicted_with_evidence_count": contradicted_with_evidence,
    }
