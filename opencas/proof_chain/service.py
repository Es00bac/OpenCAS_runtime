"""Service for recording and linking verifiable claims."""

from __future__ import annotations

from uuid import UUID

from .models import (
    ProofClaim,
    ProofClaimStatus,
    ProofClaimType,
    ProofEvidenceKind,
    ProofEvidenceLink,
)
from .store import ProofStore


class ProofChainService:
    """Record claims and attach evidence without inventing retrospective proof."""

    def __init__(self, store: ProofStore) -> None:
        self.store = store

    async def record_claim(
        self,
        *,
        claim_type: ProofClaimType | str,
        claim: str,
        subject: str = "",
        meta: dict | None = None,
    ) -> ProofClaim:
        claim_obj = ProofClaim(
            claim_type=(
                claim_type
                if isinstance(claim_type, ProofClaimType)
                else ProofClaimType(str(claim_type))
            ),
            claim=claim,
            subject=subject,
            meta=dict(meta or {}),
        )
        await self.store.save_claim(claim_obj)
        return claim_obj

    async def link_evidence(
        self,
        claim_id: UUID | str,
        *,
        evidence_kind: ProofEvidenceKind | str,
        evidence_id: str,
        summary: str = "",
        supports_claim: bool = True,
        confidence: float = 0.8,
        meta: dict | None = None,
    ) -> ProofClaim:
        claim = await self.store.get_claim(str(claim_id))
        if claim is None:
            raise KeyError(f"proof claim not found: {claim_id}")
        link = ProofEvidenceLink(
            evidence_kind=(
                evidence_kind
                if isinstance(evidence_kind, ProofEvidenceKind)
                else ProofEvidenceKind(str(evidence_kind))
            ),
            evidence_id=evidence_id,
            summary=summary,
            supports_claim=supports_claim,
            confidence=confidence,
            meta=dict(meta or {}),
        )
        claim.evidence_links.append(link)
        claim.status = _status_for_evidence(claim, link)
        await self.store.save_claim(claim)
        return claim


def _status_for_evidence(
    claim: ProofClaim,
    link: ProofEvidenceLink,
) -> ProofClaimStatus:
    if not link.supports_claim:
        return ProofClaimStatus.CONTRADICTED
    if claim.claim_type == ProofClaimType.PROMISE:
        return ProofClaimStatus.EVIDENCE_LINKED
    if link.evidence_kind in {
        ProofEvidenceKind.RECEIPT,
        ProofEvidenceKind.TOOL_RESULT,
        ProofEvidenceKind.ARTIFACT,
        ProofEvidenceKind.MAINTENANCE_OUTCOME,
        ProofEvidenceKind.DREAM_RECORD,
    }:
        return ProofClaimStatus.VERIFIED
    return ProofClaimStatus.EVIDENCE_LINKED

