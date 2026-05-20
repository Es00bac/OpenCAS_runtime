"""Models for linking claims to runtime evidence."""

from __future__ import annotations

from datetime import datetime, timezone
from enum import Enum
from typing import Any
from uuid import UUID, uuid4

from pydantic import BaseModel, Field, field_validator


class ProofClaimType(str, Enum):
    """Types of claims that can require proof."""

    ACTION = "action"
    PROMISE = "promise"
    CAPABILITY = "capability"
    SCHEDULE = "schedule"
    PROJECT = "project"
    MAINTENANCE = "maintenance"
    DREAM = "dream"


class ProofClaimStatus(str, Enum):
    """Lifecycle state of a proof claim."""

    UNVERIFIED = "unverified"
    EVIDENCE_LINKED = "evidence_linked"
    VERIFIED = "verified"
    CONTRADICTED = "contradicted"


class ProofEvidenceKind(str, Enum):
    """Kinds of evidence that can support or contradict a claim."""

    COMMITMENT = "commitment"
    SCHEDULE = "schedule"
    RECEIPT = "receipt"
    TOOL_RESULT = "tool_result"
    ARTIFACT = "artifact"
    MEMORY = "memory"
    MAINTENANCE_OUTCOME = "maintenance_outcome"
    DREAM_RECORD = "dream_record"
    RUNTIME_EVENT = "runtime_event"


class ProofEvidenceLink(BaseModel):
    """One evidence item linked to a claim."""

    link_id: UUID = Field(default_factory=uuid4)
    created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    evidence_kind: ProofEvidenceKind
    evidence_id: str
    summary: str = ""
    supports_claim: bool = True
    confidence: float = Field(default=0.8, ge=0.0, le=1.0)
    meta: dict[str, Any] = Field(default_factory=dict)

    @field_validator("evidence_id", "summary")
    @classmethod
    def _normalize_text(cls, value: str) -> str:
        return " ".join(str(value or "").split())

    @field_validator("confidence", mode="before")
    @classmethod
    def _clamp_confidence(cls, value: Any) -> float:
        try:
            parsed = float(value)
        except (TypeError, ValueError):
            return 0.8
        return round(max(0.0, min(1.0, parsed)), 3)


class ProofClaim(BaseModel):
    """A claim plus links to evidence that can prove or falsify it."""

    claim_id: UUID = Field(default_factory=uuid4)
    created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    claim_type: ProofClaimType
    claim: str
    subject: str = ""
    status: ProofClaimStatus = ProofClaimStatus.UNVERIFIED
    evidence_links: list[ProofEvidenceLink] = Field(default_factory=list)
    meta: dict[str, Any] = Field(default_factory=dict)

    @field_validator("claim", "subject")
    @classmethod
    def _normalize_text(cls, value: str) -> str:
        return " ".join(str(value or "").split())

