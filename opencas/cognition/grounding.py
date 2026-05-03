"""Evidence contracts for grounded cognition and generated synthesis."""

from __future__ import annotations

from datetime import datetime, timezone
from enum import Enum
from typing import Any

from pydantic import BaseModel, Field, field_validator


class GroundingKind(str, Enum):
    """Kinds of claims OpenCAS can surface internally or externally."""

    OBSERVED = "observed"
    LEARNED = "learned"
    DERIVED = "derived"
    SEEDED = "seeded"
    GENERATED_SYNTHESIS = "generated_synthesis"
    POLICY = "policy"
    SOMATIC_METRIC = "somatic_metric"
    RELATIONAL_METRIC = "relational_metric"
    OUTCOME = "outcome"


class GroundingSource(str, Enum):
    """Source systems that can support a grounded cognition claim."""

    MEMORY = "memory"
    SOMATIC = "somatic"
    RELATIONAL = "relational"
    IDENTITY = "identity"
    DAYDREAM = "daydream"
    OUTCOME = "outcome"
    POLICY = "policy"
    RUNTIME = "runtime"
    USER = "user"


class CognitionGrounding(BaseModel):
    """A compact evidence record for inner-life and continuity claims."""

    kind: GroundingKind
    source: GroundingSource = GroundingSource.RUNTIME
    claim: str
    subject: str = ""
    confidence: float = Field(default=1.0, ge=0.0, le=1.0)
    evidence_ids: list[str] = Field(default_factory=list)
    timestamp: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    allowed_surface: str = "internal"
    meta: dict[str, Any] = Field(default_factory=dict)

    @field_validator("confidence", mode="before")
    @classmethod
    def _clamp_confidence(cls, value: Any) -> float:
        try:
            parsed = float(value)
        except (TypeError, ValueError):
            return 1.0
        return max(0.0, min(1.0, parsed))

    @field_validator("claim", "subject", "allowed_surface")
    @classmethod
    def _normalize_text(cls, value: str) -> str:
        return " ".join(str(value or "").split())

    @field_validator("evidence_ids", mode="before")
    @classmethod
    def _normalize_evidence_ids(cls, value: Any) -> list[str]:
        if not isinstance(value, list):
            return []
        return [str(item) for item in value if str(item or "").strip()]
