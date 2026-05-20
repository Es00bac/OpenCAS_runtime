"""Data models for peripheral thread and bead continuity."""

from __future__ import annotations

from datetime import datetime, timezone
from enum import Enum
from typing import Any

from pydantic import BaseModel, Field, field_validator, model_validator

from opencas.cognition import CognitionGrounding


class ThreadStatus(str, Enum):
    """Lifecycle state for an ongoing thread of attention."""

    ACTIVE = "active"
    PERIPHERAL = "peripheral"
    CLOSED = "closed"


class BeadStatus(str, Enum):
    """Lifecycle state for a finished-enough unit attached to a thread."""

    WORKING = "working"
    CANDIDATE = "candidate"
    PERIPHERAL = "peripheral"
    ACTIVE = "active"
    DANGLING = "dangling"
    SUPERSEDED = "superseded"
    REJECTED = "rejected"


class BeadSourceKind(str, Enum):
    """Source classes that can produce bead candidates."""

    AUTONOMOUS_ARTIFACT = "autonomous_artifact"
    DAYDREAM_REFLECTION = "daydream_reflection"
    DAYDREAM_SIGNAL = "daydream_signal"
    SELF_INSPECTION_DRIFT = "self_inspection_drift"
    TOOL_CALL_TRANSIT = "tool_call_transit"
    FASCINATION = "fascination"
    COMMITMENT_GAP = "commitment_gap"
    SHADOW_INTENTION = "shadow_intention"
    SUPPRESSED_REFRAME = "suppressed_reframe"
    PROJECT_SALVAGE = "project_salvage"
    MANUAL = "manual"


class ThreadAnchor(BaseModel):
    """A durable named thread that can receive autonomous or manual beads."""

    anchor_id: str
    title: str
    kind: str = "theme"
    status: ThreadStatus = ThreadStatus.PERIPHERAL
    created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    updated_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    grounding: list[CognitionGrounding] = Field(default_factory=list)
    meta: dict[str, Any] = Field(default_factory=dict)

    @field_validator("anchor_id", "title", "kind")
    @classmethod
    def _normalize_text(cls, value: str) -> str:
        return " ".join(str(value or "").split())

    @model_validator(mode="after")
    def _require_identity(self) -> "ThreadAnchor":
        if not self.anchor_id:
            raise ValueError("anchor_id is required")
        if not self.title:
            raise ValueError("title is required")
        return self


class BeadValidationResult(BaseModel):
    """Validation result for releasing a bead from active attention."""

    complete_fields: bool
    anchor_resolved: bool
    pickup_intelligible: bool
    dangling: bool
    should_create_task: bool = False
    reasons: list[str] = Field(default_factory=list)
    confidence: float = Field(default=0.5, ge=0.0, le=1.0)

    @field_validator("reasons", mode="before")
    @classmethod
    def _normalize_reasons(cls, value: Any) -> list[str]:
        if not isinstance(value, list):
            return []
        return [" ".join(str(item or "").split()) for item in value if str(item or "").strip()]


class BeadEntry(BaseModel):
    """A finished-enough unit of meaning attached to a thread anchor."""

    bead_id: str
    thread_anchor_id: str
    title: str
    summary: str
    source_kind: BeadSourceKind
    source_ref: str
    content_hash_full: str
    content_hash_short: str
    status: BeadStatus = BeadStatus.CANDIDATE
    user_commissioned: bool = False
    created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    committed_at: datetime | None = None
    operator_signature: str = "thread_registry_v1"
    validation: BeadValidationResult | None = None
    grounding: list[CognitionGrounding] = Field(default_factory=list)
    meta: dict[str, Any] = Field(default_factory=dict)

    @field_validator("bead_id", "thread_anchor_id", "title", "summary", "source_ref")
    @classmethod
    def _normalize_text(cls, value: str) -> str:
        return " ".join(str(value or "").split())

    @model_validator(mode="after")
    def _validate_hash_contract(self) -> "BeadEntry":
        if len(self.content_hash_full) != 64:
            raise ValueError("content_hash_full must be a full SHA-256 hex digest")
        if self.content_hash_short != self.content_hash_full[:12]:
            raise ValueError("content_hash_short must be the first 12 chars of content_hash_full")
        return self
