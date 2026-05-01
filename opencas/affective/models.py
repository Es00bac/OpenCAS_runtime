"""Models for evidence-linked affective examination."""

from __future__ import annotations

from datetime import datetime, timezone
from enum import Enum
from typing import Any, Dict, Optional
from uuid import UUID, uuid4

from pydantic import BaseModel, Field

from opencas.somatic.models import AffectState, PrimaryEmotion


class AffectiveSourceType(str, Enum):
    """Concrete evidence surfaces that can be examined affectively."""

    USER_MESSAGE = "user_message"
    ASSISTANT_MESSAGE = "assistant_message"
    TOOL_RESULT = "tool_result"
    RETRIEVED_MEMORY = "retrieved_memory"
    WORK_RESULT = "work_result"
    CONTINUITY_GAP = "continuity_gap"
    CONSOLIDATION_EVENT = "consolidation_event"


class AffectiveTarget(str, Enum):
    """What the examined evidence is primarily about."""

    SELF = "self"
    USER = "user"
    PROJECT = "project"
    SYSTEM = "system"
    RELATIONSHIP = "relationship"
    MEMORY = "memory"
    TOOL = "tool"
    UNKNOWN = "unknown"


class AffectiveActionPressure(str, Enum):
    """Bounded next-action pressure derived from an examination."""

    VERIFY = "verify"
    ASK_CLARIFYING_QUESTION = "ask_clarifying_question"
    REPAIR_TRUST = "repair_trust"
    RESUME_COMMITMENT = "resume_commitment"
    REST = "rest"
    CONTINUE = "continue"
    REFRAME = "reframe"
    ARCHIVE_ONLY = "archive_only"


class AffectiveConsumedBy(str, Enum):
    """Consumer that has consumed an examination pressure."""

    NONE = "none"
    PROMPT = "prompt"
    RETRIEVAL = "retrieval"
    EXECUTIVE = "executive"
    DASHBOARD = "dashboard"


class AffectiveExamination(BaseModel):
    """A durable affective appraisal tied to one concrete source."""

    examination_id: UUID = Field(default_factory=uuid4)
    created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    session_id: Optional[str] = None
    source_type: AffectiveSourceType
    source_id: str
    source_excerpt: str = ""
    source_hash: str = ""
    target: AffectiveTarget = AffectiveTarget.UNKNOWN
    affect: AffectState = Field(
        default_factory=lambda: AffectState(primary_emotion=PrimaryEmotion.NEUTRAL)
    )
    intensity: float = Field(default=0.0, ge=0.0, le=1.0)
    confidence: float = Field(default=0.5, ge=0.0, le=1.0)
    action_pressure: AffectiveActionPressure = AffectiveActionPressure.ARCHIVE_ONLY
    bounded_reason: str = ""
    consumed_by: AffectiveConsumedBy = AffectiveConsumedBy.NONE
    expires_at: Optional[datetime] = None
    appraisal_version: str = "v1"
    meta: Dict[str, Any] = Field(default_factory=dict)

    def pressure_metadata(self) -> Dict[str, Any]:
        """Compact metadata safe to attach to tool/work outputs."""
        return {
            "examination_id": str(self.examination_id),
            "action_pressure": self.action_pressure.value,
            "primary_emotion": self.affect.primary_emotion.value,
            "confidence": self.confidence,
            "bounded_reason": self.bounded_reason,
            "already_recognized": bool(self.meta.get("already_recognized")),
        }
