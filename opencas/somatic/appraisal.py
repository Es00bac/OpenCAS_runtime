"""Somatic appraisal event taxonomy for OpenCAS."""

from datetime import datetime, timezone
from enum import Enum
from typing import Any, Dict, Optional
from uuid import UUID, uuid4

from pydantic import BaseModel, Field

from .models import AffectState


class AppraisalEventType(str, Enum):
    """Typed appraisal events that can trigger somatic state updates."""

    USER_INPUT_RECEIVED = "user_input_received"
    SELF_RESPONSE_GENERATED = "self_response_generated"
    TOOL_EXECUTED = "tool_executed"
    TOOL_REJECTED = "tool_rejected"
    DAYDREAM_GENERATED = "daydream_generated"
    GOAL_ACHIEVED = "goal_achieved"
    GOAL_BLOCKED = "goal_blocked"
    CONFLICT_DETECTED = "conflict_detected"
    REFLECTION_RESOLVED = "reflection_resolved"
    SELF_COMPASSION_OFFERED = "self_compassion_offered"


class SomaticAppraisalEvent(BaseModel):
    """A typed appraisal event linking a trigger to an affective response."""

    event_id: UUID = Field(default_factory=uuid4)
    timestamp: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    event_type: AppraisalEventType
    source_text: str = ""
    affect_state: Optional[AffectState] = None
    trigger_event_id: Optional[str] = None
    meta: Dict[str, Any] = Field(default_factory=dict)
