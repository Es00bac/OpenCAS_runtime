"""Telemetry event models for OpenCAS."""

from datetime import datetime, timezone
from enum import Enum
from typing import Any, Dict, List, Optional
from uuid import UUID, uuid4

from pydantic import BaseModel, Field


class EventKind(str, Enum):
    """Categories of telemetry events."""

    SESSION_START = "session_start"
    SESSION_END = "session_end"
    TURN = "turn"
    MEMORY_WRITE = "memory_write"
    MEMORY_COMPACT = "memory_compact"
    CONSOLIDATION_RUN = "consolidation_run"
    SELF_APPROVAL = "self_approval"
    ESCALATION = "escalation"
    TOM_EVAL = "tom_eval"
    CREATIVE_PROMOTION = "creative_promotion"
    BOOTSTRAP_STAGE = "bootstrap_stage"
    DIAGNOSTIC_RUN = "diagnostic_run"
    TOOL_CALL = "tool_call"
    LLM_CALL = "llm_call"
    ERROR = "error"
    WARNING = "warning"
    SPAN_START = "span_start"
    SPAN_END = "span_end"
    # Living handshake: explicit memory-action linkage events
    MEMORY_NOTED = "memory_noted"           # Marker: memory explicitly acknowledged
    MEMORY_ACTIVATED = "memory_activated"   # Marker: memory activated into action
    ACTION_BACKLINK = "action_backlink"     # Marker: action linked back to memory


class TelemetryEvent(BaseModel):
    """A single telemetry event with living handshake support.

    The "noted-as-such" marker (noted_as_such) turns the log from a passive
    record into a living handshake between memory and action. When an event
    is marked as noted, it means the system has explicitly acknowledged and
    integrated this memory into its working context.

    The continuity_backlink creates a bidirectional bridge: it points to the
    event_id of the action that this memory enabled, making the causal chain
    visible and traversable.

    The activated_at timestamp records when a memory was last activated into
    action, turning continuity into something visible and queryable.
    """

    event_id: UUID = Field(default_factory=uuid4)
    timestamp: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    kind: EventKind
    session_id: Optional[str] = None
    span_id: Optional[str] = None
    parent_span_id: Optional[str] = None
    message: Optional[str] = None
    payload: Dict[str, Any] = Field(default_factory=dict)

    # Living handshake fields: memory-action continuity
    noted_as_such: bool = Field(
        default=False,
        description="Marker: this event was explicitly acknowledged and integrated into memory."
    )
    continuity_backlink: Optional[str] = Field(
        default=None,
        description="Event ID of the action this memory enabled. Bidirectional bridge."
    )
    activated_at: Optional[datetime] = Field(
        default=None,
        description="When this memory was last activated into action."
    )

    def to_jsonl(self) -> str:
        return self.model_dump_json() + "\n"

    @classmethod
    def from_jsonl(cls, line: str) -> "TelemetryEvent":
        return cls.model_validate_json(line.strip())

    def mark_noted(self) -> "TelemetryEvent":
        """Mark this event as noted-as-such. Returns self for chaining."""
        self.noted_as_such = True
        return self

    def link_to_action(self, action_event_id: str) -> "TelemetryEvent":
        """Create a backlink to the action this memory enabled."""
        self.continuity_backlink = action_event_id
        return self

    def activate(self) -> "TelemetryEvent":
        """Mark this memory as activated now. Updates activated_at timestamp."""
        self.activated_at = datetime.now(timezone.utc)
        return self