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
    ERROR = "error"
    SPAN_START = "span_start"
    SPAN_END = "span_end"


class TelemetryEvent(BaseModel):
    """A single telemetry event."""

    event_id: UUID = Field(default_factory=uuid4)
    timestamp: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    kind: EventKind
    session_id: Optional[str] = None
    span_id: Optional[str] = None
    parent_span_id: Optional[str] = None
    message: Optional[str] = None
    payload: Dict[str, Any] = Field(default_factory=dict)

    def to_jsonl(self) -> str:
        return self.model_dump_json() + "\n"

    @classmethod
    def from_jsonl(cls, line: str) -> "TelemetryEvent":
        return cls.model_validate_json(line.strip())
