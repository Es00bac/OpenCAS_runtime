"""Data models for the conversational refusal path."""

from datetime import datetime, timezone
from enum import Enum
from typing import Any, Dict, List, Optional
from uuid import UUID, uuid4

from pydantic import BaseModel, Field


class RefusalCategory(str, Enum):
    """Categories of conversational refusal."""

    BOUNDARY_VIOLATION = "boundary_violation"
    HARMFUL_REQUEST = "harmful_request"
    POLICY_HOOK_BLOCK = "policy_hook_block"


class ConversationalRequest(BaseModel):
    """A normalized conversational input for refusal evaluation."""

    request_id: UUID = Field(default_factory=uuid4)
    timestamp: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    text: str
    session_id: Optional[str] = None
    meta: Dict[str, Any] = Field(default_factory=dict)


class RefusalDecision(BaseModel):
    """Outcome of a conversational refusal evaluation."""

    decision_id: UUID = Field(default_factory=uuid4)
    request_id: UUID
    refused: bool
    category: Optional[RefusalCategory] = None
    reasoning: str = ""
    suggested_response: Optional[str] = None
