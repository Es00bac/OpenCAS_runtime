"""Data models for Theory of Mind (ToM)."""

from datetime import datetime, timezone
from enum import Enum
from typing import Any, Dict, List, Optional
from uuid import UUID, uuid4

from pydantic import BaseModel, Field


class BeliefSubject(str, Enum):
    SELF = "self"
    USER = "user"


class Belief(BaseModel):
    """A tracked belief about self or user."""

    belief_id: UUID = Field(default_factory=uuid4)
    timestamp: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    subject: BeliefSubject
    predicate: str
    confidence: float = Field(default=1.0, ge=0.0, le=1.0)
    evidence_ids: List[str] = Field(default_factory=list)
    belief_revision_score: float = Field(default=0.0)
    meta: Dict[str, Any] = Field(default_factory=dict)


class IntentionStatus(str, Enum):
    ACTIVE = "active"
    COMPLETED = "completed"
    ABANDONED = "abandoned"


class Intention(BaseModel):
    """A tracked intention for self or inferred for user."""

    intention_id: UUID = Field(default_factory=uuid4)
    timestamp: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    actor: BeliefSubject
    content: str
    status: IntentionStatus = IntentionStatus.ACTIVE
    resolved_at: Optional[datetime] = None
    meta: Dict[str, Any] = Field(default_factory=dict)


class MetacognitiveResult(BaseModel):
    """Result of a metacognitive consistency check."""

    check_id: UUID = Field(default_factory=uuid4)
    timestamp: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    contradictions: List[str] = Field(default_factory=list)
    warnings: List[str] = Field(default_factory=list)
    belief_count: int = 0
    intention_count: int = 0
