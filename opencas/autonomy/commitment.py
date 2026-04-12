"""Commitment / Goal data models for structured objective tracking."""

from __future__ import annotations

from datetime import datetime, timezone
from enum import Enum
from typing import Any, Dict, List, Optional
from uuid import UUID, uuid4

from pydantic import BaseModel, Field


class CommitmentStatus(str, Enum):
    """Lifecycle states of a commitment."""

    ACTIVE = "active"
    BLOCKED = "blocked"
    COMPLETED = "completed"
    ABANDONED = "abandoned"


class Commitment(BaseModel):
    """A durable, structured commitment linking goals to work and tasks."""

    commitment_id: UUID = Field(default_factory=uuid4)
    created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    updated_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    content: str
    status: CommitmentStatus = CommitmentStatus.ACTIVE
    priority: float = Field(default=5.0, ge=1.0, le=10.0)
    deadline: Optional[datetime] = None
    linked_work_ids: List[str] = Field(default_factory=list)
    linked_task_ids: List[str] = Field(default_factory=list)
    tags: List[str] = Field(default_factory=list)
    meta: Dict[str, Any] = Field(default_factory=dict)
