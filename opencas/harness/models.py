"""Data models for the agentic harness and research notebook layer."""

from datetime import datetime, timezone
from enum import Enum
from typing import Any, Dict, List, Optional
from uuid import UUID, uuid4

from pydantic import BaseModel, Field


class ObjectiveStatus(str, Enum):
    """Lifecycle states of an objective loop."""

    PENDING = "pending"
    ACTIVE = "active"
    PAUSED = "paused"
    COMPLETED = "completed"
    FAILED = "failed"


class NotebookEntryKind(str, Enum):
    """Kinds of entries in a research notebook."""

    OBSERVATION = "observation"
    HYPOTHESIS = "hypothesis"
    EXPERIMENT = "experiment"
    RESULT = "result"
    INSIGHT = "insight"
    QUESTION = "question"


class DeliverableSchema(BaseModel):
    """Schema for a harness-defined deliverable."""

    schema_id: UUID = Field(default_factory=uuid4)
    created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    name: str
    description: str = ""
    acceptance_criteria: List[str] = Field(default_factory=list)
    expected_artifacts: List[str] = Field(default_factory=list)
    meta: Dict[str, Any] = Field(default_factory=dict)


class ObjectiveLoopContract(BaseModel):
    """Bounded work contract required before an objective loop can run."""

    goal: str
    expected_output: str
    success_check: str
    stop_condition: str
    max_attempt_budget: int = Field(default=1, ge=1)
    reframe_path: str = ""


class NotebookEntry(BaseModel):
    """A single entry in a research notebook."""

    entry_id: UUID = Field(default_factory=uuid4)
    created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    kind: NotebookEntryKind
    content: str
    source_episode_ids: List[str] = Field(default_factory=list)
    source_task_ids: List[str] = Field(default_factory=list)
    meta: Dict[str, Any] = Field(default_factory=dict)


class ResearchNotebook(BaseModel):
    """A persistent research notebook for an objective loop."""

    notebook_id: UUID = Field(default_factory=uuid4)
    created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    updated_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    title: str
    description: str = ""
    entries: List[NotebookEntry] = Field(default_factory=list)
    deliverable_schema: Optional[DeliverableSchema] = None
    meta: Dict[str, Any] = Field(default_factory=dict)


class ObjectiveLoop(BaseModel):
    """A long-horizon objective with research notebook and generated tasks."""

    loop_id: UUID = Field(default_factory=uuid4)
    created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    updated_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    status: ObjectiveStatus = ObjectiveStatus.PENDING
    title: str
    description: str = ""
    notebook_id: Optional[str] = None
    generated_task_ids: List[str] = Field(default_factory=list)
    completion_criteria: List[str] = Field(default_factory=list)
    meta: Dict[str, Any] = Field(default_factory=dict)
