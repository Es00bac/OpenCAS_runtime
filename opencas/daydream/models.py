"""Data models for daydream reflections and conflict registry."""

from datetime import datetime, timezone
from typing import Any, Dict, List, Optional
from uuid import UUID, uuid4

from pydantic import BaseModel, Field

from opencas.somatic.models import SomaticSnapshot


class DaydreamReflection(BaseModel):
    """A structured reflection produced during daydream generation."""

    reflection_id: UUID = Field(default_factory=uuid4)
    created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    spark_content: str
    recollection: str = ""
    interpretation: str = ""
    synthesis: str = ""
    open_question: Optional[str] = None
    changed_self_view: str = ""
    tension_hints: List[str] = Field(default_factory=list)
    alignment_score: float = Field(default=0.0, ge=0.0, le=1.0)
    novelty_score: float = Field(default=0.0, ge=0.0, le=1.0)
    fascination_thread: Optional[str] = None
    experience_context: Dict[str, Any] = Field(default_factory=dict)
    keeper: bool = False


class DaydreamSpark(BaseModel):
    """A raw daydream spark before routing into work."""

    spark_id: str
    created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    mode: str = ""
    trigger: str = ""
    interest: str = ""
    summary: str = ""
    label: str = ""
    kind: str = ""
    intensity: float = 0.0
    objective: str = ""
    tags: List[str] = Field(default_factory=list)
    task_id: Optional[str] = None
    raw: Dict[str, Any] = Field(default_factory=dict)


class DaydreamInitiative(BaseModel):
    """A routed spark with intended rung, task, and artifact lineage."""

    initiative_id: str
    spark_id: Optional[str] = None
    created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    mode: str = ""
    trigger: str = ""
    interest: str = ""
    summary: str = ""
    label: str = ""
    kind: str = ""
    intensity: float = 0.0
    rung: str = ""
    desired_rung: str = ""
    objective: str = ""
    focus: str = ""
    source_kind: str = ""
    source_label: str = ""
    artifact_paths: List[str] = Field(default_factory=list)
    task_id: Optional[str] = None
    route_debug: Dict[str, Any] = Field(default_factory=dict)
    tags: List[str] = Field(default_factory=list)
    raw: Dict[str, Any] = Field(default_factory=dict)


class DaydreamOutcome(BaseModel):
    """Outcome record for a daydream-promoted task."""

    task_id: str
    recorded_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    outcome: str = ""
    value_delivered: bool = False
    raw: Dict[str, Any] = Field(default_factory=dict)


class DaydreamNotification(BaseModel):
    """Notification emitted for a daydream spark."""

    notification_id: str
    spark_id: Optional[str] = None
    chat_id: Optional[str] = None
    sent_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    label: str = ""
    intensity: float = 0.0
    kind: str = ""
    raw: Dict[str, Any] = Field(default_factory=dict)


class ConflictRecord(BaseModel):
    """A detected tension tracked over time."""

    conflict_id: UUID = Field(default_factory=uuid4)
    created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    kind: str
    description: str
    source_daydream_id: Optional[str] = None
    occurrence_count: int = 1
    resolved: bool = False
    auto_resolved: bool = False
    somatic_context: Optional[SomaticSnapshot] = None
    resolved_at: Optional[datetime] = None
    resolution_notes: str = ""
