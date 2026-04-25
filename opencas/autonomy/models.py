"""Data models for the autonomy subsystem."""

from datetime import datetime, timezone
from enum import Enum
from typing import Any, Dict, List, Optional
from uuid import UUID, uuid4

from pydantic import BaseModel, Field


class ApprovalLevel(str, Enum):
    """Self-approval decision levels."""

    CAN_DO_NOW = "can_do_now"
    CAN_DO_WITH_CAUTION = "can_do_with_caution"
    CAN_DO_AFTER_MORE_EVIDENCE = "can_do_after_more_evidence"
    MUST_ESCALATE = "must_escalate"


class ActionRiskTier(str, Enum):
    """Taxonomy of action risk tiers."""

    READONLY = "readonly"          # e.g., read file, list dir
    WORKSPACE_WRITE = "workspace_write"  # edit files in workspace
    SHELL_LOCAL = "shell_local"    # local shell commands
    NETWORK = "network"            # external HTTP/API calls
    EXTERNAL_WRITE = "external_write"    # writes outside workspace, emails, posts
    DESTRUCTIVE = "destructive"    # rm -rf, drop tables, force push


class ActionRequest(BaseModel):
    """A request for self-approval evaluation."""

    action_id: UUID = Field(default_factory=uuid4)
    timestamp: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    tier: ActionRiskTier
    description: str
    tool_name: Optional[str] = None
    target_path: Optional[str] = None
    payload: Dict[str, Any] = Field(default_factory=dict)
    memory_evidence_ids: List[str] = Field(default_factory=list)


class ApprovalDecision(BaseModel):
    """Outcome of a self-approval evaluation."""

    decision_id: UUID = Field(default_factory=uuid4)
    timestamp: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    level: ApprovalLevel
    action_id: UUID
    confidence: float = Field(ge=0.0, le=1.0)
    reasoning: str
    suggested_evidence: List[str] = Field(default_factory=list)
    score: float = Field(default=0.0, ge=0.0, le=1.0)


class WorkStage(str, Enum):
    """Stages of the creative ladder."""

    SPARK = "spark"
    NOTE = "note"
    ARTIFACT = "artifact"
    MICRO_TASK = "micro_task"
    PROJECT_SEED = "project_seed"
    PROJECT = "project"
    DURABLE_WORK_STREAM = "durable_work_stream"


class WorkObject(BaseModel):
    """A unit of work on the creative ladder."""

    work_id: UUID = Field(default_factory=uuid4)
    created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    updated_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    stage: WorkStage = WorkStage.SPARK
    content: str
    embedding_id: Optional[str] = None
    source_memory_ids: List[str] = Field(default_factory=list)
    promotion_score: float = Field(default=0.0)
    access_count: int = 0
    last_accessed: Optional[datetime] = None
    meta: Dict[str, Any] = Field(default_factory=dict)
    dependency_ids: List[str] = Field(default_factory=list)
    blocked_by: List[str] = Field(default_factory=list)
    project_id: Optional[str] = None
    commitment_id: Optional[str] = None
    portfolio_id: Optional[str] = None


class ProjectPlan(BaseModel):
    """Decomposition of a project into tasks with dependencies."""

    plan_id: UUID = Field(default_factory=uuid4)
    project_work_id: str
    tasks: List[Dict[str, Any]] = Field(default_factory=list)
    summary: str = ""
