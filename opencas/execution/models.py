"""Data models for execution and repair."""

from datetime import datetime, timezone
from enum import Enum
from typing import Any, Dict, List, Optional
from uuid import UUID, uuid4

from pydantic import BaseModel, Field


class ExecutionStage(str, Enum):
    QUEUED = "queued"
    PLANNING = "planning"
    EXECUTING = "executing"
    VERIFYING = "verifying"
    RECOVERING = "recovering"
    NEEDS_APPROVAL = "needs_approval"
    NEEDS_CLARIFICATION = "needs_clarification"
    DONE = "done"
    FAILED = "failed"


class ExecutionPhase(str, Enum):
    """Explicit execution lifecycle phases."""

    DETECT = "detect"
    SNAPSHOT = "snapshot"
    PLAN = "plan"
    EXECUTE = "execute"
    VERIFY = "verify"
    POSTCHECK = "postcheck"


class PhaseRecord(BaseModel):
    """Record of a single phase execution."""

    phase: ExecutionPhase
    started_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    ended_at: Optional[datetime] = None
    success: Optional[bool] = None
    output: Optional[str] = None


class RepairTask(BaseModel):
    """A background repair or execution task."""

    task_id: UUID = Field(default_factory=uuid4)
    created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    updated_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    objective: str
    stage: ExecutionStage = ExecutionStage.QUEUED
    status: str = "queued"
    artifacts: List[str] = Field(default_factory=list)
    attempt: int = 0
    max_attempts: int = 3
    verification_command: Optional[str] = None
    meta: Dict[str, Any] = Field(default_factory=dict)
    phases: List[PhaseRecord] = Field(default_factory=list)
    scratch_dir: Optional[str] = None
    checkpoint_commit: Optional[str] = None
    convergence_hashes: List[str] = Field(default_factory=list)
    retry_backoff_seconds: float = 1.0
    depends_on: List[str] = Field(default_factory=list)
    project_id: Optional[str] = None
    commitment_id: Optional[str] = None
    lane: Optional[str] = None


class TaskTransitionRecord(BaseModel):
    """Record of a task stage transition."""

    transition_id: UUID = Field(default_factory=uuid4)
    task_id: UUID
    from_stage: ExecutionStage
    to_stage: ExecutionStage
    reason: Optional[str] = None
    timestamp: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))


class RepairResult(BaseModel):
    """Result of a repair task execution."""

    task_id: UUID
    success: bool
    stage: ExecutionStage
    output: str
    artifacts: List[str] = Field(default_factory=list)
    timestamp: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))


class ExecutionReceipt(BaseModel):
    """Durable audit record of a completed repair task lifecycle."""

    receipt_id: UUID = Field(default_factory=uuid4)
    task_id: UUID
    objective: str
    plan: Optional[str] = None
    phases: List[PhaseRecord] = Field(default_factory=list)
    verification_result: Optional[bool] = None
    checkpoint_commit: Optional[str] = None
    created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    completed_at: Optional[datetime] = None
    success: bool = False
    output: str = ""
