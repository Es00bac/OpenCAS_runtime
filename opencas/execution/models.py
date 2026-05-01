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


class RetryMode(str, Enum):
    """Preferred retry posture after an attempt is salvaged."""

    CONTINUE_RETRY = "continue_retry"
    RESUME_EXISTING_ARTIFACT = "resume_existing_artifact"
    NARROW_EDIT = "narrow_edit"
    DETERMINISTIC_REVIEW = "deterministic_review"
    PAUSE_PROJECT = "pause_project"
    COMPLETE_PARTIAL_AND_STOP = "complete_partial_and_stop"


class AttemptOutcome(str, Enum):
    """Normalized salvage outcome for an execution attempt."""

    PARTIAL = "partial"
    FAILED = "failed"
    VERIFY_FAILED = "verify_failed"
    GUARD_STOPPED = "guard_stopped"
    DONE = "done"


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


class AttemptSalvagePacket(BaseModel):
    """Deterministic salvage summary for a single execution attempt."""

    packet_id: UUID
    task_id: UUID
    attempt: int
    project_signature: Optional[str] = None
    project_id: Optional[str] = None
    objective: str
    canonical_artifact_path: Optional[str] = None
    artifact_paths_touched: List[str] = Field(default_factory=list)
    plan_digest: str = ""
    execution_digest: str = ""
    verification_digest: Optional[str] = None
    tool_signature: str = ""
    divergence_signature: str
    outcome: AttemptOutcome
    partial_value: str = ""
    discovered_constraints: List[str] = Field(default_factory=list)
    unresolved_questions: List[str] = Field(default_factory=list)
    best_next_step: str
    recommended_mode: RetryMode
    meaningful_progress_signal: str = ""
    llm_spend_class: str = "broad"
    created_at: datetime


class RetryDecision(BaseModel):
    """Decision for whether the retry governor allows the next attempt."""

    allowed: bool
    reason: str
    mode: RetryMode
    reuse_packet_id: Optional[UUID] = None
