"""Unified task lifecycle state machine for WorkObjects and RepairTasks."""

from __future__ import annotations

from datetime import datetime, timezone
from enum import Enum
from typing import Any, Dict, List, Optional, Set
from uuid import UUID, uuid4

from pydantic import BaseModel, Field


class LifecycleStage(str, Enum):
    """Unified stages spanning creative ladder and execution pipeline."""

    SPARK = "spark"
    NOTE = "note"
    ARTIFACT = "artifact"
    QUEUED = "queued"
    PLANNING = "planning"
    EXECUTING = "executing"
    VERIFYING = "verifying"
    REPORTING = "reporting"
    DONE = "done"
    FAILED = "failed"
    NEEDS_APPROVAL = "needs_approval"
    NEEDS_CLARIFICATION = "needs_clarification"


class StageTransition(BaseModel):
    """Record of a single lifecycle stage transition."""

    transition_id: UUID = Field(default_factory=uuid4)
    task_id: str
    from_stage: LifecycleStage
    to_stage: LifecycleStage
    reason: Optional[str] = None
    timestamp: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    context: Dict[str, Any] = Field(default_factory=dict)


class TaskLifecycleMachine:
    """Pure state machine for valid lifecycle transitions."""

    VALID_TRANSITIONS: Dict[LifecycleStage, Set[LifecycleStage]] = {
        LifecycleStage.SPARK: {LifecycleStage.NOTE, LifecycleStage.ARTIFACT, LifecycleStage.QUEUED, LifecycleStage.FAILED},
        LifecycleStage.NOTE: {LifecycleStage.ARTIFACT, LifecycleStage.QUEUED, LifecycleStage.FAILED},
        LifecycleStage.ARTIFACT: {LifecycleStage.QUEUED, LifecycleStage.FAILED},
        LifecycleStage.QUEUED: {
            LifecycleStage.PLANNING,
            LifecycleStage.EXECUTING,
            LifecycleStage.NEEDS_APPROVAL,
            LifecycleStage.NEEDS_CLARIFICATION,
            LifecycleStage.FAILED,
        },
        LifecycleStage.PLANNING: {LifecycleStage.EXECUTING, LifecycleStage.NEEDS_CLARIFICATION, LifecycleStage.FAILED},
        LifecycleStage.EXECUTING: {
            LifecycleStage.VERIFYING,
            LifecycleStage.NEEDS_APPROVAL,
            LifecycleStage.NEEDS_CLARIFICATION,
            LifecycleStage.FAILED,
            LifecycleStage.QUEUED,
            LifecycleStage.DONE,
        },
        LifecycleStage.VERIFYING: {LifecycleStage.REPORTING, LifecycleStage.FAILED, LifecycleStage.EXECUTING},
        LifecycleStage.REPORTING: {LifecycleStage.DONE, LifecycleStage.FAILED},
        LifecycleStage.FAILED: {LifecycleStage.QUEUED},
        LifecycleStage.NEEDS_APPROVAL: {LifecycleStage.QUEUED, LifecycleStage.EXECUTING, LifecycleStage.FAILED},
        LifecycleStage.NEEDS_CLARIFICATION: {LifecycleStage.QUEUED, LifecycleStage.PLANNING, LifecycleStage.EXECUTING, LifecycleStage.FAILED},
    }

    @classmethod
    def is_valid(cls, from_stage: LifecycleStage, to_stage: LifecycleStage) -> bool:
        """Return True if the transition is allowed."""
        return to_stage in cls.VALID_TRANSITIONS.get(from_stage, set())

    @classmethod
    def transition(
        cls,
        task_id: str,
        from_stage: LifecycleStage,
        to_stage: LifecycleStage,
        reason: Optional[str] = None,
        context: Optional[Dict[str, Any]] = None,
    ) -> StageTransition:
        """Validate and return a StageTransition record."""
        if not cls.is_valid(from_stage, to_stage):
            raise ValueError(
                f"Invalid lifecycle transition from {from_stage.value} to {to_stage.value}"
            )
        return StageTransition(
            task_id=task_id,
            from_stage=from_stage,
            to_stage=to_stage,
            reason=reason,
            context=context or {},
        )
