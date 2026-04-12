"""Executive workspace: scored, rebuildable view of current priorities."""

from __future__ import annotations

from datetime import datetime, timezone
from enum import Enum
from typing import Any, Dict, List, Optional
from uuid import UUID, uuid4

from pydantic import BaseModel, Field

from opencas.autonomy.models import WorkObject

from .commitment import Commitment, CommitmentStatus


class WorkspaceItemKind(str, Enum):
    """Kinds of items that can appear in the executive workspace."""

    GOAL = "goal"
    COMMITMENT = "commitment"
    PROJECT = "project"
    PORTFOLIO = "portfolio"
    TASK = "task"


class WorkspaceAffinity(str, Enum):
    """Affinity/profile of a workspace item."""

    BACKGROUND = "background"
    PERSONAL = "personal"
    OPERATOR = "operator"


class ExecutionMode(str, Enum):
    """Preferred execution mode for a workspace item."""

    RESPOND_INLINE = "respond_inline"
    FOREGROUND_TOOLS = "foreground_tools"
    BACKGROUND_AGENT = "background_agent"


class WorkspaceItem(BaseModel):
    """A single scored item in the executive workspace."""

    item_id: UUID = Field(default_factory=uuid4)
    kind: WorkspaceItemKind
    content: str
    urgency_score: float = Field(default=0.0, ge=0.0, le=1.0)
    importance_score: float = Field(default=0.0, ge=0.0, le=1.0)
    attention_score: float = Field(default=0.0, ge=0.0, le=1.0)
    total_score: float = Field(default=0.0, ge=0.0, le=1.0)
    affinity: WorkspaceAffinity = WorkspaceAffinity.BACKGROUND
    execution_mode: ExecutionMode = ExecutionMode.BACKGROUND_AGENT
    commitment_id: Optional[str] = None
    project_id: Optional[str] = None
    portfolio_id: Optional[str] = None
    meta: Dict[str, Any] = Field(default_factory=dict)


class PortfolioBoost(BaseModel):
    """Boost data for a portfolio cluster."""

    portfolio_id: str
    spark_count: int = 0
    boost: float = 0.0


class ExecutiveWorkspace(BaseModel):
    """A rebuildable, scored view of what the agent should focus on now."""

    focus: Optional[WorkspaceItem] = None
    queue: List[WorkspaceItem] = Field(default_factory=list)
    rebuild_timestamp: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))

    @classmethod
    def rebuild(
        cls,
        commitments: List[Commitment],
        work_objects: List[WorkObject],
        portfolio_boosts: Optional[Dict[str, PortfolioBoost]] = None,
        now: Optional[datetime] = None,
    ) -> "ExecutiveWorkspace":
        """Rebuild workspace from current commitments and work objects."""
        if now is None:
            now = datetime.now(timezone.utc)
        portfolio_boosts = portfolio_boosts or {}
        items: List[WorkspaceItem] = []

        for commitment in commitments:
            if commitment.status != CommitmentStatus.ACTIVE:
                continue
            urgency = cls._urgency_from_deadline(commitment.deadline, now)
            importance = commitment.priority / 10.0
            attention = 0.5
            total = round(0.4 * urgency + 0.4 * importance + 0.2 * attention, 4)
            items.append(
                WorkspaceItem(
                    item_id=commitment.commitment_id,
                    kind=WorkspaceItemKind.COMMITMENT,
                    content=commitment.content,
                    urgency_score=urgency,
                    importance_score=importance,
                    attention_score=attention,
                    total_score=total,
                    affinity=WorkspaceAffinity.PERSONAL,
                    execution_mode=ExecutionMode.BACKGROUND_AGENT,
                    commitment_id=str(commitment.commitment_id),
                )
            )

        for work in work_objects:
            if work.stage.value in ("done", "failed"):
                continue
            importance = min(1.0, work.promotion_score)
            attention = cls._attention_from_access(work)
            urgency = 0.3 if work.blocked_by else 0.5
            total = round(0.4 * urgency + 0.4 * importance + 0.2 * attention, 4)
            boost = portfolio_boosts.get(work.portfolio_id or "", PortfolioBoost(portfolio_id="")).boost
            total = round(min(1.0, total + boost), 4)
            mode = ExecutionMode.BACKGROUND_AGENT
            if work.stage.value in ("micro_task", "project"):
                mode = ExecutionMode.FOREGROUND_TOOLS
            items.append(
                WorkspaceItem(
                    item_id=work.work_id,
                    kind=WorkspaceItemKind.TASK,
                    content=work.content,
                    urgency_score=urgency,
                    importance_score=importance,
                    attention_score=attention,
                    total_score=total,
                    affinity=WorkspaceAffinity.BACKGROUND,
                    execution_mode=mode,
                    commitment_id=work.commitment_id,
                    project_id=work.project_id,
                    portfolio_id=work.portfolio_id,
                    meta={"stage": work.stage.value},
                )
            )

        items.sort(key=lambda i: i.total_score, reverse=True)
        queue = items[:32]
        focus = queue[0] if queue else None
        return cls(focus=focus, queue=queue, rebuild_timestamp=now)

    @staticmethod
    def _urgency_from_deadline(
        deadline: Optional[datetime],
        now: datetime,
    ) -> float:
        if deadline is None:
            return 0.3
        hours_remaining = (deadline - now).total_seconds() / 3600.0
        if hours_remaining <= 0:
            return 1.0
        if hours_remaining <= 24:
            return 0.8
        if hours_remaining <= 72:
            return 0.6
        return 0.3

    @staticmethod
    def _attention_from_access(work: WorkObject) -> float:
        if work.access_count == 0:
            return 0.2
        base = min(1.0, work.access_count / 10.0)
        if work.last_accessed is None:
            return base
        now = datetime.now(timezone.utc)
        hours_since = max(0.0, (now - work.last_accessed).total_seconds() / 3600.0)
        recency_decay = max(0.0, 1.0 - (hours_since / 168.0))  # decay over 7 days
        return round(base * recency_decay, 4)
