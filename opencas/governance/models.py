"""Data models for the governance subsystem."""

from datetime import datetime, timezone
from typing import Optional
from uuid import UUID, uuid4

from pydantic import BaseModel, Field

from opencas.autonomy.models import ActionRiskTier


class ApprovalLedgerEntry(BaseModel):
    """A durable record of a self-approval decision."""

    entry_id: UUID = Field(default_factory=uuid4)
    decision_id: UUID
    action_id: UUID
    level: str
    score: float
    reasoning: str
    tool_name: Optional[str] = None
    tier: ActionRiskTier
    somatic_state: Optional[str] = None
    created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
