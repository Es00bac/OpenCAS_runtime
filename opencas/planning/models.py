"""Planning persistence models for OpenCAS."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional
from uuid import UUID, uuid4


@dataclass
class PlanEntry:
    """A durable plan with content and status."""

    plan_id: str
    status: str  # "draft", "active", "completed", "abandoned"
    content: str
    created_at: datetime
    updated_at: datetime
    project_id: Optional[str] = None
    task_id: Optional[str] = None


@dataclass
class PlanAction:
    """A recorded action taken while a plan is active."""

    action_id: str
    plan_id: str
    tool_name: str
    args: Dict[str, Any] = field(default_factory=dict)
    result_summary: str = ""
    success: bool = False
    timestamp: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
