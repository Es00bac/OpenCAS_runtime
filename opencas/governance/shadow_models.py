"""Models for the ShadowRegistry blocked-intention ledger."""

from __future__ import annotations

import hashlib
import json
from datetime import datetime, timezone
from enum import Enum
from typing import Any, Dict, Optional
from uuid import uuid4

from pydantic import BaseModel, Field, model_validator


class BlockReason(str, Enum):
    """Canonical reasons why an intention was blocked."""

    APPROVAL_DENIED = "approval_denied"
    HOOK_BLOCKED = "hook_blocked"
    VALIDATION_BLOCKED = "validation_blocked"
    SAFETY_BLOCKED = "safety_blocked"
    TOOL_LOOP_GUARD_BLOCKED = "tool_loop_guard_blocked"
    RETRY_BLOCKED = "retry_blocked"
    UNKNOWN_BLOCKED = "unknown_blocked"


class DecompositionStage(str, Enum):
    """Lifecycle state for a blocked intention."""

    RAW = "raw_intent"
    FERMENTING = "fermenting"
    DIGESTED = "digested"
    MATURE = "mature"
    SEED = "seed"


class ClusterTriageStatus(str, Enum):
    """Operator triage state for one recurring blocked-intention cluster."""

    ACTIVE = "active"
    DISMISSED = "dismissed"


class BlockedIntention(BaseModel):
    """Durable record of a blocked action with enough context for later reuse."""

    id: str = Field(default_factory=lambda: uuid4().hex[:16])
    captured_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    tool_name: str
    intent_summary: str
    raw_parameters: Dict[str, Any] = Field(default_factory=dict)
    block_reason: BlockReason
    block_context: str
    session_id: Optional[str] = None
    task_id: Optional[str] = None
    action_id: Optional[str] = None
    artifact: Optional[str] = None
    target_kind: Optional[str] = None
    target_id: Optional[str] = None
    risk_tier: Optional[str] = None
    decision_level: Optional[str] = None
    capture_source: Optional[str] = None
    decomposition_stage: DecompositionStage = DecompositionStage.RAW
    agent_state: Dict[str, Any] = Field(default_factory=dict)
    fingerprint: Optional[str] = None

    @model_validator(mode="after")
    def _populate_fingerprint(self) -> "BlockedIntention":
        if not self.fingerprint:
            self.fingerprint = compute_blocked_intention_fingerprint(
                tool_name=self.tool_name,
                intent_summary=self.intent_summary,
                raw_parameters=self.raw_parameters,
                block_reason=self.block_reason,
            )
        return self


class ShadowClusterTriageState(BaseModel):
    """Durable operator triage metadata for one blocked-intention cluster."""

    fingerprint: str
    triage_status: ClusterTriageStatus = ClusterTriageStatus.ACTIVE
    annotation: Optional[str] = None
    triaged_at: Optional[datetime] = None
    dismissed_at: Optional[datetime] = None


def compute_blocked_intention_fingerprint(
    *,
    tool_name: str,
    intent_summary: str,
    raw_parameters: Dict[str, Any],
    block_reason: BlockReason,
) -> str:
    """Create a stable fingerprint for clustering similar blocked intentions."""

    payload = json.dumps(
        _normalize_parameters_for_fingerprint(raw_parameters),
        sort_keys=True,
        ensure_ascii=True,
        default=str,
    )
    content = f"{tool_name}|{intent_summary}|{block_reason.value}|{payload}"
    return hashlib.sha256(content.encode("utf-8")).hexdigest()[:16]


def _normalize_parameters_for_fingerprint(value: Any) -> Any:
    volatile_keys = {
        "attempt",
        "timestamp",
        "captured_at",
        "updated_at",
        "created_at",
        "session_id",
        "task_id",
        "action_id",
    }
    if isinstance(value, dict):
        return {
            key: _normalize_parameters_for_fingerprint(item)
            for key, item in sorted(value.items())
            if key not in volatile_keys
        }
    if isinstance(value, list):
        return [_normalize_parameters_for_fingerprint(item) for item in value]
    return value
