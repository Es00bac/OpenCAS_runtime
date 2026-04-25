"""Commitment / Goal data models for structured objective tracking."""

from __future__ import annotations

from datetime import datetime, timezone
from enum import Enum
from typing import Any, Dict, List, Optional
from uuid import UUID, uuid4

from pydantic import BaseModel, Field


class CommitmentStatus(str, Enum):
    """Lifecycle states of a commitment."""

    ACTIVE = "active"
    BLOCKED = "blocked"
    COMPLETED = "completed"
    ABANDONED = "abandoned"


class Commitment(BaseModel):
    """A durable, structured commitment linking goals to work and tasks."""

    commitment_id: UUID = Field(default_factory=uuid4)
    created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    updated_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    content: str
    status: CommitmentStatus = CommitmentStatus.ACTIVE
    priority: float = Field(default=5.0, ge=1.0, le=10.0)
    deadline: Optional[datetime] = None
    linked_work_ids: List[str] = Field(default_factory=list)
    linked_task_ids: List[str] = Field(default_factory=list)
    tags: List[str] = Field(default_factory=list)
    meta: Dict[str, Any] = Field(default_factory=dict)


def commitment_lifecycle_snapshot(commitment: Commitment) -> Dict[str, Any]:
    """Return operator-facing lifecycle and provenance detail for a commitment."""
    meta = dict(commitment.meta or {})
    source = str(meta.get("source", "")).strip() or None
    raw_excerpt = (
        str(meta.get("source_sentence", "")).strip()
        or str(meta.get("raw_excerpt", "")).strip()
        or None
    )
    merged_from = meta.get("merged_from_commitment_ids") or []
    if not isinstance(merged_from, list):
        merged_from = []
    return {
        "source": source,
        "raw_excerpt": raw_excerpt,
        "normalization_source": str(meta.get("normalization_source", "")).strip() or None,
        "capture_confidence": meta.get("capture_confidence"),
        "blocked_reason": str(meta.get("blocked_reason", "")).strip() or None,
        "resume_reason": str(meta.get("resume_reason", "")).strip() or None,
        "previous_blocked_reason": str(meta.get("previous_blocked_reason", "")).strip() or None,
        "resume_policy": str(meta.get("resume_policy", "")).strip() or None,
        "merge_rationale": str(meta.get("consolidation_merge_rationale", "")).strip() or None,
        "merged_from_commitment_ids": merged_from,
        "extracted_from_chat": source == "nightly_consolidation",
        "source_session_id": (
            str(meta.get("source_session_id", "")).strip()
            or str(meta.get("session_id", "")).strip()
            or None
        ),
        "source_episode_id": str(meta.get("source_episode_id", "")).strip() or None,
        "previous_user_turn": str(meta.get("previous_user_turn", "")).strip() or None,
        "role_source": str(meta.get("role_source", "")).strip() or None,
    }


def commitment_operator_snapshot(
    commitment: Commitment,
    *,
    include_meta: bool = False,
) -> Dict[str, Any]:
    """Return a stable operator-facing commitment payload."""
    payload = {
        "commitment_id": str(commitment.commitment_id),
        "created_at": commitment.created_at.isoformat(),
        "updated_at": commitment.updated_at.isoformat(),
        "content": commitment.content,
        "status": commitment.status.value,
        "priority": commitment.priority,
        "tags": list(commitment.tags),
        "deadline": commitment.deadline.isoformat() if commitment.deadline else None,
        "linked_work_ids": list(commitment.linked_work_ids),
        "linked_task_ids": list(commitment.linked_task_ids),
        "lifecycle": commitment_lifecycle_snapshot(commitment),
    }
    if include_meta:
        payload["meta"] = dict(commitment.meta or {})
    return payload
