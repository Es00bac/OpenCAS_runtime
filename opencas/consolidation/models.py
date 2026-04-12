"""Models for the nightly consolidation engine."""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Any, Dict, List
from uuid import UUID, uuid4

from pydantic import BaseModel, Field


class SalienceUpdate(BaseModel):
    """Result of reweighting a memory's salience."""

    memory_id: str
    old_salience: float
    new_salience: float


class RejectedMerge(BaseModel):
    """A cluster merge that was rejected and should be skipped in future cycles."""

    cluster_hash: str
    episode_ids: List[str]
    reason: str = ""
    rejected_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))


class ConsolidationResult(BaseModel):
    """Summary of a consolidation run."""

    result_id: UUID = Field(default_factory=uuid4)
    timestamp: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    candidate_episodes: int = 0
    clusters_formed: int = 0
    memories_created: int = 0
    signals_promoted: int = 0
    memories_updated: int = 0
    salience_updates: List[SalienceUpdate] = Field(default_factory=list)
    episode_salience_updates: int = 0
    episodes_pruned: int = 0
    edges_created: int = 0
    identity_core_promotions: int = 0
    identity_updates: Dict[str, Any] = Field(default_factory=dict)
    merges_rejected: int = 0
    orphans_recovered: int = 0
    beliefs_decayed: int = 0
