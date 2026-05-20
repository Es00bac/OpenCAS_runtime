"""Models for compact autobiographical reconstruction."""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field


class SessionAnchor(BaseModel):
    """Deterministic skeleton for one conversation session."""

    session_id: str
    created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    ended_at: Optional[datetime] = None
    duration_s: Optional[float] = None
    agent_name: str = "OpenCAS"
    source_system: Optional[str] = None
    episode_kind_tally: Dict[str, int] = Field(default_factory=dict)
    decision_episode_ids: List[str] = Field(default_factory=list)
    affect_peaks: List[dict[str, Any]] = Field(default_factory=list)
    recorded_affect_summary: Optional[str] = None
    artifact_paths_touched: List[str] = Field(default_factory=list)
    commitments_opened: List[str] = Field(default_factory=list)
    commitments_closed: List[str] = Field(default_factory=list)
    identity_mutagen_episode_ids: List[str] = Field(default_factory=list)
    compaction_record_ids: List[str] = Field(default_factory=list)
    narrative_bridge_message_ids: List[str] = Field(default_factory=list)
    continuity_breadcrumb_ids: List[str] = Field(default_factory=list)
    evidence_episode_ids: List[str] = Field(default_factory=list)
    recall_failure_episode_ids: List[str] = Field(default_factory=list)
    recall_recovery_episode_ids: List[str] = Field(default_factory=list)
    evidence_hash: Optional[str] = None
    evidence_strength: Optional[str] = "low"
    gist: Optional[str] = None
    gist_version: int = 0
    confidence: Optional[str] = None
    gaps_noted: List[str] = Field(default_factory=list)
    consolidation_run_id_at_time: Optional[str] = None
    version: int = 1


class EvidenceItem(BaseModel):
    """One compact piece of recall evidence."""

    timestamp: str
    kind: str
    label: str
    excerpt: str
    session_id: Optional[str] = None


class NextBestLookup(BaseModel):
    """Suggested follow-up lookup when recall evidence is thin."""

    tool: str
    args: Dict[str, Any] = Field(default_factory=dict)
    reason: str


class SessionRef(BaseModel):
    """Compact session citation for a recall packet."""

    session_id: str
    date: str
    duration_s: Optional[float] = None


class AutobiographyRecallResult(BaseModel):
    """Answer-shaped recall packet for tool and prompt use."""

    query: str
    essence: str
    confidence: str
    evidence_scope: str
    strongest_evidence: List[EvidenceItem] = Field(default_factory=list)
    gaps: List[str] = Field(default_factory=list)
    next_best_lookup: List[NextBestLookup] = Field(default_factory=list)
    session_refs: List[SessionRef] = Field(default_factory=list)
    project_arc: Optional[Dict[str, Any]] = None

    def to_dict(self) -> dict[str, Any]:
        return self.model_dump()
