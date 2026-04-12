"""Memory data models for OpenCAS."""

from datetime import datetime, timezone
from enum import Enum
from typing import Any, Dict, List, Optional
from uuid import UUID, uuid4

from pydantic import BaseModel, Field

from opencas.somatic.models import AffectState


class EpisodeKind(str, Enum):
    """Types of episodic records."""

    TURN = "turn"
    OBSERVATION = "observation"
    ACTION = "action"
    ARTIFACT = "artifact"
    COMPACTION = "compaction"
    CONSOLIDATION = "consolidation"


class EdgeKind(str, Enum):
    """Types of associative links between episodes."""

    SEMANTIC = "semantic"
    EMOTIONAL = "emotional"
    TEMPORAL = "temporal"
    CONCEPTUAL = "conceptual"
    RELATIONAL = "relational"
    CAUSAL = "causal"


class Episode(BaseModel):
    """A single episodic record."""

    episode_id: UUID = Field(default_factory=uuid4)
    created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    kind: EpisodeKind
    session_id: Optional[str] = None
    content: str
    embedding_id: Optional[str] = None
    somatic_tag: Optional[str] = None
    affect: Optional[AffectState] = None
    salience: float = Field(default=1.0, ge=0.0, le=10.0)
    compacted: bool = False
    identity_core: bool = False
    confidence_score: float = Field(default=0.8, ge=0.0, le=1.0)
    access_count: int = Field(default=0, ge=0)
    last_accessed: Optional[datetime] = None
    used_successfully: int = Field(default=0, ge=0)
    used_unsuccessfully: int = Field(default=0, ge=0)
    payload: Dict[str, Any] = Field(default_factory=dict)


class Memory(BaseModel):
    """A distilled semantic memory extracted from episodes."""

    memory_id: UUID = Field(default_factory=uuid4)
    created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    updated_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    content: str
    embedding_id: Optional[str] = None
    source_episode_ids: List[str] = Field(default_factory=list)
    tags: List[str] = Field(default_factory=list)
    salience: float = Field(default=1.0, ge=0.0, le=10.0)
    access_count: int = 0
    last_accessed: Optional[datetime] = None


class EpisodeEdge(BaseModel):
    """A weighted associative link between two episodes."""

    edge_id: UUID = Field(default_factory=uuid4)
    source_id: str
    target_id: str
    kind: EdgeKind = Field(default=EdgeKind.SEMANTIC)
    semantic_weight: float = Field(default=0.0, ge=0.0, le=1.0)
    emotional_weight: float = Field(default=0.0, ge=0.0, le=1.0)
    recency_weight: float = Field(default=0.0, ge=0.0, le=1.0)
    structural_weight: float = Field(default=0.0, ge=0.0, le=1.0)
    salience_weight: float = Field(default=0.0, ge=0.0, le=1.0)
    causal_weight: float = Field(default=0.0, ge=0.0, le=1.0)
    verification_weight: float = Field(default=0.0, ge=0.0, le=1.0)
    actor_affinity_weight: float = Field(default=0.0, ge=0.0, le=1.0)
    confidence: float = Field(default=0.0, ge=0.0, le=1.0)
    created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))


class CompactionRecord(BaseModel):
    """A record of a compaction operation."""

    compaction_id: UUID = Field(default_factory=uuid4)
    created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    episode_ids: List[str]
    summary: str
    removed_count: int
