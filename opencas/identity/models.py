"""Identity and user-model data types for OpenCAS."""

from datetime import datetime, timezone
from typing import Any, Dict, List, Optional
from uuid import UUID, uuid4

from pydantic import BaseModel, Field


class SelfModel(BaseModel):
    """The agent's live self-model."""

    model_id: UUID = Field(default_factory=uuid4)
    updated_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    name: str = "OpenCAS"
    version: str = "0.1.0"
    narrative: Optional[str] = None
    values: List[str] = Field(default_factory=list)
    traits: List[str] = Field(default_factory=list)
    current_goals: List[str] = Field(default_factory=list)
    current_intention: Optional[str] = None
    recent_activity: List[Dict[str, Any]] = Field(default_factory=list)
    self_beliefs: Dict[str, Any] = Field(default_factory=dict)
    relational_state_id: Optional[str] = None
    source_system: Optional[str] = None
    imported_identity_profile: Dict[str, Any] = Field(default_factory=dict)
    memory_anchors: List[Dict[str, Any]] = Field(default_factory=list)
    recent_themes: List[Dict[str, Any]] = Field(default_factory=list)
    identity_rebuild_audit: Dict[str, Any] = Field(default_factory=dict)


class UserModel(BaseModel):
    """The agent's model of the user/operator."""

    model_id: UUID = Field(default_factory=uuid4)
    updated_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    explicit_preferences: Dict[str, Any] = Field(default_factory=dict)
    inferred_goals: List[str] = Field(default_factory=list)
    known_boundaries: List[str] = Field(default_factory=list)
    trust_level: float = Field(default=0.5, ge=0.0, le=1.0)
    uncertainty_areas: List[str] = Field(default_factory=list)
    partner_user_id: Optional[str] = None
    partner_musubi: Optional[float] = None
    partner_trust_raw: Optional[float] = None
    partner_musubi_raw: Optional[float] = None


class ContinuityState(BaseModel):
    """Boot-to-boot continuity bookkeeping."""

    state_id: UUID = Field(default_factory=uuid4)
    updated_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    last_session_id: Optional[str] = None
    last_shutdown_time: Optional[datetime] = None
    boot_count: int = 0
    compaction_count: int = 0
    version: str = "0.1.0"
    source_system: Optional[str] = None
    temporal_bridges: Dict[str, Any] = Field(default_factory=dict)
    integrity_report: Dict[str, Any] = Field(default_factory=dict)

    # Phase 9: Continuous Present
    continuous_present_score: float = Field(default=1.0, ge=0.0, le=1.0)
    last_continuity_monologue: Optional[str] = None
    continuity_decay_rate: float = 0.95  # multiplied per hour of sleep
    continuity_recovery_rate: float = 0.05  # added per meaningful turn
