"""Data models for the relational resonance (musubi) subsystem."""

from datetime import datetime, timezone
from enum import Enum
from typing import Any, Dict, Optional
from uuid import UUID, uuid4

from pydantic import BaseModel, Field


class ResonanceDimension(str, Enum):
    """Dimensions of relational resonance."""

    TRUST = "trust"              # reliability, boundary respect
    RESONANCE = "resonance"      # emotional/creative alignment
    PRESENCE = "presence"        # quality of recent attention/contact
    ATTUNEMENT = "attunement"    # understanding of each other's intent


class MusubiState(BaseModel):
    """Current relational field snapshot."""

    state_id: UUID = Field(default_factory=uuid4)
    updated_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    musubi: float = Field(default=0.0, ge=-1.0, le=1.0)
    dimensions: Dict[str, float] = Field(default_factory=dict)
    source_tag: Optional[str] = None

    def model_post_init(self, __context: Any) -> None:
        ensure_keys = [ResonanceDimension.TRUST, ResonanceDimension.RESONANCE,
                       ResonanceDimension.PRESENCE, ResonanceDimension.ATTUNEMENT]
        for key in ensure_keys:
            if key.value not in self.dimensions:
                self.dimensions[key.value] = 0.0


class MusubiRecord(BaseModel):
    """A single event in the musubi time-series."""

    record_id: UUID = Field(default_factory=uuid4)
    created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    musubi_before: float = 0.0
    musubi_after: float = 0.0
    delta: float = 0.0
    dimension_deltas: Dict[str, float] = Field(default_factory=dict)
    trigger_event: str = "unknown"
    episode_id: Optional[str] = None
    note: Optional[str] = None
