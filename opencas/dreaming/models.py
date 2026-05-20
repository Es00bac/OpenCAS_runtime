"""Models for nightly dreams, distinct from waking daydreams."""

from __future__ import annotations

from datetime import datetime, timezone
from enum import Enum
from typing import Any
from uuid import UUID, uuid4

from pydantic import BaseModel, Field, field_validator

from opencas.cognition import CognitionGrounding


class DreamMode(str, Enum):
    """Depth modes for nightly dreaming."""

    LIGHT = "light"
    MEDIUM = "medium"
    HEAVY = "heavy"


class DreamRecord(BaseModel):
    """A persisted nightly dream synthesis record."""

    dream_id: UUID = Field(default_factory=uuid4)
    created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    mode: DreamMode = DreamMode.LIGHT
    source: str = "nightly_consolidation"
    consolidation_result_id: str = ""
    summary: str = ""
    narrative: str = ""
    insights: list[str] = Field(default_factory=list)
    curiosity_seeds: list[str] = Field(default_factory=list)
    action_candidates: list[str] = Field(default_factory=list)
    grounding: list[CognitionGrounding] = Field(default_factory=list)
    artifact_path: str = ""
    meta: dict[str, Any] = Field(default_factory=dict)

    @field_validator(
        "source",
        "consolidation_result_id",
        "summary",
        "narrative",
        "artifact_path",
    )
    @classmethod
    def _normalize_text(cls, value: str) -> str:
        return " ".join(str(value or "").split())

    @field_validator("insights", "curiosity_seeds", "action_candidates", mode="before")
    @classmethod
    def _normalize_list(cls, value: Any) -> list[str]:
        if not isinstance(value, list):
            return []
        cleaned: list[str] = []
        for item in value:
            text = " ".join(str(item or "").split())
            if text:
                cleaned.append(text)
        return cleaned

