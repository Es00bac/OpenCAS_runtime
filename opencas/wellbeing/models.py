"""Data models for operational wellbeing and self-maintenance."""

from __future__ import annotations

from datetime import datetime, timezone
from enum import Enum
from typing import Any
from uuid import UUID, uuid4

from pydantic import BaseModel, Field, field_validator, model_validator

from opencas.cognition import CognitionGrounding


class WellbeingDimension(str, Enum):
    """Operational integrity dimensions tracked by the wellbeing model."""

    COHERENCE = "coherence"
    AUTONOMY = "autonomy"
    RECOVERY_NEED = "recovery_need"
    PROMISE_LOAD = "promise_load"
    BOUNDARY_PRESSURE = "boundary_pressure"
    RELATIONSHIP_PRESSURE = "relationship_pressure"
    CURIOSITY = "curiosity"
    DRIFT_LOAD = "drift_load"
    TRUTH_PRESSURE = "truth_pressure"


class MaintenanceActionType(str, Enum):
    """Bounded self-maintenance routes."""

    RECOVER = "recover"
    PRESERVE_PROMISES = "preserve_promises"
    TRUTH_REPAIR = "truth_repair"
    BOUNDARY_CHECK = "boundary_check"
    DEEP_THINK = "deep_think"
    CURIOSITY_INCUBATE = "curiosity_incubate"
    SELF_MODIFICATION_PROPOSAL = "self_modification_proposal"
    OPERATOR_SUMMARY = "operator_summary"


class WellbeingState(BaseModel):
    """Current operational wellbeing estimate.

    These scores are control-plane signals, not claims about phenomenal
    experience. Higher pressure dimensions increase operational risk; higher
    coherence, autonomy, and curiosity reduce it.
    """

    state_id: UUID = Field(default_factory=uuid4)
    created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    coherence: float = Field(default=0.7, ge=0.0, le=1.0)
    autonomy: float = Field(default=0.7, ge=0.0, le=1.0)
    recovery_need: float = Field(default=0.0, ge=0.0, le=1.0)
    promise_load: float = Field(default=0.0, ge=0.0, le=1.0)
    boundary_pressure: float = Field(default=0.0, ge=0.0, le=1.0)
    relationship_pressure: float = Field(default=0.0, ge=0.0, le=1.0)
    curiosity: float = Field(default=0.5, ge=0.0, le=1.0)
    drift_load: float = Field(default=0.0, ge=0.0, le=1.0)
    truth_pressure: float = Field(default=0.0, ge=0.0, le=1.0)
    overall_risk: float = Field(default=0.0, ge=0.0, le=1.0)
    grounding: list[CognitionGrounding] = Field(default_factory=list)
    meta: dict[str, Any] = Field(default_factory=dict)

    @field_validator(
        "coherence",
        "autonomy",
        "recovery_need",
        "promise_load",
        "boundary_pressure",
        "relationship_pressure",
        "curiosity",
        "drift_load",
        "truth_pressure",
        "overall_risk",
        mode="before",
    )
    @classmethod
    def _clamp_dimension(cls, value: Any) -> float:
        try:
            parsed = float(value)
        except (TypeError, ValueError):
            return 0.0
        return round(max(0.0, min(1.0, parsed)), 3)

    @model_validator(mode="after")
    def _derive_overall_risk(self) -> "WellbeingState":
        pressures = [
            1.0 - self.coherence,
            1.0 - self.autonomy,
            1.0 - self.curiosity,
            self.recovery_need,
            self.promise_load,
            self.boundary_pressure,
            self.relationship_pressure,
            self.drift_load,
            self.truth_pressure,
        ]
        derived = round(max(0.0, min(1.0, sum(pressures) / len(pressures))), 3)
        self.overall_risk = max(self.overall_risk, derived)
        return self


class MaintenanceAction(BaseModel):
    """A bounded action proposed by self-maintenance planning."""

    action_type: MaintenanceActionType
    reason: str
    priority: float = Field(default=0.5, ge=0.0, le=1.0)
    grounding: list[CognitionGrounding] = Field(default_factory=list)
    meta: dict[str, Any] = Field(default_factory=dict)

    @field_validator("reason")
    @classmethod
    def _normalize_reason(cls, value: str) -> str:
        return " ".join(str(value or "").split())

    @field_validator("priority", mode="before")
    @classmethod
    def _clamp_priority(cls, value: Any) -> float:
        try:
            parsed = float(value)
        except (TypeError, ValueError):
            return 0.5
        return round(max(0.0, min(1.0, parsed)), 3)


class MaintenanceOutcome(BaseModel):
    """Observed outcome of a maintenance action."""

    outcome_id: UUID = Field(default_factory=uuid4)
    created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    action_type: str
    before_risk: float = Field(default=0.0, ge=0.0, le=1.0)
    after_risk: float = Field(default=0.0, ge=0.0, le=1.0)
    outcome: str = ""
    evidence_ids: list[str] = Field(default_factory=list)
    meta: dict[str, Any] = Field(default_factory=dict)

    @field_validator("action_type", "outcome")
    @classmethod
    def _normalize_text(cls, value: str) -> str:
        return " ".join(str(value or "").split())

    @field_validator("before_risk", "after_risk", mode="before")
    @classmethod
    def _clamp_risk(cls, value: Any) -> float:
        try:
            parsed = float(value)
        except (TypeError, ValueError):
            return 0.0
        return round(max(0.0, min(1.0, parsed)), 3)


class LearnedMaintenanceBehavior(BaseModel):
    """Learned routing hint from a maintenance outcome."""

    action_type: str
    effect_direction: str
    confidence: float = Field(default=0.5, ge=0.0, le=1.0)
    reinforce_action: bool = False
    avoid_action: bool = False
    inspect_next_time: bool = False
    evidence_ids: list[str] = Field(default_factory=list)
    meta: dict[str, Any] = Field(default_factory=dict)

    @field_validator("action_type", "effect_direction")
    @classmethod
    def _normalize_text(cls, value: str) -> str:
        return " ".join(str(value or "").split())


class WellbeingAssessment(BaseModel):
    """A derived wellbeing state plus the dimensions that require attention."""

    assessment_id: UUID = Field(default_factory=uuid4)
    created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    state: WellbeingState
    concern_dimensions: list[WellbeingDimension] = Field(default_factory=list)
    grounding: list[CognitionGrounding] = Field(default_factory=list)
    meta: dict[str, Any] = Field(default_factory=dict)

    @model_validator(mode="after")
    def _derive_concern_dimensions(self) -> "WellbeingAssessment":
        if self.concern_dimensions:
            return self
        concerns: list[WellbeingDimension] = []
        if self.state.coherence < 0.55:
            concerns.append(WellbeingDimension.COHERENCE)
        if self.state.autonomy < 0.55:
            concerns.append(WellbeingDimension.AUTONOMY)
        if self.state.curiosity < 0.35:
            concerns.append(WellbeingDimension.CURIOSITY)
        pressure_thresholds = {
            WellbeingDimension.RECOVERY_NEED: self.state.recovery_need,
            WellbeingDimension.PROMISE_LOAD: self.state.promise_load,
            WellbeingDimension.BOUNDARY_PRESSURE: self.state.boundary_pressure,
            WellbeingDimension.RELATIONSHIP_PRESSURE: self.state.relationship_pressure,
            WellbeingDimension.DRIFT_LOAD: self.state.drift_load,
            WellbeingDimension.TRUTH_PRESSURE: self.state.truth_pressure,
        }
        for dimension, value in pressure_thresholds.items():
            if value >= 0.4:
                concerns.append(dimension)
        self.concern_dimensions = concerns
        return self


class WellbeingEvent(BaseModel):
    """Durable event describing a wellbeing-related runtime observation."""

    event_id: UUID = Field(default_factory=uuid4)
    created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    event_type: str
    summary: str
    grounding: list[CognitionGrounding] = Field(default_factory=list)
    meta: dict[str, Any] = Field(default_factory=dict)

    @field_validator("event_type", "summary")
    @classmethod
    def _normalize_text(cls, value: str) -> str:
        return " ".join(str(value or "").split())


class WellbeingRecommendation(BaseModel):
    """A persisted set of maintenance actions and their current lifecycle state."""

    recommendation_id: UUID = Field(default_factory=uuid4)
    created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    reason: str
    actions: list[MaintenanceAction] = Field(default_factory=list)
    status: str = "proposed"
    grounding: list[CognitionGrounding] = Field(default_factory=list)
    meta: dict[str, Any] = Field(default_factory=dict)

    @field_validator("reason", "status")
    @classmethod
    def _normalize_text(cls, value: str) -> str:
        return " ".join(str(value or "").split())


class SelfModificationProposal(BaseModel):
    """Review-required proposal for changing prompts, config, skills, or code."""

    proposal_id: UUID = Field(default_factory=uuid4)
    created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    title: str
    rationale: str
    target_kind: str
    evidence_ids: list[str] = Field(default_factory=list)
    review_required: bool = True
    status: str = "proposed"
    artifact_path: str = ""
    risk_summary: str = ""
    meta: dict[str, Any] = Field(default_factory=dict)

    @field_validator("title", "rationale", "target_kind", "status", "artifact_path", "risk_summary")
    @classmethod
    def _normalize_text(cls, value: str) -> str:
        return " ".join(str(value or "").split())

    @field_validator("evidence_ids", mode="before")
    @classmethod
    def _normalize_evidence_ids(cls, value: Any) -> list[str]:
        if not isinstance(value, list):
            return []
        return [str(item) for item in value if str(item or "").strip()]
