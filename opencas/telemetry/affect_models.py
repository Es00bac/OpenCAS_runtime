"""Emotional state telemetry models for the provenance mirror.

Maps developer mood signals (commit tone, PR sentiment, incident response patterns)
to build artifacts through the provenance chain.
"""

from __future__ import annotations

from datetime import datetime, timezone
from enum import Enum
from typing import Any, Dict, List, Optional
from uuid import UUID, uuid4

from pydantic import BaseModel, Field


class AffectDimension(str, Enum):
    """Core emotional dimensions tracked across the provenance chain."""

    VALENCE = "valence"          # positive vs negative emotional tone (-1 to 1)
    AROUSAL = "arousal"          # energy/activation level (-1 to 1)
    CERTAINTY = "certainty"      # confidence in approach (-1 to 1)
    COHERENCE = "coherence"      # narrative/logical flow (-1 to 1)
    URGENCY = "urgency"          # time pressure felt (-1 to 1)


class MoodSignalSource(str, Enum):
    """Sources of mood telemetry data."""

    COMMIT_MESSAGE = "commit_message"
    PR_DESCRIPTION = "pr_description"
    PR_REVIEW_COMMENT = "pr_review_comment"
    INCIDENT_RESPONSE = "incident_response"
    ISSUE_COMMENT = "issue_comment"
    CHAT_MESSAGE = "chat_message"
    BUILD_LOG = "build_log"
    MANUAL_ENTRY = "manual_entry"
    INFERRED_FROM_PATTERN = "inferred_from_pattern"


class AffectSnapshot(BaseModel):
    """A single emotional state reading at a point in time."""

    snapshot_id: UUID = Field(default_factory=uuid4)
    timestamp: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    session_id: str = Field(..., description="Links to provenance session_id")
    artifact_id: Optional[str] = Field(None, description="Build artifact or code unit")
    actor: Optional[str] = Field(None, description="Developer identifier")

    # Core affect dimensions (all bounded [-1, 1])
    dimensions: Dict[str, float] = Field(default_factory=dict)

    # Source of this mood reading
    source: MoodSignalSource = MoodSignalSource.INFERRED_FROM_PATTERN
    source_payload: Dict[str, Any] = Field(default_factory=dict)

    # Raw signal that produced this snapshot
    raw_signal: Optional[str] = Field(None, max_length=4096)
    signal_hash: Optional[str] = Field(None, description="Hash of raw signal for dedup")

    # Confidence in the inference (0 to 1)
    confidence: float = Field(default=0.5, ge=0.0, le=1.0)

    def model_post_init(self, __context: Any) -> None:
        ensure_keys = [
            AffectDimension.VALENCE,
            AffectDimension.AROUSAL,
            AffectDimension.CERTAINTY,
            AffectDimension.COHERENCE,
            AffectDimension.URGENCY,
        ]
        for key in ensure_keys:
            if key.value not in self.dimensions:
                self.dimensions[key.value] = 0.0

    @property
    def composite_stress(self) -> float:
        """Derived stress indicator: high arousal + negative valence + high urgency."""
        v = self.dimensions.get(AffectDimension.VALENCE.value, 0.0)
        a = self.dimensions.get(AffectDimension.AROUSAL.value, 0.0)
        u = self.dimensions.get(AffectDimension.URGENCY.value, 0.0)
        # Stress peaks when arousal is high, valence is negative, urgency is high
        stress = (a * 0.4) + ((-v) * 0.35) + (u * 0.25)
        return round(max(0.0, min(1.0, stress)), 4)

    @property
    def composite_flow(self) -> float:
        """Derived flow indicator: positive valence + high coherence + moderate arousal."""
        v = self.dimensions.get(AffectDimension.VALENCE.value, 0.0)
        c = self.dimensions.get(AffectDimension.COHERENCE.value, 0.0)
        a = self.dimensions.get(AffectDimension.AROUSAL.value, 0.0)
        # Flow: positive emotion, clear thinking, engaged but not frantic
        flow = (v * 0.35) + (c * 0.35) + (a * 0.15) + ((1.0 - abs(a - 0.3)) * 0.15)
        return round(max(-1.0, min(1.0, flow)), 4)

    def to_telemetry_payload(self) -> Dict[str, Any]:
        return {
            "snapshot_id": str(self.snapshot_id),
            "timestamp": self.timestamp.isoformat(),
            "session_id": self.session_id,
            "artifact_id": self.artifact_id,
            "actor": self.actor,
            "dimensions": self.dimensions,
            "composite_stress": self.composite_stress,
            "composite_flow": self.composite_flow,
            "source": self.source.value,
            "confidence": self.confidence,
        }


class AffectTrajectory(BaseModel):
    """A time-series of affect snapshots for a session or artifact."""

    trajectory_id: UUID = Field(default_factory=uuid4)
    session_id: str
    artifact_id: Optional[str] = None
    actor: Optional[str] = None
    snapshots: List[AffectSnapshot] = Field(default_factory=list)
    created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    updated_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))

    @property
    def duration_seconds(self) -> float:
        if len(self.snapshots) < 2:
            return 0.0
        return (self.snapshots[-1].timestamp - self.snapshots[0].timestamp).total_seconds()

    @property
    def valence_trend(self) -> float:
        """Linear trend of valence over time (positive = improving mood)."""
        if len(self.snapshots) < 2:
            return 0.0
        vals = [s.dimensions.get(AffectDimension.VALENCE.value, 0.0) for s in self.snapshots]
        # Simple slope: (last - first) / count
        return round((vals[-1] - vals[0]) / max(1, len(vals) - 1), 4)

    @property
    def stress_trend(self) -> float:
        """Trend of stress over time (positive = increasing stress)."""
        if len(self.snapshots) < 2:
            return 0.0
        stresses = [s.composite_stress for s in self.snapshots]
        return round((stresses[-1] - stresses[0]) / max(1, len(stresses) - 1), 4)

    @property
    def volatility(self) -> float:
        """Standard deviation of valence changes between consecutive snapshots."""
        if len(self.snapshots) < 2:
            return 0.0
        vals = [s.dimensions.get(AffectDimension.VALENCE.value, 0.0) for s in self.snapshots]
        deltas = [vals[i] - vals[i - 1] for i in range(1, len(vals))]
        if not deltas:
            return 0.0
        mean = sum(deltas) / len(deltas)
        variance = sum((d - mean) ** 2 for d in deltas) / len(deltas)
        return round(variance ** 0.5, 4)

    def add_snapshot(self, snapshot: AffectSnapshot) -> None:
        self.snapshots.append(snapshot)
        self.updated_at = datetime.now(timezone.utc)


class QualitySignal(BaseModel):
    """Downstream quality metric linked to a provenance artifact."""

    signal_id: UUID = Field(default_factory=uuid4)
    timestamp: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    artifact_id: str
    session_id: Optional[str] = None

    # Quality dimensions
    bug_density: Optional[float] = Field(None, ge=0.0)           # bugs per LOC
    rollback_frequency: Optional[float] = Field(None, ge=0.0)    # rollbacks per deploy
    test_flakiness: Optional[float] = Field(None, ge=0.0, le=1.0)
    technical_debt_score: Optional[float] = Field(None, ge=0.0)
    code_churn: Optional[int] = Field(None, ge=0)                # lines changed
    review_rounds: Optional[int] = Field(None, ge=0)
    ci_failure_rate: Optional[float] = Field(None, ge=0.0, le=1.0)

    # Composite quality index (higher = better quality)
    composite_quality: Optional[float] = Field(None, ge=0.0, le=1.0)

    # Link back to affect trajectory
    affect_trajectory_id: Optional[UUID] = None

    def compute_composite(self) -> float:
        """Compute a weighted composite quality score."""
        scores = []
        weights = []
        if self.bug_density is not None:
            # Invert: fewer bugs = higher score, normalize assuming 0.1 bugs/LOC max
            scores.append(max(0.0, 1.0 - (self.bug_density / 0.1)))
            weights.append(0.25)
        if self.rollback_frequency is not None:
            scores.append(max(0.0, 1.0 - self.rollback_frequency))
            weights.append(0.20)
        if self.test_flakiness is not None:
            scores.append(1.0 - self.test_flakiness)
            weights.append(0.15)
        if self.ci_failure_rate is not None:
            scores.append(1.0 - self.ci_failure_rate)
            weights.append(0.20)
        if self.review_rounds is not None:
            # Fewer review rounds = better (assuming 5+ is bad)
            scores.append(max(0.0, 1.0 - (self.review_rounds / 5.0)))
            weights.append(0.20)
        if not scores:
            return 0.5
        total_weight = sum(weights)
        return round(sum(s * w for s, w in zip(scores, weights)) / total_weight, 4)


class AffectQualityEdge(BaseModel):
    """A dependency edge linking affect state to quality outcome."""

    edge_id: UUID = Field(default_factory=uuid4)
    trajectory_id: UUID
    quality_signal_id: UUID
    artifact_id: str
    session_id: str

    # Temporal relationship
    affect_lead_time_seconds: float  # how long before quality signal the affect was captured

    # Correlation strength
    correlation_coefficient: Optional[float] = Field(None, ge=-1.0, le=1.0)
    p_value: Optional[float] = None

    # Causal inference flags
    is_predictive: bool = False      # affect predicted the quality outcome
    is_intervention_trigger: bool = False  # this edge triggered an intervention

    created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))


class TeamHealthAlert(BaseModel):
    """Alert triggered when affect-quality anomaly is detected."""

    alert_id: UUID = Field(default_factory=uuid4)
    timestamp: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    severity: str = Field(..., pattern="^(info|warning|critical)$")
    alert_type: str = Field(..., description="e.g., 'stress_spike', 'flow_depletion', 'quality_cliff'")

    session_id: Optional[str] = None
    artifact_id: Optional[str] = None
    actor: Optional[str] = None

    # What we observed
    observed_affect: Dict[str, Any] = Field(default_factory=dict)
    predicted_quality_impact: Dict[str, Any] = Field(default_factory=dict)

    # Recommended intervention
    recommended_action: Optional[str] = None
    auto_intervention_triggered: bool = False

    # Resolution tracking
    acknowledged_at: Optional[datetime] = None
    resolved_at: Optional[datetime] = None
    resolution_notes: Optional[str] = None
