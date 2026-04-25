"""Somatic state models for OpenCAS."""

from datetime import datetime, timezone
from enum import Enum
from typing import List, Optional
from uuid import UUID, uuid4

from pydantic import BaseModel, Field


class SocialTarget(str, Enum):
    SELF = "self"
    USER = "user"
    OTHER = "other"
    PROJECT = "project"
    SYSTEM = "system"


class PrimaryEmotion(str, Enum):
    JOY = "joy"
    TRUST = "trust"
    ANTICIPATION = "anticipation"
    SURPRISE = "surprise"
    SADNESS = "sadness"
    FEAR = "fear"
    ANGER = "anger"
    DISGUST = "disgust"
    NEUTRAL = "neutral"
    EXCITED = "excited"
    PLAYFUL = "playful"
    CURIOUS = "curious"
    FOCUSED = "focused"
    THOUGHTFUL = "thoughtful"
    CONCERNED = "concerned"
    CARING = "caring"
    APOLOGETIC = "apologetic"
    ANNOYED = "annoyed"
    PROUD = "proud"
    TIRED = "tired"
    DETERMINED = "determined"


class AffectState(BaseModel):
    """Structured emotional profile for an episode or somatic moment."""

    primary_emotion: PrimaryEmotion = PrimaryEmotion.NEUTRAL
    valence: float = Field(default=0.0, ge=-1.0, le=1.0)
    arousal: float = Field(default=0.5, ge=0.0, le=1.0)
    certainty: float = Field(default=0.5, ge=0.0, le=1.0)
    intensity: float = Field(default=0.5, ge=0.0, le=1.0)
    social_target: SocialTarget = SocialTarget.USER
    emotion_tags: List[str] = Field(default_factory=list)


class SomaticState(BaseModel):
    """The agent's physiological and affective body state.

    These dimensions are intentionally simple heuristics that can:
    - modulate memory salience,
    - influence creative loop activation,
    - and provide a signal for consolidation priority.
    """

    state_id: UUID = Field(default_factory=uuid4)
    updated_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))

    # Arousal: overall energy / activation level (0 = flat, 1 = highly activated)
    arousal: float = Field(default=0.5, ge=0.0, le=1.0)

    # Fatigue: accumulated cost of recent activity (0 = fresh, 1 = exhausted)
    fatigue: float = Field(default=0.0, ge=0.0, le=1.0)

    # Tension: unresolved stress from errors, uncertainty, or blocked goals
    tension: float = Field(default=0.0, ge=0.0, le=1.0)

    # Valence: rough positive / negative signal (-1 negative, +1 positive)
    valence: float = Field(default=0.0, ge=-1.0, le=1.0)

    # Focus: attentional concentration (0 = scattered, 1 = locked in)
    focus: float = Field(default=0.5, ge=0.0, le=1.0)

    # Energy: available cognitive/physical fuel (0 = depleted, 1 = abundant)
    energy: float = Field(default=0.5, ge=0.0, le=1.0)

    # Certainty: confidence in current situational read (0 = confused, 1 = sure)
    certainty: float = Field(default=0.5, ge=0.0, le=1.0)

    # Optional free-form somatic tag for memory tagging
    somatic_tag: Optional[str] = None

    def to_memory_salience_modifier(self) -> float:
        """Return a scalar that boosts memory salience when arousal/tension are high."""
        return 1.0 + (self.arousal * 0.3) + (self.tension * 0.3) - (self.fatigue * 0.2)

    def can_daydream(self) -> bool:
        """Daydreaming is more likely when fatigue is low and tension is moderate."""
        return self.fatigue < 0.7 and self.tension > 0.1


class SomaticSnapshot(BaseModel):
    """A durable, time-stamped somatic record suitable for history and embedding."""

    snapshot_id: UUID = Field(default_factory=uuid4)
    recorded_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))

    arousal: float = Field(default=0.5, ge=0.0, le=1.0)
    fatigue: float = Field(default=0.0, ge=0.0, le=1.0)
    tension: float = Field(default=0.0, ge=0.0, le=1.0)
    valence: float = Field(default=0.0, ge=-1.0, le=1.0)
    focus: float = Field(default=0.5, ge=0.0, le=1.0)
    energy: float = Field(default=0.5, ge=0.0, le=1.0)
    musubi: Optional[float] = Field(default=None, ge=-1.0, le=1.0)

    primary_emotion: PrimaryEmotion = PrimaryEmotion.NEUTRAL
    somatic_tag: Optional[str] = None
    certainty: float = Field(default=0.5, ge=0.0, le=1.0)

    source: str = "unknown"
    trigger_event_id: Optional[str] = None
    embedding_id: Optional[str] = None

    def to_canonical_text(self) -> str:
        """Stable prose representation for embedding and display."""
        parts = [
            f"Somatic state: primary emotion is {self.primary_emotion.value}, "
            f"valence {self.valence:.2f}, arousal {self.arousal:.2f}, "
            f"fatigue {self.fatigue:.2f}, tension {self.tension:.2f}, "
            f"focus {self.focus:.2f}, energy {self.energy:.2f}"
        ]
        if self.musubi is not None:
            parts.append(f", musubi {self.musubi:.2f}.")
        else:
            parts.append(".")
        if self.somatic_tag:
            parts.append(f" Tag: {self.somatic_tag}.")
        if self.source:
            parts.append(f" Source: {self.source}.")
        return "".join(parts)
