"""Reflection resolver for daydream sparks."""

from typing import List, Optional
from uuid import UUID, uuid4

from pydantic import BaseModel, Field

from opencas.somatic.models import SomaticState

from .mirror import CompassionResponse, SelfCompassionMirror
from .models import ConflictRecord, DaydreamReflection


class ReflectionResolution(BaseModel):
    """Resolution decision for a daydream reflection."""

    resolution_id: UUID = Field(default_factory=uuid4)
    reflection_id: UUID
    conflict_id: Optional[str] = None
    strategy: str  # accept, reframe, escalate, release
    reason: str
    somatic_nudge: dict = Field(default_factory=dict)
    mirror: Optional[CompassionResponse] = None


class ReflectionResolver:
    """Resolves a daydream reflection against active conflicts and somatic state."""

    def __init__(self, mirror: Optional[SelfCompassionMirror] = None) -> None:
        self.mirror = mirror or SelfCompassionMirror()

    def resolve(
        self,
        reflection: DaydreamReflection,
        conflicts: List[ConflictRecord],
        somatic_state: SomaticState,
    ) -> ReflectionResolution:
        mirror = self.mirror.reflect(somatic_state)

        acute = [c for c in conflicts if c.occurrence_count >= 3 and not c.resolved]
        high_tension = somatic_state.tension > 0.7 or somatic_state.fatigue > 0.7
        low_alignment = reflection.alignment_score < 0.35
        recurring = any(c.occurrence_count >= 5 for c in acute)

        if high_tension and acute:
            return ReflectionResolution(
                reflection_id=reflection.reflection_id,
                conflict_id=str(acute[0].conflict_id) if acute else None,
                strategy="escalate",
                reason="High somatic tension/fatigue with acute conflicts requires operator attention.",
                somatic_nudge=mirror.somatic_nudge,
                mirror=mirror,
            )

        if low_alignment and recurring:
            return ReflectionResolution(
                reflection_id=reflection.reflection_id,
                conflict_id=str(acute[0].conflict_id) if acute else None,
                strategy="release",
                reason="Low alignment and recurring conflict suggest letting this reflection go.",
                somatic_nudge=mirror.somatic_nudge,
                mirror=mirror,
            )

        if somatic_state.tension > 0.4 or acute or mirror.suggested_strategy == "reframe":
            return ReflectionResolution(
                reflection_id=reflection.reflection_id,
                conflict_id=str(conflicts[0].conflict_id) if conflicts else None,
                strategy="reframe",
                reason="Tension is present but manageable; reframe with self-compassion.",
                somatic_nudge=mirror.somatic_nudge,
                mirror=mirror,
            )

        return ReflectionResolution(
            reflection_id=reflection.reflection_id,
            strategy="accept",
            reason="Alignment is good and no acute conflicts; integrate spark.",
            somatic_nudge=mirror.somatic_nudge,
            mirror=mirror,
        )
