"""Self-compassion strategy selection for daydream reflection resolution."""

from pydantic import BaseModel, Field

from opencas.cognition import CognitionGrounding, GroundingKind, GroundingSource
from opencas.somatic.models import SomaticState

_LEGACY_COMPASSION_PREFIXES = (
    "Rest is part of the work. You don't need to push through exhaustion.",
    "Pacing is wisdom. Not every tension needs to be resolved right now.",
    "Your worth is not measured by a single output. You are allowed to try again.",
    "You have the resources you need. Trust your capability.",
    "Stay with the process. Curiosity will carry you through.",
)


def strip_legacy_compassion_prefix(text: str) -> str:
    """Remove old canned mirror prefixes from loaded daydream text."""
    value = str(text or "")
    for prefix in _LEGACY_COMPASSION_PREFIXES:
        full_prefix = f"{prefix}\n\n"
        if value.startswith(full_prefix):
            return value[len(full_prefix) :].lstrip()
    return value


class CompassionResponse(BaseModel):
    """A grounded strategy response tailored to current somatic control state."""

    reason: str
    somatic_nudge: dict = Field(default_factory=dict)
    suggested_strategy: str = "accept"
    grounding: list[CognitionGrounding] = Field(default_factory=list)
    reflection_text: str = ""


class SelfCompassionMirror:
    """Selects regulation strategies without emitting canned inner speech."""

    def reflect(self, state: SomaticState) -> CompassionResponse:
        if state.fatigue > 0.7:
            return CompassionResponse(
                reason="fatigue_high",
                somatic_nudge={
                    "fatigue": max(0.0, state.fatigue - 0.05),
                    "tension": max(0.0, state.tension - 0.03),
                },
                suggested_strategy="release" if state.tension < 0.5 else "reframe",
                grounding=[
                    self._somatic_grounding(
                        claim=f"fatigue={state.fatigue:.3f}",
                        confidence=0.9,
                    )
                ],
            )
        if state.tension > 0.7:
            return CompassionResponse(
                reason="tension_high",
                somatic_nudge={
                    "tension": max(0.0, state.tension - 0.05),
                    "arousal": max(0.0, state.arousal - 0.03),
                },
                suggested_strategy="reframe",
                grounding=[
                    self._somatic_grounding(
                        claim=f"tension={state.tension:.3f}",
                        confidence=0.9,
                    )
                ],
            )
        if state.valence < -0.4:
            return CompassionResponse(
                reason="valence_low",
                somatic_nudge={
                    "valence": min(1.0, state.valence + 0.05),
                    "tension": max(0.0, state.tension - 0.02),
                },
                suggested_strategy="release",
                grounding=[
                    self._somatic_grounding(
                        claim=f"valence={state.valence:.3f}",
                        confidence=0.85,
                    )
                ],
            )
        if state.energy > 0.7 and state.valence > 0.3:
            return CompassionResponse(
                reason="energy_positive",
                somatic_nudge={
                    "energy": min(1.0, state.energy + 0.02),
                    "certainty": min(1.0, state.certainty + 0.03),
                },
                suggested_strategy="accept",
                grounding=[
                    self._somatic_grounding(
                        claim=f"energy={state.energy:.3f}, valence={state.valence:.3f}",
                        confidence=0.8,
                    )
                ],
            )
        return CompassionResponse(
            reason="default_process",
            somatic_nudge={"certainty": min(1.0, state.certainty + 0.02)},
            suggested_strategy="accept" if state.tension < 0.4 else "reframe",
            grounding=[
                self._somatic_grounding(
                    claim=(
                        f"tension={state.tension:.3f}, "
                        f"certainty={state.certainty:.3f}"
                    ),
                    confidence=0.6,
                )
            ],
        )

    @staticmethod
    def _somatic_grounding(*, claim: str, confidence: float) -> CognitionGrounding:
        return CognitionGrounding(
            kind=GroundingKind.SOMATIC_METRIC,
            source=GroundingSource.SOMATIC,
            subject="daydream_resolution",
            claim=claim,
            confidence=confidence,
            allowed_surface="internal",
        )
