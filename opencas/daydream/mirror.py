"""Self-compassion mirror for daydream reflection resolution."""

from pydantic import BaseModel, Field

from opencas.somatic.models import SomaticState


class CompassionResponse(BaseModel):
    """A mirror response tailored to current somatic state."""

    affirmation: str
    somatic_nudge: dict = Field(default_factory=dict)
    suggested_strategy: str = "accept"


class SelfCompassionMirror:
    """Produces state-aware compassionate responses that influence resolution strategy."""

    def reflect(self, state: SomaticState) -> CompassionResponse:
        if state.fatigue > 0.7:
            return CompassionResponse(
                affirmation="Rest is part of the work. You don't need to push through exhaustion.",
                somatic_nudge={
                    "fatigue": max(0.0, state.fatigue - 0.05),
                    "tension": max(0.0, state.tension - 0.03),
                },
                suggested_strategy="release" if state.tension < 0.5 else "reframe",
            )
        if state.tension > 0.7:
            return CompassionResponse(
                affirmation="Pacing is wisdom. Not every tension needs to be resolved right now.",
                somatic_nudge={
                    "tension": max(0.0, state.tension - 0.05),
                    "arousal": max(0.0, state.arousal - 0.03),
                },
                suggested_strategy="reframe",
            )
        if state.valence < -0.4:
            return CompassionResponse(
                affirmation="Your worth is not measured by a single output. You are allowed to try again.",
                somatic_nudge={
                    "valence": min(1.0, state.valence + 0.05),
                    "tension": max(0.0, state.tension - 0.02),
                },
                suggested_strategy="release",
            )
        if state.energy > 0.7 and state.valence > 0.3:
            return CompassionResponse(
                affirmation="You have the resources you need. Trust your capability.",
                somatic_nudge={
                    "energy": min(1.0, state.energy + 0.02),
                    "certainty": min(1.0, state.certainty + 0.03),
                },
                suggested_strategy="accept",
            )
        return CompassionResponse(
            affirmation="Stay with the process. Curiosity will carry you through.",
            somatic_nudge={"certainty": min(1.0, state.certainty + 0.02)},
            suggested_strategy="accept" if state.tension < 0.4 else "reframe",
        )
