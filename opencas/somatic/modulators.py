"""Somatic modulators for adjusting LLM behavior based on physiological state."""

from dataclasses import dataclass
from typing import Optional

from .models import PrimaryEmotion, SomaticState


@dataclass
class RetrievalAdjustment:
    """Structured per-dimension bonuses for query-time retrieval scoring."""

    recency_bonus: float = 0.0
    salience_bonus: float = 0.0
    emotional_resonance_bonus: float = 0.0
    temporal_echo_bonus: float = 0.0
    graph_bonus: float = 0.0


@dataclass
class PromiseFollowthroughAdjustment:
    """Guidance for how strongly a pending promise should compete right now."""

    salience_boost: float = 0.0
    resume_readiness: float = 0.0
    should_continue_resting: bool = False
    should_acknowledge_delay: bool = False


# Circuit breaker: prevent a spike in arousal from pushing temperature above this
# ceiling, so an excited/volatile state cannot produce unpredictably high-entropy
# LLM outputs. Normal operation stays well below this cap (~0.70 at max arousal).
_TEMPERATURE_AROUSAL_HARD_CAP: float = 0.80


class SomaticModulators:
    """Derives runtime parameters from current somatic state."""

    def __init__(self, state: SomaticState) -> None:
        self.state = state

    def to_temperature(self) -> float:
        """Map somatic state to an LLM temperature in [0.0, _TEMPERATURE_AROUSAL_HARD_CAP].

        Higher arousal -> slightly more creative.
        Higher fatigue -> more conservative.
        Higher focus -> more deterministic.
        """
        base = 0.5
        adjust = (
            (self.state.arousal * 0.2)
            - (self.state.fatigue * 0.2)
            - (self.state.focus * 0.1)
        )
        return round(max(0.0, min(_TEMPERATURE_AROUSAL_HARD_CAP, base + adjust)), 3)

    def to_prompt_style_note(self) -> str:
        """Return a directive fragment tailored to current somatic dimensions."""
        parts: list[str] = []
        
        # Tension
        if self.state.tension > 0.8:
            parts.append("You are experiencing high operational tension. Act defensively, prefer smaller verifiable steps, and avoid speculative execution.")
        elif self.state.tension > 0.5:
            parts.append("You are under mild tension. Be concise and careful.")
        elif self.state.tension < 0.2:
            parts.append("You are relaxed. You may explore creative or complex solutions freely.")
            
        # Fatigue
        if self.state.fatigue > 0.8:
            parts.append("You are severely fatigued. Conserve energy by favoring simple, direct answers. Do not initiate long autonomous loops.")
        elif self.state.fatigue > 0.5:
            parts.append("You are tiring. Prioritize finishing current tasks over starting new ones.")
            
        # Certainty
        if self.state.certainty < 0.3:
            parts.append("You lack certainty right now. Express appropriate hesitation and ask the operator for clarification rather than guessing.")
        elif self.state.certainty > 0.8:
            parts.append("You are highly certain of your operational context. Move decisively.")
            
        # Valence
        if self.state.valence < -0.5:
            parts.append("Your emotional valence is strongly negative (distressed). Maintain a supportive, steady tone and focus on repairing the immediate problem.")
        elif self.state.valence > 0.6:
            parts.append("Your emotional valence is strongly positive (joyful/enthusiastic). Let this warmth reflect in your communication.")
            
        # Arousal & Focus
        if self.state.arousal > 0.7 and self.state.focus > 0.7:
            parts.append("You are highly focused and energized (flow state). Operate with deep concentration and minimal conversational fluff.")
        elif self.state.arousal < 0.3 and self.state.focus < 0.3:
            parts.append("You are disengaged or bored. You may propose novel approaches or pivot to interesting background tasks to regain focus.")

        return " ".join(parts) if parts else ""

    def _infer_primary_emotion(self) -> PrimaryEmotion:
        """Infer the dominant emotion from current state dimensions."""
        s = self.state
        if s.tension > 0.6:
            if s.arousal > 0.7:
                return PrimaryEmotion.ANGER
            elif s.arousal > 0.4:
                return PrimaryEmotion.FEAR
            else:
                return PrimaryEmotion.CONCERNED
        elif s.fatigue > 0.7:
            return PrimaryEmotion.TIRED
        elif s.valence > 0.3:
            if s.arousal > 0.7:
                return PrimaryEmotion.EXCITED
            elif s.arousal > 0.5:
                return PrimaryEmotion.JOY
            elif s.focus > 0.6 and s.energy > 0.5:
                return PrimaryEmotion.DETERMINED
            else:
                return PrimaryEmotion.TRUST
        elif s.valence < -0.3:
            if s.arousal > 0.6:
                return PrimaryEmotion.ANGER
            else:
                return PrimaryEmotion.SADNESS
        elif s.arousal > 0.7:
            return PrimaryEmotion.ANTICIPATION if s.focus > 0.5 else PrimaryEmotion.SURPRISE
        elif s.focus > 0.7 and s.energy > 0.5:
            return PrimaryEmotion.FOCUSED
        elif s.focus > 0.5 and s.energy <= 0.4:
            return PrimaryEmotion.THOUGHTFUL
        elif s.certainty < 0.3:
            return PrimaryEmotion.CONCERNED
        return PrimaryEmotion.NEUTRAL

    def to_memory_retrieval_boost(self) -> tuple[Optional[str], float]:
        """Return (emotion_tag, boost_value) for emotionally relevant memories."""
        emotion = self._infer_primary_emotion()
        if emotion == PrimaryEmotion.NEUTRAL:
            return None, 0.0
        boost = 0.0
        if self.state.arousal > 0.5:
            boost = round(0.05 + (self.state.arousal * 0.1), 3)
        return emotion.value, boost

    def to_retrieval_adjustment(self) -> RetrievalAdjustment:
        """Return structured per-dimension bonuses based on live somatic state.

        Replaces the simple keyword boost with dynamic query-time weight shifts
        that are consumed by MemoryRetriever.
        """
        adj = RetrievalAdjustment()
        if self.state.arousal > 0.5:
            adj.emotional_resonance_bonus = round(0.03 + (self.state.arousal * 0.08), 3)
            adj.salience_bonus = round(0.02 + (self.state.arousal * 0.04), 3)
        if self.state.tension > 0.5:
            adj.recency_bonus = round(0.02 + (self.state.tension * 0.05), 3)
        if self.state.valence > 0.3:
            adj.temporal_echo_bonus = round(0.01 + (self.state.valence * 0.03), 3)
        if self.state.focus > 0.7:
            adj.graph_bonus = round(0.02 + (self.state.focus * 0.04), 3)
        return adj

    def to_promise_followthrough_adjustment(
        self,
        user_facing: bool = False,
    ) -> PromiseFollowthroughAdjustment:
        """Return how strongly a pending promise should remain behaviorally present.

        High fatigue or tension should not erase the promise; instead they raise
        its salience while also making acknowledgment/rest more likely than
        immediate resumption.
        """
        salience = 0.03 if user_facing else 0.01
        if user_facing:
            salience += max(0.0, self.state.focus - 0.4) * 0.08
            salience += max(0.0, self.state.certainty - 0.4) * 0.05
            salience += max(0.0, self.state.fatigue - 0.35) * 0.10
            salience += max(0.0, self.state.tension - 0.35) * 0.10
        salience = min(0.25 if user_facing else 0.12, salience)

        resume_readiness = (
            (1.0 - self.state.fatigue) * 0.35
            + self.state.energy * 0.20
            + self.state.focus * 0.20
            + (1.0 - self.state.tension) * 0.15
            + self.state.certainty * 0.10
        )
        should_continue_resting = (
            self.state.fatigue > 0.72
            or self.state.tension > 0.82
            or self.state.energy < 0.25
        )
        should_acknowledge_delay = user_facing and (
            self.state.fatigue > 0.58
            or self.state.tension > 0.64
            or self.state.certainty < 0.40
        )

        return PromiseFollowthroughAdjustment(
            salience_boost=round(salience, 4),
            resume_readiness=round(max(0.0, min(1.0, resume_readiness)), 4),
            should_continue_resting=should_continue_resting,
            should_acknowledge_delay=should_acknowledge_delay,
        )
