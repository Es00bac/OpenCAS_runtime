"""Emotional (affect alignment) scorer."""

from __future__ import annotations

from typing import Any, Optional

from opencas.memory import Episode
from opencas.somatic.models import AffectState


class EmotionalScorer:
    """Score emotional alignment between two episodes."""

    async def score(
        self,
        ep_a: Episode,
        ep_b: Episode,
        context: Optional[Any] = None,
    ) -> float:
        return self._affect_alignment(ep_a.affect, ep_b.affect)

    @staticmethod
    def _affect_alignment(
        affect_a: Optional[AffectState],
        affect_b: Optional[AffectState],
    ) -> float:
        if affect_a is None or affect_b is None:
            return 0.0
        score = 0.0
        if affect_a.primary_emotion == affect_b.primary_emotion:
            score += 0.5
        valence_diff = abs((affect_a.valence or 0.0) - (affect_b.valence or 0.0))
        arousal_diff = abs((affect_a.arousal or 0.0) - (affect_b.arousal or 0.0))
        score += max(0.0, 0.5 - valence_diff) * 0.3
        score += max(0.0, 0.5 - arousal_diff) * 0.2
        return round(min(1.0, score), 4)
