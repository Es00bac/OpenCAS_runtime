"""Ranking helpers for memory retrieval results."""

from __future__ import annotations

import math
from collections import deque
from typing import List, Optional

from .models import RetrievalResult
from .resonance import compute_emotional_resonance


def apply_temporal_decay(score: float, age_days: float, half_life_days: float = 60.0) -> float:
    """Apply exponential temporal decay to a score."""
    return score * math.exp(-math.log(2) * age_days / half_life_days)


def apply_diversity_penalty(
    results: List[RetrievalResult],
    window: int = 5,
    penalty: float = 0.05,
) -> List[RetrievalResult]:
    """Penalize duplicate primary emotions and source families in a sliding window."""
    if not results:
        return results

    def _emotion_of(result: RetrievalResult) -> Optional[str]:
        episode = getattr(result, "episode", None)
        if episode is not None and getattr(episode, "affect", None) is not None:
            return str(episode.affect.primary_emotion.value)
        return None

    def _family_of(result: RetrievalResult) -> Optional[str]:
        episode = getattr(result, "episode", None)
        if episode is not None:
            return str(episode.kind.value)
        return "memory"

    penalized: List[RetrievalResult] = []
    recent_emotions: deque[Optional[str]] = deque(maxlen=window)
    recent_families: deque[Optional[str]] = deque(maxlen=window)

    for result in results:
        emotion = _emotion_of(result)
        family = _family_of(result)
        deductions = 0
        if emotion is not None and emotion in recent_emotions:
            deductions += 1
        if family in recent_families:
            deductions += 1
        new_score = max(0.0, result.score - penalty * deductions)
        penalized.append(result.model_copy(update={"score": new_score}))
        recent_emotions.append(emotion)
        recent_families.append(family)

    penalized.sort(key=lambda item: item.score, reverse=True)
    return penalized


def apply_emotion_boost(
    results: List[RetrievalResult],
    tag: str,
    boost: float,
    query_affect=None,
) -> List[RetrievalResult]:
    """Blend emotion_boost_tag into emotional resonance instead of substring matching."""
    if not results:
        return results
    boosted: List[RetrievalResult] = []
    for result in results:
        new_score = result.score
        episode = getattr(result, "episode", None)
        if episode is not None and getattr(episode, "affect", None) is not None:
            if episode.affect.primary_emotion.value.lower() == tag.lower():
                new_score = result.score + boost
            if query_affect is not None:
                resonance = compute_emotional_resonance(query_affect, episode.affect)
                new_score = max(new_score, result.score + resonance * boost)
        boosted.append(result.model_copy(update={"score": new_score}))
    boosted.sort(key=lambda item: item.score, reverse=True)
    return boosted
