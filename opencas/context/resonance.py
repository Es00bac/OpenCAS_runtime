"""Pure scoring helpers for emotionally-aware memory retrieval.

These functions are ported and adapted from LegacyPrototype-v4's MemoryFabric
multi-signal fusion engine.
"""

from __future__ import annotations

import math
from datetime import datetime, timezone
from typing import Optional

from opencas.memory import EpisodeEdge
from opencas.somatic.models import AffectState


def _clamp(value: float, min_v: float, max_v: float) -> float:
    return max(min_v, min(max_v, value))


def compute_emotional_resonance(
    query_affect: Optional[AffectState],
    episode_affect: Optional[AffectState],
) -> float:
    """Return [0, 1] resonance between query and episode emotional states."""
    if episode_affect is None:
        return 0.0

    if query_affect is None:
        return _clamp(
            episode_affect.intensity * 0.45 + (episode_affect.valence + 1.0) * 0.2,
            0.0,
            1.0,
        )

    valence_alignment = 1.0 - abs(query_affect.valence - episode_affect.valence) / 2.0
    arousal_alignment = 1.0 - abs(query_affect.arousal - episode_affect.arousal)
    intensity_alignment = 1.0 - abs(query_affect.intensity - episode_affect.intensity)

    primary_bonus = 0.14 if query_affect.primary_emotion == episode_affect.primary_emotion else 0.0
    social_bonus = 0.10 if query_affect.social_target == episode_affect.social_target else 0.0

    return _clamp(
        valence_alignment * 0.45
        + arousal_alignment * 0.25
        + intensity_alignment * 0.20
        + primary_bonus
        + social_bonus,
        0.0,
        1.0,
    )


def compute_temporal_echo(
    query_dt: datetime,
    episode_dt: datetime,
) -> float:
    """Return [0, 1] temporal similarity bonus between two datetimes.

    Rewards same weekday, seasonal proximity, and long-range (>2 day) memories.
    """
    if episode_dt is None:
        return 0.0

    # Ensure both are timezone-aware
    q = query_dt if query_dt.tzinfo else query_dt.replace(tzinfo=timezone.utc)
    e = episode_dt if episode_dt.tzinfo else episode_dt.replace(tzinfo=timezone.utc)

    weekday_match = 0.12 if q.weekday() == e.weekday() else 0.0

    q_doy = q.timetuple().tm_yday
    e_doy = e.timetuple().tm_yday
    doy_delta = abs(q_doy - e_doy)
    seasonal_match = max(0.0, 1.0 - min(doy_delta, 183) / 183.0) * 0.28

    day_distance = abs((q - e).days)
    long_range_bonus = 0.16 if day_distance >= 2 else 0.0

    return _clamp(weekday_match + seasonal_match + long_range_bonus, 0.0, 1.0)


def compute_reliability_score(
    used_successfully: int,
    used_unsuccessfully: int,
) -> float:
    """Return [0, 1] reliability score from usage feedback counters.

    Neutral prior is 0.8. Laplace-smoothed so poor history decays toward 0.5
    and strong history approaches 1.0.
    """
    total = used_successfully + used_unsuccessfully
    if total == 0:
        return 0.8

    smoothed = (used_successfully + 1.0) / (total + 2.0)
    return _clamp(0.5 + (smoothed - 0.5) * 0.6, 0.0, 1.0)


def compute_edge_strength(edge: EpisodeEdge) -> float:
    """Multi-weight edge fusion for graph resonance scoring.

    Goes beyond raw confidence to exploit the rich EpisodeEdge taxonomy.
    """
    return _clamp(
        edge.confidence * 0.55
        + edge.semantic_weight * 0.20
        + edge.emotional_weight * 0.15
        + edge.recency_weight * 0.05
        + edge.structural_weight * 0.05,
        0.0,
        1.0,
    )
