"""Unit tests for resonance scoring helpers."""

from datetime import datetime, timezone

import pytest

from opencas.context.resonance import (
    compute_edge_strength,
    compute_emotional_resonance,
    compute_reliability_score,
    compute_temporal_echo,
)
from opencas.memory import EpisodeEdge
from opencas.somatic.models import AffectState, PrimaryEmotion, SocialTarget


def test_emotional_resonance_no_episode_affect() -> None:
    assert compute_emotional_resonance(None, None) == 0.0


def test_emotional_resonance_no_query_affect() -> None:
    ep = AffectState(
        primary_emotion=PrimaryEmotion.JOY,
        valence=0.8,
        arousal=0.7,
        intensity=0.9,
    )
    # Falls back to intensity*0.45 + (valence+1)*0.2
    expected = 0.9 * 0.45 + (0.8 + 1.0) * 0.2
    assert compute_emotional_resonance(None, ep) == pytest.approx(expected, abs=0.01)


def test_emotional_resonance_identical_states() -> None:
    aff = AffectState(
        primary_emotion=PrimaryEmotion.JOY,
        valence=0.5,
        arousal=0.6,
        intensity=0.7,
        social_target=SocialTarget.USER,
    )
    score = compute_emotional_resonance(aff, aff)
    # Raw sum would be 0.45+0.25+0.20+0.14+0.10=1.14, but result is clamped to 1.0
    assert score == pytest.approx(1.0, abs=0.01)


def test_emotional_resonance_different_primary() -> None:
    q = AffectState(
        primary_emotion=PrimaryEmotion.ANGER,
        valence=-0.5,
        arousal=0.8,
        intensity=0.9,
        social_target=SocialTarget.OTHER,
    )
    e = AffectState(
        primary_emotion=PrimaryEmotion.JOY,
        valence=0.5,
        arousal=0.2,
        intensity=0.3,
        social_target=SocialTarget.USER,
    )
    score = compute_emotional_resonance(q, e)
    valence_alignment = 1.0 - abs(-0.5 - 0.5) / 2.0  # 0.5
    arousal_alignment = 1.0 - abs(0.8 - 0.2)  # 0.4
    intensity_alignment = 1.0 - abs(0.9 - 0.3)  # 0.4
    # No primary or social bonuses since emotions and targets differ
    expected = valence_alignment * 0.45 + arousal_alignment * 0.25 + intensity_alignment * 0.20
    assert score == pytest.approx(expected, abs=0.01)


def test_temporal_echo_none() -> None:
    now = datetime.now(timezone.utc)
    assert compute_temporal_echo(now, None) == 0.0


def test_temporal_echo_weekday_match() -> None:
    # Two datetimes on the same weekday but far apart in year
    dt1 = datetime(2024, 1, 8, tzinfo=timezone.utc)  # Monday
    dt2 = datetime(2024, 6, 10, tzinfo=timezone.utc)  # Monday
    score = compute_temporal_echo(dt1, dt2)
    weekday_match = 0.12
    # doy delta ~154 -> seasonal_match ~ (1 - 154/183) * 0.28 ≈ 0.044
    day_distance = abs((dt1 - dt2).days)
    long_range = 0.16 if day_distance >= 2 else 0.0
    expected = weekday_match + 0.044 + long_range
    assert score == pytest.approx(expected, abs=0.02)


def test_temporal_echo_long_range_bonus() -> None:
    dt1 = datetime(2024, 1, 1, tzinfo=timezone.utc)
    dt2 = datetime(2024, 1, 10, tzinfo=timezone.utc)
    score = compute_temporal_echo(dt1, dt2)
    assert score >= 0.16  # long_range_bonus present


def test_reliability_empty_history() -> None:
    assert compute_reliability_score(0, 0) == 0.8


def test_reliability_perfect_history() -> None:
    # smoothed = (100+1)/(100+2) ≈ 0.9901 -> 0.5 + (0.4901)*0.6 ≈ 0.794
    score = compute_reliability_score(100, 0)
    assert score == pytest.approx(0.794, abs=0.01)


def test_reliability_poor_history() -> None:
    score = compute_reliability_score(0, 100)
    # smoothed = 1/102 ≈ 0.0098 -> 0.5 + (-0.4902)*0.6 ≈ 0.206
    assert score < 0.3


def test_reliability_trend_improves() -> None:
    poor = compute_reliability_score(1, 9)
    good = compute_reliability_score(9, 1)
    assert good > poor


def test_compute_edge_strength_defaults() -> None:
    edge = EpisodeEdge(
        source_id="a",
        target_id="b",
        confidence=1.0,
        semantic_weight=0.0,
        emotional_weight=0.0,
        recency_weight=0.0,
        structural_weight=0.0,
    )
    assert compute_edge_strength(edge) == pytest.approx(0.55, abs=0.01)


def test_compute_edge_strength_maxed() -> None:
    edge = EpisodeEdge(
        source_id="a",
        target_id="b",
        confidence=1.0,
        semantic_weight=1.0,
        emotional_weight=1.0,
        recency_weight=1.0,
        structural_weight=1.0,
    )
    score = compute_edge_strength(edge)
    # 1.0*0.55 + 1.0*0.20 + 1.0*0.15 + 1.0*0.05 + 1.0*0.05 = 1.0
    assert score == pytest.approx(1.0, abs=0.01)


def test_compute_edge_strength_clamped() -> None:
    from types import SimpleNamespace
    # Bypass Pydantic validation to test runtime clamping
    edge = SimpleNamespace(
        confidence=2.0,
        semantic_weight=1.0,
        emotional_weight=1.0,
        recency_weight=1.0,
        structural_weight=1.0,
    )
    assert compute_edge_strength(edge) == pytest.approx(1.0, abs=0.01)
