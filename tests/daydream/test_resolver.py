"""Tests for the reflection resolver."""

import pytest

from opencas.daydream.models import ConflictRecord, DaydreamReflection
from opencas.daydream.resolver import ReflectionResolver
from opencas.somatic.models import SomaticState


@pytest.fixture
def resolver():
    return ReflectionResolver()


def test_resolver_accept(resolver):
    reflection = DaydreamReflection(spark_content="good idea", alignment_score=0.8, keeper=True)
    state = SomaticState(tension=0.2, fatigue=0.2)
    resolution = resolver.resolve(reflection, [], state)
    assert resolution.strategy == "accept"
    assert resolution.reflection_id == reflection.reflection_id
    assert resolution.mirror is not None


def test_resolver_reframe_with_tension(resolver):
    reflection = DaydreamReflection(spark_content="tense idea", alignment_score=0.5, keeper=True)
    state = SomaticState(tension=0.6, fatigue=0.3)
    resolution = resolver.resolve(reflection, [], state)
    assert resolution.strategy == "reframe"
    assert resolution.mirror is not None


def test_resolver_escalate_high_tension_fatigue(resolver):
    reflection = DaydreamReflection(spark_content="overwhelming", alignment_score=0.5, keeper=True)
    state = SomaticState(tension=0.8, fatigue=0.8)
    conflicts = [
        ConflictRecord(kind="energy_vs_ambition", description="too much", occurrence_count=3),
    ]
    resolution = resolver.resolve(reflection, conflicts, state)
    assert resolution.strategy == "escalate"
    assert resolution.conflict_id is not None


def test_resolver_release_low_alignment_recurring(resolver):
    reflection = DaydreamReflection(spark_content="weak spark", alignment_score=0.2, keeper=False)
    state = SomaticState(tension=0.3, fatigue=0.3)
    conflicts = [
        ConflictRecord(kind="obligation_vs_curiosity", description="meh", occurrence_count=5),
    ]
    resolution = resolver.resolve(reflection, conflicts, state)
    assert resolution.strategy == "release"
    assert resolution.conflict_id is not None


def test_resolver_reframe_with_acute_conflict(resolver):
    reflection = DaydreamReflection(spark_content="conflicted", alignment_score=0.6, keeper=True)
    state = SomaticState(tension=0.2, fatigue=0.2)
    conflicts = [
        ConflictRecord(kind="action_vs_avoidance", description="pull", occurrence_count=3),
    ]
    resolution = resolver.resolve(reflection, conflicts, state)
    assert resolution.strategy == "reframe"
    assert resolution.conflict_id is not None
