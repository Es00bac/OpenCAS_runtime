"""Tests for the self-compassion mirror."""

import pytest

from opencas.daydream.mirror import SelfCompassionMirror, strip_legacy_compassion_prefix
from opencas.somatic.models import SomaticState


@pytest.fixture
def mirror():
    return SelfCompassionMirror()


def test_mirror_fatigue_suggests_release(mirror):
    state = SomaticState(fatigue=0.8, tension=0.3)
    resp = mirror.reflect(state)
    assert not hasattr(resp, "affirmation")
    assert resp.reason == "fatigue_high"
    assert resp.suggested_strategy == "release"
    assert resp.somatic_nudge["fatigue"] < state.fatigue
    assert resp.grounding[0].kind.value == "somatic_metric"


def test_mirror_fatigue_high_tension_suggests_reframe(mirror):
    state = SomaticState(fatigue=0.8, tension=0.8)
    resp = mirror.reflect(state)
    assert not hasattr(resp, "affirmation")
    assert resp.reason == "fatigue_high"
    assert resp.suggested_strategy == "reframe"


def test_mirror_tension_suggests_reframe(mirror):
    state = SomaticState(tension=0.8, arousal=0.6)
    resp = mirror.reflect(state)
    assert not hasattr(resp, "affirmation")
    assert resp.reason == "tension_high"
    assert resp.suggested_strategy == "reframe"
    assert resp.somatic_nudge["tension"] < state.tension


def test_mirror_low_valence_suggests_release(mirror):
    state = SomaticState(valence=-0.6)
    resp = mirror.reflect(state)
    assert not hasattr(resp, "affirmation")
    assert resp.reason == "valence_low"
    assert resp.suggested_strategy == "release"
    assert resp.somatic_nudge["valence"] > state.valence


def test_mirror_high_energy_accepts(mirror):
    state = SomaticState(energy=0.8, valence=0.5)
    resp = mirror.reflect(state)
    assert not hasattr(resp, "affirmation")
    assert resp.reason == "energy_positive"
    assert resp.suggested_strategy == "accept"
    assert resp.somatic_nudge["energy"] >= state.energy


def test_mirror_default(mirror):
    state = SomaticState()
    resp = mirror.reflect(state)
    assert not hasattr(resp, "affirmation")
    assert resp.reason == "default_process"
    assert resp.suggested_strategy in ("accept", "reframe")


def test_legacy_compassion_prefix_is_removed_from_stored_reflection_text():
    text = (
        "Stay with the process. Curiosity will carry you through.\n\n"
        "A grounded thought remains after the old prefix is removed."
    )

    assert strip_legacy_compassion_prefix(text) == (
        "A grounded thought remains after the old prefix is removed."
    )


def test_legacy_compassion_prefix_leaves_normal_text_unchanged():
    text = "A grounded thought without an old mirror prefix."

    assert strip_legacy_compassion_prefix(text) == text
