"""Tests for the self-compassion mirror."""

import pytest

from opencas.daydream.mirror import SelfCompassionMirror
from opencas.somatic.models import SomaticState


@pytest.fixture
def mirror():
    return SelfCompassionMirror()


def test_mirror_fatigue_suggests_release(mirror):
    state = SomaticState(fatigue=0.8, tension=0.3)
    resp = mirror.reflect(state)
    assert "rest" in resp.affirmation.lower()
    assert resp.suggested_strategy == "release"
    assert resp.somatic_nudge["fatigue"] < state.fatigue


def test_mirror_fatigue_high_tension_suggests_reframe(mirror):
    state = SomaticState(fatigue=0.8, tension=0.8)
    resp = mirror.reflect(state)
    assert "rest" in resp.affirmation.lower()
    assert resp.suggested_strategy == "reframe"


def test_mirror_tension_suggests_reframe(mirror):
    state = SomaticState(tension=0.8, arousal=0.6)
    resp = mirror.reflect(state)
    assert "pacing" in resp.affirmation.lower() or "tension" in resp.affirmation.lower()
    assert resp.suggested_strategy == "reframe"
    assert resp.somatic_nudge["tension"] < state.tension


def test_mirror_low_valence_suggests_release(mirror):
    state = SomaticState(valence=-0.6)
    resp = mirror.reflect(state)
    assert "worth" in resp.affirmation.lower() or "measured" in resp.affirmation.lower()
    assert resp.suggested_strategy == "release"
    assert resp.somatic_nudge["valence"] > state.valence


def test_mirror_high_energy_accepts(mirror):
    state = SomaticState(energy=0.8, valence=0.5)
    resp = mirror.reflect(state)
    assert "capability" in resp.affirmation.lower() or "resources" in resp.affirmation.lower()
    assert resp.suggested_strategy == "accept"
    assert resp.somatic_nudge["energy"] >= state.energy


def test_mirror_default(mirror):
    state = SomaticState()
    resp = mirror.reflect(state)
    assert "process" in resp.affirmation.lower() or "curiosity" in resp.affirmation.lower()
    assert resp.suggested_strategy in ("accept", "reframe")
