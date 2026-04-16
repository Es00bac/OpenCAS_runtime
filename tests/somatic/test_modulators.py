"""Tests for SomaticModulators."""

import pytest
from opencas.somatic import SomaticModulators, SomaticState
from opencas.somatic.models import PrimaryEmotion


def test_temperature_base() -> None:
    state = SomaticState(arousal=0.5, fatigue=0.0, focus=0.5)
    mod = SomaticModulators(state)
    temp = mod.to_temperature()
    assert 0.0 <= temp <= 1.0
    # Base 0.5 + (0.5*0.2) - (0*0.2) - (0.5*0.1) = 0.5 + 0.1 - 0 - 0.05 = 0.55
    assert temp == pytest.approx(0.55)


def test_temperature_clamped() -> None:
    state = SomaticState(arousal=1.0, fatigue=0.0, focus=0.0)
    mod = SomaticModulators(state)
    assert mod.to_temperature() == pytest.approx(0.7)

    state = SomaticState(arousal=0.0, fatigue=1.0, focus=1.0)
    mod = SomaticModulators(state)
    assert mod.to_temperature() == pytest.approx(0.2)


def test_prompt_style_note_tension() -> None:
    state = SomaticState(tension=0.7)
    mod = SomaticModulators(state)
    note = mod.to_prompt_style_note()
    assert "concise" in note.lower()


def test_prompt_style_note_fatigue() -> None:
    state = SomaticState(fatigue=0.8)
    mod = SomaticModulators(state)
    note = mod.to_prompt_style_note()
    assert "simple" in note.lower()


def test_prompt_style_note_certainty() -> None:
    state = SomaticState(certainty=0.2)
    mod = SomaticModulators(state)
    note = mod.to_prompt_style_note()
    assert "uncertainty" in note.lower()


def test_prompt_style_note_valence() -> None:
    state = SomaticState(valence=-0.5)
    mod = SomaticModulators(state)
    note = mod.to_prompt_style_note()
    assert "supportive" in note.lower()


def test_prompt_style_note_empty_when_neutral() -> None:
    state = SomaticState()
    mod = SomaticModulators(state)
    assert mod.to_prompt_style_note() == ""


def test_memory_retrieval_boost_neutral() -> None:
    state = SomaticState()
    mod = SomaticModulators(state)
    tag, boost = mod.to_memory_retrieval_boost()
    assert tag is None
    assert boost == 0.0


def test_memory_retrieval_boost_tired_high_arousal() -> None:
    state = SomaticState(fatigue=0.8, arousal=0.6)
    mod = SomaticModulators(state)
    tag, boost = mod.to_memory_retrieval_boost()
    assert tag == PrimaryEmotion.TIRED.value
    assert boost == pytest.approx(0.11)


def test_memory_retrieval_boost_anger() -> None:
    state = SomaticState(tension=0.8, arousal=0.9)
    mod = SomaticModulators(state)
    tag, boost = mod.to_memory_retrieval_boost()
    assert tag == PrimaryEmotion.ANGER.value
    assert boost > 0.0
