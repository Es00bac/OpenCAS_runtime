"""Tests for AffectState appraisal."""

from pathlib import Path

from opencas.identity import IdentityManager, IdentityStore
from opencas.somatic import SomaticManager
from opencas.somatic.models import PrimaryEmotion, SocialTarget


def make_manager(tmp_path: Path) -> SomaticManager:
    return SomaticManager(tmp_path / "somatic.json")


def test_appraise_joy() -> None:
    mgr = make_manager(Path("/tmp/test_affect_joy"))
    affect = mgr.appraise("I am so happy and joyful today!")
    assert affect.primary_emotion == PrimaryEmotion.JOY
    assert affect.valence > 0.0
    assert affect.arousal > 0.0
    assert affect.social_target == SocialTarget.USER


def test_appraise_sadness() -> None:
    mgr = make_manager(Path("/tmp/test_affect_sad"))
    affect = mgr.appraise("I feel sad and depressed about the result")
    assert affect.primary_emotion == PrimaryEmotion.SADNESS
    assert affect.valence < 0.0


def test_appraise_anger() -> None:
    mgr = make_manager(Path("/tmp/test_affect_angry"))
    affect = mgr.appraise("This makes me furious and frustrated")
    assert affect.primary_emotion == PrimaryEmotion.ANGER
    assert affect.valence < 0.0
    assert affect.arousal > 0.5


def test_appraise_neutral() -> None:
    mgr = make_manager(Path("/tmp/test_affect_neutral"))
    affect = mgr.appraise("The sky is blue")
    assert affect.primary_emotion == PrimaryEmotion.NEUTRAL
    assert affect.valence == 0.0


def test_appraise_outcome_positive() -> None:
    mgr = make_manager(Path("/tmp/test_affect_pos"))
    affect = mgr.appraise("We failed", outcome="positive")
    assert affect.valence > -0.7  # positive outcome softens negative text


def test_appraise_outcome_negative() -> None:
    mgr = make_manager(Path("/tmp/test_affect_neg"))
    affect = mgr.appraise("We succeeded", outcome="negative")
    assert affect.valence < 0.7  # negative outcome dampens positive text
