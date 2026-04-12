"""Tests for the somatic module."""

from pathlib import Path
from opencas.somatic import SomaticManager


def test_somatic_load_and_save(tmp_path: Path) -> None:
    path = tmp_path / "somatic.json"
    mgr = SomaticManager(path)
    assert mgr.state.arousal == 0.5

    mgr.set_arousal(0.8)
    assert mgr.state.arousal == 0.8

    mgr2 = SomaticManager(path)
    assert mgr2.state.arousal == 0.8


def test_somatic_clamping(tmp_path: Path) -> None:
    mgr = SomaticManager(tmp_path / "somatic.json")
    mgr.set_fatigue(1.5)
    assert mgr.state.fatigue == 1.0
    mgr.set_valence(-2.0)
    assert mgr.state.valence == -1.0


def test_somatic_decay(tmp_path: Path) -> None:
    mgr = SomaticManager(tmp_path / "somatic.json")
    mgr.set_fatigue(0.5)
    mgr.set_tension(0.5)
    mgr.decay()
    assert mgr.state.fatigue == 0.52
    assert mgr.state.tension == 0.49


def test_somatic_bump_from_work(tmp_path: Path) -> None:
    mgr = SomaticManager(tmp_path / "somatic.json")
    mgr.set_fatigue(0.0)
    mgr.set_tension(0.5)
    mgr.set_valence(0.0)

    mgr.bump_from_work(intensity=0.2, success=True)
    assert mgr.state.fatigue > 0.0
    assert mgr.state.tension < 0.5
    assert mgr.state.valence > 0.0

    mgr.set_fatigue(0.0)
    mgr.set_tension(0.5)
    mgr.set_valence(0.0)
    mgr.bump_from_work(intensity=0.2, success=False)
    assert mgr.state.tension > 0.5
    assert mgr.state.valence < 0.0


def test_somatic_salience_modifier(tmp_path: Path) -> None:
    mgr = SomaticManager(tmp_path / "somatic.json")
    mgr.set_arousal(1.0)
    mgr.set_tension(1.0)
    mgr.set_fatigue(0.0)
    assert mgr.state.to_memory_salience_modifier() > 1.5
