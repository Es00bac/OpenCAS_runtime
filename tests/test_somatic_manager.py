from pathlib import Path

from opencas.somatic import SomaticManager
from opencas.somatic.models import AffectState, PrimaryEmotion


def test_reflect_structural_load_uses_continuity_pressure_for_parked_residue(tmp_path: Path) -> None:
    somatic = SomaticManager(tmp_path / "somatic.json")

    somatic.reflect_structural_load(
        weighted_queue_load=0.0,
        queue_depth=0,
        active_goal_count=0,
        parked_goal_count=10,
    )

    assert somatic.state.somatic_tag == "continuity_pressure"


def test_reflect_structural_load_keeps_crowded_for_live_clutter(tmp_path: Path) -> None:
    somatic = SomaticManager(tmp_path / "somatic.json")

    somatic.reflect_structural_load(
        weighted_queue_load=1.2,
        queue_depth=2,
        active_goal_count=1,
        parked_goal_count=10,
    )

    assert somatic.state.somatic_tag == "crowded"


def test_bump_from_work_moves_focus_and_energy(tmp_path: Path) -> None:
    somatic = SomaticManager(tmp_path / "somatic.json")
    start_focus = somatic.state.focus
    start_energy = somatic.state.energy

    somatic.bump_from_work(intensity=0.4, success=True)

    assert somatic.state.focus > start_focus
    assert somatic.state.energy < start_energy


def test_decay_rest_recovers_energy_and_relaxes_focus(tmp_path: Path) -> None:
    somatic = SomaticManager(tmp_path / "somatic.json")
    somatic.set_arousal(0.2)
    somatic.set_tension(0.1)
    somatic.set_energy(0.5)
    somatic.set_focus(0.8)

    somatic.decay()

    assert somatic.state.energy > 0.5
    assert somatic.state.focus < 0.8


def test_warm_reconciliation_nudges_focus_up(tmp_path: Path) -> None:
    somatic = SomaticManager(tmp_path / "somatic.json")
    somatic.set_focus(0.45)
    pre_state = somatic.state.model_copy()
    affect = AffectState(
        primary_emotion=PrimaryEmotion.CARING,
        valence=0.6,
        arousal=0.3,
        certainty=0.8,
        intensity=0.7,
    )

    import asyncio

    asyncio.run(somatic.reconcile(pre_state, affect))

    assert somatic.state.focus > 0.45


def test_appraisal_nudge_moves_focus_and_energy(tmp_path: Path) -> None:
    somatic = SomaticManager(tmp_path / "somatic.json")
    start_focus = somatic.state.focus
    start_energy = somatic.state.energy
    affect = AffectState(
        primary_emotion=PrimaryEmotion.DETERMINED,
        valence=0.1,
        arousal=0.6,
        certainty=0.8,
        intensity=0.7,
    )

    somatic.nudge_from_appraisal(affect)

    assert somatic.state.focus > start_focus
    assert somatic.state.energy < start_energy
