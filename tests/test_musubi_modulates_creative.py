"""Tests that musubi modulates creative ladder scoring."""

from pathlib import Path

from opencas.autonomy import WorkObject, WorkStage
from opencas.autonomy.creative_ladder import CreativeLadder
from opencas.autonomy.executive import ExecutiveState
from opencas.identity import IdentityManager, IdentityStore
from opencas.relational import RelationalEngine, MusubiStore, MusubiState


def make_identity(tmp_path: Path) -> IdentityManager:
    store = IdentityStore(tmp_path / "identity")
    mgr = IdentityManager(store)
    mgr.load()
    return mgr


def test_musubi_boosts_creative_score(tmp_path: Path) -> None:
    identity = make_identity(tmp_path)
    executive = ExecutiveState(identity=identity)
    executive.add_goal("learn rust")

    store = MusubiStore(tmp_path / "relational.db")
    rel = RelationalEngine(store)
    rel._state = MusubiState(musubi=0.8)

    ladder_with = CreativeLadder(executive=executive, relational=rel)
    ladder_without = CreativeLadder(executive=executive)

    work = WorkObject(content="I want to learn rust today", stage=WorkStage.SPARK)
    score_with = ladder_with.evaluate(work)
    score_without = ladder_without.evaluate(work)

    assert score_with > score_without


def test_musubi_boost_requires_goal_alignment(tmp_path: Path) -> None:
    identity = make_identity(tmp_path)
    executive = ExecutiveState(identity=identity)
    executive.add_goal("fitness")

    store = MusubiStore(tmp_path / "relational.db")
    rel = RelationalEngine(store)
    rel._state = MusubiState(musubi=0.8)

    ladder = CreativeLadder(executive=executive, relational=rel)

    # Work aligned with goals gets boost
    aligned = WorkObject(content="fitness app idea", stage=WorkStage.SPARK)
    score_aligned = ladder.evaluate(aligned)

    # Work not aligned with goals gets no boost
    unaligned = WorkObject(content="random poetry", stage=WorkStage.SPARK)
    score_unaligned = ladder.evaluate(unaligned)

    assert score_aligned > score_unaligned
