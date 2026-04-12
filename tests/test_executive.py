"""Tests for the executive state tracker."""

from pathlib import Path
import pytest

from opencas.autonomy import WorkObject, WorkStage
from opencas.autonomy.executive import ExecutiveState
from opencas.identity import IdentityManager, IdentityStore
from opencas.somatic import SomaticManager
from opencas.telemetry import TelemetryStore, Tracer


@pytest.fixture
def identity(tmp_path: Path):
    store = IdentityStore(tmp_path / "identity")
    mgr = IdentityManager(store)
    mgr.load()
    return mgr


@pytest.fixture
def somatic(tmp_path: Path):
    return SomaticManager(tmp_path / "somatic.json")


@pytest.fixture
def tracer(tmp_path: Path):
    store = TelemetryStore(tmp_path / "telemetry")
    return Tracer(store)


@pytest.fixture
def executive(identity, somatic, tracer):
    return ExecutiveState(identity=identity, somatic=somatic, tracer=tracer)


def test_intention_set(executive: ExecutiveState) -> None:
    executive.set_intention("plan the day")
    assert executive.intention == "plan the day"


def test_goals_sync_to_identity(executive: ExecutiveState, identity: IdentityManager) -> None:
    executive.add_goal("learn rust")
    assert "learn rust" in identity.self_model.current_goals
    executive.remove_goal("learn rust")
    assert "learn rust" not in identity.self_model.current_goals


def test_enqueue_and_dequeue(executive: ExecutiveState) -> None:
    w1 = WorkObject(content="a", stage=WorkStage.SPARK, promotion_score=1.0)
    w2 = WorkObject(content="b", stage=WorkStage.NOTE, promotion_score=2.0)
    assert executive.enqueue(w1) is True
    assert executive.enqueue(w2) is True
    assert executive.capacity_remaining == 3

    highest = executive.dequeue()
    assert highest is not None
    assert highest.work_id == w2.work_id  # higher score first


def test_capacity_limit(executive: ExecutiveState) -> None:
    for i in range(5):
        assert executive.enqueue(WorkObject(content=str(i))) is True
    rejected = WorkObject(content="overflow")
    assert executive.enqueue(rejected) is False
    assert executive.is_overloaded is True


def test_recommend_pause_overload(executive: ExecutiveState) -> None:
    for i in range(5):
        executive.enqueue(WorkObject(content=str(i)))
    assert executive.recommend_pause() is True


def test_recommend_pause_fatigue(executive: ExecutiveState, somatic: SomaticManager) -> None:
    somatic.set_fatigue(0.8)
    assert executive.recommend_pause() is True


def test_snapshot(executive: ExecutiveState) -> None:
    executive.add_goal("test goal")
    executive.enqueue(WorkObject(content="work", stage=WorkStage.SPARK))
    snap = executive.snapshot()
    assert snap["intention"] is None
    assert snap["active_goals"] == ["test goal"]
    assert snap["capacity_remaining"] == 4
    assert snap["queue_size"] == 1
    assert snap["queue_stages"] == ["spark"]
    assert "timestamp" in snap


def test_remove_work(executive: ExecutiveState) -> None:
    w = WorkObject(content="x")
    executive.enqueue(w)
    assert executive.remove_work(str(w.work_id)) is True
    assert executive.remove_work(str(w.work_id)) is False
