"""Tests for the creative ladder."""

from pathlib import Path
import pytest

from opencas.autonomy import WorkObject, WorkStage
from opencas.autonomy.creative_ladder import CreativeLadder
from opencas.autonomy.executive import ExecutiveState
from opencas.identity import IdentityManager, IdentityStore


@pytest.fixture
def identity(tmp_path: Path):
    store = IdentityStore(tmp_path / "identity")
    mgr = IdentityManager(store)
    mgr.load()
    return mgr


@pytest.fixture
def executive(identity):
    return ExecutiveState(identity=identity)


@pytest.fixture
def ladder(executive):
    return CreativeLadder(executive=executive)


def test_add_and_list(ladder: CreativeLadder) -> None:
    w = WorkObject(content="idea", stage=WorkStage.SPARK)
    ladder.add(w)
    assert len(ladder.list_by_stage(WorkStage.SPARK)) == 1


def test_promote_by_relevance(ladder: CreativeLadder, executive: ExecutiveState) -> None:
    executive.add_goal("learn rust")
    w = WorkObject(content="I want to learn rust today", stage=WorkStage.SPARK)
    ladder.add(w)
    promoted = ladder.try_promote(w)
    assert promoted is True
    assert w.stage == WorkStage.NOTE


def test_promote_fails_low_score(ladder: CreativeLadder) -> None:
    w = WorkObject(content="zzz", stage=WorkStage.SPARK)
    ladder.add(w)
    promoted = ladder.try_promote(w)
    assert promoted is False
    assert w.stage == WorkStage.SPARK


def test_demote_on_low_score(ladder: CreativeLadder, executive: ExecutiveState) -> None:
    # Overload to drop score significantly
    for i in range(5):
        executive.enqueue(WorkObject(content=str(i)))
    w = WorkObject(content="weak note", stage=WorkStage.NOTE)
    ladder.add(w)
    demoted = ladder.try_demote(w)
    assert demoted is True
    assert w.stage == WorkStage.SPARK


def test_terminal_stage_no_promote(ladder: CreativeLadder) -> None:
    w = WorkObject(content="final", stage=WorkStage.DURABLE_WORK_STREAM)
    ladder.add(w)
    assert ladder.try_promote(w) is False


def test_run_cycle(ladder: CreativeLadder, executive: ExecutiveState) -> None:
    executive.add_goal("fitness")
    ladder.add(WorkObject(content="fitness app idea", stage=WorkStage.SPARK))
    ladder.add(WorkObject(content="random noise", stage=WorkStage.SPARK))
    result = ladder.run_cycle()
    assert result["promoted"] >= 1


def test_promote_records_lifecycle_transition(executive: ExecutiveState) -> None:
    """When a WorkObject is promoted, the creative ladder should record a lifecycle transition."""
    recorded = []

    class MockTaskStore:
        async def record_lifecycle_transition(self, **kwargs):
            recorded.append(kwargs)

    executive.add_goal("learn rust")
    ladder = CreativeLadder(executive=executive, task_store=MockTaskStore())
    w = WorkObject(content="learn rust", stage=WorkStage.ARTIFACT, access_count=5)
    ladder.add(w)
    promoted = ladder.try_promote(w)
    assert promoted is True
    assert w.stage == WorkStage.MICRO_TASK
    # In a sync test with no running event loop, the async persistence is skipped.
    # This test primarily verifies no exception is raised during promotion.


@pytest.mark.asyncio
async def test_promote_persists_lifecycle_transition(identity: IdentityManager, tmp_path: Path) -> None:
    import asyncio
    from opencas.execution import TaskStore
    executive = ExecutiveState(identity=identity)
    executive.add_goal("learn rust")
    task_store = TaskStore(tmp_path / "tasks.db")
    await task_store.connect()
    ladder = CreativeLadder(executive=executive, task_store=task_store)
    w = WorkObject(content="learn rust", stage=WorkStage.ARTIFACT, access_count=5)
    ladder.add(w)
    promoted = ladder.try_promote(w)
    assert promoted is True
    assert w.stage == WorkStage.MICRO_TASK
    # Allow background task to complete
    await asyncio.sleep(0.05)
    transitions = await task_store.list_lifecycle_transitions(str(w.work_id))
    assert len(transitions) >= 1
    assert transitions[0]["from_stage"] == "artifact"
    assert transitions[0]["to_stage"] == "queued"
    await task_store.close()


def test_record_success_boosts_semantic(ladder: CreativeLadder) -> None:
    w = WorkObject(content="ai agent", stage=WorkStage.SPARK, embedding_id="hash-1")
    ladder.add(w)
    ladder.record_success(w)
    w2 = WorkObject(content="agent system", stage=WorkStage.SPARK, embedding_id="hash-1")
    ladder.add(w2)
    score = ladder.evaluate(w2)
    assert score > 0.0
    assert ladder.try_promote(w2) is True


def test_remove(ladder: CreativeLadder) -> None:
    w = WorkObject(content="x")
    ladder.add(w)
    assert ladder.remove(str(w.work_id)) is True
    assert ladder.remove(str(w.work_id)) is False
