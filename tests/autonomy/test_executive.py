"""Tests for ExecutiveState."""

from pathlib import Path

import pytest
import pytest_asyncio

from opencas.autonomy.commitment import Commitment, CommitmentStatus
from opencas.autonomy.commitment_store import CommitmentStore
from opencas.autonomy.executive import ExecutiveState, ExecutiveSnapshot
from opencas.autonomy.models import WorkObject, WorkStage
from opencas.identity import IdentityManager, IdentityStore


@pytest.fixture
def identity(tmp_path: Path):
    store = IdentityStore(tmp_path / "identity")
    mgr = IdentityManager(store)
    mgr.load()
    return mgr


@pytest.fixture
def executive(identity: IdentityManager, tmp_path: Path):
    exec_state = ExecutiveState(identity=identity)
    exec_state._snapshot_path = tmp_path / "executive.json"
    return exec_state


def test_load_save_snapshot(executive: ExecutiveState, tmp_path: Path) -> None:
    executive.set_intention("test intention")
    executive.add_goal("goal one")
    executive.add_goal("goal two")

    executive.save_snapshot()

    # Create fresh executive and load
    fresh = ExecutiveState(identity=executive.identity)
    fresh.load_snapshot(tmp_path / "executive.json")

    assert fresh.intention == "test intention"
    assert "goal one" in fresh.active_goals
    assert "goal two" in fresh.active_goals


def test_restore_goals_from_identity(identity: IdentityManager, tmp_path: Path) -> None:
    identity.self_model.current_goals = ["hydrate goals", "from identity"]
    identity.save()

    executive = ExecutiveState(identity=identity)
    count = executive.restore_goals_from_identity()

    assert count == 2
    assert "hydrate goals" in executive.active_goals
    assert "from identity" in executive.active_goals


@pytest.mark.asyncio
async def test_check_goal_resolution(executive: ExecutiveState) -> None:
    executive.add_goal("rewrite the readme")
    executive.add_goal("fix the login bug")
    executive.add_goal("unrelated goal")

    resolved = await executive.check_goal_resolution("I have rewritten the README file for clarity.")

    assert "rewrite the readme" in resolved
    assert "fix the login bug" not in resolved
    assert "unrelated goal" not in resolved
    assert "rewrite the readme" not in executive.active_goals


@pytest_asyncio.fixture
async def exec_with_commitments(tmp_path: Path):
    store = CommitmentStore(tmp_path / "commitments.db")
    await store.connect()
    identity_store = IdentityStore(tmp_path / "identity")
    identity = IdentityManager(identity_store)
    identity.load()
    executive = ExecutiveState(identity=identity, commitment_store=store)
    yield executive, store
    await store.close()


@pytest.mark.asyncio
async def test_check_goal_resolution_with_commitments(exec_with_commitments) -> None:
    executive, store = exec_with_commitments
    c = Commitment(content="rewrite the readme", priority=8.0)
    await store.save(c)

    resolved = await executive.check_goal_resolution("I have rewritten the README file for clarity.")

    assert "rewrite the readme" in resolved
    fetched = await store.get(str(c.commitment_id))
    assert fetched.status == CommitmentStatus.COMPLETED


@pytest.mark.asyncio
async def test_enqueue_accepts_commitment_id(exec_with_commitments) -> None:
    executive, _ = exec_with_commitments
    wo = WorkObject(content="do work", stage=WorkStage.MICRO_TASK)
    assert executive.enqueue(wo, commitment_id="c-123") is True
    assert wo.commitment_id == "c-123"


def test_dequeue_ordering_and_capacity(executive: ExecutiveState) -> None:
    # Lower score / later created should come after higher score / earlier created
    w1 = WorkObject(content="low", stage=WorkStage.MICRO_TASK, promotion_score=0.1)
    w2 = WorkObject(content="high", stage=WorkStage.MICRO_TASK, promotion_score=0.9)
    w3 = WorkObject(content="medium", stage=WorkStage.MICRO_TASK, promotion_score=0.5)

    executive.enqueue(w1)
    executive.enqueue(w2)
    executive.enqueue(w3)

    first = executive.dequeue()
    second = executive.dequeue()
    third = executive.dequeue()

    assert first is not None and first.content == "high"
    assert second is not None and second.content == "medium"
    assert third is not None and third.content == "low"
    assert executive.dequeue() is None


def test_enqueue_respects_capacity(identity: IdentityManager) -> None:
    executive = ExecutiveState(identity=identity)
    executive._max_capacity = 2

    w1 = WorkObject(content="one", stage=WorkStage.MICRO_TASK)
    w2 = WorkObject(content="two", stage=WorkStage.MICRO_TASK)
    w3 = WorkObject(content="three", stage=WorkStage.MICRO_TASK)

    assert executive.enqueue(w1) is True
    assert executive.enqueue(w2) is True
    assert executive.enqueue(w3) is False
    assert len(executive.task_queue) == 2
