"""Tests for the executive state tracker."""

from pathlib import Path
import pytest

from opencas.autonomy import WorkObject, WorkStage
from opencas.autonomy.commitment import Commitment, CommitmentStatus
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


def test_dequeue_prioritizes_user_facing_commitment_work(executive: ExecutiveState) -> None:
    linked = WorkObject(content="linked", stage=WorkStage.MICRO_TASK, promotion_score=0.3)
    background = WorkObject(content="background", stage=WorkStage.MICRO_TASK, promotion_score=0.8)
    commitment = Commitment(
        content="Return to linked work",
        priority=9.0,
        meta={"source": "assistant_response"},
    )
    ExecutiveState.apply_commitment_execution_bias(linked, commitment)

    assert executive.enqueue(background) is True
    assert executive.enqueue(linked) is True

    highest = executive.dequeue()
    assert highest is not None
    assert highest.work_id == linked.work_id


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


@pytest.mark.asyncio
async def test_restore_queue_skips_non_active_commitment_work(
    identity: IdentityManager,
    somatic: SomaticManager,
    tracer: Tracer,
) -> None:
    class FakeCommitmentStore:
        def __init__(self, commitments):
            self.commitments = {str(c.commitment_id): c for c in commitments}

        async def get(self, commitment_id: str):
            return self.commitments.get(commitment_id)

    class FakeWorkStore:
        def __init__(self, work_items):
            self.work_items = {str(w.work_id): w for w in work_items}

        async def list_ready(self, limit: int = 100):
            return list(self.work_items.values())[:limit]

        async def save(self, work):
            self.work_items[str(work.work_id)] = work

    executive = ExecutiveState(
        identity=identity,
        somatic=somatic,
        tracer=tracer,
    )

    active_commitment = Commitment(content="Active commitment")
    blocked_commitment = Commitment(
        content="Blocked commitment",
        status=CommitmentStatus.BLOCKED,
        meta={"blocked_reason": "manual_hold"},
    )
    active_work = WorkObject(
        content="active work",
        stage=WorkStage.MICRO_TASK,
        commitment_id=str(active_commitment.commitment_id),
    )
    blocked_work = WorkObject(
        content="blocked work",
        stage=WorkStage.MICRO_TASK,
        commitment_id=str(blocked_commitment.commitment_id),
    )
    executive.commitment_store = FakeCommitmentStore([active_commitment, blocked_commitment])
    executive.work_store = FakeWorkStore([active_work, blocked_work])

    restored = await executive.restore_queue()
    queued_ids = {str(item.work_id) for item in executive.task_queue}
    assert restored == 1
    assert str(active_work.work_id) in queued_ids
    assert str(blocked_work.work_id) not in queued_ids


@pytest.mark.asyncio
async def test_resume_deferred_work_only_unblocks_auto_resumable_commitments(
    identity: IdentityManager,
    somatic: SomaticManager,
    tracer: Tracer,
) -> None:
    class FakeCommitmentStore:
        def __init__(self, commitments):
            self.commitments = {str(c.commitment_id): c for c in commitments}

        async def list_by_status(self, status, limit: int = 100):
            return [
                commitment
                for commitment in self.commitments.values()
                if commitment.status == status
            ][:limit]

        async def get(self, commitment_id: str):
            return self.commitments.get(commitment_id)

        async def save(self, commitment):
            self.commitments[str(commitment.commitment_id)] = commitment

    class FakeWorkStore:
        def __init__(self, work_items):
            self.work_items = {str(w.work_id): w for w in work_items}

        async def get(self, work_id: str):
            return self.work_items.get(work_id)

        async def list_ready(self, limit: int = 100):
            return list(self.work_items.values())[:limit]

        async def save(self, work):
            self.work_items[str(work.work_id)] = work

    executive = ExecutiveState(
        identity=identity,
        somatic=somatic,
        tracer=tracer,
    )

    resumable = Commitment(
        content="Resume after rest",
        status=CommitmentStatus.BLOCKED,
        tags=["self_commitment"],
        meta={
            "source": "assistant_response",
            "resume_policy": "auto_on_executive_recovery",
            "blocked_reason": "executive_fatigue",
        },
    )
    manual_hold = Commitment(
        content="Do not resume automatically",
        status=CommitmentStatus.BLOCKED,
        meta={"blocked_reason": "manual_hold"},
    )
    resumable_work = WorkObject(
        content="resume work",
        stage=WorkStage.MICRO_TASK,
        commitment_id=str(resumable.commitment_id),
    )
    manual_hold_work = WorkObject(
        content="manual hold work",
        stage=WorkStage.MICRO_TASK,
        commitment_id=str(manual_hold.commitment_id),
    )
    resumable.linked_work_ids.append(str(resumable_work.work_id))
    manual_hold.linked_work_ids.append(str(manual_hold_work.work_id))
    executive.commitment_store = FakeCommitmentStore([resumable, manual_hold])
    executive.work_store = FakeWorkStore([resumable_work, manual_hold_work])

    result = await executive.resume_deferred_work()
    await __import__("asyncio").sleep(0)
    queued_ids = {str(item.work_id) for item in executive.task_queue}
    resumed = await executive.commitment_store.get(str(resumable.commitment_id))
    held = await executive.commitment_store.get(str(manual_hold.commitment_id))

    assert result["unblocked_commitments"] == 1
    assert resumed is not None
    assert resumed.status == CommitmentStatus.ACTIVE
    assert resumed.meta["resume_reason"] == "executive_recovery"
    assert held is not None
    assert held.status == CommitmentStatus.BLOCKED
    assert str(resumable_work.work_id) in queued_ids
    assert str(manual_hold_work.work_id) not in queued_ids
