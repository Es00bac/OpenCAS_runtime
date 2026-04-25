"""Tests for Commitment, CommitmentStore, and executive integration."""

from pathlib import Path
import pytest
import pytest_asyncio

from opencas.autonomy.commitment import Commitment, CommitmentStatus
from opencas.autonomy.commitment_store import CommitmentStore
from opencas.autonomy.executive import ExecutiveState
from opencas.autonomy.models import WorkObject, WorkStage
from opencas.autonomy.workspace import ExecutiveWorkspace
from opencas.identity import IdentityManager, IdentityStore


@pytest.fixture
def identity(tmp_path: Path):
    store = IdentityStore(tmp_path / "identity")
    mgr = IdentityManager(store)
    mgr.load()
    return mgr


@pytest_asyncio.fixture
async def commitment_store(tmp_path: Path) -> CommitmentStore:
    store = CommitmentStore(tmp_path / "commitments.db")
    await store.connect()
    yield store
    await store.close()


@pytest.mark.asyncio
async def test_commitment_crud(commitment_store: CommitmentStore) -> None:
    c = Commitment(content="rewrite readme", priority=8.0)
    await commitment_store.save(c)

    fetched = await commitment_store.get(str(c.commitment_id))
    assert fetched is not None
    assert fetched.content == "rewrite readme"
    assert fetched.priority == 8.0
    assert fetched.status == CommitmentStatus.ACTIVE


@pytest.mark.asyncio
async def test_list_active_ordered_by_priority(commitment_store: CommitmentStore) -> None:
    c1 = Commitment(content="low", priority=3.0)
    c2 = Commitment(content="high", priority=9.0)
    await commitment_store.save(c1)
    await commitment_store.save(c2)

    active = await commitment_store.list_active()
    assert [a.content for a in active] == ["high", "low"]


@pytest.mark.asyncio
async def test_update_status(commitment_store: CommitmentStore) -> None:
    c = Commitment(content="task")
    await commitment_store.save(c)
    ok = await commitment_store.update_status(str(c.commitment_id), CommitmentStatus.COMPLETED)
    assert ok is True

    fetched = await commitment_store.get(str(c.commitment_id))
    assert fetched.status == CommitmentStatus.COMPLETED


@pytest.mark.asyncio
async def test_link_work_and_task(commitment_store: CommitmentStore) -> None:
    c = Commitment(content="task")
    await commitment_store.save(c)
    await commitment_store.link_work(str(c.commitment_id), "w-1")
    await commitment_store.link_task(str(c.commitment_id), "t-1")

    fetched = await commitment_store.get(str(c.commitment_id))
    assert "w-1" in fetched.linked_work_ids
    assert "t-1" in fetched.linked_task_ids


@pytest.mark.asyncio
async def test_delete(commitment_store: CommitmentStore) -> None:
    c = Commitment(content="dispose")
    await commitment_store.save(c)
    ok = await commitment_store.delete(str(c.commitment_id))
    assert ok is True
    assert await commitment_store.get(str(c.commitment_id)) is None


@pytest.mark.asyncio
async def test_workspace_rebuild_includes_commitments(
    commitment_store: CommitmentStore, identity: IdentityManager
) -> None:
    c = Commitment(content="urgent goal", priority=10.0)
    await commitment_store.save(c)

    commitments = await commitment_store.list_active()
    workspace = ExecutiveWorkspace.rebuild(commitments=commitments, work_objects=[])

    assert workspace.focus is not None
    assert workspace.focus.content == "urgent goal"
    assert workspace.focus.kind.value == "commitment"


@pytest.mark.asyncio
async def test_work_object_commitment_id_persisted(commitment_store: CommitmentStore) -> None:
    c = Commitment(content="linked work", priority=7.0)
    await commitment_store.save(c)

    wo = WorkObject(content="do the work", stage=WorkStage.MICRO_TASK)
    wo.commitment_id = str(c.commitment_id)
    assert wo.commitment_id == str(c.commitment_id)
