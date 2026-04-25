"""Tests for ExecutiveState persistence via WorkStore."""

import pytest
import pytest_asyncio

from opencas.autonomy import WorkObject, WorkStage
from opencas.autonomy.executive import ExecutiveState
from opencas.autonomy.work_store import WorkStore
from opencas.identity import IdentityManager, IdentityStore


@pytest_asyncio.fixture
async def store(tmp_path):
    ws = WorkStore(tmp_path / "work.db")
    await ws.connect()
    yield ws
    await ws.close()


@pytest.fixture
def identity(tmp_path):
    s = IdentityStore(tmp_path / "identity")
    mgr = IdentityManager(s)
    mgr.load()
    return mgr


@pytest.mark.asyncio
async def test_enqueue_persists_to_store(store, identity):
    exec_state = ExecutiveState(identity=identity, work_store=store)
    w = WorkObject(content="task", stage=WorkStage.MICRO_TASK)
    exec_state.enqueue(w)
    import asyncio
    await asyncio.sleep(0.05)
    fetched = await store.get(str(w.work_id))
    assert fetched is not None
    assert fetched.content == "task"


@pytest.mark.asyncio
async def test_restore_queue_loads_from_store(store, identity):
    exec_state = ExecutiveState(identity=identity, work_store=store)
    w = WorkObject(content="restored task", stage=WorkStage.MICRO_TASK)
    await store.save(w)

    exec_state2 = ExecutiveState(identity=identity, work_store=store)
    restored = await exec_state2.restore_queue()
    assert restored == 1
    assert len(exec_state2.task_queue) == 1
    assert exec_state2.task_queue[0].content == "restored task"


@pytest.mark.asyncio
async def test_restore_queue_skips_blocked_work(store, identity):
    exec_state = ExecutiveState(identity=identity, work_store=store)
    w_ready = WorkObject(content="ready", stage=WorkStage.MICRO_TASK)
    w_blocked = WorkObject(content="blocked", stage=WorkStage.MICRO_TASK, blocked_by=["dep1"])
    await store.save(w_ready)
    await store.save(w_blocked)

    exec_state2 = ExecutiveState(identity=identity, work_store=store)
    restored = await exec_state2.restore_queue()
    assert restored == 1
    assert len(exec_state2.task_queue) == 1
    assert exec_state2.task_queue[0].content == "ready"
