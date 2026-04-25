"""Tests for CreativeLadder persistence via WorkStore."""

import pytest
import pytest_asyncio

from opencas.autonomy import WorkObject, WorkStage
from opencas.autonomy.creative_ladder import CreativeLadder
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
    store = IdentityStore(tmp_path / "identity")
    mgr = IdentityManager(store)
    mgr.load()
    return mgr


@pytest.fixture
def executive(identity):
    return ExecutiveState(identity=identity)


@pytest.mark.asyncio
async def test_add_persists_to_store(store, executive):
    ladder = CreativeLadder(executive=executive, work_store=store)
    w = WorkObject(content="spark idea")
    ladder.add(w)
    # Give the async task a moment to run
    import asyncio
    await asyncio.sleep(0.05)
    fetched = await store.get(str(w.work_id))
    assert fetched is not None
    assert fetched.content == "spark idea"


@pytest.mark.asyncio
async def test_promote_persists_to_store(store, executive):
    executive.add_goal("learn rust")
    ladder = CreativeLadder(executive=executive, work_store=store)
    w = WorkObject(content="I want to learn rust today", stage=WorkStage.SPARK)
    ladder.add(w)
    ladder.try_promote(w)
    import asyncio
    await asyncio.sleep(0.05)
    fetched = await store.get(str(w.work_id))
    assert fetched.stage == WorkStage.NOTE


@pytest.mark.asyncio
async def test_remove_deletes_from_store(store, executive):
    ladder = CreativeLadder(executive=executive, work_store=store)
    w = WorkObject(content="to remove")
    ladder.add(w)
    import asyncio
    await asyncio.sleep(0.05)
    ladder.remove(str(w.work_id))
    await asyncio.sleep(0.05)
    fetched = await store.get(str(w.work_id))
    assert fetched is None
