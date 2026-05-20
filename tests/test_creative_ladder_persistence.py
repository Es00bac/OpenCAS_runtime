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
    await ladder.drain_background_tasks()
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
    await ladder.drain_background_tasks()
    fetched = await store.get(str(w.work_id))
    assert fetched.stage == WorkStage.NOTE


@pytest.mark.asyncio
async def test_remove_deletes_from_store(store, executive):
    ladder = CreativeLadder(executive=executive, work_store=store)
    w = WorkObject(content="to remove")
    ladder.add(w)
    await ladder.drain_background_tasks()
    ladder.remove(str(w.work_id))
    await ladder.drain_background_tasks()
    fetched = await store.get(str(w.work_id))
    assert fetched is None


@pytest.mark.asyncio
async def test_hydrate_from_store_reloads_persisted_work(store, executive):
    note = WorkObject(content="persisted note", stage=WorkStage.NOTE)
    await store.save(note)

    ladder = CreativeLadder(executive=executive, work_store=store)
    hydrated = await ladder.hydrate_from_store()

    assert hydrated == 1
    assert [work.content for work in ladder.list_by_stage(WorkStage.NOTE)] == [
        "persisted note"
    ]


def test_run_cycle_fallback_promotes_daydream_note_and_artifact(executive):
    ladder = CreativeLadder(executive=executive)
    note = WorkObject(
        content="A daydream note that should keep moving.",
        stage=WorkStage.NOTE,
        meta={"origin": "daydream"},
    )
    artifact = WorkObject(
        content="A daydream artifact that should become actionable.",
        stage=WorkStage.ARTIFACT,
        meta={"origin": "daydream"},
    )
    ladder.add(note)
    ladder.add(artifact)

    result = ladder.run_cycle()

    assert result["fallback_promoted"] == 2
    assert note.stage == WorkStage.ARTIFACT
    assert artifact.stage == WorkStage.MICRO_TASK
    assert ladder.last_cycle_health["choke_point"] in {"artifact", "micro_task"}


def test_run_cycle_fallback_promotes_work_once_per_cycle(executive):
    ladder = CreativeLadder(executive=executive)
    note = WorkObject(
        content="A single daydream note should not skip the artifact rung.",
        stage=WorkStage.NOTE,
        meta={"origin": "daydream"},
    )
    ladder.add(note)

    result = ladder.run_cycle()

    assert result["fallback_promoted"] == 1
    assert note.stage == WorkStage.ARTIFACT
