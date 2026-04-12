"""Tests for WorkStore persistence."""

import pytest
import pytest_asyncio

from opencas.autonomy.models import WorkObject, WorkStage
from opencas.autonomy.work_store import WorkStore


@pytest_asyncio.fixture
async def store(tmp_path):
    ws = WorkStore(tmp_path / "work.db")
    await ws.connect()
    yield ws
    await ws.close()


@pytest.mark.asyncio
async def test_save_and_get(store):
    work = WorkObject(content="test work", project_id="p1")
    await store.save(work)
    fetched = await store.get(str(work.work_id))
    assert fetched is not None
    assert fetched.content == "test work"
    assert fetched.stage == WorkStage.SPARK
    assert fetched.project_id == "p1"


@pytest.mark.asyncio
async def test_list_by_stage(store):
    w1 = WorkObject(content="spark work")
    w2 = WorkObject(content="project work", stage=WorkStage.PROJECT)
    await store.save(w1)
    await store.save(w2)
    sparks = await store.list_by_stage(WorkStage.SPARK)
    assert len(sparks) == 1
    assert sparks[0].content == "spark work"
    projects = await store.list_by_stage(WorkStage.PROJECT)
    assert len(projects) == 1


@pytest.mark.asyncio
async def test_list_by_project(store):
    w1 = WorkObject(content="a", project_id="p1")
    w2 = WorkObject(content="b", project_id="p2")
    await store.save(w1)
    await store.save(w2)
    result = await store.list_by_project("p1")
    assert len(result) == 1
    assert result[0].content == "a"


@pytest.mark.asyncio
async def test_blocked_and_ready(store):
    w1 = WorkObject(content="blocked", blocked_by=["dep1"])
    w2 = WorkObject(content="ready")
    await store.save(w1)
    await store.save(w2)
    blocked = await store.list_blocked()
    assert len(blocked) == 1
    assert blocked[0].content == "blocked"
    ready = await store.list_ready()
    assert len(ready) == 1
    assert ready[0].content == "ready"


@pytest.mark.asyncio
async def test_unblock_dependencies(store):
    dep = WorkObject(content="dependency")
    w = WorkObject(content="blocked", blocked_by=[str(dep.work_id)])
    await store.save(dep)
    await store.save(w)
    modified = await store.unblock_dependencies(str(dep.work_id))
    assert modified == 1
    fetched = await store.get(str(w.work_id))
    assert fetched.blocked_by == []


@pytest.mark.asyncio
async def test_delete(store):
    work = WorkObject(content="to delete")
    await store.save(work)
    assert await store.delete(str(work.work_id)) is True
    assert await store.get(str(work.work_id)) is None


@pytest.mark.asyncio
async def test_touch(store):
    work = WorkObject(content="touch me")
    await store.save(work)
    ok = await store.touch(str(work.work_id))
    assert ok is True
    fetched = await store.get(str(work.work_id))
    assert fetched.access_count == 1
    assert fetched.last_accessed is not None


@pytest.mark.asyncio
async def test_summary_counts(store):
    await store.save(WorkObject(content="ready one"))
    await store.save(WorkObject(content="blocked one", blocked_by=["dep"]))
    counts = await store.summary_counts()
    assert counts["total"] == 2
    assert counts["ready"] == 1
    assert counts["blocked"] == 1
