"""Tests for PlanStore."""

import pytest
import pytest_asyncio

from opencas.planning import PlanStore


@pytest_asyncio.fixture
async def store(tmp_path):
    path = tmp_path / "plans.db"
    s = PlanStore(path)
    await s.connect()
    yield s
    await s.close()


@pytest.mark.asyncio
async def test_create_and_get_plan(store):
    plan = await store.create_plan("plan-1", content="do things", project_id="p1")
    assert plan.plan_id == "plan-1"
    assert plan.status == "draft"
    fetched = await store.get_plan("plan-1")
    assert fetched is not None
    assert fetched.content == "do things"
    assert fetched.project_id == "p1"


@pytest.mark.asyncio
async def test_update_content_and_status(store):
    await store.create_plan("plan-1", content="initial")
    ok = await store.update_content("plan-1", "updated")
    assert ok is True
    ok = await store.set_status("plan-1", "active")
    assert ok is True
    fetched = await store.get_plan("plan-1")
    assert fetched.content == "updated"
    assert fetched.status == "active"


@pytest.mark.asyncio
async def test_list_active(store):
    await store.create_plan("plan-1", content="a")
    await store.create_plan("plan-2", content="b")
    await store.set_status("plan-1", "active")
    active = await store.list_active()
    assert len(active) == 1
    assert active[0].plan_id == "plan-1"


@pytest.mark.asyncio
async def test_list_active_with_project_and_task(store):
    await store.create_plan("plan-a", content="x", project_id="p1", task_id="t1")
    await store.set_status("plan-a", "active")
    await store.create_plan("plan-b", content="y", project_id="p1", task_id="t2")
    await store.set_status("plan-b", "active")
    results = await store.list_active(project_id="p1", task_id="t2")
    assert len(results) == 1
    assert results[0].plan_id == "plan-b"
    assert await store.count_active(project_id="p1") == 2
    assert await store.count_active(project_id="p1", task_id="t2") == 1


@pytest.mark.asyncio
async def test_record_action_and_get_actions(store):
    await store.create_plan("plan-1")
    await store.record_action(
        plan_id="plan-1",
        tool_name="read_file",
        args={"path": "/tmp"},
        result_summary="ok",
        success=True,
    )
    actions = await store.get_actions("plan-1")
    assert len(actions) == 1
    assert actions[0].tool_name == "read_file"
    assert actions[0].success is True


@pytest.mark.asyncio
async def test_delete_plan(store):
    await store.create_plan("plan-1")
    await store.record_action("plan-1", "t", {}, "", True)
    ok = await store.delete_plan("plan-1")
    assert ok is True
    assert await store.get_plan("plan-1") is None
    assert await store.get_actions("plan-1") == []
