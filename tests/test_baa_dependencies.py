"""Tests for BAA dependency-aware scheduling."""

import pytest
import pytest_asyncio

from opencas.execution import BoundedAssistantAgent, RepairTask
from opencas.execution.store import TaskStore
from opencas.tools import ToolRegistry


@pytest_asyncio.fixture
async def store(tmp_path):
    ts = TaskStore(tmp_path / "tasks.db")
    await ts.connect()
    yield ts
    await ts.close()


@pytest_asyncio.fixture
async def baa(store):
    tools = ToolRegistry()
    agent = BoundedAssistantAgent(tools=tools, store=store, max_concurrent=1)
    await agent.start()
    yield agent
    await agent.stop()


@pytest.mark.asyncio
async def test_task_held_until_dependency_completes(baa, store):
    dep = RepairTask(objective="dependency task")
    child = RepairTask(objective="child task", depends_on=[str(dep.task_id)])

    # Submit child first; it should be held
    f_child = await baa.submit(child)
    assert str(child.task_id) in baa._held

    # Submit dependency; it should run immediately
    f_dep = await baa.submit(dep)
    r_dep = await f_dep
    assert r_dep.success is True

    # Child should now be released and run
    r_child = await f_child
    assert r_child.success is True


@pytest.mark.asyncio
async def test_task_with_precompleted_dependency_runs_immediately(baa, store):
    dep = RepairTask(objective="already done")
    await baa.submit(dep)
    r = await baa._futures[str(dep.task_id)]
    assert r.success is True

    child = RepairTask(objective="child", depends_on=[str(dep.task_id)])
    f_child = await baa.submit(child)
    assert str(child.task_id) not in baa._held
    r_child = await f_child
    assert r_child.success is True


@pytest.mark.asyncio
async def test_held_tasks_restored_from_store(tmp_path):
    ts = TaskStore(tmp_path / "tasks.db")
    await ts.connect()
    dep = RepairTask(objective="dep")
    child = RepairTask(objective="child", depends_on=[str(dep.task_id)])
    await ts.save(child)

    tools = ToolRegistry()
    baa2 = BoundedAssistantAgent(tools=tools, store=ts, max_concurrent=1)
    await baa2.start()
    assert str(child.task_id) in baa2._held

    f_dep = await baa2.submit(dep)
    await f_dep

    # Child should have been released after dep finished
    assert str(child.task_id) not in baa2._held
    await baa2.stop()
    await ts.close()


@pytest.mark.asyncio
async def test_task_held_for_approval_and_resolved(baa):
    from opencas.execution.lifecycle import LifecycleStage
    from opencas.execution.models import ExecutionStage
    import asyncio

    task = RepairTask(objective="needs approval")
    task_id = str(task.task_id)
    task.stage = ExecutionStage.NEEDS_APPROVAL

    future = asyncio.get_running_loop().create_future()
    baa._futures[task_id] = future
    baa._held[task_id] = task

    resolved = await baa.resolve_hold(task_id)
    assert resolved is True
    assert task_id not in baa._held
    assert task.stage.value == "queued"

    r = await future
    assert r.success is True


@pytest.mark.asyncio
async def test_task_held_for_clarification_and_resolved(baa):
    from opencas.execution.lifecycle import LifecycleStage
    from opencas.execution.models import ExecutionStage
    import asyncio

    task = RepairTask(objective="needs clarification")
    task_id = str(task.task_id)
    task.stage = ExecutionStage.NEEDS_CLARIFICATION

    future = asyncio.get_running_loop().create_future()
    baa._futures[task_id] = future
    baa._held[task_id] = task

    resolved = await baa.resolve_hold(task_id)
    assert resolved is True
    assert task_id not in baa._held
    assert task.stage.value == "queued"

    r = await future
    assert r.success is True


@pytest.mark.asyncio
async def test_resolve_hold_missing_task_returns_false(baa):
    resolved = await baa.resolve_hold("nonexistent-task-id")
    assert resolved is False
