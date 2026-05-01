"""Tests for BAA dependency-aware scheduling."""

import pytest
import pytest_asyncio
from types import SimpleNamespace

from opencas.api import provenance_store as ps

from opencas.autonomy.models import WorkObject, WorkStage
from opencas.autonomy.work_store import WorkStore
from opencas.execution import BoundedAssistantAgent, RepairTask, ExecutionStage
from opencas.execution.lifecycle import LifecycleStage
from opencas.execution.lanes import CommandLane
from opencas.execution.store import TaskStore
from opencas.tools import ToolRegistry


@pytest_asyncio.fixture
async def store(tmp_path):
    ts = TaskStore(tmp_path / "tasks.db")
    await ts.connect()
    yield ts
    await ts.close()


@pytest_asyncio.fixture
async def work_store(tmp_path):
    ws = WorkStore(tmp_path / "work.db")
    await ws.connect()
    yield ws
    await ws.close()


@pytest_asyncio.fixture
async def baa(store):
    tools = ToolRegistry()
    agent = BoundedAssistantAgent(tools=tools, store=store, max_concurrent=1)
    await agent.start()
    yield agent
    await agent.stop()


def _runtime_with_work_store(work_store):
    class _Ctx:
        def __init__(self, ws):
            self.work_store = ws

    class _Runtime:
        def __init__(self, ws):
            self.ctx = _Ctx(ws)

    return _Runtime(work_store)


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
async def test_task_held_until_work_dependency_becomes_artifact(tmp_path, work_store):
    ts = TaskStore(tmp_path / "tasks.db")
    await ts.connect()
    tools = ToolRegistry()
    baa = BoundedAssistantAgent(
        tools=tools,
        store=ts,
        max_concurrent=1,
        runtime=_runtime_with_work_store(work_store),
    )
    await baa.start()

    dep_work = WorkObject(content="prepare dependency", stage=WorkStage.MICRO_TASK)
    await work_store.save(dep_work)

    child = RepairTask(objective="child", depends_on=[str(dep_work.work_id)])
    f_child = await baa.submit(child)
    assert str(child.task_id) in baa._held

    dep_work.stage = WorkStage.ARTIFACT
    await work_store.save(dep_work)
    await baa._try_release_held()

    assert str(child.task_id) not in baa._held
    r_child = await f_child
    assert r_child.success is True

    await baa.stop()
    await ts.close()


@pytest.mark.asyncio
async def test_micro_task_work_dependency_does_not_release_early(tmp_path, work_store):
    ts = TaskStore(tmp_path / "tasks.db")
    await ts.connect()
    tools = ToolRegistry()
    baa = BoundedAssistantAgent(
        tools=tools,
        store=ts,
        max_concurrent=1,
        runtime=_runtime_with_work_store(work_store),
    )
    await baa.start()

    dep_work = WorkObject(content="still in progress", stage=WorkStage.MICRO_TASK)
    await work_store.save(dep_work)

    child = RepairTask(objective="child", depends_on=[str(dep_work.work_id)])
    await baa.submit(child)
    await baa._try_release_held()

    assert str(child.task_id) in baa._held

    await baa.stop()
    await ts.close()


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


def test_active_count_excludes_held_and_queued_tasks():
    tools = ToolRegistry()
    agent = BoundedAssistantAgent(tools=tools, max_concurrent=1)
    agent._futures = {"a": object(), "b": object(), "c": object()}
    agent._held = {"held": RepairTask(objective="held")}
    agent._lanes.submit(CommandLane.BAA, object())
    assert agent.active_count == 1


@pytest.mark.asyncio
async def test_task_transition_to_operator_input_records_waiting_provenance(store, tmp_path):
    tools = ToolRegistry()
    runtime = SimpleNamespace(
        ctx=SimpleNamespace(
            config=SimpleNamespace(session_id="session-1", state_dir=tmp_path),
        )
    )
    baa = BoundedAssistantAgent(tools=tools, store=store, runtime=runtime, max_concurrent=1)

    task = RepairTask(objective="needs approval")
    task.stage = ExecutionStage.QUEUED

    await baa._transition_task(task, LifecycleStage.NEEDS_APPROVAL, "need operator input")

    records_path = tmp_path / "provenance.transitions.jsonl"
    records = [
        ps.parse_provenance_transition(line)
        for line in records_path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    record = records[-1]

    assert record.kind == ps.ProvenanceTransitionKind.WAITING
    assert record.status == "blocked"
    assert record.details["source_artifact"] == f"repair|baa|{task.task_id}"
    assert record.details["trigger_action"] == "baa.transition_task"
    assert record.details["target_entity"] == str(task.task_id)
    assert record.details["origin_action_id"] == str(task.task_id)
    assert any(event.get("event_type") == "BLOCKED" for event in task.meta["provenance_events"])
    assert any(
        event.get("triggering_artifact") == f"repair-task|default|{task.task_id}"
        for event in task.meta["provenance_events"]
    )
