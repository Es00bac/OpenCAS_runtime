"""Tests for the durable SQLite TaskStore and BAA integration."""

import asyncio
from pathlib import Path

import pytest
import pytest_asyncio

from opencas.execution import (
    BoundedAssistantAgent,
    ExecutionStage,
    RepairResult,
    RepairTask,
    TaskStore,
)
from opencas.tools import ToolRegistry


@pytest_asyncio.fixture
async def store(tmp_path: Path):
    db = tmp_path / "tasks.db"
    s = TaskStore(db)
    await s.connect()
    yield s
    await s.close()


@pytest.mark.asyncio
async def test_task_store_save_and_get(store: TaskStore) -> None:
    task = RepairTask(objective="save test")
    await store.save(task)
    fetched = await store.get(str(task.task_id))
    assert fetched is not None
    assert fetched.objective == "save test"
    assert fetched.stage == ExecutionStage.QUEUED


@pytest.mark.asyncio
async def test_task_store_list_pending_and_all(store: TaskStore) -> None:
    t1 = RepairTask(objective="pending 1")
    t2 = RepairTask(objective="pending 2")
    t3 = RepairTask(objective="done task")
    await store.save(t1)
    await store.save(t2)
    await store.save(t3)
    await store.save_result(
        RepairResult(task_id=t3.task_id, success=True, stage=ExecutionStage.DONE, output="ok")
    )

    pending = await store.list_pending()
    assert len(pending) == 2
    all_tasks = await store.list_all()
    assert len(all_tasks) == 3


@pytest.mark.asyncio
async def test_task_store_delete(store: TaskStore) -> None:
    task = RepairTask(objective="delete me")
    await store.save(task)
    deleted = await store.delete(str(task.task_id))
    assert deleted is True
    assert await store.get(str(task.task_id)) is None


@pytest.mark.asyncio
async def test_task_store_update_overwrites(store: TaskStore) -> None:
    task = RepairTask(objective="update test")
    await store.save(task)
    task.stage = ExecutionStage.EXECUTING
    task.status = "executing"
    await store.save(task)
    fetched = await store.get(str(task.task_id))
    assert fetched is not None
    assert fetched.stage == ExecutionStage.EXECUTING
    assert fetched.status == "executing"


@pytest.mark.asyncio
async def test_task_store_save_result_terminal(store: TaskStore) -> None:
    task = RepairTask(objective="result test")
    await store.save(task)
    result = RepairResult(
        task_id=task.task_id,
        success=True,
        stage=ExecutionStage.DONE,
        output="all good",
    )
    await store.save_result(result)
    fetched = await store.get(str(task.task_id))
    assert fetched is not None
    assert fetched.stage == ExecutionStage.DONE


@pytest.mark.asyncio
async def test_baa_persists_tasks_on_submit(tmp_path: Path) -> None:
    db = tmp_path / "baa.db"
    store = TaskStore(db)
    await store.connect()

    tools = ToolRegistry()
    baa = BoundedAssistantAgent(tools=tools, store=store)
    task = RepairTask(objective="persisted task")
    future = await baa.submit(task)

    fetched = await store.get(str(task.task_id))
    assert fetched is not None
    assert fetched.objective == "persisted task"

    # clean up
    await baa.stop()
    await store.close()


@pytest.mark.asyncio
async def test_baa_saves_result_after_execution(tmp_path: Path) -> None:
    db = tmp_path / "baa.db"
    store = TaskStore(db)
    await store.connect()

    tools = ToolRegistry()
    workspace = str(tmp_path)
    from opencas.autonomy.models import ActionRiskTier
    from opencas.tools import ShellToolAdapter

    shell = ShellToolAdapter(cwd=workspace, timeout=30.0)
    tools.register("bash_run_command", "Run command", shell, ActionRiskTier.SHELL_LOCAL)

    baa = BoundedAssistantAgent(tools=tools, store=store)
    task = RepairTask(
        objective="run and save",
        verification_command="echo ok",
    )
    future = await baa.submit(task)
    await baa.start()
    result = await asyncio.wait_for(future, timeout=5.0)
    await baa.stop()

    assert result.success is True
    fetched = await store.get(str(task.task_id))
    assert fetched is not None
    assert fetched.stage == ExecutionStage.DONE

    await store.close()


@pytest.mark.asyncio
async def test_baa_auto_resumes_pending_tasks(tmp_path: Path) -> None:
    db = tmp_path / "baa.db"
    store = TaskStore(db)
    await store.connect()

    # Pre-seed a task as if from a crashed session
    task = RepairTask(
        objective="resume me",
        verification_command="echo resumed",
        stage=ExecutionStage.EXECUTING,
        status="executing",
    )
    await store.save(task)

    tools = ToolRegistry()
    workspace = str(tmp_path)
    from opencas.autonomy.models import ActionRiskTier
    from opencas.tools import ShellToolAdapter

    shell = ShellToolAdapter(cwd=workspace, timeout=30.0)
    tools.register("bash_run_command", "Run command", shell, ActionRiskTier.SHELL_LOCAL)

    baa = BoundedAssistantAgent(tools=tools, store=store)
    await baa.start()

    # Wait for the resumed task to complete
    pending = await store.list_pending()
    assert len(pending) == 1
    task_id = str(pending[0].task_id)
    future = baa._futures.get(task_id)
    assert future is not None
    result = await asyncio.wait_for(future, timeout=5.0)
    await baa.stop()

    assert result.success is True
    fetched = await store.get(task_id)
    assert fetched is not None
    assert fetched.stage == ExecutionStage.DONE

    await store.close()
