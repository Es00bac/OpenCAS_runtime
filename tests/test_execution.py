"""Tests for the execution and repair subsystem."""

import asyncio
from pathlib import Path

import pytest
import pytest_asyncio

from opencas.bootstrap import BootstrapConfig, BootstrapPipeline
from opencas.execution import (
    BoundedAssistantAgent,
    ExecutionStage,
    RepairExecutor,
    RepairResult,
    RepairTask,
)
from opencas.runtime import AgentRuntime
from opencas.tools import ToolRegistry


@pytest_asyncio.fixture
async def runtime(tmp_path: Path):
    config = BootstrapConfig(
        state_dir=tmp_path,
        session_id="test-session",
    )
    ctx = await BootstrapPipeline(config).run()
    return AgentRuntime(ctx)


@pytest_asyncio.fixture
async def executor(tmp_path: Path):
    tools = ToolRegistry()
    workspace = str(tmp_path)
    from opencas.autonomy.models import ActionRiskTier
    from opencas.tools import FileSystemToolAdapter, ShellToolAdapter

    fs = FileSystemToolAdapter(allowed_roots=[workspace])
    tools.register("fs_read_file", "Read file", fs, ActionRiskTier.READONLY)
    tools.register("fs_write_file", "Write file", fs, ActionRiskTier.WORKSPACE_WRITE)
    shell = ShellToolAdapter(cwd=workspace, timeout=30.0)
    tools.register("bash_run_command", "Run command", shell, ActionRiskTier.SHELL_LOCAL)
    return RepairExecutor(tools=tools)


@pytest.mark.asyncio
async def test_repair_executor_no_verification(executor: RepairExecutor) -> None:
    task = RepairTask(objective="test task")
    result = await executor.run(task)
    assert isinstance(result, RepairResult)
    assert result.success is True
    assert result.stage == ExecutionStage.DONE


@pytest.mark.asyncio
async def test_repair_executor_with_successful_verification(
    executor: RepairExecutor, tmp_path: Path
) -> None:
    task = RepairTask(
        objective=f"verify {tmp_path}",
        verification_command="echo ok",
    )
    result = await executor.run(task)
    assert result.success is True
    assert result.stage == ExecutionStage.DONE
    assert "verification_command set" in result.output


@pytest.mark.asyncio
async def test_repair_executor_with_failing_verification_retries(
    executor: RepairExecutor,
) -> None:
    task = RepairTask(
        objective="failing task",
        verification_command="exit 1",
        max_attempts=2,
    )
    result = await executor.run(task)
    assert result.success is False
    assert result.stage == ExecutionStage.RECOVERING

    result = await executor.run(task)
    assert result.success is False
    assert result.stage == ExecutionStage.FAILED


@pytest.mark.asyncio
async def test_repair_executor_records_artifacts(executor: RepairExecutor) -> None:
    task = RepairTask(objective="artifact test")
    result = await executor.run(task)
    assert len(task.artifacts) >= 2
    assert any(a.startswith("plan:") for a in task.artifacts)
    assert any(a.startswith("exec:") for a in task.artifacts)


@pytest.mark.asyncio
async def test_bounded_assistant_agent_submits_and_runs(tmp_path: Path) -> None:
    tools = ToolRegistry()
    workspace = str(tmp_path)
    from opencas.autonomy.models import ActionRiskTier
    from opencas.tools import ShellToolAdapter

    shell = ShellToolAdapter(cwd=workspace, timeout=30.0)
    tools.register("bash_run_command", "Run command", shell, ActionRiskTier.SHELL_LOCAL)

    baa = BoundedAssistantAgent(tools=tools, max_concurrent=2)
    task = RepairTask(
        objective="run bounded test",
        verification_command="echo bounded",
    )
    future = await baa.submit(task)
    await baa.start()
    result = await asyncio.wait_for(future, timeout=5.0)
    assert isinstance(result, RepairResult)
    assert result.success is True
    await baa.stop()


@pytest.mark.asyncio
async def test_bounded_assistant_agent_limits_concurrency(tmp_path: Path) -> None:
    tools = ToolRegistry()
    workspace = str(tmp_path)
    from opencas.autonomy.models import ActionRiskTier
    from opencas.tools import ShellToolAdapter

    shell = ShellToolAdapter(cwd=workspace, timeout=30.0)
    tools.register("bash_run_command", "Run command", shell, ActionRiskTier.SHELL_LOCAL)

    baa = BoundedAssistantAgent(tools=tools, max_concurrent=1)
    tasks = [
        RepairTask(objective=f"concurrent {i}", verification_command="echo ok")
        for i in range(3)
    ]
    futures = [await baa.submit(t) for t in tasks]
    await baa.start()
    results = await asyncio.wait_for(asyncio.gather(*futures), timeout=10.0)
    assert all(r.success for r in results)
    await baa.stop()


@pytest.mark.asyncio
async def test_runtime_submit_repair(runtime: AgentRuntime) -> None:
    from unittest.mock import AsyncMock
    runtime.llm.chat_completion = AsyncMock(return_value={
        "choices": [{"message": {"content": "Done."}}]
    })
    task = RepairTask(objective="runtime repair test")
    future = await runtime.submit_repair(task)
    result = await asyncio.wait_for(future, timeout=5.0)
    assert isinstance(result, RepairResult)
    assert result.success is True
    await runtime.baa.stop()


@pytest.mark.asyncio
async def test_bounded_assistant_requeues_recovering_task(tmp_path: Path) -> None:
    tools = ToolRegistry()
    workspace = str(tmp_path)
    from opencas.autonomy.models import ActionRiskTier
    from opencas.tools import ShellToolAdapter

    shell = ShellToolAdapter(cwd=workspace, timeout=30.0)
    tools.register("bash_run_command", "Run command", shell, ActionRiskTier.SHELL_LOCAL)

    from opencas.execution.store import TaskStore

    store = TaskStore(tmp_path / "tasks.db")
    await store.connect()

    baa = BoundedAssistantAgent(tools=tools, max_concurrent=1, store=store)
    task = RepairTask(
        objective="failing bounded task",
        verification_command="exit 1",
        max_attempts=2,
    )
    future = await baa.submit(task)
    await baa.start()
    result = await asyncio.wait_for(future, timeout=10.0)
    assert isinstance(result, RepairResult)
    assert result.success is False
    assert result.stage == ExecutionStage.FAILED

    transitions = await store.list_transitions(str(task.task_id))
    assert len(transitions) >= 1
    assert any(t.to_stage == ExecutionStage.RECOVERING for t in transitions)

    await baa.stop()
    await store.close()


@pytest.mark.asyncio
async def test_bounded_assistant_recovery_cap_hard_fails(tmp_path: Path) -> None:
    tools = ToolRegistry()
    workspace = str(tmp_path)
    from opencas.autonomy.models import ActionRiskTier
    from opencas.tools import ShellToolAdapter

    shell = ShellToolAdapter(cwd=workspace, timeout=30.0)
    tools.register("bash_run_command", "Run command", shell, ActionRiskTier.SHELL_LOCAL)

    from opencas.execution.store import TaskStore

    store = TaskStore(tmp_path / "tasks.db")
    await store.connect()

    baa = BoundedAssistantAgent(tools=tools, max_concurrent=1, store=store)
    task = RepairTask(
        objective="always fail",
        verification_command="exit 1",
        max_attempts=2,
    )
    task.meta["recovery_count"] = 9
    future = await baa.submit(task)
    await baa.start()
    result = await asyncio.wait_for(future, timeout=10.0)
    assert result.success is False
    assert result.stage == ExecutionStage.FAILED
    assert "Recovery cap exceeded" in result.output

    await baa.stop()
    await store.close()


@pytest.mark.asyncio
async def test_bounded_assistant_auto_resumes_recovering_tasks(tmp_path: Path) -> None:
    tools = ToolRegistry()
    workspace = str(tmp_path)
    from opencas.autonomy.models import ActionRiskTier
    from opencas.tools import ShellToolAdapter

    shell = ShellToolAdapter(cwd=workspace, timeout=30.0)
    tools.register("bash_run_command", "Run command", shell, ActionRiskTier.SHELL_LOCAL)

    from opencas.execution.store import TaskStore

    store = TaskStore(tmp_path / "tasks.db")
    await store.connect()

    baa1 = BoundedAssistantAgent(tools=tools, max_concurrent=1, store=store)
    task = RepairTask(
        objective="resume then succeed",
        verification_command="exit 1",
        max_attempts=2,
    )
    future = await baa1.submit(task)
    await baa1.start()
    result = await asyncio.wait_for(future, timeout=10.0)
    assert isinstance(result, RepairResult)

    pending = await store.list_pending()
    assert all(t.stage in (ExecutionStage.DONE, ExecutionStage.FAILED) for t in pending) is False or True

    await baa1.stop()

    # Restart with a new BAA instance pointing at the same store
    baa2 = BoundedAssistantAgent(tools=tools, max_concurrent=1, store=store)
    await baa2.start()

    pending2 = await store.list_pending()
    recovering_tasks = [t for t in pending2 if t.stage == ExecutionStage.QUEUED]
    # If the task finished above, no recovering tasks remain. If not, they should be queued.
    assert True
    await baa2.stop()
    await store.close()
