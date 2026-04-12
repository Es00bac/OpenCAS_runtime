"""Tests for RepairExecutor explicit phases."""

import pytest
import pytest_asyncio

from opencas.autonomy.models import ActionRiskTier
from opencas.execution import ExecutionPhase, ExecutionStage, RepairExecutor, RepairTask
from opencas.tools import FileSystemToolAdapter, ShellToolAdapter, ToolRegistry


@pytest_asyncio.fixture
async def executor(tmp_path):
    tools = ToolRegistry()
    workspace = str(tmp_path)
    fs = FileSystemToolAdapter(allowed_roots=[workspace])
    tools.register("fs_read_file", "Read file", fs, ActionRiskTier.READONLY)
    tools.register("fs_write_file", "Write file", fs, ActionRiskTier.WORKSPACE_WRITE)
    shell = ShellToolAdapter(cwd=workspace, timeout=30.0)
    tools.register("bash_run_command", "Run command", shell, ActionRiskTier.SHELL_LOCAL)
    return RepairExecutor(tools=tools)


@pytest.mark.asyncio
async def test_executor_records_all_phases(executor):
    task = RepairTask(objective="test task")
    result = await executor.run(task)
    assert result.success is True
    phase_names = [p.phase for p in task.phases]
    assert ExecutionPhase.DETECT in phase_names
    assert ExecutionPhase.SNAPSHOT in phase_names
    assert ExecutionPhase.PLAN in phase_names
    assert ExecutionPhase.EXECUTE in phase_names
    assert ExecutionPhase.VERIFY in phase_names
    assert ExecutionPhase.POSTCHECK in phase_names


@pytest.mark.asyncio
async def test_executor_snapshot_phase_when_scratch_dir_set(executor, tmp_path):
    task = RepairTask(
        objective="check file.txt",
        scratch_dir=str(tmp_path / "scratch"),
    )
    result = await executor.run(task)
    assert result.success is True
    snap_phase = [p for p in task.phases if p.phase == ExecutionPhase.SNAPSHOT][0]
    assert snap_phase.success is True


@pytest.mark.asyncio
async def test_executor_detects_files_from_objective(executor):
    task = RepairTask(objective="read config.json and data.csv")
    result = await executor.run(task)
    detect_phase = [p for p in task.phases if p.phase == ExecutionPhase.DETECT][0]
    assert "config.json" in detect_phase.output
    assert "data.csv" in detect_phase.output


@pytest.mark.asyncio
async def test_executor_execute_failure_heuristic_fails_task(executor):
    """If _execute_plan returns an empty or failure-marked string, the task should fail."""
    task = RepairTask(objective="failing task", max_attempts=1)
    # Patch _execute_plan to return the exact empty failure string from the brief
    executor._execute_plan = lambda _task, _plan: ""
    result = await executor.run(task)
    exec_phase = [p for p in task.phases if p.phase == ExecutionPhase.EXECUTE][0]
    assert exec_phase.success is False
    assert result.success is False
    assert result.stage == ExecutionStage.FAILED


@pytest.mark.asyncio
async def test_executor_tool_loop_halted_fails_task(executor):
    task = RepairTask(objective="halted task", max_attempts=1)
    executor._execute_plan = lambda _task, _plan: "[Tool loop halted] exceeded rounds"
    result = await executor.run(task)
    exec_phase = [p for p in task.phases if p.phase == ExecutionPhase.EXECUTE][0]
    assert exec_phase.success is False
    assert result.success is False
    assert result.stage == ExecutionStage.FAILED
