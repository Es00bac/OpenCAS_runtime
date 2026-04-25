"""Tests for RepairExecutor convergence guard and backoff."""

import asyncio
from unittest.mock import patch

import pytest
import pytest_asyncio

from opencas.autonomy.models import ActionRiskTier
from opencas.execution import ExecutionStage, RepairExecutor, RepairTask
from opencas.tools import ShellToolAdapter, ToolRegistry


@pytest_asyncio.fixture
async def executor(tmp_path):
    tools = ToolRegistry()
    workspace = str(tmp_path)
    shell = ShellToolAdapter(cwd=workspace, timeout=30.0)
    tools.register("bash_run_command", "Run command", shell, ActionRiskTier.SHELL_LOCAL)
    return RepairExecutor(tools=tools)


@pytest.mark.asyncio
async def test_convergence_guard_aborts_loop(executor):
    task = RepairTask(
        objective="failing task",
        verification_command="exit 1",
        max_attempts=3,
    )
    # First attempt: recovering
    r1 = await executor.run(task)
    assert r1.stage == ExecutionStage.RECOVERING

    # Second attempt: same artifacts/output -> should hit convergence guard and fail
    r2 = await executor.run(task)
    assert r2.stage == ExecutionStage.FAILED
    assert "non-improving loop" in r2.output.lower()


@pytest.mark.asyncio
async def test_exponential_backoff_between_retries(executor):
    task = RepairTask(
        objective="failing task",
        verification_command="exit 1",
        max_attempts=3,
        retry_backoff_seconds=0.1,
    )
    with patch("asyncio.sleep") as mock_sleep:
        r1 = await executor.run(task)
        assert r1.stage == ExecutionStage.RECOVERING
        mock_sleep.assert_not_called()

        r2 = await executor.run(task)
        assert r2.stage == ExecutionStage.FAILED  # convergence guard may trigger
        # If not converged, backoff should sleep
        if "non-improving" not in r2.output:
            mock_sleep.assert_any_call(0.1)
            assert task.retry_backoff_seconds == 0.2


@pytest.mark.asyncio
async def test_backoff_doubles_each_time(executor):
    task = RepairTask(
        objective="backoff test with unique output",
        verification_command="exit 1",
        max_attempts=4,
        retry_backoff_seconds=0.05,
    )
    sleeps = []
    original_sleep = asyncio.sleep

    async def _track_sleep(duration):
        sleeps.append(duration)
        await original_sleep(duration)

    with patch("asyncio.sleep", side_effect=_track_sleep):
        # Attempt 1: no sleep, returns RECOVERING
        r1 = await executor.run(task)
        assert r1.stage == ExecutionStage.RECOVERING
        assert sleeps == []

        # Attempt 2: sleeps 0.05 before running, then doubles to 0.1
        # Convergence guard will detect same output and abort
        r2 = await executor.run(task)
        assert r2.stage == ExecutionStage.FAILED
        assert "non-improving loop" in r2.output.lower()

    assert sleeps == [0.05]
    assert task.retry_backoff_seconds == 0.1
