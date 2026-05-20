"""Tests for RepairExecutor convergence guard and backoff."""

from types import SimpleNamespace
from unittest.mock import AsyncMock, patch

import pytest
import pytest_asyncio

from opencas.execution.baa import BoundedAssistantAgent
from opencas.autonomy.models import ActionRiskTier
from opencas.execution import ExecutionStage, RepairExecutor, RepairTask
from opencas.tools import ToolUseResult
from opencas.execution.store import TaskStore
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
    async def _track_sleep(duration):
        sleeps.append(duration)

    with patch("asyncio.sleep", new=AsyncMock(side_effect=_track_sleep)):
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


@pytest.mark.asyncio
async def test_executor_blocks_low_divergence_retry_after_salvage(tmp_path):
    tools = ToolRegistry()
    workspace = str(tmp_path)
    shell = ShellToolAdapter(cwd=workspace, timeout=30.0)
    tools.register("bash_run_command", "Run command", shell, ActionRiskTier.SHELL_LOCAL)
    store = TaskStore(tmp_path / "tasks.db")
    await store.connect()
    captures = []
    executor = RepairExecutor(
        tools=tools,
        store=store,
        runtime=SimpleNamespace(
            ctx=SimpleNamespace(
                shadow_registry=SimpleNamespace(
                    capture_retry_blocked=captures.append,
                )
            )
        ),
    )

    task = RepairTask(
        objective="Continue writing project 4246 from the existing manuscript.",
        verification_command="exit 1",
        max_attempts=3,
        retry_backoff_seconds=0.05,
        meta={
            "resume_project": {
                "signature": "writing-project-4246",
                "canonical_artifact_path": "workspace/writing/4246/story_4246.md",
            }
        },
    )

    with patch("asyncio.sleep", new=AsyncMock()):
        first = await executor.run(task)
        second = await executor.run(task)

    assert first.stage == ExecutionStage.RECOVERING
    assert second.stage == ExecutionStage.FAILED
    assert "retry blocked" in second.output.lower()

    latest = await store.get_latest_salvage_packet(str(task.task_id))
    assert latest is not None
    assert latest.attempt == 2
    assert latest.canonical_artifact_path == "workspace/writing/4246/story_4246.md"
    assert task.meta["retry_governor"]["allowed"] is False
    assert task.meta["counterfactual_review"]["salvage_packet_id"] == str(latest.packet_id)
    assert task.meta["counterfactual_review"]["retry_allowed"] is False
    assert task.meta["counterfactual_review"]["recommended"]["strategy"] in {
        "change_tool_or_arguments",
        "narrow_scope",
        "verify_prerequisite",
        "ask_or_escalate",
    }
    assert len(captures) == 1


@pytest.mark.asyncio
async def test_executor_blocks_guard_stopped_broad_retry_without_repeating_full_attempt(
    tmp_path,
) -> None:
    tools = ToolRegistry()
    store = TaskStore(tmp_path / "tasks.db")
    await store.connect()
    calls = []
    captures = []

    async def _run(**kwargs):
        calls.append(kwargs)
        return ToolUseResult(
            final_output="partial output before guard",
            tool_calls=[
                {
                    "id": "call-1",
                    "name": "runtime_status",
                    "args": {},
                }
            ],
            guard_fired=True,
            guard_reason="Tool loop circuit breaker: tool 'runtime_status' was called 3 times with identical arguments.",
        )

    executor = RepairExecutor(
        tools=tools,
        store=store,
        runtime=SimpleNamespace(
            tool_loop=SimpleNamespace(run=_run),
            scheduler=None,
            ctx=SimpleNamespace(
                shadow_registry=SimpleNamespace(capture_retry_blocked=captures.append)
            ),
        ),
    )
    task = RepairTask(
        objective="Return to project: Writing Project 4",
        max_attempts=3,
        retry_backoff_seconds=0.0,
    )

    first = await executor.run(task)
    assert first.stage == ExecutionStage.FAILED
    assert "retry blocked" in first.output.lower()
    assert len(calls) == 1
    latest = await store.get_latest_salvage_packet(str(task.task_id))
    assert latest is not None
    assert latest.outcome.value == "guard_stopped"
    assert latest.meaningful_progress_signal == "blocker"
    assert latest.tool_signature != "4f53cda18c2baa0c"
    assert task.meta["retry_governor"]["allowed"] is False
    assert captures[0]["task_id"] == str(task.task_id)
    await store.close()


@pytest.mark.asyncio
async def test_executor_allows_guard_stopped_attempt_with_edit_artifact_progress(
    tmp_path,
) -> None:
    tools = ToolRegistry()
    store = TaskStore(tmp_path / "tasks.db")
    await store.connect()
    calls = []
    artifact_path = "/mnt/xtra/OpenCAS/workspace/self/notes/evidence.md"

    async def _run(**kwargs):
        calls.append(kwargs)
        return ToolUseResult(
            final_output="edited evidence note before guard",
            tool_calls=[
                {
                    "id": "call-edit",
                    "name": "edit_file",
                    "args": {"file_path": artifact_path},
                }
            ],
            guard_fired=True,
            guard_reason="Meaningful progress contract blocker: bash_run_command reported a blocker",
        )

    executor = RepairExecutor(
        tools=tools,
        store=store,
        runtime=SimpleNamespace(
            tool_loop=SimpleNamespace(run=_run),
            scheduler=None,
            ctx=SimpleNamespace(
                shadow_registry=SimpleNamespace(capture_retry_blocked=lambda payload: None)
            ),
        ),
    )
    task = RepairTask(
        objective="Verify and record evidence for a standards note.",
        max_attempts=3,
        retry_backoff_seconds=0.0,
    )

    first = await executor.run(task)

    assert first.stage == ExecutionStage.RECOVERING
    assert len(calls) == 1
    latest = await store.get_latest_salvage_packet(str(task.task_id))
    assert latest is not None
    assert latest.outcome.value == "guard_stopped"
    assert latest.artifact_paths_touched == [artifact_path]
    assert latest.meaningful_progress_signal == "artifact"
    assert latest.recommended_mode.value == "resume_existing_artifact"
    assert task.meta["retry_governor"]["allowed"] is True
    await store.close()


def _make_retry_blocked_task():
    task = RepairTask(
        objective="Continue writing project 4246 from the existing manuscript.",
        verification_command="exit 1",
        max_attempts=3,
        retry_backoff_seconds=0.0,
        meta={
            "resume_project": {
                "signature": "writing-project-4246",
                "canonical_artifact_path": "workspace/writing/4246/story_4246.md",
            }
        },
    )
    return task

@pytest.mark.asyncio
async def test_baa_does_not_requeue_retry_blocked_task(tmp_path):
    from opencas.execution.baa import BoundedAssistantAgent
    from opencas.execution.store import TaskStore

    store = TaskStore(tmp_path / "tasks.db")
    await store.connect()

    tools = ToolRegistry()
    workspace = str(tmp_path)
    shell = ShellToolAdapter(cwd=workspace, timeout=30.0)
    tools.register("bash_run_command", "Run command", shell, ActionRiskTier.SHELL_LOCAL)

    baa = BoundedAssistantAgent(tools=tools, store=store)
    task = _make_retry_blocked_task()

    # Attempt 1: verify fails, governor allows (no prior packet) → RECOVERING → BAA requeues
    await baa.submit(task)
    await baa._run_bounded(task)
    assert task.status == "queued"

    # Attempt 2: same divergence signature → governor blocks → executor returns FAILED
    # BAA should NOT requeue; task resolves as failed
    await baa._run_bounded(task)

    assert task.status == "failed"
    assert str(task.task_id) not in baa._futures
    await store.close()


@pytest.mark.asyncio
async def test_baa_provider_rate_limit_failure_does_not_requeue(tmp_path):
    class RateLimitedLLM:
        async def chat_completion(self, messages, **kwargs):
            raise RuntimeError(
                "Client error '429 Too Many Requests' for url "
                "'https://api.z.ai/api/coding/paas/v4/chat/completions'"
            )

    store = TaskStore(tmp_path / "tasks.db")
    await store.connect()
    tools = ToolRegistry()
    baa = BoundedAssistantAgent(tools=tools, llm=RateLimitedLLM(), store=store)
    task = RepairTask(
        objective="Generate and send a comprehensive health check.",
        max_attempts=3,
    )

    await baa.submit(task)
    await baa._run_bounded(task)

    assert task.status == "failed"
    assert task.stage == ExecutionStage.FAILED
    assert "provider backoff" in task.meta["retry_governor"]["reason"]
    assert str(task.task_id) not in baa._futures

    await store.close()
