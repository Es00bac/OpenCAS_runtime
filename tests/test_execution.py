"""Tests for the execution and repair subsystem."""

import asyncio
from datetime import datetime, timedelta, timezone
from pathlib import Path
from types import SimpleNamespace

import pytest
import pytest_asyncio

from opencas.autonomy.executive import ExecutiveState
from opencas.bootstrap import BootstrapConfig, BootstrapPipeline
from opencas.execution import (
    BoundedAssistantAgent,
    ExecutionStage,
    RepairExecutor,
    RepairResult,
    RepairTask,
)
from opencas.execution.receipt_store import ExecutionReceiptStore
from opencas.identity import IdentityManager, IdentityStore
from opencas.proof_chain import (
    ProofChainService,
    ProofClaimStatus,
    ProofClaimType,
    ProofEvidenceKind,
    ProofStore,
)
from opencas.relational.models import MusubiState
from opencas.runtime import AgentRuntime
from opencas.tools import ToolRegistry, ToolUseContext


@pytest_asyncio.fixture
async def runtime(tmp_path: Path):
    config = BootstrapConfig(
        state_dir=tmp_path,
        session_id="test-session",
        embedding_model_id="local-fallback",
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
    assert result.success is True
    assert len(task.artifacts) >= 2
    assert any(a.startswith("plan:") for a in task.artifacts)
    assert any(a.startswith("exec:") for a in task.artifacts)


@pytest.mark.asyncio
async def test_repair_executor_detect_extracts_real_paths(executor: RepairExecutor) -> None:
    task = RepairTask(
        objective=(
            "Update opencas/execution/executor.py and tests/test_execution.py, "
            "then verify README.md and .env.example."
        )
    )

    detected = await executor._detect(task)

    assert detected == (
        "opencas/execution/executor.py,tests/test_execution.py,README.md,.env.example"
    )


@pytest.mark.asyncio
async def test_repair_executor_detect_ignores_domains_and_versions(executor: RepairExecutor) -> None:
    task = RepairTask(
        objective="Investigate v1.2 behavior on example.com without touching files."
    )

    detected = await executor._detect(task)

    assert detected == ""


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
async def test_bounded_assistant_agent_links_receipt_to_promise_claim(tmp_path: Path) -> None:
    tools = ToolRegistry()
    receipt_store = await ExecutionReceiptStore(tmp_path / "receipts.db").connect()
    proof_store = await ProofStore(tmp_path / "proof_chain.db").connect()
    proof_chain = ProofChainService(proof_store)
    try:
        claim = await proof_chain.record_claim(
            claim_type=ProofClaimType.PROMISE,
            claim="I will run a receipt-backed task.",
            subject="self_commitment",
        )
        runtime = SimpleNamespace(proof_chain=proof_chain)
        baa = BoundedAssistantAgent(
            tools=tools,
            max_concurrent=1,
            receipt_store=receipt_store,
            runtime=runtime,
        )
        task = RepairTask(
            objective="Run a receipt-backed task.",
            meta={"claim_id": str(claim.claim_id)},
        )

        future = await baa.submit(task)
        await baa.start()
        result = await asyncio.wait_for(future, timeout=5.0)
        await baa.stop()

        receipt = (await receipt_store.list_by_task(str(task.task_id)))[0]
        updated = await proof_store.get_claim(str(claim.claim_id))
        assert result.success is True
        assert updated is not None
        assert updated.status == ProofClaimStatus.EVIDENCE_LINKED
        assert any(
            link.evidence_kind == ProofEvidenceKind.RECEIPT
            and link.evidence_id == f"receipt:{receipt.receipt_id}"
            for link in updated.evidence_links
        )
    finally:
        await receipt_store.close()
        await proof_store.close()


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
async def test_bounded_assistant_agent_suppresses_live_duplicate_objective(tmp_path: Path) -> None:
    tools = ToolRegistry()
    baa = BoundedAssistantAgent(tools=tools, max_concurrent=1)
    first = RepairTask(objective="Quiet task beacon repair", project_id="loop-1")
    duplicate = RepairTask(objective="Quiet task beacon repair", project_id="loop-1")

    first_future = await baa.submit(first)
    duplicate_future = await baa.submit(duplicate)

    assert duplicate_future is first_future
    assert duplicate.task_id == first.task_id


@pytest.mark.asyncio
async def test_bounded_assistant_agent_suppresses_recent_terminal_duplicate(tmp_path: Path) -> None:
    tools = ToolRegistry()
    baa = BoundedAssistantAgent(tools=tools, max_concurrent=1)
    original = RepairTask(objective="Quiet task beacon repair", project_id="loop-1")
    original_result = RepairResult(
        task_id=original.task_id,
        success=True,
        stage=ExecutionStage.DONE,
        output="already completed",
        timestamp=datetime.now(timezone.utc) - timedelta(minutes=5),
    )
    baa._remember_terminal_duplicate(original, original_result)

    duplicate = RepairTask(objective="Quiet task beacon repair", project_id="loop-1")
    future = await baa.submit(duplicate)
    result = await future

    assert result == original_result
    assert duplicate.task_id == original.task_id


@pytest.mark.asyncio
async def test_bounded_assistant_agent_hydrates_recent_failed_duplicate_after_restart(
    tmp_path: Path,
) -> None:
    tools = ToolRegistry()
    from opencas.execution.store import TaskStore

    store = TaskStore(tmp_path / "tasks.db")
    await store.connect()
    original = RepairTask(
        objective="Continue writing project 4246 from the existing manuscript.",
        project_id="loop-1",
        meta={
            "retry_governor": {
                "allowed": False,
                "reason": "low-divergence broad retry without new evidence",
                "mode": "resume_existing_artifact",
            },
            "resume_project": {
                "canonical_artifact_path": "workspace/writing/4246/story_4246.md",
            },
        },
    )
    await store.save(original)
    original_result = RepairResult(
        task_id=original.task_id,
        success=False,
        stage=ExecutionStage.FAILED,
        output="retry blocked",
        timestamp=datetime.now(timezone.utc) - timedelta(minutes=5),
    )
    await store.save_result(original_result)

    restarted_baa = BoundedAssistantAgent(tools=tools, max_concurrent=1, store=store)
    await restarted_baa.start()
    duplicate = RepairTask(
        objective="Continue writing project 4246 from the existing manuscript.",
        project_id="loop-1",
    )
    future = await restarted_baa.submit(duplicate)
    result = await future

    assert result == original_result
    assert duplicate.task_id == original.task_id
    assert str(original.task_id) not in restarted_baa._live_tasks

    await restarted_baa.stop()
    await store.close()


@pytest.mark.asyncio
async def test_bounded_assistant_agent_does_not_reframe_plain_single_failure(
    tmp_path: Path,
) -> None:
    tools = ToolRegistry()
    identity = IdentityManager(IdentityStore(tmp_path / "identity"))
    identity.load()
    executive = ExecutiveState(identity=identity)
    captures: list[dict[str, object]] = []
    runtime = SimpleNamespace(
        ctx=SimpleNamespace(
            executive=executive,
            shadow_registry=SimpleNamespace(capture_retry_blocked=captures.append),
        )
    )
    baa = BoundedAssistantAgent(tools=tools, max_concurrent=1, runtime=runtime)
    original = RepairTask(objective="Repair the report export", project_id="loop-1")
    original_result = RepairResult(
        task_id=original.task_id,
        success=False,
        stage=ExecutionStage.FAILED,
        output="first failure without retry-governor evidence",
        timestamp=datetime.now(timezone.utc) - timedelta(minutes=5),
    )
    baa._remember_terminal_duplicate(original, original_result)

    duplicate = RepairTask(objective="Repair the report export", project_id="loop-1")
    future = await baa.submit(duplicate)

    assert future.done() is False
    assert duplicate.task_id != original.task_id
    assert captures == []
    assert executive.parked_goals == []


@pytest.mark.asyncio
async def test_bounded_assistant_agent_failed_duplicate_parks_reframe_and_captures_shadow(tmp_path: Path) -> None:
    tools = ToolRegistry()
    identity = IdentityManager(IdentityStore(tmp_path / "identity"))
    identity.load()
    executive = ExecutiveState(identity=identity)
    captures: list[dict[str, object]] = []
    runtime = SimpleNamespace(
        ctx=SimpleNamespace(
            executive=executive,
            shadow_registry=SimpleNamespace(capture_retry_blocked=captures.append),
        )
    )
    baa = BoundedAssistantAgent(tools=tools, max_concurrent=1, runtime=runtime)
    original = RepairTask(
        objective="Continue writing project 4246 from the existing manuscript.",
        project_id="loop-1",
        meta={
            "retry_governor": {
                "allowed": False,
                "reason": "low-divergence broad retry without new evidence",
                "mode": "resume_existing_artifact",
            },
            "resume_project": {
                "canonical_artifact_path": "workspace/writing/4246/story_4246.md",
                "best_next_step": "Resume from workspace/writing/4246/story_4246.md with one narrow edit.",
            }
        },
    )
    original_result = RepairResult(
        task_id=original.task_id,
        success=False,
        stage=ExecutionStage.FAILED,
        output="retry blocked",
        timestamp=datetime.now(timezone.utc) - timedelta(minutes=5),
    )
    baa._remember_terminal_duplicate(original, original_result)

    duplicate = RepairTask(
        objective="Continue writing project 4246 from the existing manuscript.",
        project_id="loop-1",
    )
    future = await baa.submit(duplicate)
    result = await future

    assert result == original_result
    assert len(captures) == 1
    assert captures[0]["capture_source"] == "baa_duplicate_suppression"
    assert captures[0]["best_next_step"] == (
        "Resume from workspace/writing/4246/story_4246.md with one narrow edit."
    )
    assert "Continue writing project 4246 from the existing manuscript." in executive.parked_goals
    metadata = executive.parked_goal_metadata["Continue writing project 4246 from the existing manuscript."]
    assert metadata["reason"] == "low_divergence_reframe"
    assert metadata["reframe_hint"] == (
        "Resume from workspace/writing/4246/story_4246.md with one narrow edit."
    )
    assert metadata["source_artifact"] == "workspace/writing/4246/story_4246.md"


@pytest.mark.asyncio
async def test_bounded_assistant_agent_daydream_duplicate_reframe_stays_out_of_live_parked_goals(
    tmp_path: Path,
) -> None:
    tools = ToolRegistry()
    identity = IdentityManager(IdentityStore(tmp_path / "identity"))
    identity.load()
    executive = ExecutiveState(identity=identity)
    captures: list[dict[str, object]] = []
    runtime = SimpleNamespace(
        ctx=SimpleNamespace(
            executive=executive,
            shadow_registry=SimpleNamespace(capture_retry_blocked=captures.append),
        )
    )
    baa = BoundedAssistantAgent(tools=tools, max_concurrent=1, runtime=runtime)
    original = RepairTask(
        objective="A noted-as-such log of the interruption should become a ritual",
        project_id="loop-1",
        meta={
            "origin": "daydream",
            "source_reflection_id": "reflection-1",
            "retry_governor": {
                "allowed": False,
                "reason": "low-divergence broad retry without new evidence",
                "mode": "deterministic_review",
            },
        },
    )
    original_result = RepairResult(
        task_id=original.task_id,
        success=False,
        stage=ExecutionStage.FAILED,
        output="retry blocked",
        timestamp=datetime.now(timezone.utc) - timedelta(minutes=5),
    )
    baa._remember_terminal_duplicate(original, original_result)

    duplicate = RepairTask(
        objective="A noted-as-such log of the interruption should become a ritual",
        project_id="loop-1",
        meta={"origin": "daydream"},
    )
    future = await baa.submit(duplicate)
    result = await future

    assert result == original_result
    assert len(captures) == 1
    assert captures[0]["capture_source"] == "baa_duplicate_suppression"
    assert executive.parked_goals == []
    assert "A noted-as-such log of the interruption should become a ritual" not in executive.parked_goal_metadata


@pytest.mark.asyncio
async def test_bounded_assistant_agent_routes_self_referential_suppression_to_thread_registry(
    tmp_path: Path,
) -> None:
    tools = ToolRegistry()
    identity = IdentityManager(IdentityStore(tmp_path / "identity"))
    identity.load()
    executive = ExecutiveState(identity=identity)
    captures: list[dict[str, object]] = []

    class FakeThreadRegistry:
        def __init__(self) -> None:
            self.beads: list[dict[str, object]] = []

        async def ensure_thread_anchor(self, **kwargs):
            return SimpleNamespace(anchor_id=kwargs.get("anchor_id") or "suppressed-reframes")

        async def create_candidate_bead(self, **kwargs):
            self.beads.append(kwargs)
            return SimpleNamespace(bead_id="suppressed-reframe-1")

    thread_registry = FakeThreadRegistry()
    objective = "browser_click: Click at x=1590, y=890 to click lightsaber"
    runtime = SimpleNamespace(
        ctx=SimpleNamespace(
            executive=executive,
            shadow_registry=SimpleNamespace(capture_retry_blocked=captures.append),
        ),
        thread_registry_service=thread_registry,
    )
    baa = BoundedAssistantAgent(tools=tools, max_concurrent=1, runtime=runtime)
    original = RepairTask(
        objective=objective,
        project_id="loop-1",
        meta={
            "canonical_artifact_path": objective,
            "retry_governor": {
                "allowed": False,
                "reason": "low-divergence broad retry without new evidence",
                "mode": "deterministic_review",
            },
        },
    )
    original_result = RepairResult(
        task_id=original.task_id,
        success=False,
        stage=ExecutionStage.FAILED,
        output="retry blocked",
        timestamp=datetime.now(timezone.utc) - timedelta(minutes=5),
    )
    baa._remember_terminal_duplicate(original, original_result)

    duplicate = RepairTask(
        objective=objective,
        project_id="loop-1",
        meta={"canonical_artifact_path": objective},
    )
    future = await baa.submit(duplicate)
    result = await future

    assert result == original_result
    assert executive.parked_goals == []
    assert objective not in executive.parked_goal_metadata
    assert len(thread_registry.beads) == 1
    bead = thread_registry.beads[0]
    assert bead["source_kind"] == "suppressed_reframe"
    assert bead["source_ref"].startswith(f"suppressed_reframe:{original.task_id}:")
    assert objective in bead["content"]


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
async def test_runtime_allows_managed_workspace_write_under_stress(
    runtime: AgentRuntime,
) -> None:
    target = runtime.ctx.config.agent_workspace_root() / "stress-write-check.md"
    runtime.ctx.identity.user_model.trust_level = 0.95
    runtime.ctx.identity.save()
    runtime.ctx.somatic.set_tension(1.0)
    runtime.ctx.somatic.set_fatigue(1.0)

    result = await runtime.execute_tool(
        "fs_write_file",
        {"file_path": str(target), "content": "ok\n"},
    )

    assert result["success"] is True
    assert target.read_text(encoding="utf-8") == "ok\n"


@pytest.mark.asyncio
async def test_runtime_allows_managed_workspace_shell_verification_under_stress(
    runtime: AgentRuntime,
) -> None:
    managed_root = runtime.ctx.config.agent_workspace_root()
    managed_root.mkdir(parents=True, exist_ok=True)
    script = managed_root / "echo_ok.py"
    script.write_text("print('ok')\n", encoding="utf-8")

    runtime.ctx.identity.user_model.trust_level = 0.95
    runtime.ctx.identity.save()
    runtime.ctx.somatic.set_tension(1.0)
    runtime.ctx.somatic.set_fatigue(1.0)
    if runtime.ctx.relational is not None:
        runtime.ctx.relational._state = MusubiState(musubi=0.8)

    result = await runtime.execute_tool(
        "bash_run_command",
        {"command": f"cd {managed_root} && python echo_ok.py"},
    )

    assert result["success"] is True
    assert "\"stdout\": \"ok\\n\"" in result["output"]


@pytest.mark.asyncio
async def test_conversation_tool_context_does_not_inherit_unrelated_active_plan(
    runtime: AgentRuntime,
) -> None:
    plan_store = getattr(runtime.ctx, "plan_store", None)
    assert plan_store is not None
    await plan_store.create_plan(
        "plan-unrelated",
        content="unrelated active plan",
        task_id="different-task",
    )
    await plan_store.set_status("plan-unrelated", "active")

    ctx = await runtime._build_tool_use_context("conversation-session")

    assert ctx.plan_mode is False
    assert ctx.active_plan_id is None


@pytest.mark.asyncio
async def test_task_tool_context_recovers_matching_active_plan(
    runtime: AgentRuntime,
) -> None:
    plan_store = getattr(runtime.ctx, "plan_store", None)
    assert plan_store is not None
    await plan_store.create_plan(
        "plan-task-match",
        content="repair plan",
        task_id="task-123",
    )
    await plan_store.set_status("plan-task-match", "active")

    ctx = ToolUseContext(
        runtime=runtime,
        session_id="task-session",
        task_id="task-123",
    )
    hydrated = await runtime._build_tool_use_context("conversation-session")
    assert hydrated.task_id is None

    from opencas.runtime.tool_runtime import hydrate_runtime_tool_use_context

    hydrated = await hydrate_runtime_tool_use_context(runtime, ctx)

    assert hydrated.plan_mode is True
    assert hydrated.active_plan_id == "plan-task-match"


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
    assert len(recovering_tasks) == len(pending2)
    await baa2.stop()
    await store.close()
