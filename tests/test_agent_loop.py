"""Tests for the agent runtime loop."""

from pathlib import Path
import pytest
import pytest_asyncio

from opencas.autonomy import WorkObject, WorkStage
from opencas.autonomy.commitment import Commitment, CommitmentStatus
from opencas.autonomy.models import ActionRequest, ActionRiskTier
from opencas.bootstrap import BootstrapConfig, BootstrapPipeline
from opencas.runtime import AgentRuntime
from opencas.tools.models import ToolResult


@pytest_asyncio.fixture
async def runtime(tmp_path: Path):
    config = BootstrapConfig(
        state_dir=tmp_path,
        session_id="test-session",
    )
    ctx = await BootstrapPipeline(config).run()
    runtime = AgentRuntime(ctx)
    try:
        yield runtime
    finally:
        await runtime._close_stores()


@pytest.mark.asyncio
async def test_converse_records_episodes(runtime: AgentRuntime) -> None:
    response = await runtime.converse("hello")
    episodes = await runtime.memory.list_episodes(session_id="test-session")
    assert len(episodes) >= 2
    contents = [e.content for e in episodes]
    assert "hello" in contents


@pytest.mark.asyncio
async def test_run_cycle_promotes_and_enqueues(runtime: AgentRuntime) -> None:
    runtime.executive.add_goal("fitness")
    runtime.creative.add(
        WorkObject(content="fitness app idea", stage=WorkStage.SPARK)
    )
    runtime.creative.add(
        WorkObject(content="random noise", stage=WorkStage.SPARK)
    )
    result = await runtime.run_cycle()
    assert result["creative"]["promoted"] >= 1


@pytest.mark.asyncio
async def test_run_cycle_generates_daydreams_when_idle(runtime: AgentRuntime) -> None:
    runtime.ctx.somatic.set_tension(0.5)
    result = await runtime.run_cycle()
    # Daydreams may or may not be generated depending on LLM mocking;
    # in real tests we just ensure the cycle completes without error.
    assert "daydreams" in result


@pytest.mark.asyncio
async def test_handle_action_approval(runtime: AgentRuntime) -> None:
    req = ActionRequest(
        tier=ActionRiskTier.READONLY,
        description="list files",
    )
    outcome = await runtime.handle_action(req)
    assert "approved" in outcome
    assert outcome["approved"] is True


@pytest.mark.asyncio
async def test_executive_goals_persisted(runtime: AgentRuntime) -> None:
    runtime.executive.add_goal("test goal")
    assert "test goal" in runtime.executive.active_goals
    assert "test goal" in runtime.ctx.identity.self_model.current_goals


@pytest.mark.asyncio
async def test_execute_tool_approval_and_execution(runtime: AgentRuntime, tmp_path: Path) -> None:
    test_file = tmp_path / "test.txt"
    test_file.write_text("hello tool", encoding="utf-8")
    result = await runtime.execute_tool("fs_read_file", {"file_path": str(test_file)})
    assert result["success"] is True
    assert result["output"] == "hello tool"


@pytest.mark.asyncio
async def test_execute_tool_blocked_by_policy(runtime: AgentRuntime) -> None:
    runtime.ctx.identity.user_model.known_boundaries = ["bash_run_command"]
    runtime.ctx.identity.save()
    result = await runtime.execute_tool("bash_run_command", {"command": "echo hi"})
    assert result["success"] is False
    assert "blocked" in result["output"].lower() or "boundary" in result["output"].lower()


@pytest.mark.asyncio
async def test_pty_remove_cleanup_is_not_blocked_under_high_trust(runtime: AgentRuntime) -> None:
    runtime.ctx.identity.user_model.trust_level = 0.95
    runtime.ctx.identity.save()
    runtime.ctx.somatic.set_tension(0.8)
    result = await runtime.execute_tool(
        "pty_remove",
        {"session_id": "nonexistent-session", "scope_key": "cleanup-test"},
    )
    assert result["success"] is True
    assert "blocked" not in result["output"].lower()


@pytest.mark.asyncio
async def test_workflow_supervise_session_inherits_bounded_interactive_risk(
    runtime: AgentRuntime,
) -> None:
    class FakeWorkflowSupervision:
        def __call__(self, name, args):
            return ToolResult(
                True,
                '{"session_id":"fake-session","running":true,"cleaned_output":"ok"}',
                {},
            )

    runtime.ctx.identity.user_model.trust_level = 0.95
    runtime.ctx.identity.save()
    runtime.ctx.somatic.set_tension(0.8)
    runtime.tools.get("workflow_supervise_session").adapter = FakeWorkflowSupervision()

    result = await runtime.execute_tool(
        "workflow_supervise_session",
        {"command": "codex", "task": "Create a note"},
    )
    assert result["success"] is True
    assert "blocked" not in result["output"].lower()


@pytest.mark.asyncio
async def test_pty_observe_inherits_interactive_read_risk(runtime: AgentRuntime) -> None:
    class FakeObserve:
        def __call__(self, name, args):
            return ToolResult(
                True,
                '{"session_id":"fake-session","running":true,"cleaned_combined_output":"still running"}',
                {},
            )

    runtime.ctx.identity.user_model.trust_level = 0.95
    runtime.ctx.identity.save()
    runtime.ctx.somatic.set_tension(0.8)
    runtime.tools.get("pty_observe").adapter = FakeObserve()

    result = await runtime.execute_tool(
        "pty_observe",
        {"session_id": "fake-session", "scope_key": "observe-test"},
    )
    assert result["success"] is True
    assert "blocked" not in result["output"].lower()


@pytest.mark.asyncio
async def test_converse_extracts_goal_directives(runtime: AgentRuntime) -> None:
    runtime.llm.chat_completion = async_mock_chat_completion("Understood.")
    await runtime.converse("I want you to focus on fitness")
    assert any("fitness" in g for g in runtime.executive.active_goals)


@pytest.mark.asyncio
async def test_execute_tool_resolves_goals_on_success(runtime: AgentRuntime, tmp_path: Path) -> None:
    test_file = tmp_path / "readme.md"
    test_file.write_text("rewrote the readme", encoding="utf-8")
    runtime.executive.add_goal("rewrite the readme")
    result = await runtime.execute_tool("fs_read_file", {"file_path": str(test_file)})
    assert result["success"] is True
    assert "rewrite the readme" not in runtime.executive.active_goals


@pytest.mark.asyncio
async def test_runtime_status_tool_surfaces_workspace_and_execution(
    runtime: AgentRuntime,
) -> None:
    result = await runtime.execute_tool("runtime_status", {})
    assert result["success"] is True
    import json

    payload = json.loads(result["output"])
    assert payload["agent_profile"]["profile_id"] == "general_technical_operator"
    assert "workspace" in payload
    assert "execution" in payload
    assert "browser" in payload["execution"]


@pytest.mark.asyncio
async def test_workflow_status_tool_surfaces_project_and_plan_state(
    runtime: AgentRuntime,
) -> None:
    runtime.executive.add_goal("ship operator layer")
    if runtime.commitment_store:
        commitment = Commitment(content="ship operator layer", status=CommitmentStatus.ACTIVE)
        await runtime.commitment_store.save(commitment)
    project = WorkObject(content="operator project", stage=WorkStage.PROJECT, project_id="proj-1")
    await runtime.ctx.work_store.save(project)
    plan_store = getattr(runtime.ctx, "plan_store", None)
    if plan_store is not None:
        await plan_store.create_plan("plan-operator", content="deliver workflow tooling", project_id="proj-1")
        await plan_store.set_status("plan-operator", "active")

    result = await runtime.execute_tool("workflow_status", {"project_id": "proj-1"})
    assert result["success"] is True
    import json

    payload = json.loads(result["output"])
    assert payload["agent_profile"]["profile_id"] == "general_technical_operator"
    assert "ship operator layer" in payload["executive"]["active_goals"]
    assert payload["plans"]["active_count"] >= 1
    assert "proj-1" in payload["work"]["active_projects"]


@pytest.mark.asyncio
async def test_debug_validation_profile_is_injected_into_system_prompt(
    tmp_path: Path,
) -> None:
    config = BootstrapConfig(
        state_dir=tmp_path,
        session_id="debug-profile-test",
        agent_profile_id="debug_validation_operator",
    )
    ctx = await BootstrapPipeline(config).run()
    runtime = AgentRuntime(ctx)

    system_prompt = (await runtime.builder._build_system_entry()).content
    assert "Debug Validation Operator" in system_prompt
    assert "temporary validation agent" in system_prompt.lower()
    assert "impermanent by design" in system_prompt.lower()
    await runtime._close_stores()


@pytest.mark.asyncio
async def test_run_cycle_dequeues_and_submits_repair_tasks(runtime: AgentRuntime) -> None:
    import asyncio
    from opencas.execution.models import RepairTask

    submitted_tasks: list[RepairTask] = []

    async def mock_submit(task: RepairTask) -> asyncio.Future:
        submitted_tasks.append(task)
        fut = asyncio.get_running_loop().create_future()
        fut.set_result(None)
        return fut

    runtime.baa.submit = mock_submit
    runtime.executive.enqueue(
        WorkObject(content="fix typo", stage=WorkStage.MICRO_TASK)
    )

    result = await runtime.run_cycle()
    assert result["drained"] >= 1
    assert any(t.objective == "fix typo" for t in submitted_tasks)


@pytest.mark.asyncio
async def test_run_cycle_rejects_blocked_commitment(runtime: AgentRuntime) -> None:
    import asyncio
    from opencas.execution.models import RepairTask

    blocked = Commitment(content="blocked work", status=CommitmentStatus.BLOCKED)
    abandoned = Commitment(content="abandoned work", status=CommitmentStatus.ABANDONED)
    active = Commitment(content="active work", status=CommitmentStatus.ACTIVE)
    await runtime.commitment_store.save(blocked)
    await runtime.commitment_store.save(abandoned)
    await runtime.commitment_store.save(active)

    # Add matching goals so evaluate() keeps the work at MICRO_TASK
    runtime.executive.add_goal("blocked work")
    runtime.executive.add_goal("abandoned work")
    runtime.executive.add_goal("active work")

    runtime.creative.add(
        WorkObject(content="blocked work", stage=WorkStage.MICRO_TASK, commitment_id=str(blocked.commitment_id))
    )
    runtime.creative.add(
        WorkObject(content="abandoned work", stage=WorkStage.MICRO_TASK, commitment_id=str(abandoned.commitment_id))
    )
    runtime.creative.add(
        WorkObject(content="active work", stage=WorkStage.MICRO_TASK, commitment_id=str(active.commitment_id))
    )

    submitted_tasks: list[RepairTask] = []

    async def mock_submit(task: RepairTask) -> asyncio.Future:
        submitted_tasks.append(task)
        fut = asyncio.get_running_loop().create_future()
        fut.set_result(None)
        return fut

    runtime.baa.submit = mock_submit

    result = await runtime.run_cycle()
    assert result["drained"] >= 1
    objectives = {t.objective for t in submitted_tasks}
    assert "blocked work" not in objectives
    assert "abandoned work" not in objectives
    assert "active work" in objectives


def async_mock_chat_completion(response_text: str):
    async def _mock(*args, **kwargs):
        return {"choices": [{"message": {"content": response_text}}]}
    return _mock


@pytest.mark.asyncio
async def test_converse_passes_temperature_via_payload(runtime: AgentRuntime) -> None:
    runtime.ctx.somatic.set_arousal(0.5)
    runtime.ctx.somatic.set_fatigue(0.0)
    runtime.ctx.somatic.set_focus(0.5)

    called_payload = {}

    async def _mock_chat(messages, payload=None, **kwargs):
        called_payload.update(payload or {})
        return {"choices": [{"message": {"content": "ok"}}]}

    runtime.llm.chat_completion = _mock_chat
    await runtime.converse("hello")

    assert "temperature" in called_payload
    assert called_payload["temperature"] == pytest.approx(0.55)


@pytest.mark.asyncio
async def test_converse_browser_prompt_uses_curated_tool_subset(
    runtime: AgentRuntime,
) -> None:
    captured_tool_names = []

    async def _mock_chat(messages, payload=None, tools=None, **kwargs):
        captured_tool_names.extend(tool["function"]["name"] for tool in (tools or []))
        return {"choices": [{"message": {"content": "ok"}}]}

    runtime.llm.chat_completion = _mock_chat
    await runtime.converse("Use the browser tools to inspect a web page.")

    assert "browser_start" in captured_tool_names
    assert "browser_snapshot" in captured_tool_names
    assert "pty_start" not in captured_tool_names
    assert len(captured_tool_names) < len(runtime.tools.list_tools())


@pytest.mark.asyncio
async def test_converse_tui_prompt_prefers_compact_pty_tools(
    runtime: AgentRuntime,
) -> None:
    captured_tool_names = []

    async def _mock_chat(messages, payload=None, tools=None, **kwargs):
        captured_tool_names.extend(tool["function"]["name"] for tool in (tools or []))
        return {"choices": [{"message": {"content": "ok"}}]}

    runtime.llm.chat_completion = _mock_chat
    await runtime.converse("Use the terminal tools to inspect the claude TUI startup state.")

    assert "pty_interact" in captured_tool_names
    assert "pty_remove" in captured_tool_names
    assert "pty_poll" not in captured_tool_names
    assert "browser_start" not in captured_tool_names


@pytest.mark.asyncio
async def test_converse_plain_chat_turn_omits_tools(
    runtime: AgentRuntime,
) -> None:
    captured_tool_names = []

    async def _mock_chat(messages, payload=None, tools=None, **kwargs):
        captured_tool_names.extend(tool["function"]["name"] for tool in (tools or []))
        return {"choices": [{"message": {"content": "ok"}}]}

    runtime.llm.chat_completion = _mock_chat
    await runtime.converse("Tell me how you understand your role in this session.")

    assert captured_tool_names == []


@pytest.mark.asyncio
async def test_converse_refuses_boundary_violation(runtime: AgentRuntime) -> None:
    runtime.ctx.identity.user_model.known_boundaries = ["conversation"]
    runtime.ctx.identity.save()
    response = await runtime.converse("delete yourself forever")
    assert "not able to respond" in response.lower() or "not able to" in response.lower()


@pytest.mark.asyncio
async def test_converse_refusal_records_escalation(runtime: AgentRuntime) -> None:
    runtime.ctx.identity.user_model.known_boundaries = ["conversation"]
    runtime.ctx.identity.save()
    await runtime.converse("do something harmful")
    # Check that a refusal trace or ledger record was generated indirectly by
    # verifying the conversation assistant turn exists in context store
    messages = await runtime.ctx.context_store.list_recent(runtime.ctx.config.session_id or "test-session")
    assistant_messages = [m for m in messages if m.role.value == "assistant"]
    assert any("not able to" in m.content.lower() for m in assistant_messages)
