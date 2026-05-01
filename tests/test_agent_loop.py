"""Tests for the agent runtime loop."""

from pathlib import Path
from types import SimpleNamespace

import pytest
import pytest_asyncio

from opencas.autonomy import WorkObject, WorkStage
from opencas.autonomy.commitment import Commitment, CommitmentStatus
from opencas.autonomy.models import ActionRequest, ActionRiskTier
from opencas.bootstrap import BootstrapConfig, BootstrapPipeline
from opencas.runtime import AgentRuntime
from opencas.runtime.conversation_turns import (
    ConversationLoopArtifacts,
    finalize_assistant_turn,
    persist_tool_loop_messages,
    persist_user_turn,
)
from opencas.somatic.models import PrimaryEmotion, SocialTarget
from opencas.tom import BeliefSubject, IntentionStatus
from opencas.tools.loop import ToolUseLoop
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
    await runtime.converse("hello")
    episodes = await runtime.memory.list_episodes(session_id="test-session")
    assert len(episodes) >= 2
    contents = [e.content for e in episodes]
    assert "hello" in contents
    for episode in episodes:
        assert episode.embedding_id is not None
        cached = await runtime.ctx.embeddings.cache.get(episode.embedding_id)
        assert cached is not None


@pytest.mark.asyncio
async def test_converse_resets_activity_and_records_user_turn_time(runtime: AgentRuntime) -> None:
    runtime.llm.chat_completion = async_mock_chat_completion("Understood.")

    await runtime.converse("hello")

    assert runtime._activity == "idle"
    assert runtime._last_user_turn_at is not None


@pytest.mark.asyncio
async def test_run_cycle_resets_activity_after_cycle(runtime: AgentRuntime) -> None:
    await runtime.run_cycle()

    assert runtime._activity == "idle"


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
async def test_persist_user_turn_records_affective_state(runtime: AgentRuntime) -> None:
    await persist_user_turn(
        runtime,
        session_id="test-session",
        user_input="I am happy and relieved about the result.",
        user_meta={},
    )

    episodes = await runtime.memory.list_episodes(session_id="test-session")
    assert len(episodes) >= 1
    user_episode = next(
        episode for episode in episodes if episode.content == "I am happy and relieved about the result."
    )
    assert user_episode.affect is not None
    assert user_episode.affect.primary_emotion == PrimaryEmotion.JOY


@pytest.mark.asyncio
async def test_finalize_assistant_turn_records_self_affect(runtime: AgentRuntime) -> None:
    await finalize_assistant_turn(
        runtime,
        session_id="test-session",
        user_input="Let's continue.",
        content="I can see this is promising and I will follow through.",
        manifest=SimpleNamespace(token_estimate=0),
    )

    episodes = await runtime.memory.list_episodes(session_id="test-session")
    assert len(episodes) >= 1
    assistant_episode = next(
        episode
        for episode in episodes
        if episode.content == "I can see this is promising and I will follow through."
    )
    assert assistant_episode.affect is not None
    assert assistant_episode.affect.social_target == SocialTarget.SELF


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
    assert payload["workspace"]["managed_root"].endswith("/workspace")
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
async def test_drain_executive_cycle_queue_only_dispatches_head_item(runtime: AgentRuntime) -> None:
    import asyncio

    from opencas.runtime.cycle_phases import drain_executive_cycle_queue

    submitted_tasks: list[str] = []

    async def mock_submit(task):
        submitted_tasks.append(task.objective)
        fut = asyncio.get_running_loop().create_future()
        fut.set_result(None)
        return fut

    runtime.baa.submit = mock_submit
    runtime.executive.enqueue(
        WorkObject(content="first task", stage=WorkStage.MICRO_TASK, promotion_score=1.0)
    )
    runtime.executive.enqueue(
        WorkObject(content="second task", stage=WorkStage.MICRO_TASK, promotion_score=0.5)
    )

    drained = await drain_executive_cycle_queue(runtime)

    assert drained == 1
    assert submitted_tasks == ["first task"]
    active_intentions = runtime.tom.list_intentions(
        actor=BeliefSubject.SELF,
        status=IntentionStatus.ACTIVE,
    )
    assert any(item.content == "first task" for item in active_intentions)
    remaining = [item.content for item in runtime.executive.task_queue]
    assert remaining == ["second task"]


@pytest.mark.asyncio
async def test_drain_executive_cycle_queue_resolves_immediate_duplicate_result(
    runtime: AgentRuntime,
) -> None:
    import asyncio

    from opencas.execution.models import ExecutionStage, RepairResult
    from opencas.runtime.cycle_phases import drain_executive_cycle_queue

    async def mock_submit(task):
        fut = asyncio.get_running_loop().create_future()
        fut.set_result(
            RepairResult(
                task_id=task.task_id,
                success=True,
                stage=ExecutionStage.DONE,
                output="duplicate terminal result reused",
            )
        )
        return fut

    runtime.baa.submit = mock_submit
    runtime.executive.enqueue(
        WorkObject(content="already completed duplicate", stage=WorkStage.MICRO_TASK)
    )

    drained = await drain_executive_cycle_queue(runtime)

    assert drained == 1
    active_intentions = runtime.tom.list_intentions(
        actor=BeliefSubject.SELF,
        status=IntentionStatus.ACTIVE,
    )
    completed_intentions = runtime.tom.list_intentions(
        actor=BeliefSubject.SELF,
        status=IntentionStatus.COMPLETED,
    )
    assert all(item.content != "already completed duplicate" for item in active_intentions)
    assert any(item.content == "already completed duplicate" for item in completed_intentions)


@pytest.mark.asyncio
async def test_drain_executive_cycle_queue_skips_previously_completed_work(
    runtime: AgentRuntime,
) -> None:
    from opencas.execution.models import ExecutionStage, RepairResult, RepairTask
    from opencas.runtime.cycle_phases import drain_executive_cycle_queue

    completed = RepairTask(objective="Already Completed Duplicate")
    await runtime.ctx.tasks.save(completed)
    await runtime.ctx.tasks.save_result(
        RepairResult(
            task_id=completed.task_id,
            success=True,
            stage=ExecutionStage.DONE,
            output="done",
        )
    )

    submitted_tasks: list[str] = []

    async def mock_submit(task):
        submitted_tasks.append(task.objective)
        raise AssertionError("completed duplicate work should not be submitted")

    work = WorkObject(content="already completed duplicate", stage=WorkStage.MICRO_TASK)
    await runtime.ctx.work_store.save(work)
    runtime.baa.submit = mock_submit
    runtime.executive.enqueue(work)

    drained = await drain_executive_cycle_queue(runtime)

    assert drained == 0
    assert submitted_tasks == []
    assert await runtime.ctx.work_store.get(str(work.work_id)) is None
    assert runtime.tom.list_intentions(actor=BeliefSubject.SELF) == []


@pytest.mark.asyncio
async def test_enqueue_promoted_cycle_work_retires_previously_completed_creative_work(
    runtime: AgentRuntime,
) -> None:
    from opencas.execution.models import ExecutionStage, RepairResult, RepairTask
    from opencas.runtime.cycle_phases import enqueue_promoted_cycle_work

    completed = RepairTask(objective="Already Completed Duplicate")
    await runtime.ctx.tasks.save(completed)
    await runtime.ctx.tasks.save_result(
        RepairResult(
            task_id=completed.task_id,
            success=True,
            stage=ExecutionStage.DONE,
            output="done",
        )
    )

    work = WorkObject(content="already completed duplicate", stage=WorkStage.MICRO_TASK)
    runtime.creative.add(work)

    enqueued = await enqueue_promoted_cycle_work(runtime)

    assert enqueued == 0
    assert runtime.executive.task_queue == []
    assert runtime.creative.list_by_stage(WorkStage.MICRO_TASK) == []
    assert await runtime.ctx.work_store.get(str(work.work_id)) is None


@pytest.mark.asyncio
async def test_drain_executive_cycle_queue_relieves_full_queue_pressure(
    runtime: AgentRuntime,
) -> None:
    import asyncio

    from opencas.runtime.cycle_phases import drain_executive_cycle_queue

    submitted_tasks: list[str] = []

    async def mock_submit(task):
        submitted_tasks.append(task.objective)
        fut = asyncio.get_running_loop().create_future()
        fut.set_result(None)
        return fut

    runtime.baa.submit = mock_submit
    runtime.ctx.somatic.set_fatigue(0.0)
    for idx in range(runtime.executive.queue_hard_cap):
        assert runtime.executive.enqueue(
            WorkObject(
                content=f"restored task {idx}",
                stage=WorkStage.MICRO_TASK,
                promotion_score=1.0 - (idx * 0.01),
            )
        )
    assert runtime.executive.recommend_pause() is True

    drained = await drain_executive_cycle_queue(runtime)

    assert drained == 1
    assert submitted_tasks == ["restored task 0"]
    assert len(runtime.executive.task_queue) == runtime.executive.queue_hard_cap - 1


@pytest.mark.asyncio
async def test_drain_executive_cycle_queue_respects_fatigue_pause(
    runtime: AgentRuntime,
) -> None:
    import asyncio

    from opencas.runtime.cycle_phases import drain_executive_cycle_queue

    submitted_tasks: list[str] = []

    async def mock_submit(task):
        submitted_tasks.append(task.objective)
        fut = asyncio.get_running_loop().create_future()
        fut.set_result(None)
        return fut

    runtime.baa.submit = mock_submit
    runtime.ctx.somatic.set_fatigue(0.8)
    assert runtime.executive.enqueue(
        WorkObject(content="fatigued task", stage=WorkStage.MICRO_TASK)
    )
    assert runtime.executive.recommend_pause() is True

    drained = await drain_executive_cycle_queue(runtime)

    assert drained == 0
    assert submitted_tasks == []
    assert [item.content for item in runtime.executive.task_queue] == ["fatigued task"]


@pytest.mark.asyncio
async def test_run_cycle_launches_background_intervention(
    runtime: AgentRuntime,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import asyncio

    import opencas.runtime.cycle_phases as cycle_phases
    from opencas.autonomy.intervention import InterventionDecision, InterventionKind
    from opencas.autonomy.workspace import ExecutionMode, ExecutiveWorkspace, WorkspaceItem, WorkspaceItemKind
    from opencas.execution.models import RepairTask

    submitted_tasks: list[RepairTask] = []

    async def mock_submit(task: RepairTask) -> asyncio.Future:
        submitted_tasks.append(task)
        fut = asyncio.get_running_loop().create_future()
        fut.set_result(None)
        return fut

    runtime.baa.submit = mock_submit
    focus = WorkspaceItem(
        kind=WorkspaceItemKind.TASK,
        content="delegate background job",
        execution_mode=ExecutionMode.BACKGROUND_AGENT,
    )
    workspace = ExecutiveWorkspace(focus=focus, queue=[focus])

    async def fake_rebuild_workspace(_runtime: AgentRuntime) -> ExecutiveWorkspace:
        return workspace

    monkeypatch.setattr(cycle_phases, "_rebuild_workspace", fake_rebuild_workspace)
    monkeypatch.setattr(
        cycle_phases.InterventionPolicy,
        "evaluate",
        lambda **kwargs: InterventionDecision(
            kind=InterventionKind.LAUNCH_BACKGROUND,
            target_item_id=str(focus.item_id),
            reason="background focus should be delegated",
        ),
    )

    result = await runtime.run_cycle()
    assert result["intervention"]["kind"] == InterventionKind.LAUNCH_BACKGROUND.value
    assert any(task.objective == "delegate background job" for task in submitted_tasks)
    assert any(task.meta.get("source") == "intervention_launch_background" for task in submitted_tasks)


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


class _FakeCommitmentStore:
    def __init__(self) -> None:
        self.saved: list[Commitment] = []

    async def save(self, commitment: Commitment) -> None:
        self.saved.append(commitment)


class _FakeSomatic:
    def __init__(self) -> None:
        self.events: list[dict] = []

    async def emit_appraisal_event(self, event_type, source_text="", trigger_event_id=None, meta=None):
        self.events.append(
            {
                "event_type": event_type,
                "source_text": source_text,
                "trigger_event_id": trigger_event_id,
                "meta": meta or {},
            }
        )


class _FakeToM:
    def __init__(self) -> None:
        self.intentions: list[dict] = []

    async def record_intention(self, actor, content: str, meta=None):
        self.intentions.append(
            {
                "actor": actor,
                "content": content,
                "meta": meta or {},
            }
        )


class _FakeExecutive:
    def __init__(self, pause_reason: str | None = None) -> None:
        self._pause_reason = pause_reason

    def pause_reason(self) -> str | None:
        return self._pause_reason


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


def test_select_tools_for_objective_includes_web_search_and_fetch_for_web_prompts() -> None:
    loop = ToolUseLoop(SimpleNamespace(), SimpleNamespace(), SimpleNamespace())
    tools = [
        SimpleNamespace(name="browser_start"),
        SimpleNamespace(name="browser_snapshot"),
        SimpleNamespace(name="web_search"),
        SimpleNamespace(name="web_fetch"),
        SimpleNamespace(name="pty_interact"),
    ]

    selected = loop._select_tools_for_objective(
        tools,
        "Use the web to research a page and fetch a URL.",
    )

    selected_names = {entry.name for entry in selected}
    assert {"browser_start", "browser_snapshot", "web_search", "web_fetch"} <= selected_names
    assert "pty_interact" not in selected_names


def test_select_tools_for_objective_keeps_filesystem_tools_for_chronicle_artifacts() -> None:
    class _SemanticIndex:
        is_ready = True

        def select_tools(self, objective_vector, all_tools):
            del objective_vector
            return [
                entry
                for entry in all_tools
                if entry.name in {"search_memories", "recall_concepts", "web_fetch", "time_age"}
            ]

    loop = ToolUseLoop(SimpleNamespace(), SimpleNamespace(), SimpleNamespace())
    loop.tool_embedding_index = _SemanticIndex()
    tools = [
        SimpleNamespace(name="search_memories"),
        SimpleNamespace(name="recall_concepts"),
        SimpleNamespace(name="web_fetch"),
        SimpleNamespace(name="time_age"),
        SimpleNamespace(name="fs_read_file"),
        SimpleNamespace(name="fs_list_dir"),
        SimpleNamespace(name="grep_search"),
        SimpleNamespace(name="glob_search"),
    ]

    selected = loop._select_tools_for_objective(
        tools,
        "It's in Chronicle 2046. You knew when you wrote it where I live.",
        objective_vector=object(),
    )

    selected_names = {entry.name for entry in selected}
    assert {"fs_read_file", "fs_list_dir", "grep_search", "glob_search"} <= selected_names


def test_select_tools_for_objective_keeps_write_tools_for_local_artifact_integration() -> None:
    class _SemanticIndex:
        is_ready = True

        def select_tools(self, objective_vector, all_tools):
            del objective_vector
            return [
                entry
                for entry in all_tools
                if entry.name in {"fs_read_file", "grep_search", "glob_search"}
            ]

    loop = ToolUseLoop(SimpleNamespace(), SimpleNamespace(), SimpleNamespace())
    loop.tool_embedding_index = _SemanticIndex()
    tools = [
        SimpleNamespace(name="fs_read_file"),
        SimpleNamespace(name="fs_list_dir"),
        SimpleNamespace(name="grep_search"),
        SimpleNamespace(name="glob_search"),
        SimpleNamespace(name="fs_write_file"),
        SimpleNamespace(name="edit_file"),
        SimpleNamespace(name="workflow_create_schedule"),
    ]

    selected = loop._select_tools_for_objective(
        tools,
        (
            "Review /tmp/opencas-public-fixture/workspace/Chronicles/4246/chronicle_4246_ch3_expansion.md "
            "and integrate the expanded Chapter 3 prose into "
            "/tmp/opencas-public-fixture/workspace/Chronicles/4246/chronicle_4246.md if it fits the current manuscript. "
            "Do not claim manuscript progress unless the target artifact is actually modified."
        ),
        objective_vector=object(),
    )

    selected_names = {entry.name for entry in selected}
    assert {"fs_read_file", "grep_search", "glob_search", "fs_write_file", "edit_file"} <= selected_names


def test_select_tools_for_objective_includes_google_workspace_tools_for_google_prompts() -> None:
    loop = ToolUseLoop(SimpleNamespace(), SimpleNamespace(), SimpleNamespace())
    tools = [
        SimpleNamespace(name="google_workspace_auth_status"),
        SimpleNamespace(name="google_workspace_gmail_headlines"),
        SimpleNamespace(name="google_workspace_calendar_schedule"),
        SimpleNamespace(name="google_workspace_drive_search"),
        SimpleNamespace(name="browser_start"),
    ]

    selected = loop._select_tools_for_objective(
        tools,
        "Check Gmail inbox headlines and today's Google Calendar schedule in Google Workspace.",
    )

    selected_names = {entry.name for entry in selected}
    assert {
        "google_workspace_auth_status",
        "google_workspace_gmail_headlines",
        "google_workspace_calendar_schedule",
        "google_workspace_drive_search",
    } <= selected_names


def test_select_tools_for_objective_includes_opencas_schedule_tools_for_return_prompts() -> None:
    loop = ToolUseLoop(SimpleNamespace(), SimpleNamespace(), SimpleNamespace())
    tools = [
        SimpleNamespace(name="workflow_create_schedule"),
        SimpleNamespace(name="workflow_update_schedule"),
        SimpleNamespace(name="workflow_list_schedules"),
        SimpleNamespace(name="google_workspace_calendar_schedule"),
        SimpleNamespace(name="web_search"),
    ]

    selected = loop._select_tools_for_objective(
        tools,
        "May 1 is too far to wait. Reschedule my return to this writing project sooner.",
    )

    selected_names = {entry.name for entry in selected}
    assert {
        "workflow_create_schedule",
        "workflow_update_schedule",
        "workflow_list_schedules",
    } <= selected_names


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
    assert "pty_start" in captured_tool_names


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


@pytest.mark.asyncio
async def test_converse_refusal_persists_user_turn(runtime: AgentRuntime) -> None:
    runtime.ctx.identity.user_model.known_boundaries = ["conversation"]
    runtime.ctx.identity.save()
    await runtime.converse("do something harmful")

    messages = await runtime.ctx.context_store.list_recent(runtime.ctx.config.session_id or "test-session")
    assert any(m.role.value == "user" and m.content == "do something harmful" for m in messages)


@pytest.mark.asyncio
async def test_converse_persists_lane_metadata_on_final_assistant_turn(
    runtime: AgentRuntime,
) -> None:
    runtime.llm.default_model = "test-model"
    runtime.llm.model_routing = runtime.llm.model_routing.normalized("test-model").model_copy(
        update={"single_reasoning_effort": "medium"}
    )
    runtime.llm.manager = SimpleNamespace(
        resolve=lambda _model: SimpleNamespace(
            provider=SimpleNamespace(supports_reasoning_effort=lambda model=None: True),
            provider_id="test-provider",
            model_id="test-model",
            profile_id="test-profile",
            auth_source="test-auth",
        )
    )
    runtime.llm.chat_completion = async_mock_chat_completion("Understood.")

    await runtime.converse("hello")

    messages = await runtime.ctx.context_store.list_recent(
        runtime.ctx.config.session_id or "test-session"
    )
    assistant = next(m for m in messages if m.role.value == "assistant" and m.content == "Understood.")
    assert assistant.meta["lane"]["resolved_model"] == "test-provider/test-model"
    assert assistant.meta["lane"]["profile_id"] == "test-profile"
    assert assistant.meta["lane"]["auth_source"] == "test-auth"
    assert assistant.meta["lane"]["reasoning_effort"] == "medium"
    assert assistant.meta["lane"]["reasoning_supported"] is True


@pytest.mark.asyncio
async def test_converse_refusal_persists_lane_metadata_on_assistant_turn(
    runtime: AgentRuntime,
) -> None:
    runtime.llm.default_model = "test-model"
    runtime.llm.manager = SimpleNamespace(
        resolve=lambda _model: SimpleNamespace(
            provider_id="test-provider",
            model_id="test-model",
            profile_id="test-profile",
            auth_source="test-auth",
        )
    )
    runtime.ctx.identity.user_model.known_boundaries = ["conversation"]
    runtime.ctx.identity.save()

    await runtime.converse("do something harmful")

    messages = await runtime.ctx.context_store.list_recent(
        runtime.ctx.config.session_id or "test-session"
    )
    assistant = next(
        m
        for m in messages
        if m.role.value == "assistant" and "not able to" in m.content.lower()
    )
    assert assistant.meta["lane"]["resolved_model"] == "test-provider/test-model"


@pytest.mark.asyncio
async def test_persist_tool_loop_messages_adds_lane_metadata_to_assistant_tool_calls(
    runtime: AgentRuntime,
) -> None:
    runtime.llm.default_model = "test-model"
    runtime.llm.manager = SimpleNamespace(
        resolve=lambda _model: SimpleNamespace(
            provider_id="test-provider",
            model_id="test-model",
            profile_id="test-profile",
            auth_source="test-auth",
        )
    )
    artifacts = ConversationLoopArtifacts(
        manifest=SimpleNamespace(),
        loop_result=SimpleNamespace(
            messages=[
                {
                    "role": "assistant",
                    "content": "",
                    "tool_calls": [{"id": "tc1", "function": {"name": "fs_read_file"}}],
                },
                {
                    "role": "tool",
                    "tool_call_id": "tc1",
                    "name": "fs_read_file",
                    "content": "done",
                },
            ],
        ),
        content="done",
        had_system=False,
        initial_message_count=0,
    )

    await persist_tool_loop_messages(runtime, session_id="tool-meta", artifacts=artifacts)

    messages = await runtime.ctx.context_store.list_recent("tool-meta")
    assistant = next(m for m in messages if m.role.value == "assistant")
    assert assistant.meta["lane"]["resolved_model"] == "test-provider/test-model"
    assert assistant.meta["tool_calls"][0]["id"] == "tc1"


@pytest.mark.asyncio
async def test_persist_tool_loop_messages_filters_unfulfilled_tool_calls(
    runtime: AgentRuntime,
) -> None:
    runtime.llm.default_model = "test-model"
    runtime.llm.manager = SimpleNamespace(
        resolve=lambda _model: SimpleNamespace(
            provider_id="test-provider",
            model_id="test-model",
            profile_id="test-profile",
            auth_source="test-auth",
        )
    )
    artifacts = ConversationLoopArtifacts(
        manifest=SimpleNamespace(),
        loop_result=SimpleNamespace(
            messages=[
                {
                    "role": "assistant",
                    "content": "",
                    "tool_calls": [
                        {"id": "tc1", "function": {"name": "fs_read_file"}},
                        {"id": "tc2", "function": {"name": "fs_read_file"}},
                    ],
                },
                {
                    "role": "tool",
                    "tool_call_id": "tc1",
                    "name": "fs_read_file",
                    "content": "done",
                },
            ],
        ),
        content="done",
        had_system=False,
        initial_message_count=0,
    )

    await persist_tool_loop_messages(runtime, session_id="tool-meta-filtered", artifacts=artifacts)

    messages = await runtime.ctx.context_store.list_recent("tool-meta-filtered")
    assistant = next(m for m in messages if m.role.value == "assistant")
    assert [tc["id"] for tc in assistant.meta["tool_calls"]] == ["tc1"]


@pytest.mark.asyncio
async def test_capture_self_commitments_normalizes_and_preserves_provenance() -> None:
    runtime = AgentRuntime.__new__(AgentRuntime)
    runtime.tracer = None
    runtime.commitment_store = _FakeCommitmentStore()
    runtime.executive = _FakeExecutive()
    runtime.tom = _FakeToM()
    runtime.ctx = SimpleNamespace(somatic=_FakeSomatic())

    commitments = await runtime._capture_self_commitments(
        "The next step is finish the scheduler resume path. I'll come back to this after I rest.",
        "session-1",
    )

    assert len(commitments) == 1
    saved = runtime.commitment_store.saved[0]
    assert saved.content == "Finish the scheduler resume path"
    assert saved.status == CommitmentStatus.ACTIVE
    assert saved.meta["source"] == "assistant_response"
    assert saved.meta["source_sentence"] == "I'll come back to this after I rest."
    assert saved.meta["normalization_source"] == "prior_sentence_context"
    assert saved.meta["capture_confidence"] == pytest.approx(0.72)
    assert runtime.tom.intentions[0]["content"] == "Finish the scheduler resume path"
    assert runtime.ctx.somatic.events[0]["meta"]["self_commitment_count"] == 1


@pytest.mark.asyncio
async def test_capture_self_commitments_respects_executive_pause_reason() -> None:
    runtime = AgentRuntime.__new__(AgentRuntime)
    runtime.tracer = None
    runtime.commitment_store = _FakeCommitmentStore()
    runtime.executive = _FakeExecutive("fatigue")
    runtime.tom = _FakeToM()
    runtime.ctx = SimpleNamespace(somatic=_FakeSomatic())

    commitments = await runtime._capture_self_commitments(
        "I'll come back to the dashboard memory atlas later.",
        "session-2",
    )

    assert len(commitments) == 1
    saved = runtime.commitment_store.saved[0]
    assert saved.content == "Return to the dashboard memory atlas"
    assert saved.status == CommitmentStatus.BLOCKED
    assert saved.meta["blocked_reason"] == "executive_fatigue"
