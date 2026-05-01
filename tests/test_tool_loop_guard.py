"""Tests for ToolLoopGuard circuit breaker."""

from types import SimpleNamespace
from uuid import uuid4

import pytest

from opencas.autonomy.models import ActionRiskTier
from opencas.execution import RepairTask
from opencas.execution.store import TaskStore
from opencas.planning import PlanStore
from opencas.tools import ToolRegistry, ToolUseContext, ToolUseLoop
from opencas.tools.loop_guard import ToolLoopGuard


class TestToolLoopGuard:
    def test_initial_calls_allowed(self):
        guard = ToolLoopGuard()
        for i in range(ToolLoopGuard.MAX_ROUNDS):
            assert guard.record_call("s1", "fs_read_file", {"path": f"/tmp/{i}"}) is None

    def test_max_rounds_circuit_breaker(self):
        guard = ToolLoopGuard()
        for i in range(ToolLoopGuard.MAX_ROUNDS):
            guard.record_call("s1", "fs_read_file", {"path": f"/tmp/{i}"})

        reason = guard.record_call("s1", "fs_read_file", {"path": "/tmp/extra"})
        assert reason is not None
        assert f"exceeded {ToolLoopGuard.MAX_ROUNDS}" in reason

    def test_identical_call_circuit_breaker(self):
        guard = ToolLoopGuard()
        args = {"path": "/tmp"}
        assert guard.record_call("s1", "fs_read_file", args) is None
        assert guard.record_call("s1", "fs_read_file", args) is None
        reason = guard.record_call("s1", "fs_read_file", args)
        assert reason is not None
        assert "fs_read_file" in reason
        assert "3 times" in reason

    def test_different_tools_do_not_trigger_identical_guard(self):
        guard = ToolLoopGuard()
        for _ in range(5):
            assert guard.record_call("s1", "tool_a", {"x": 1}) is None
            assert guard.record_call("s1", "tool_b", {"x": 1}) is None

    def test_reset_clears_state(self):
        guard = ToolLoopGuard()
        for i in range(ToolLoopGuard.MAX_ROUNDS):
            guard.record_call("s1", "fs_read_file", {"path": f"/tmp/{i}"})

        guard.reset("s1")
        assert guard.record_call("s1", "fs_read_file", {"path": "/tmp"}) is None

    def test_isolation_per_session(self):
        guard = ToolLoopGuard()
        for i in range(ToolLoopGuard.MAX_ROUNDS):
            guard.record_call("s1", "tool", {"i": i})
        assert guard.record_call("s1", "tool", {"i": 99}) is not None
        assert guard.record_call("s2", "tool", {"i": 99}) is None


@pytest.mark.asyncio
async def test_tool_loop_guard_returns_partial_progress_summary():
    class _FakeLLM:
        def __init__(self) -> None:
            self.model_routing = SimpleNamespace(auto_escalation=True)
            self._call_count = 0

        async def chat_completion(self, *args, **kwargs):
            self._call_count += 1
            start = 0 if self._call_count == 1 else 20
            count = 20 if self._call_count == 1 else 5
            tool_calls = [
                {
                    "id": f"tc-{start + idx + 1}",
                    "function": {
                        "name": "write_note",
                        "arguments": f'{{"index": {start + idx + 1}}}',
                    },
                }
                for idx in range(count)
            ]
            return {"choices": [{"message": {"tool_calls": tool_calls}}]}

    class _FakeRuntime:
        def __init__(self) -> None:
            self.shadow_captures = []
            self.ctx = SimpleNamespace(
                config=SimpleNamespace(state_dir="/tmp"),
                plan_store=None,
                shadow_registry=SimpleNamespace(
                    capture_tool_loop_guard=self.shadow_captures.append,
                ),
            )
            self.executed: list[str] = []

        async def execute_tool(self, name, args, *, session_id=None, task_id=None):
            self.executed.append(name)
            return {"success": True, "output": f"{name} ok", "metadata": {}}

        async def _record_episode(self, *args, **kwargs):
            return None

    tools = ToolRegistry()
    tools.register(
        "write_note",
        "Write a note",
        lambda _name, _args: None,
        ActionRiskTier.WORKSPACE_WRITE,
    )
    runtime = _FakeRuntime()
    loop = ToolUseLoop(
        llm=_FakeLLM(),
        tools=tools,
        approval=SimpleNamespace(),
    )

    result = await loop.run(
        objective="Create a large writing scaffold",
        messages=[{"role": "user", "content": "Start writing files."}],
        ctx=ToolUseContext(runtime=runtime, session_id="session-1"),
    )

    assert result.guard_fired is True
    assert "exceeded 24" in (result.guard_reason or "")
    assert "[Tool loop halted]" not in result.final_output
    assert "partial progress" in result.final_output.lower()
    assert "write_note x24" in result.final_output
    assert "write_note x1" in result.final_output
    assert len(runtime.executed) == ToolLoopGuard.MAX_ROUNDS
    assert len(runtime.shadow_captures) == 1
    assert runtime.shadow_captures[0]["session_id"] == "session-1"
    assert runtime.shadow_captures[0]["task_id"] is None
    assert runtime.shadow_captures[0]["dominant_tool"] == "write_note"
    tool_messages = [message for message in result.messages if message.get("role") == "tool"]
    assert len(tool_messages) == ToolLoopGuard.MAX_ROUNDS


@pytest.mark.asyncio
async def test_tool_loop_research_context_gets_expanded_call_budget():
    class _FakeLLM:
        def __init__(self) -> None:
            self.model_routing = SimpleNamespace(auto_escalation=True)
            self._call_count = 0

        async def chat_completion(self, *args, **kwargs):
            self._call_count += 1
            if self._call_count > 1:
                return {"choices": [{"message": {"content": "research complete"}}]}
            tool_calls = [
                {
                    "id": f"search-{idx}",
                    "function": {
                        "name": "web_search",
                        "arguments": f'{{"query": "rare name etymology {idx}"}}',
                    },
                }
                for idx in range(ToolLoopGuard.MAX_ROUNDS + 6)
            ]
            return {"choices": [{"message": {"tool_calls": tool_calls}}]}

    class _FakeRuntime:
        def __init__(self) -> None:
            self.ctx = SimpleNamespace(config=SimpleNamespace(state_dir="/tmp"), plan_store=None)
            self.executed: list[dict] = []

        async def execute_tool(self, name, args, *, session_id=None, task_id=None):
            self.executed.append({"name": name, "args": args})
            return {
                "success": True,
                "output": f"evidence for {args['query']}",
                "metadata": {},
            }

        async def _record_episode(self, *args, **kwargs):
            return None

    tools = ToolRegistry()
    tools.register(
        "web_search",
        "Search the web",
        lambda _name, _args: None,
        ActionRiskTier.READONLY,
    )
    runtime = _FakeRuntime()
    loop = ToolUseLoop(
        llm=_FakeLLM(),
        tools=tools,
        approval=SimpleNamespace(),
    )

    result = await loop.run(
        objective="Research rare Basque and Welsh name etymologies for a manuscript.",
        messages=[
            {
                "role": "user",
                "content": "Research the names before using them.",
            }
        ],
        ctx=ToolUseContext(runtime=runtime, session_id="research-session"),
    )

    assert result.guard_fired is False
    assert result.final_output == "research complete"
    assert len(runtime.executed) == ToolLoopGuard.MAX_ROUNDS + 6


def test_tool_loop_budget_expands_from_prior_research_context_on_continue():
    loop = ToolUseLoop(
        llm=SimpleNamespace(),
        tools=ToolRegistry(),
        approval=SimpleNamespace(),
    )
    budget = loop._select_tool_call_budget(
        objective="Continue.",
        messages=[
            {
                "role": "system",
                "content": (
                    "Recent context: the agent was doing source-backed name research, "
                    "cross-reference checks, and etymology verification when a tool "
                    "loop circuit breaker stopped the work."
                ),
            },
            {
                "role": "user",
                "content": "Keep going from where you left off.",
            },
        ],
        ctx=ToolUseContext(runtime=SimpleNamespace(), session_id="research-session"),
    )

    assert ToolLoopGuard.MAX_ROUNDS < budget <= ToolUseLoop.HARD_TOOL_CALL_BUDGET


def test_tool_loop_explicit_budget_is_honored_and_capped():
    loop = ToolUseLoop(
        llm=SimpleNamespace(),
        tools=ToolRegistry(),
        approval=SimpleNamespace(),
    )

    assert (
        loop._select_tool_call_budget(
            objective="Research deeply.",
            messages=[],
            ctx=ToolUseContext(
                runtime=SimpleNamespace(),
                session_id="research-session",
                tool_call_budget=40,
            ),
        )
        == 40
    )
    assert (
        loop._select_tool_call_budget(
            objective="Research deeply.",
            messages=[],
            ctx=ToolUseContext(
                runtime=SimpleNamespace(),
                session_id="research-session",
                tool_call_budget=999,
            ),
        )
        == ToolUseLoop.HARD_TOOL_CALL_BUDGET
    )


@pytest.mark.asyncio
async def test_tool_loop_injects_shadow_registry_guidance_into_system_prompt():
    captured_messages = []

    class _FakeLLM:
        def __init__(self) -> None:
            self.model_routing = SimpleNamespace(auto_escalation=True)

        async def chat_completion(self, *args, **kwargs):
            captured_messages.extend(kwargs.get("messages") or [])
            return {"choices": [{"message": {"content": "done"}}]}

    class _FakeRuntime:
        def __init__(self) -> None:
            self.ctx = SimpleNamespace(
                config=SimpleNamespace(state_dir="/tmp"),
                plan_store=None,
                shadow_registry=SimpleNamespace(
                    build_planning_context=lambda **_kwargs: {
                        "available": True,
                        "prompt_block": (
                            "Related blocked-intention clusters:\n"
                            "- 2x retry_blocked around retry:workspace/Chronicles/4246/chronicle.md\n"
                            "Safer alternatives:\n"
                            "- Prefer one narrow edit tied to the canonical artifact, then rerun verification."
                        ),
                    },
                ),
            )

    loop = ToolUseLoop(
        llm=_FakeLLM(),
        tools=ToolRegistry(),
        approval=SimpleNamespace(),
    )

    result = await loop.run(
        objective="Revise the chronicle draft without restarting the whole plan",
        messages=[{"role": "user", "content": "Continue revising the chronicle draft."}],
        ctx=ToolUseContext(runtime=_FakeRuntime(), session_id="session-1"),
    )

    assert result.final_output == "done"
    assert captured_messages
    system_message = captured_messages[0]
    assert system_message["role"] == "system"
    assert "Related blocked-intention clusters:" in system_message["content"]
    assert "Prefer one narrow edit tied to the canonical artifact" in system_message["content"]


@pytest.mark.asyncio
async def test_tool_loop_passes_task_canonical_artifact_to_shadow_registry(tmp_path):
    captured_artifacts = []
    task_store = TaskStore(tmp_path / "tasks.db")
    await task_store.connect()
    task = RepairTask(
        task_id=uuid4(),
        objective="Continue Chronicle 4246 from the existing manuscript.",
        meta={
            "resume_project": {
                "canonical_artifact_path": "workspace/Chronicles/4246/chronicle_4246.md",
            }
        },
    )
    await task_store.save(task)

    class _FakeLLM:
        def __init__(self) -> None:
            self.model_routing = SimpleNamespace(auto_escalation=True)

        async def chat_completion(self, *args, **kwargs):
            return {"choices": [{"message": {"content": "done"}}]}

    class _FakeRuntime:
        def __init__(self) -> None:
            self.ctx = SimpleNamespace(
                config=SimpleNamespace(state_dir=tmp_path / ".opencas"),
                tasks=task_store,
                plan_store=None,
                shadow_registry=SimpleNamespace(
                    build_planning_context=lambda **kwargs: captured_artifacts.append(kwargs.get("artifact")) or {
                        "available": True,
                        "prompt_block": "Safer alternatives:\n- Prefer one narrow edit.",
                    }
                ),
            )

    loop = ToolUseLoop(
        llm=_FakeLLM(),
        tools=ToolRegistry(),
        approval=SimpleNamespace(),
    )

    result = await loop.run(
        objective="Revise the chronicle draft without restarting the whole plan",
        messages=[{"role": "user", "content": "Continue revising the chronicle draft."}],
        ctx=ToolUseContext(
            runtime=_FakeRuntime(),
            session_id="session-1",
            task_id=str(task.task_id),
        ),
    )

    assert result.final_output == "done"
    assert captured_artifacts == ["workspace/Chronicles/4246/chronicle_4246.md"]
    await task_store.close()


@pytest.mark.asyncio
async def test_tool_loop_resolves_matching_active_task_artifact_for_shadow_registry(tmp_path):
    captured_artifacts = []
    task_store = TaskStore(tmp_path / "tasks.db")
    await task_store.connect()
    matching = RepairTask(
        task_id=uuid4(),
        objective="Continue Chronicle 4246 from the existing manuscript.",
        meta={
            "resume_project": {
                "canonical_artifact_path": "workspace/Chronicles/4246/chronicle_4246.md",
            }
        },
    )
    unrelated = RepairTask(
        task_id=uuid4(),
        objective="Refactor the dashboard auth middleware",
        meta={
            "resume_project": {
                "canonical_artifact_path": "workspace/Infra/auth_router.py",
            }
        },
    )
    await task_store.save(unrelated)
    await task_store.save(matching)

    class _FakeLLM:
        def __init__(self) -> None:
            self.model_routing = SimpleNamespace(auto_escalation=True)

        async def chat_completion(self, *args, **kwargs):
            return {"choices": [{"message": {"content": "done"}}]}

    class _FakeRuntime:
        def __init__(self) -> None:
            self.ctx = SimpleNamespace(
                config=SimpleNamespace(state_dir=tmp_path / ".opencas"),
                tasks=task_store,
                plan_store=None,
                shadow_registry=SimpleNamespace(
                    build_planning_context=lambda **kwargs: captured_artifacts.append(kwargs.get("artifact")) or {
                        "available": True,
                        "prompt_block": "Safer alternatives:\n- Prefer one narrow edit.",
                    }
                ),
            )

    loop = ToolUseLoop(
        llm=_FakeLLM(),
        tools=ToolRegistry(),
        approval=SimpleNamespace(),
    )

    result = await loop.run(
        objective="Continue Chronicle 4246 from the existing manuscript with a narrow revision.",
        messages=[{"role": "user", "content": "Continue revising the chronicle draft."}],
        ctx=ToolUseContext(runtime=_FakeRuntime(), session_id="session-1"),
    )

    assert result.final_output == "done"
    assert captured_artifacts == ["workspace/Chronicles/4246/chronicle_4246.md"]
    await task_store.close()


@pytest.mark.asyncio
async def test_tool_loop_passes_active_plan_artifact_to_shadow_registry(tmp_path):
    captured_artifacts = []
    task_store = TaskStore(tmp_path / "tasks.db")
    await task_store.connect()
    plan_store = PlanStore(tmp_path / "plans.db")
    await plan_store.connect()

    task = RepairTask(
        task_id=uuid4(),
        objective="Continue Chronicle 4246 from the existing manuscript.",
        meta={
            "resume_project": {
                "canonical_artifact_path": "workspace/Chronicles/4246/chronicle_4246.md",
            }
        },
    )
    await task_store.save(task)
    await plan_store.create_plan(
        "plan-chronicle",
        content="Narrow revision plan for Chronicle 4246.",
        task_id=str(task.task_id),
    )
    await plan_store.set_status("plan-chronicle", "active")

    class _FakeLLM:
        def __init__(self) -> None:
            self.model_routing = SimpleNamespace(auto_escalation=True)

        async def chat_completion(self, *args, **kwargs):
            return {"choices": [{"message": {"content": "done"}}]}

    class _FakeRuntime:
        def __init__(self) -> None:
            self.ctx = SimpleNamespace(
                config=SimpleNamespace(state_dir=tmp_path / ".opencas"),
                tasks=task_store,
                plan_store=plan_store,
                shadow_registry=SimpleNamespace(
                    build_planning_context=lambda **kwargs: captured_artifacts.append(kwargs.get("artifact")) or {
                        "available": True,
                        "prompt_block": "Safer alternatives:\n- Prefer one narrow edit.",
                    }
                ),
            )

    loop = ToolUseLoop(
        llm=_FakeLLM(),
        tools=ToolRegistry(),
        approval=SimpleNamespace(),
    )

    result = await loop.run(
        objective="Continue Chronicle 4246 from the active plan with one narrow revision.",
        messages=[{"role": "user", "content": "Continue revising the chronicle draft."}],
        ctx=ToolUseContext(
            runtime=_FakeRuntime(),
            session_id="session-1",
            active_plan_id="plan-chronicle",
        ),
    )

    assert result.final_output == "done"
    assert captured_artifacts == ["workspace/Chronicles/4246/chronicle_4246.md"]
    await plan_store.close()
    await task_store.close()
