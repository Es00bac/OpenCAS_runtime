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


def test_tool_result_messages_are_prompt_bounded():
    loop = ToolUseLoop(llm=SimpleNamespace(), tools=ToolRegistry(), approval=SimpleNamespace())
    message = loop._build_tool_result_message(
        {"id": "call-tasks", "name": "workflow_list_tasks"},
        {"success": True, "output": "A" * 120_000, "metadata": {}},
    )

    assert message["role"] == "tool"
    assert message["tool_call_id"] == "call-tasks"
    assert len(message["content"]) <= ToolUseLoop.MAX_TOOL_RESULT_PROMPT_CHARS
    assert "Tool output truncated" in message["content"]


def test_tool_result_prompt_bound_expands_for_large_context_model():
    llm = SimpleNamespace(
        model_context_metadata=lambda: {
            "context_window": 262_144,
            "prompt_context_budget": 224_134,
            "budget_source": "model_metadata",
        }
    )
    loop = ToolUseLoop(llm=llm, tools=ToolRegistry(), approval=SimpleNamespace())
    message = loop._build_tool_result_message(
        {"id": "call-manuscript", "name": "fs_read_file"},
        {"success": True, "output": "A" * 120_000, "metadata": {}},
    )

    assert len(message["content"]) == 120_000
    assert "Tool output truncated" not in message["content"]


def test_tool_result_prompt_bound_still_caps_very_large_outputs():
    llm = SimpleNamespace(
        model_context_metadata=lambda: {
            "context_window": 262_144,
            "prompt_context_budget": 224_134,
            "budget_source": "model_metadata",
        }
    )
    loop = ToolUseLoop(llm=llm, tools=ToolRegistry(), approval=SimpleNamespace())
    message = loop._build_tool_result_message(
        {"id": "call-manuscript", "name": "fs_read_file"},
        {"success": True, "output": "A" * 400_000, "metadata": {}},
    )

    assert 120_000 < len(message["content"]) <= loop._tool_result_prompt_char_limit()
    assert "Tool output truncated" in message["content"]


def test_truncated_file_listing_tool_result_warns_absence_is_not_evidence():
    loop = ToolUseLoop(llm=SimpleNamespace(), tools=ToolRegistry(), approval=SimpleNamespace())
    message = loop._build_tool_result_message(
        {"id": "call-files", "name": "fs_list_dir"},
        {
            "success": True,
            "output": '{"ok": true, "entries": []}',
            "metadata": {
                "truncated": True,
                "total_count": 12,
                "returned_count": 5,
                "next_offset": 5,
            },
        },
    )

    assert "Result set is incomplete" in message["content"]
    assert "Do not infer absence" in message["content"]
    assert "next_offset=5" in message["content"]


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


@pytest.mark.asyncio
async def test_tool_loop_requires_external_web_evidence_before_current_forum_report():
    class _FakeLLM:
        def __init__(self) -> None:
            self.model_routing = SimpleNamespace(auto_escalation=True)
            self.calls: list[list[dict]] = []

        async def chat_completion(self, messages, *args, **kwargs):
            del args, kwargs
            self.calls.append([dict(message) for message in messages])
            if len(self.calls) == 1:
                return {
                    "choices": [
                        {
                            "message": {
                                "content": "I cannot report current 5ch threads because no browser/search/fetch evidence was supplied."
                            }
                        }
                    ]
                }
            if len(self.calls) == 2:
                return {
                    "choices": [
                        {
                            "message": {
                                "tool_calls": [
                                    {
                                        "id": "search-1",
                                        "function": {
                                            "name": "web_search",
                                            "arguments": '{"query": "site:5ch.net current threads Japan discussion"}',
                                        },
                                    }
                                ]
                            }
                        }
                    ]
                }
            return {"choices": [{"message": {"content": "grounded report from search evidence"}}]}

    class _FakeRuntime:
        def __init__(self) -> None:
            self.ctx = SimpleNamespace(config=SimpleNamespace(state_dir="/tmp"), plan_store=None)
            self.executed: list[dict] = []

        async def execute_tool(self, name, args, *, session_id=None, task_id=None):
            del session_id, task_id
            self.executed.append({"name": name, "args": args})
            return {
                "success": True,
                "output": "5ch search result evidence",
                "metadata": {"query": args.get("query")},
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
    llm = _FakeLLM()
    loop = ToolUseLoop(
        llm=llm,
        tools=tools,
        approval=SimpleNamespace(),
    )

    result = await loop.run(
        objective=(
            "Go to 2channel/5ch, find five interesting current threads, "
            "and report what people are talking about."
        ),
        messages=[
            {
                "role": "user",
                "content": "Find five interesting current 5ch threads.",
            }
        ],
        ctx=ToolUseContext(runtime=_FakeRuntime(), session_id="forum-research-session"),
    )

    assert result.final_output == "grounded report from search evidence"
    assert [call["name"] for call in result.tool_calls] == ["web_search"]
    assert "current/external web research evidence" in llm.calls[1][-1]["content"]


@pytest.mark.asyncio
async def test_tool_loop_auto_routes_external_web_evidence_when_model_ignores_retry():
    class _FakeLLM:
        def __init__(self) -> None:
            self.model_routing = SimpleNamespace(auto_escalation=True)
            self.calls: list[list[dict]] = []

        async def chat_completion(self, messages, *args, **kwargs):
            del args, kwargs
            self.calls.append([dict(message) for message in messages])
            if len(self.calls) < 3:
                return {
                    "choices": [
                        {
                            "message": {
                                "content": "I cannot report current forum threads because no web evidence was supplied."
                            }
                        }
                    ]
                }
            assert any(message.get("role") == "tool" for message in messages)
            return {"choices": [{"message": {"content": "grounded report after automatic search"}}]}

    class _FakeRuntime:
        def __init__(self) -> None:
            self.ctx = SimpleNamespace(config=SimpleNamespace(state_dir="/tmp"), plan_store=None)
            self.executed: list[dict] = []

        async def execute_tool(self, name, args, *, session_id=None, task_id=None):
            del session_id, task_id
            self.executed.append({"name": name, "args": args})
            return {
                "success": True,
                "output": "automatic web evidence",
                "metadata": {"query": args.get("query")},
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
        objective="Find one current public 5ch thread before answering.",
        messages=[{"role": "user", "content": "Find one current public 5ch thread."}],
        ctx=ToolUseContext(runtime=runtime, session_id="auto-evidence-session"),
    )

    assert result.final_output == "grounded report after automatic search"
    assert [call["name"] for call in result.tool_calls] == ["web_search"]
    assert runtime.executed == [
        {
            "name": "web_search",
            "args": {"query": "5ch 5ちゃん current thread board listing Japan discussion"},
        }
    ]


@pytest.mark.asyncio
async def test_tool_loop_respects_max_complexity_cap_for_prefetched_recall():
    complexities = []

    class _FakeLLM:
        def __init__(self) -> None:
            self.model_routing = SimpleNamespace(auto_escalation=True)
            self._call_count = 0

        async def chat_completion(self, *args, **kwargs):
            self._call_count += 1
            complexities.append(kwargs.get("complexity"))
            if self._call_count <= 2:
                return {
                    "choices": [
                        {
                            "message": {
                                "tool_calls": [
                                    {
                                        "id": f"recall-{self._call_count}",
                                        "function": {
                                            "name": "search_memories",
                                            "arguments": '{"query": "atlas repair"}',
                                        },
                                    }
                                ]
                            }
                        }
                    ]
                }
            return {"choices": [{"message": {"content": "grounded recall"}}]}

    class _FakeRuntime:
        def __init__(self) -> None:
            self.ctx = SimpleNamespace(config=SimpleNamespace(state_dir="/tmp"), plan_store=None)

        async def execute_tool(self, name, args, *, session_id=None, task_id=None):
            return {"success": True, "output": "atlas repair evidence", "metadata": {}}

        async def _record_episode(self, *args, **kwargs):
            return None

    tools = ToolRegistry()
    tools.register(
        "search_memories",
        "Search memories",
        lambda _name, _args: None,
        ActionRiskTier.READONLY,
    )
    loop = ToolUseLoop(
        llm=_FakeLLM(),
        tools=tools,
        approval=SimpleNamespace(),
    )

    result = await loop.run(
        objective="Recall the atlas repair thread from available memory evidence.",
        messages=[{"role": "user", "content": "What do you recall about atlas repair?"}],
        ctx=ToolUseContext(
            runtime=_FakeRuntime(),
            session_id="recall-session",
            initial_complexity="light",
            max_complexity="light",
        ),
    )

    assert result.final_output == "grounded recall"
    assert complexities == ["light", "light", "light"]


@pytest.mark.asyncio
async def test_tool_loop_renders_turn_scratchpad_before_next_llm_iteration():
    captured_rounds = []

    class _FakeLLM:
        def __init__(self) -> None:
            self.model_routing = SimpleNamespace(auto_escalation=True)
            self._call_count = 0

        async def chat_completion(self, *args, **kwargs):
            self._call_count += 1
            captured_rounds.append([dict(message) for message in kwargs.get("messages") or []])
            if self._call_count == 1:
                return {
                    "choices": [
                        {
                            "message": {
                                "tool_calls": [
                                    {
                                        "id": "read-1",
                                        "function": {
                                            "name": "fs_read_file",
                                            "arguments": '{"file_path": "/tmp/same.txt"}',
                                        },
                                    }
                                ]
                            }
                        }
                    ]
                }
            return {"choices": [{"message": {"content": "done"}}]}

    class _FakeRuntime:
        def __init__(self) -> None:
            self.ctx = SimpleNamespace(config=SimpleNamespace(state_dir="/tmp"), plan_store=None)

        async def execute_tool(self, name, args, *, session_id=None, task_id=None):
            return {
                "success": True,
                "output": "alpha\nbeta\n",
                "metadata": {"path": args["file_path"]},
            }

        async def _record_episode(self, *args, **kwargs):
            return None

    tools = ToolRegistry()
    tools.register(
        "fs_read_file",
        "Read a file",
        lambda _name, _args: None,
        ActionRiskTier.READONLY,
    )
    loop = ToolUseLoop(
        llm=_FakeLLM(),
        tools=tools,
        approval=SimpleNamespace(),
    )

    result = await loop.run(
        objective="Read /tmp/same.txt and report what it says.",
        messages=[{"role": "user", "content": "Read /tmp/same.txt."}],
        ctx=ToolUseContext(runtime=_FakeRuntime(), session_id="session-1"),
    )

    assert result.final_output == "done"
    assert len(captured_rounds) == 2
    scratchpad_messages = [
        message
        for message in captured_rounds[1]
        if message.get("role") == "user"
        and "Recent tool results this turn" in str(message.get("content", ""))
    ]
    assert scratchpad_messages
    content = scratchpad_messages[-1]["content"]
    assert "fs_read_file" in content
    assert "/tmp/same.txt" in content
    assert "alpha" in content


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
                            "- 2x retry_blocked around retry:workspace/writing/4246/creative_writing.md\n"
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
        objective="Revise the creative_writing draft without restarting the whole plan",
        messages=[{"role": "user", "content": "Continue revising the creative_writing draft."}],
        ctx=ToolUseContext(runtime=_FakeRuntime(), session_id="session-1"),
    )

    assert result.final_output == "done"
    assert captured_messages
    system_message = captured_messages[0]
    assert system_message["role"] == "system"
    assert "Related blocked-intention clusters:" in system_message["content"]
    assert "Prefer one narrow edit tied to the canonical artifact" in system_message["content"]


@pytest.mark.asyncio
async def test_tool_loop_injects_runtime_capability_context_into_existing_system_prompt():
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
                shadow_registry=None,
            )
            self.tools = SimpleNamespace(
                list_tools=lambda: [
                    SimpleNamespace(
                        name="runtime_status",
                        risk_tier=SimpleNamespace(value="readonly"),
                    )
                ]
            )

    loop = ToolUseLoop(
        llm=_FakeLLM(),
        tools=ToolRegistry(),
        approval=SimpleNamespace(),
    )

    result = await loop.run(
        objective="Read this corrected capability report and verify it.",
        messages=[
            {"role": "system", "content": "Existing Bulma system prompt."},
            {"role": "user", "content": "Read this corrected capability report."},
        ],
        ctx=ToolUseContext(runtime=_FakeRuntime(), session_id="session-1"),
    )

    assert result.final_output == "done"
    assert captured_messages
    system_message = captured_messages[0]
    assert system_message["role"] == "system"
    assert "Existing Bulma system prompt." in system_message["content"]
    assert "Runtime capability evidence:" in system_message["content"]
    assert "runtime_status" in system_message["content"]


@pytest.mark.asyncio
async def test_tool_loop_passes_task_canonical_artifact_to_shadow_registry(tmp_path):
    captured_artifacts = []
    task_store = TaskStore(tmp_path / "tasks.db")
    await task_store.connect()
    task = RepairTask(
        task_id=uuid4(),
        objective="Continue writing project 4246 from the existing manuscript.",
        meta={
            "resume_project": {
                "canonical_artifact_path": "workspace/writing/4246/story_4246.md",
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
        objective="Revise the creative_writing draft without restarting the whole plan",
        messages=[{"role": "user", "content": "Continue revising the creative_writing draft."}],
        ctx=ToolUseContext(
            runtime=_FakeRuntime(),
            session_id="session-1",
            task_id=str(task.task_id),
        ),
    )

    assert result.final_output == "done"
    assert captured_artifacts == ["workspace/writing/4246/story_4246.md"]
    await task_store.close()


@pytest.mark.asyncio
async def test_tool_loop_resolves_matching_active_task_artifact_for_shadow_registry(tmp_path):
    captured_artifacts = []
    task_store = TaskStore(tmp_path / "tasks.db")
    await task_store.connect()
    matching = RepairTask(
        task_id=uuid4(),
        objective="Continue writing project 4246 from the existing manuscript.",
        meta={
            "resume_project": {
                "canonical_artifact_path": "workspace/writing/4246/story_4246.md",
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
        objective="Continue writing project 4246 from the existing manuscript with a narrow revision.",
        messages=[{"role": "user", "content": "Continue revising the creative_writing draft."}],
        ctx=ToolUseContext(runtime=_FakeRuntime(), session_id="session-1"),
    )

    assert result.final_output == "done"
    assert captured_artifacts == ["workspace/writing/4246/story_4246.md"]
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
        objective="Continue writing project 4246 from the existing manuscript.",
        meta={
            "resume_project": {
                "canonical_artifact_path": "workspace/writing/4246/story_4246.md",
            }
        },
    )
    await task_store.save(task)
    await plan_store.create_plan(
        "plan-creative_writing",
        content="Narrow revision plan for writing project 4246.",
        task_id=str(task.task_id),
    )
    await plan_store.set_status("plan-creative_writing", "active")

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
        objective="Continue writing project 4246 from the active plan with one narrow revision.",
        messages=[{"role": "user", "content": "Continue revising the creative_writing draft."}],
        ctx=ToolUseContext(
            runtime=_FakeRuntime(),
            session_id="session-1",
            active_plan_id="plan-creative_writing",
        ),
    )

    assert result.final_output == "done"
    assert captured_artifacts == ["workspace/writing/4246/story_4246.md"]
    await plan_store.close()
    await task_store.close()
