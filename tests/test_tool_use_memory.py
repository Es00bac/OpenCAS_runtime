from __future__ import annotations

from types import SimpleNamespace

import pytest

from opencas.autonomy.models import ActionRiskTier
from opencas.cognition import CognitiveStateStore, LearnedSkill
from opencas.tools import ToolRegistry, ToolUseContext, ToolUseLoop
from opencas.tools.tool_use_memory import ToolUseMemoryStore


def test_tool_use_memory_learns_schedule_submit_baa_rule(tmp_path) -> None:
    store = ToolUseMemoryStore(tmp_path)

    store.record_result(
        objective="Return to the Writing Project manuscript and continue unfinished writing.",
        tool_name="workflow_create_schedule",
        args={"action": "reminder_only"},
        result={
            "success": False,
            "output": (
                "Unfinished writing/project return schedules must use "
                "action=submit_baa, not reminder_only."
            ),
            "metadata": {},
        },
    )

    context = store.build_context(
        objective="Schedule a return to continue Writing Project manuscript revision.",
        available_tool_names=["workflow_create_schedule"],
    )

    assert "Tool-use memory hints" in context
    assert "workflow_create_schedule" in context
    assert "submit_baa" in context
    assert "reminder_only" in context


def test_tool_use_memory_keeps_context_compact_and_relevant(tmp_path) -> None:
    store = ToolUseMemoryStore(tmp_path)
    for index in range(8):
        store.record_lesson(
            objective=f"Task family {index}",
            tool_name=f"tool_{index}",
            outcome="success",
            summary=f"Use tool_{index} for task family {index}.",
        )
    store.record_lesson(
        objective="Writing Project manuscript scheduling",
        tool_name="workflow_create_schedule",
        outcome="failure",
        summary=(
            "workflow_create_schedule: unfinished writing/project return schedules "
            "need action=submit_baa instead of reminder_only."
        ),
    )

    context = store.build_context(
        objective="Writing Project manuscript return schedule",
        available_tool_names=["workflow_create_schedule"],
        limit=3,
    )

    assert "workflow_create_schedule" in context
    assert "tool_7" not in context
    assert len([line for line in context.splitlines() if line.startswith("- ")]) <= 3


@pytest.mark.asyncio
async def test_tool_loop_injects_relevant_tool_use_memory(tmp_path) -> None:
    class _CapturingLLM:
        def __init__(self) -> None:
            self.model_routing = SimpleNamespace(auto_escalation=True)
            self.messages = None

        async def chat_completion(self, *, messages, **kwargs):
            self.messages = messages
            return {"choices": [{"message": {"content": "done"}}]}

    class _FakeRuntime:
        def __init__(self) -> None:
            self.ctx = SimpleNamespace(
                config=SimpleNamespace(state_dir=tmp_path),
                plan_store=None,
                tool_use_memory=ToolUseMemoryStore(tmp_path),
            )

        async def execute_tool(self, name, args, *, session_id=None, task_id=None):
            return {"success": True, "output": f"{name} ok", "metadata": {}}

        async def _record_episode(self, *args, **kwargs):
            return None

    tools = ToolRegistry()
    tools.register(
        "workflow_create_schedule",
        "Create a schedule",
        lambda _name, _args: None,
        ActionRiskTier.WORKSPACE_WRITE,
    )
    tools.register(
        "fs_write_file",
        "Write a file",
        lambda _name, _args: None,
        ActionRiskTier.WORKSPACE_WRITE,
    )
    runtime = _FakeRuntime()
    runtime.ctx.tool_use_memory.record_lesson(
        objective="Writing Project manuscript scheduling",
        tool_name="workflow_create_schedule",
        outcome="failure",
        summary=(
            "workflow_create_schedule: unfinished writing/project return schedules "
            "need action=submit_baa instead of reminder_only."
        ),
    )
    runtime.ctx.tool_use_memory.record_lesson(
        objective="Unrelated math task",
        tool_name="calculate",
        outcome="success",
        summary="calculate: use for arithmetic.",
    )
    llm = _CapturingLLM()

    await ToolUseLoop(
        llm=llm,
        tools=tools,
        approval=SimpleNamespace(),
    ).run(
        objective="Schedule a return to continue Writing Project manuscript writing.",
        messages=[{"role": "user", "content": "Keep working later."}],
        ctx=ToolUseContext(runtime=runtime, session_id="memory-session"),
    )

    system_content = llm.messages[0]["content"]
    assert "Tool-use memory hints" in system_content
    assert "submit_baa" in system_content
    assert "calculate" not in system_content


def test_required_tool_names_include_skill_create_for_teaching_requests() -> None:
    required = ToolUseLoop._required_tool_names_for_objective(
        "Teach Bulma a reusable skill creation procedure for this workflow."
    )

    assert "cognitive_skill_library_search" in required
    assert "cognitive_skill_create" in required


@pytest.mark.asyncio
async def test_tool_loop_includes_tools_from_cognitive_learned_skills(tmp_path) -> None:
    class _CapturingLLM:
        def __init__(self) -> None:
            self.model_routing = SimpleNamespace(auto_escalation=True)
            self.tools = None

        async def chat_completion(self, *, tools=None, **kwargs):
            self.tools = tools or []
            return {"choices": [{"message": {"content": "done"}}]}

    cognitive = CognitiveStateStore(tmp_path / "cognitive.db")
    await cognitive.connect()
    try:
        await cognitive.upsert_learned_skill(
            LearnedSkill(
                name="weather station capture",
                description="Use special_weather_tool to capture weather station evidence.",
                tool_sequence=["special_weather_tool"],
                preconditions=["weather station"],
                success_count=4,
                failure_count=0,
                risk_tier="readonly",
            )
        )
        tools = ToolRegistry()
        tools.register(
            "fs_read_file",
            "Read a file",
            lambda _name, _args: None,
            ActionRiskTier.READONLY,
        )
        tools.register(
            "special_weather_tool",
            "Capture weather station evidence",
            lambda _name, _args: None,
            ActionRiskTier.READONLY,
        )
        runtime = SimpleNamespace(
            cognitive_state_store=cognitive,
            ctx=SimpleNamespace(config=SimpleNamespace(state_dir=tmp_path), plan_store=None),
        )
        llm = _CapturingLLM()

        await ToolUseLoop(llm=llm, tools=tools, approval=SimpleNamespace()).run(
            objective="Check the weather station capture.",
            messages=[{"role": "user", "content": "Check the weather station."}],
            ctx=ToolUseContext(runtime=runtime, session_id="skill-session"),
        )

        tool_names = {tool["function"]["name"] for tool in llm.tools}
        assert "special_weather_tool" in tool_names
    finally:
        await cognitive.close()


@pytest.mark.asyncio
async def test_tool_loop_requires_evidence_pass_for_active_uncertainty(tmp_path) -> None:
    class _EvidenceLLM:
        def __init__(self) -> None:
            self.model_routing = SimpleNamespace(auto_escalation=True)
            self.calls = 0

        async def chat_completion(self, *, messages, tools=None, **kwargs):
            self.calls += 1
            if self.calls == 1:
                return {"choices": [{"message": {"content": "I think it is fine."}}]}
            if self.calls == 2:
                return {
                    "choices": [
                        {
                            "message": {
                                "tool_calls": [
                                    {
                                        "id": "call-memory",
                                        "function": {
                                            "name": "search_memories",
                                            "arguments": '{"query": "schedule evidence"}',
                                        },
                                    }
                                ]
                            }
                        }
                    ]
                }
            return {"choices": [{"message": {"content": "done with evidence"}}]}

    class _FakeRuntime:
        def __init__(self, cognitive) -> None:
            self.executed: list[str] = []
            self.cognitive_state_store = cognitive
            self.ctx = SimpleNamespace(
                config=SimpleNamespace(state_dir=tmp_path),
                plan_store=None,
                cognitive_state_store=cognitive,
            )

        async def execute_tool(self, name, args, *, session_id=None, task_id=None):
            self.executed.append(name)
            return {"success": True, "output": "schedule evidence found", "metadata": {}}

        async def _record_episode(self, *args, **kwargs):
            return None

    cognitive = CognitiveStateStore(tmp_path / "cognitive.db")
    await cognitive.connect()
    try:
        await cognitive.upsert_working_memory(
            "uncertainty_seeking",
            "Uncertainty pressure requires checking evidence before assertion.",
            priority=0.84,
            source="test",
        )
        tools = ToolRegistry()
        tools.register(
            "search_memories",
            "Search autobiographical memory.",
            lambda _name, _args: None,
            ActionRiskTier.READONLY,
        )
        runtime = _FakeRuntime(cognitive)
        llm = _EvidenceLLM()

        result = await ToolUseLoop(llm=llm, tools=tools, approval=SimpleNamespace()).run(
            objective="Tell me whether the schedule evidence exists.",
            messages=[{"role": "user", "content": "Is the schedule evidence there?"}],
            ctx=ToolUseContext(runtime=runtime, session_id="uncertainty-session"),
        )

        assert result.final_output == "done with evidence"
        assert runtime.executed == ["search_memories"]
        assert llm.calls == 3
    finally:
        await cognitive.close()
