from __future__ import annotations

from types import SimpleNamespace

import pytest

from opencas.autonomy.models import ActionRiskTier
from opencas.tools import ToolRegistry, ToolUseContext, ToolUseLoop
from opencas.tools.tool_use_memory import ToolUseMemoryStore


def test_tool_use_memory_learns_schedule_submit_baa_rule(tmp_path) -> None:
    store = ToolUseMemoryStore(tmp_path)

    store.record_result(
        objective="Return to the Chronicle manuscript and continue unfinished writing.",
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
        objective="Schedule a return to continue Chronicle manuscript revision.",
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
        objective="Chronicle manuscript scheduling",
        tool_name="workflow_create_schedule",
        outcome="failure",
        summary=(
            "workflow_create_schedule: unfinished writing/project return schedules "
            "need action=submit_baa instead of reminder_only."
        ),
    )

    context = store.build_context(
        objective="Chronicle manuscript return schedule",
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
        objective="Chronicle manuscript scheduling",
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
        objective="Schedule a return to continue Chronicle manuscript writing.",
        messages=[{"role": "user", "content": "Keep working later."}],
        ctx=ToolUseContext(runtime=runtime, session_id="memory-session"),
    )

    system_content = llm.messages[0]["content"]
    assert "Tool-use memory hints" in system_content
    assert "submit_baa" in system_content
    assert "calculate" not in system_content
