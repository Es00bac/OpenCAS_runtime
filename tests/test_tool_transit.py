from __future__ import annotations

from types import SimpleNamespace

import pytest

from opencas.autonomy.models import ActionRiskTier
from opencas.somatic.models import SomaticState
from opencas.tools import ToolRegistry, ToolUseContext, ToolUseLoop


@pytest.mark.asyncio
async def test_tool_loop_returns_call_transits_and_chain_summary() -> None:
    class _FakeLLM:
        def __init__(self) -> None:
            self.model_routing = SimpleNamespace(auto_escalation=True)
            self.calls = 0

        async def chat_completion(self, *args, **kwargs):
            self.calls += 1
            if self.calls == 1:
                return {
                    "choices": [
                        {
                            "message": {
                                "content": "I need live workflow evidence before answering.",
                                "tool_calls": [
                                    {
                                        "id": "call-status",
                                        "function": {
                                            "name": "workflow_status",
                                            "arguments": "{}",
                                        },
                                    },
                                    {
                                        "id": "call-schedules",
                                        "function": {
                                            "name": "workflow_list_schedules",
                                            "arguments": "{}",
                                        },
                                    },
                                ],
                            }
                        }
                    ]
                }
            return {"choices": [{"message": {"content": "done"}}]}

    class _FakeRuntime:
        def __init__(self) -> None:
            self.ctx = SimpleNamespace(
                config=SimpleNamespace(state_dir="/tmp"),
                plan_store=None,
                shadow_registry=None,
                somatic=SimpleNamespace(state=SomaticState(tension=0.0, fatigue=0.0, certainty=0.55)),
            )

        async def execute_tool(self, name, args, *, session_id=None, task_id=None):
            if name == "workflow_list_schedules":
                return {"success": False, "output": "blocked", "metadata": {}}
            return {"success": True, "output": "active schedule found", "metadata": {}}

        async def _record_episode(self, *args, **kwargs):
            return None

    tools = ToolRegistry()
    tools.register(
        "workflow_status",
        "Inspect workflow status.",
        lambda _name, _args: None,
        ActionRiskTier.READONLY,
    )
    tools.register(
        "workflow_list_schedules",
        "List schedules.",
        lambda _name, _args: None,
        ActionRiskTier.READONLY,
    )

    result = await ToolUseLoop(
        llm=_FakeLLM(),
        tools=tools,
        approval=SimpleNamespace(),
    ).run(
        objective="Check whether the health schedule exists.",
        messages=[{"role": "user", "content": "Check whether the health schedule exists."}],
        ctx=ToolUseContext(runtime=_FakeRuntime(), session_id="transit-session"),
    )

    assert [item.call_id for item in result.tool_call_transits] == ["call-status", "call-schedules"]
    assert {item.trust_context for item in result.tool_call_transits} == {"operator_requested"}
    assert result.tool_call_transits[0].entry_intent == "I need live workflow evidence before answering."
    assert result.tool_call_transits[1].result_shape["tags"] == ["failed", "nonempty", "blocked"]
    assert result.tool_chain_summary is not None
    assert result.tool_chain_summary.call_ids == ["call-status", "call-schedules"]
    assert result.tool_chain_summary.undercoupled_somatic is True
