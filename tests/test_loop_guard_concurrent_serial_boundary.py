"""Regression test for concurrent-to-serial batch guard boundary gap.

Bug: When a concurrent batch of readonly tool calls executes, the guard
counts each call. But there was no guard check between the concurrent batch
and the serial batch. This meant if the concurrent batch pushed the session
over max_rounds, serial calls would still execute past the limit.
"""

from types import SimpleNamespace

import pytest

from opencas.autonomy.models import ActionRiskTier
from opencas.tools import ToolRegistry, ToolUseContext, ToolUseLoop


@pytest.mark.asyncio
async def test_guard_fires_between_concurrent_and_serial_batches():
    """If concurrent readonly calls exhaust the budget, serial calls must not run."""

    class _FakeLLM:
        def __init__(self) -> None:
            self.model_routing = SimpleNamespace(auto_escalation=True)
            self._call_count = 0

        async def chat_completion(self, *args, **kwargs):
            self._call_count += 1
            if self._call_count == 1:
                # 25 readonly concurrent calls (exceeds budget of 24).
                # Use calculate (non-observational) with distinct args so
                # progress guard does not fire before loop guard.
                tool_calls = [
                    {
                        "id": f"calc-{i}",
                        "function": {
                            "name": "calculate",
                            "arguments": f'{{"expression": "{i} + 1"}}',
                        },
                    }
                    for i in range(25)
                ]
                return {"choices": [{"message": {"tool_calls": tool_calls}}]}
            return {"choices": [{"message": {"tool_calls": []}}]}

    class _FakeRuntime:
        def __init__(self) -> None:
            self.executed = []
            self.ctx = SimpleNamespace(
                config=SimpleNamespace(state_dir="/tmp"),
                plan_store=None,
                shadow_registry=SimpleNamespace(
                    capture_tool_loop_guard=lambda _: None,
                ),
            )

        async def execute_tool(self, name, args, *, session_id=None, task_id=None):
            self.executed.append(name)
            return {"success": True, "output": f"result={args.get('expression', '')}", "metadata": {}}

        async def _record_episode(self, *args, **kwargs):
            return None

    tools = ToolRegistry()
    tools.register(
        "calculate",
        "Calculate expression",
        lambda _name, _args: None,
        ActionRiskTier.READONLY,
    )
    tools.register(
        "fs_write_file",
        "Write file",
        lambda _name, _args: None,
        ActionRiskTier.WORKSPACE_WRITE,
    )

    runtime = _FakeRuntime()
    loop = ToolUseLoop(llm=_FakeLLM(), tools=tools, approval=SimpleNamespace())

    result = await loop.run(
        objective="Calculate many things then write a summary.",
        messages=[{"role": "user", "content": "Calculate and write summary."}],
        ctx=ToolUseContext(
            runtime=runtime,
            session_id="boundary-session",
            tool_call_budget=24,
        ),
    )

    assert result.guard_fired is True
    assert "exceeded 24" in (result.guard_reason or "")
    assert "fs_write_file" not in runtime.executed


@pytest.mark.asyncio
async def test_serial_batch_guarded_after_concurrent_batch_at_budget_limit():
    """When concurrent batch exactly hits the budget, serial calls must be blocked."""

    class _FakeLLM:
        def __init__(self) -> None:
            self.model_routing = SimpleNamespace(auto_escalation=True)
            self._call_count = 0

        async def chat_completion(self, *args, **kwargs):
            self._call_count += 1
            if self._call_count == 1:
                # 24 readonly concurrent calls (exactly at budget limit)
                # PLUS 1 serial write call that should be blocked
                tool_calls = [
                    {
                        "id": f"calc-{i}",
                        "function": {
                            "name": "calculate",
                            "arguments": f'{{"expression": "{i} + 1"}}',
                        },
                    }
                    for i in range(24)
                ] + [
                    {
                        "id": "write-1",
                        "function": {
                            "name": "fs_write_file",
                            "arguments": '{"file_path": "/tmp/summary.txt", "content": "done"}',
                        },
                    }
                ]
                return {"choices": [{"message": {"tool_calls": tool_calls}}]}
            return {"choices": [{"message": {"tool_calls": []}}]}

    class _FakeRuntime:
        def __init__(self) -> None:
            self.executed = []
            self.ctx = SimpleNamespace(
                config=SimpleNamespace(state_dir="/tmp"),
                plan_store=None,
                shadow_registry=SimpleNamespace(
                    capture_tool_loop_guard=lambda _: None,
                ),
            )

        async def execute_tool(self, name, args, *, session_id=None, task_id=None):
            self.executed.append(name)
            return {"success": True, "output": f"result={args.get('expression', '')}", "metadata": {}}

        async def _record_episode(self, *args, **kwargs):
            return None

    tools = ToolRegistry()
    tools.register(
        "calculate",
        "Calculate expression",
        lambda _name, _args: None,
        ActionRiskTier.READONLY,
    )
    tools.register(
        "fs_write_file",
        "Write file",
        lambda _name, _args: None,
        ActionRiskTier.WORKSPACE_WRITE,
    )

    runtime = _FakeRuntime()
    loop = ToolUseLoop(llm=_FakeLLM(), tools=tools, approval=SimpleNamespace())

    result = await loop.run(
        objective="Calculate things then write summary.",
        messages=[{"role": "user", "content": "Calculate and write summary."}],
        ctx=ToolUseContext(
            runtime=runtime,
            session_id="limit-session",
            tool_call_budget=24,
        ),
    )

    # The concurrent batch executes 24 calls (at the budget limit).
    # The serial write call MUST be blocked by the guard.
    assert result.guard_fired is True
    assert "exceeded 24" in (result.guard_reason or "")
    assert "fs_write_file" not in runtime.executed
    assert runtime.executed.count("calculate") == 24
