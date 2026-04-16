"""Tests for ToolLoopGuard circuit breaker."""

from types import SimpleNamespace

import pytest

from opencas.autonomy.models import ActionRiskTier
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
            self.ctx = SimpleNamespace(config=SimpleNamespace(state_dir="/tmp"), plan_store=None)
            self.executed: list[str] = []

        async def execute_tool(self, name, args):
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
    tool_messages = [message for message in result.messages if message.get("role") == "tool"]
    assert len(tool_messages) == ToolLoopGuard.MAX_ROUNDS
