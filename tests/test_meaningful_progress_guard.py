"""Tests for the meaningful-progress contract in tool loops."""

from types import SimpleNamespace

import pytest

from opencas.autonomy.models import ActionRiskTier
from opencas.tools import ToolRegistry, ToolUseContext, ToolUseLoop
from opencas.tools.progress_guard import MeaningfulProgressGuard


def test_blocks_immediate_approval_dead_end() -> None:
    guard = MeaningfulProgressGuard()

    reason = guard.record_result(
        "fs_write_file",
        {"path": "workspace/note.md"},
        {
            "success": False,
            "output": "Tool execution blocked: approval required",
            "metadata": {},
        },
    )

    assert reason is not None
    assert "blocker" in reason.lower()


def test_web_fetch_forbidden_is_recoverable_research_friction() -> None:
    guard = MeaningfulProgressGuard()

    reason = guard.record_result(
        "web_fetch",
        {"url": "https://example.test/restricted"},
        {
            "success": False,
            "output": (
                "Client error '403 Forbidden' for url "
                "'https://example.test/restricted'"
            ),
            "metadata": {},
        },
    )

    assert reason is None
    assert guard.last_assessment is not None
    assert guard.last_assessment.signal == "recoverable_web_failure"
    assert guard.last_assessment.terminal is False


def test_blocks_consecutive_no_progress_failures() -> None:
    guard = MeaningfulProgressGuard(max_consecutive_no_progress=3)

    assert guard.record_result(
        "web_fetch",
        {"url": "https://a.test"},
        {"success": False, "output": "timeout", "metadata": {}},
    ) is None
    assert guard.record_result(
        "web_fetch",
        {"url": "https://b.test"},
        {"success": False, "output": "timeout", "metadata": {}},
    ) is None
    reason = guard.record_result(
        "web_fetch",
        {"url": "https://c.test"},
        {"success": False, "output": "timeout", "metadata": {}},
    )

    assert reason is not None
    assert "no meaningful progress" in reason.lower()


def test_artifact_metadata_resets_no_progress_counter() -> None:
    guard = MeaningfulProgressGuard(max_consecutive_no_progress=2)

    assert guard.record_result(
        "web_fetch",
        {"url": "https://a.test"},
        {"success": False, "output": "timeout", "metadata": {}},
    ) is None
    assert guard.record_result(
        "fs_write_file",
        {"path": "workspace/note.md"},
        {
            "success": True,
            "output": "ok",
            "metadata": {"artifact_path": "workspace/note.md"},
        },
    ) is None
    assert guard.record_result(
        "web_fetch",
        {"url": "https://b.test"},
        {"success": False, "output": "timeout", "metadata": {}},
    ) is None


def test_repeated_stale_evidence_is_not_meaningful_forever() -> None:
    guard = MeaningfulProgressGuard(repeated_evidence_limit=3)
    result = {"success": True, "output": "status: still processing", "metadata": {}}

    assert guard.record_result("workflow_status", {"id": "1"}, result) is None
    assert guard.record_result("workflow_status", {"id": "1"}, result) is None
    reason = guard.record_result("workflow_status", {"id": "1"}, result)

    assert reason is not None
    assert "repeated stale evidence" in reason.lower()


@pytest.mark.asyncio
async def test_tool_loop_stops_on_consecutive_non_meaningful_failures() -> None:
    class _FakeLLM:
        def __init__(self) -> None:
            self.model_routing = SimpleNamespace(auto_escalation=True)
            self._call_count = 0

        async def chat_completion(self, *args, **kwargs):
            self._call_count += 1
            return {
                "choices": [
                    {
                        "message": {
                            "tool_calls": [
                                {
                                    "id": f"tc-{self._call_count}",
                                    "function": {
                                        "name": "web_fetch",
                                        "arguments": f'{{"url": "https://example.test/{self._call_count}"}}',
                                    },
                                }
                            ]
                        }
                    }
                ]
            }

    class _FakeRuntime:
        def __init__(self) -> None:
            self.executed = []
            self.shadow_captures = []
            self.ctx = SimpleNamespace(
                config=SimpleNamespace(state_dir="/tmp"),
                plan_store=None,
                shadow_registry=SimpleNamespace(
                    capture_tool_loop_guard=self.shadow_captures.append,
                ),
            )

        async def execute_tool(self, name, args, *, session_id=None, task_id=None):
            self.executed.append(args["url"])
            return {"success": False, "output": "timeout", "metadata": {}}

        async def _record_episode(self, *args, **kwargs):
            return None

    tools = ToolRegistry()
    tools.register(
        "web_fetch",
        "Fetch URL",
        lambda _name, _args: None,
        ActionRiskTier.READONLY,
    )
    runtime = _FakeRuntime()
    loop = ToolUseLoop(llm=_FakeLLM(), tools=tools, approval=SimpleNamespace())

    result = await loop.run(
        objective="Fetch status from several places",
        messages=[{"role": "user", "content": "Fetch status from several places."}],
        ctx=ToolUseContext(runtime=runtime, session_id="progress-session"),
    )

    assert result.guard_fired is True
    assert result.guard_reason is not None
    assert "no meaningful progress" in result.guard_reason.lower()
    assert len(runtime.executed) == 4
    assert len(runtime.shadow_captures) == 1
    assert runtime.shadow_captures[0]["guard_reason"] == result.guard_reason


@pytest.mark.asyncio
async def test_tool_loop_allows_meaningful_artifact_progress_before_later_failure() -> None:
    class _FakeLLM:
        def __init__(self) -> None:
            self.model_routing = SimpleNamespace(auto_escalation=True)
            self._call_count = 0

        async def chat_completion(self, *args, **kwargs):
            self._call_count += 1
            if self._call_count <= 5:
                tool_name = "fs_write_file" if self._call_count == 2 else "web_fetch"
                arguments = (
                    '{"path": "workspace/note.md", "content": "done"}'
                    if tool_name == "fs_write_file"
                    else f'{{"url": "https://example.test/{self._call_count}"}}'
                )
                return {
                    "choices": [
                        {
                            "message": {
                                "tool_calls": [
                                    {
                                        "id": f"tc-{self._call_count}",
                                        "function": {
                                            "name": tool_name,
                                            "arguments": arguments,
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
            self.ctx = SimpleNamespace(
                config=SimpleNamespace(state_dir="/tmp"),
                plan_store=None,
                shadow_registry=None,
            )

        async def execute_tool(self, name, args, *, session_id=None, task_id=None):
            if name == "fs_write_file":
                return {
                    "success": True,
                    "output": "wrote workspace/note.md",
                    "metadata": {"artifact_path": "workspace/note.md"},
                }
            return {"success": False, "output": "timeout", "metadata": {}}

        async def _record_episode(self, *args, **kwargs):
            return None

    tools = ToolRegistry()
    tools.register(
        "web_fetch",
        "Fetch URL",
        lambda _name, _args: None,
        ActionRiskTier.READONLY,
    )
    tools.register(
        "fs_write_file",
        "Write file",
        lambda _name, _args: None,
        ActionRiskTier.WORKSPACE_WRITE,
    )

    result = await ToolUseLoop(
        llm=_FakeLLM(),
        tools=tools,
        approval=SimpleNamespace(),
    ).run(
        objective="Write one note after checking status",
        messages=[{"role": "user", "content": "Write one note after checking status."}],
        ctx=ToolUseContext(runtime=_FakeRuntime(), session_id="progress-session"),
    )

    assert result.guard_fired is False
    assert result.final_output == "done"
