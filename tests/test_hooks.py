"""Tests for the HookBus infrastructure."""

import pytest

from opencas.autonomy.models import ActionRiskTier
from opencas.infra import (
    POST_TOOL_EXECUTE,
    PRE_COMMAND_EXECUTE,
    PRE_FILE_WRITE,
    PRE_TOOL_EXECUTE,
    HookBus,
    HookResult,
)
from opencas.tools import ToolRegistry, ToolResult


def test_hook_bus_allows_when_empty() -> None:
    bus = HookBus()
    result = bus.run(PRE_TOOL_EXECUTE, {"tool_name": "echo"})
    assert result.allowed is True


def test_hook_bus_short_circuits_on_deny() -> None:
    bus = HookBus()

    def allow_handler(_name, ctx):
        return HookResult(allowed=True)

    def deny_handler(_name, ctx):
        return HookResult(allowed=False, reason="blocked")

    bus.register(PRE_TOOL_EXECUTE, allow_handler)
    bus.register(PRE_TOOL_EXECUTE, deny_handler)
    bus.register(PRE_TOOL_EXECUTE, allow_handler)

    result = bus.run(PRE_TOOL_EXECUTE, {"tool_name": "echo"})
    assert result.allowed is False
    assert result.reason == "blocked"


def test_hook_bus_mutates_context() -> None:
    bus = HookBus()

    def mutate_handler(_name, ctx):
        ctx["mutated"] = True
        return HookResult(allowed=True, mutated_context=ctx)

    bus.register(PRE_TOOL_EXECUTE, mutate_handler)
    result = bus.run(PRE_TOOL_EXECUTE, {"tool_name": "echo"})
    assert result.allowed is True
    assert result.mutated_context["mutated"] is True


def test_hook_bus_unregister() -> None:
    bus = HookBus()

    def handler(_name, ctx):
        return HookResult(allowed=False)

    bus.register(PRE_COMMAND_EXECUTE, handler)
    bus.unregister(PRE_COMMAND_EXECUTE, handler)
    result = bus.run(PRE_COMMAND_EXECUTE, {})
    assert result.allowed is True


def test_predefined_hooks_exist() -> None:
    assert PRE_TOOL_EXECUTE == "PRE_TOOL_EXECUTE"
    assert PRE_COMMAND_EXECUTE == "PRE_COMMAND_EXECUTE"
    assert PRE_FILE_WRITE == "PRE_FILE_WRITE"


@pytest.mark.asyncio
async def test_post_tool_hook_exception_isolated_for_successful_tool() -> None:
    class RecordingTracer:
        def __init__(self):
            self.events = []

        def log(self, kind, message, payload=None):
            self.events.append((kind, message, payload or {}))

    tracer = RecordingTracer()
    bus = HookBus()

    def raising_post_handler(_name, _ctx):
        raise RuntimeError("post hook failed")

    async def successful_adapter(_name, _args):
        return ToolResult(success=True, output="ok", metadata={})

    bus.register(POST_TOOL_EXECUTE, raising_post_handler)
    registry = ToolRegistry(tracer=tracer, hook_bus=bus)
    registry.register("write_note", "Write note", successful_adapter, ActionRiskTier.WORKSPACE_WRITE)

    result = await registry.execute_async("write_note", {"path": "note.txt"})

    assert result.success is True
    assert result.output == "ok"
    hook_failures = [event for event in tracer.events if event[1] == "hook_handler_failed"]
    assert hook_failures
    assert hook_failures[-1][2]["hook_name"] == POST_TOOL_EXECUTE
    assert "post hook failed" in hook_failures[-1][2]["error"]
