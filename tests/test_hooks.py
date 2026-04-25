"""Tests for the HookBus infrastructure."""

from opencas.infra import (
    PRE_COMMAND_EXECUTE,
    PRE_FILE_WRITE,
    PRE_TOOL_EXECUTE,
    HookBus,
    HookResult,
)


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
