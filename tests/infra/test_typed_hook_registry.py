"""Tests for TypedHookRegistry."""

from opencas.infra.hook_registry import HookResult, HookSpec, TypedHookRegistry


def test_register_and_run() -> None:
    reg = TypedHookRegistry()
    reg.register_spec(HookSpec(name="test_hook"))
    calls = []

    def handler(_, ctx):
        calls.append(ctx["value"])
        return HookResult(allowed=True)

    reg.register("test_hook", handler)
    result = reg.run("test_hook", {"value": 42})
    assert result.allowed is True
    assert calls == [42]


def test_priority_ordering() -> None:
    reg = TypedHookRegistry()
    reg.register_spec(HookSpec(name="priority_hook"))
    order = []

    def low(_, ctx):
        order.append("low")
        return HookResult(allowed=True)

    def high(_, ctx):
        order.append("high")
        return HookResult(allowed=True)

    reg.register("priority_hook", low, priority=1)
    reg.register("priority_hook", high, priority=10)
    reg.run("priority_hook", {})
    assert order == ["high", "low"]


def test_short_circuit() -> None:
    reg = TypedHookRegistry()
    reg.register_spec(HookSpec(name="block_hook"))

    def blocker(_, ctx):
        return HookResult(allowed=False, reason="blocked")

    def never_called(_, ctx):
        raise RuntimeError("should not run")

    reg.register("block_hook", blocker, priority=10)
    reg.register("block_hook", never_called, priority=1)
    result = reg.run("block_hook", {})
    assert result.allowed is False
    assert result.reason == "blocked"


def test_mutated_context() -> None:
    reg = TypedHookRegistry()
    reg.register_spec(HookSpec(name="mutate_hook"))

    def mutator(_, ctx):
        ctx["x"] = ctx.get("x", 0) + 1
        return HookResult(allowed=True, mutated_context=ctx)

    reg.register("mutate_hook", mutator)
    result = reg.run("mutate_hook", {"x": 0})
    assert result.mutated_context["x"] == 1


def test_unregister() -> None:
    reg = TypedHookRegistry()
    reg.register_spec(HookSpec(name="unreg_hook"))

    def handler(_, ctx):
        return HookResult(allowed=False)

    reg.register("unreg_hook", handler)
    reg.unregister("unreg_hook", handler)
    result = reg.run("unreg_hook", {})
    assert result.allowed is True


def test_clear_source() -> None:
    reg = TypedHookRegistry()
    reg.register_spec(HookSpec(name="src_hook"))

    def h1(_, ctx):
        return HookResult(allowed=False)

    def h2(_, ctx):
        return HookResult(allowed=False)

    reg.register("src_hook", h1, source="plugin_a")
    reg.register("src_hook", h2, source="plugin_b")
    reg.clear_source("plugin_a")
    result = reg.run("src_hook", {})
    # h2 still blocks, so run returns allowed=False
    assert result.allowed is False
    reg.unregister("src_hook", h2)
    result = reg.run("src_hook", {})
    assert result.allowed is True


def test_expected_kwargs_validation() -> None:
    reg = TypedHookRegistry()
    reg.register_spec(
        HookSpec(name="validated_hook", expected_kwargs={"tool_name": "str", "args": "dict"})
    )

    def handler(_, ctx):
        return HookResult(allowed=True)

    reg.register("validated_hook", handler)

    # Missing "args" should fail validation
    result = reg.run("validated_hook", {"tool_name": "fs_read_file"})
    assert result.allowed is False
    assert "Missing required hook context key" in result.reason

    # With both keys present, handler runs
    result = reg.run("validated_hook", {"tool_name": "fs_read_file", "args": {}})
    assert result.allowed is True
