"""Hook handlers that project blocked runtime intentions into ShadowRegistry."""

from __future__ import annotations

from typing import Any, Dict

from opencas.infra.hook_bus import (
    HookResult,
    POST_ACTION_DECISION,
    POST_TOOL_EXECUTE,
)


def register_runtime_shadow_registry_hooks(runtime: Any) -> None:
    """Attach blocked-intention capture hooks once per runtime."""

    if getattr(runtime, "_shadow_registry_hooks_registered", False):
        return

    hook_bus = getattr(getattr(runtime, "ctx", None), "hook_bus", None)
    shadow_registry = getattr(getattr(runtime, "ctx", None), "shadow_registry", None)
    if hook_bus is None or shadow_registry is None:
        return

    hook_bus.register(
        POST_ACTION_DECISION,
        lambda hook_name, ctx: _post_action_decision(runtime, hook_name, ctx),
        priority=-105,
    )
    hook_bus.register(
        POST_TOOL_EXECUTE,
        lambda hook_name, ctx: _post_tool_execute(runtime, hook_name, ctx),
        priority=-105,
    )
    runtime._shadow_registry_hooks_registered = True


def _post_action_decision(runtime: Any, _hook_name: str, ctx: Dict[str, Any]) -> HookResult:
    shadow_registry = getattr(getattr(runtime, "ctx", None), "shadow_registry", None)
    if shadow_registry is not None:
        shadow_registry.capture_action_decision(ctx)
    return HookResult(allowed=True)


def _post_tool_execute(runtime: Any, _hook_name: str, ctx: Dict[str, Any]) -> HookResult:
    shadow_registry = getattr(getattr(runtime, "ctx", None), "shadow_registry", None)
    if shadow_registry is not None:
        shadow_registry.capture_tool_block(ctx)
    return HookResult(allowed=True)
