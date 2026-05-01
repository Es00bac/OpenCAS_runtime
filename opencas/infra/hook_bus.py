"""Runtime hook bus for dynamic policy intercepts."""

from __future__ import annotations

import dataclasses
from typing import Any, Callable, Dict, List, Optional


@dataclasses.dataclass
class HookResult:
    """Result of running a hook."""

    allowed: bool
    reason: Optional[str] = None
    mutated_context: Optional[Dict[str, Any]] = None


HookHandler = Callable[[str, Dict[str, Any]], HookResult]

PRE_TOOL_EXECUTE = "PRE_TOOL_EXECUTE"
POST_TOOL_EXECUTE = "POST_TOOL_EXECUTE"
POST_ACTION_DECISION = "POST_ACTION_DECISION"
POST_SESSION_LIFECYCLE = "POST_SESSION_LIFECYCLE"
PRE_COMMAND_EXECUTE = "PRE_COMMAND_EXECUTE"
POST_COMMAND_EXECUTE = "POST_COMMAND_EXECUTE"
PRE_FILE_WRITE = "PRE_FILE_WRITE"
POST_FILE_WRITE = "POST_FILE_WRITE"
PRE_CONVERSATION_RESPONSE = "PRE_CONVERSATION_RESPONSE"
POST_CONVERSATION_RESPONSE = "POST_CONVERSATION_RESPONSE"


class HookBus:
    """Lightweight hook bus supporting registration, mutation, and short-circuit."""

    def __init__(self, typed_registry=None) -> None:
        self._handlers: Dict[str, List[HookHandler]] = {}
        self._typed_registry = typed_registry
        self._wrapper_map: Dict[tuple[str, HookHandler], List[Any]] = {}

    def register(self, hook_name: str, handler: HookHandler, priority: int = 0) -> None:
        """Register a handler for *hook_name*."""
        self._handlers.setdefault(hook_name, []).append(handler)
        if self._typed_registry is not None:
            from .hook_registry import HookResult as TypedHookResult

            def wrapper(_, ctx):
                res = handler(hook_name, ctx)
                return TypedHookResult(
                    allowed=res.allowed,
                    reason=res.reason,
                    mutated_context=res.mutated_context,
                )

            wrapper._original_handler = handler  # type: ignore[attr-defined]
            self._typed_registry.register(hook_name, wrapper, priority=priority)
            key = (hook_name, handler)
            self._wrapper_map.setdefault(key, []).append(wrapper)

    def unregister(self, hook_name: str, handler: HookHandler) -> None:
        """Remove a handler from *hook_name*."""
        handlers = self._handlers.get(hook_name, [])
        if handler in handlers:
            handlers.remove(handler)
        if self._typed_registry is not None:
            key = (hook_name, handler)
            wrappers = self._wrapper_map.get(key, [])
            if wrappers:
                wrapper = wrappers.pop(0)
                self._typed_registry.unregister(hook_name, wrapper)
            if not wrappers:
                self._wrapper_map.pop(key, None)

    def run(self, hook_name: str, context: Dict[str, Any]) -> HookResult:
        """Run all handlers for *hook_name* in registration order.

        Short-circuits on the first handler that returns *allowed=False*.
        Applies any *mutated_context* from the last successful handler.
        """
        if self._typed_registry is not None:
            from .hook_registry import HookResult as TypedHookResult
            result = self._typed_registry.run(hook_name, context)
            return HookResult(
                allowed=result.allowed,
                reason=result.reason,
                mutated_context=result.mutated_context,
            )
        handlers = list(self._handlers.get(hook_name, []))
        current_context = dict(context)
        for handler in handlers:
            result = handler(hook_name, current_context)
            if result.mutated_context is not None:
                current_context = result.mutated_context
            if not result.allowed:
                return HookResult(
                    allowed=False,
                    reason=result.reason,
                    mutated_context=current_context,
                )
        return HookResult(allowed=True, mutated_context=current_context)
