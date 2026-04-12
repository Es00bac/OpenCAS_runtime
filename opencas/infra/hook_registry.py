"""Typed hook registry with priority-aware execution."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional

from pydantic import BaseModel, Field


class HookSpec(BaseModel):
    """Specification for a typed hook."""

    name: str
    expected_kwargs: Dict[str, str] = Field(default_factory=dict)
    return_schema: Optional[str] = None


@dataclass
class HookResult:
    """Result of running a hook handler."""

    allowed: bool
    reason: Optional[str] = None
    mutated_context: Optional[Dict[str, Any]] = None


@dataclass
class HookRegistration:
    """Registration of a handler for a hook."""

    hook_name: str
    handler: Callable[[str, Dict[str, Any]], HookResult]
    priority: int = 0
    source: Optional[str] = None


class TypedHookRegistry:
    """Priority-aware typed hook registry."""

    def __init__(self) -> None:
        self._specs: Dict[str, HookSpec] = {}
        self._handlers: Dict[str, List[HookRegistration]] = {}

    def register_spec(self, spec: HookSpec) -> None:
        """Declare a hook specification."""
        self._specs[spec.name] = spec

    def register(
        self,
        hook_name: str,
        handler: Callable[[str, Dict[str, Any]], HookResult],
        priority: int = 0,
        source: Optional[str] = None,
    ) -> None:
        """Register a handler for *hook_name*."""
        self._handlers.setdefault(hook_name, []).append(
            HookRegistration(
                hook_name=hook_name,
                handler=handler,
                priority=priority,
                source=source,
            )
        )

    def unregister(
        self,
        hook_name: str,
        handler: Callable[[str, Dict[str, Any]], HookResult],
    ) -> None:
        """Remove a handler from *hook_name*."""
        regs = self._handlers.get(hook_name, [])
        self._handlers[hook_name] = [r for r in regs if r.handler is not handler]

    def run(self, hook_name: str, context: Dict[str, Any]) -> HookResult:
        """Run all handlers for *hook_name* sorted by priority (descending).

        Short-circuits on the first handler that returns *allowed=False*.
        Applies any *mutated_context* from successful handlers.
        Validates that *context* contains the expected kwargs declared by the spec.
        """
        spec = self._specs.get(hook_name)
        if spec is not None and spec.expected_kwargs:
            for key in spec.expected_kwargs:
                if key not in context:
                    return HookResult(
                        allowed=False,
                        reason=f"Missing required hook context key: {key}",
                    )
        handlers = list(self._handlers.get(hook_name, []))
        handlers.sort(key=lambda r: r.priority, reverse=True)
        current_context = dict(context)
        for reg in handlers:
            result = reg.handler(hook_name, current_context)
            if result.mutated_context is not None:
                current_context = result.mutated_context
            if not result.allowed:
                return HookResult(
                    allowed=False,
                    reason=result.reason,
                    mutated_context=current_context,
                )
        return HookResult(allowed=True, mutated_context=current_context)

    def list_handlers(self, hook_name: str) -> List[HookRegistration]:
        """Return registered handlers for *hook_name* in priority order."""
        handlers = list(self._handlers.get(hook_name, []))
        handlers.sort(key=lambda r: r.priority, reverse=True)
        return handlers

    def clear_source(self, source: str) -> None:
        """Remove all handlers originating from *source*."""
        for hook_name in list(self._handlers.keys()):
            regs = self._handlers[hook_name]
            self._handlers[hook_name] = [r for r in regs if r.source != source]
