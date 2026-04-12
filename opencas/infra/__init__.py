"""Infrastructure utilities for OpenCAS."""

from .event_bus import BaaCompletedEvent, BaaPauseEvent, BaaProgressEvent, EventBus, HealthCheckEvent
from .hook_bus import (
    PRE_COMMAND_EXECUTE,
    PRE_CONVERSATION_RESPONSE,
    PRE_FILE_WRITE,
    PRE_TOOL_EXECUTE,
    HookBus,
    HookResult,
)
from .hook_registry import HookSpec, TypedHookRegistry

__all__ = [
    "BaaCompletedEvent",
    "BaaPauseEvent",
    "BaaProgressEvent",
    "HealthCheckEvent",
    "EventBus",
    "HookBus",
    "HookResult",
    "HookSpec",
    "TypedHookRegistry",
    "PRE_TOOL_EXECUTE",
    "PRE_COMMAND_EXECUTE",
    "PRE_FILE_WRITE",
    "PRE_CONVERSATION_RESPONSE",
]
