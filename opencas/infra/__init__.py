"""Infrastructure utilities for OpenCAS."""

from .event_bus import BaaCompletedEvent, BaaPauseEvent, BaaProgressEvent, EventBus, HealthCheckEvent
from .hook_bus import (
    POST_ACTION_DECISION,
    POST_COMMAND_EXECUTE,
    POST_CONVERSATION_RESPONSE,
    POST_FILE_WRITE,
    POST_SESSION_LIFECYCLE,
    POST_TOOL_EXECUTE,
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
    "POST_TOOL_EXECUTE",
    "POST_ACTION_DECISION",
    "POST_SESSION_LIFECYCLE",
    "PRE_COMMAND_EXECUTE",
    "POST_COMMAND_EXECUTE",
    "PRE_FILE_WRITE",
    "POST_FILE_WRITE",
    "PRE_CONVERSATION_RESPONSE",
    "POST_CONVERSATION_RESPONSE",
]
