"""Runtime module for OpenCAS: main agent loop and session management."""

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .agent_loop import AgentRuntime
    from .daydream import DaydreamGenerator
    from .readiness import AgentReadiness, ReadinessState

__all__ = ["AgentRuntime", "DaydreamGenerator", "AgentReadiness", "ReadinessState"]


# Lazy imports to avoid circular dependencies with bootstrap
_import_map = {
    "AgentRuntime": (".agent_loop", "AgentRuntime"),
    "DaydreamGenerator": (".daydream", "DaydreamGenerator"),
    "AgentReadiness": (".readiness", "AgentReadiness"),
    "ReadinessState": (".readiness", "ReadinessState"),
}


def __getattr__(name: str):
    if name in _import_map:
        import importlib
        module_path, obj_name = _import_map[name]
        module = importlib.import_module(module_path, package=__name__)
        return getattr(module, obj_name)
    raise AttributeError(f"module '{__name__}' has no attribute '{name}'")
