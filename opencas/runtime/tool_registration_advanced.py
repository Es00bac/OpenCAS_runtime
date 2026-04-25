"""Advanced tool registration helpers for AgentRuntime."""

from __future__ import annotations

from typing import Any

from .tool_registration_advanced_coding import register_advanced_coding_tools
from .tool_registration_advanced_integrations import (
    register_advanced_integration_tools,
)
from .tool_registration_advanced_planning import register_advanced_planning_tools


def register_advanced_tools(runtime: Any) -> None:
    register_advanced_coding_tools(runtime)
    register_advanced_planning_tools(runtime)
    register_advanced_integration_tools(runtime)
