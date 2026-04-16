"""Workflow, web, and browser registration helpers for AgentRuntime."""

from __future__ import annotations

from typing import Any, Sequence

from .tool_registration_browser_web import register_web_and_browser_tools
from .tool_registration_workflow_state import register_workflow_state_tools
from .tool_registration_workflow_tasking import register_workflow_tasking_tools


def register_workflow_tools(
    runtime: Any,
    *,
    roots: Sequence[str],
) -> None:
    """Register workflow-facing status, tasking, and web/browser tools."""
    del roots  # kept for call-site stability while registration stays workspace-scoped by adapter config
    register_workflow_state_tools(runtime)
    register_workflow_tasking_tools(runtime)
    register_web_and_browser_tools(runtime)
