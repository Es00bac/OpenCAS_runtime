"""Default tool registration helpers for AgentRuntime."""

from __future__ import annotations

from typing import Any

from opencas.tools.validation import create_default_tool_validation_pipeline

from .tool_registration_advanced import register_advanced_tools
from .tool_registration_foundation import register_foundation_tools
from .tool_registration_workflow import register_workflow_tools
from .tool_registration_memory import register_memory_tools


def register_runtime_default_tools(runtime: Any) -> None:
    roots = [str(r) for r in runtime.ctx.sandbox.allowed_roots]
    if not roots:
        roots = [str(runtime.ctx.config.primary_workspace_root())]
    default_cwd = roots[0]
    runtime.tools.validation_pipeline = create_default_tool_validation_pipeline(
        roots=roots,
        max_write_bytes=500_000,
    )
    register_foundation_tools(runtime, roots=roots, default_cwd=default_cwd)
    register_workflow_tools(runtime, roots=roots)
    register_advanced_tools(runtime)
    register_memory_tools(runtime)
