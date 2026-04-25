"""Foundation tool registration helpers for AgentRuntime."""

from __future__ import annotations

from typing import Any, Sequence

from .tool_registration_foundation_fs import register_foundation_fs_tools
from .tool_registration_foundation_process import register_foundation_process_tools


def register_foundation_tools(
    runtime: Any,
    *,
    roots: Sequence[str],
    default_cwd: str,
) -> None:
    """Register the runtime's core filesystem and local execution tools."""
    register_foundation_fs_tools(runtime, roots=roots)
    register_foundation_process_tools(runtime, roots=roots, default_cwd=default_cwd)
