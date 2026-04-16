"""Managed-workspace path helpers for workflow tools."""

from __future__ import annotations

from pathlib import Path
from typing import Any


def managed_workspace_root(runtime: Any) -> Path:
    """Return the dedicated root for workflow-created files."""
    config = runtime.ctx.config
    root = Path(config.agent_workspace_root())
    root.mkdir(parents=True, exist_ok=True)
    return root


def resolve_managed_output_path(
    runtime: Any,
    requested_path: str,
    *,
    default_relative_path: Path,
) -> Path:
    """Resolve workflow outputs inside the managed workspace root only."""
    workspace_root = managed_workspace_root(runtime)
    candidate = (
        workspace_root / default_relative_path
        if not requested_path
        else Path(requested_path).expanduser()
    )
    if not candidate.is_absolute():
        candidate = workspace_root / candidate
    resolved = candidate.resolve()
    if not resolved.is_relative_to(workspace_root):
        raise ValueError(
            f"Output path must stay within the managed workspace root: {workspace_root}"
        )
    return resolved
