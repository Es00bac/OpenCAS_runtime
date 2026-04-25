"""Shared bootstrap defaults for repo-local maintenance scripts.

These helpers keep maintenance utilities aligned with the managed workspace
policy so path-policy changes land in one place instead of being re-encoded
across small scripts.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from opencas.bootstrap import BootstrapConfig


def build_repo_local_bootstrap_config(
    repo_root: Path,
    *,
    session_id: str,
    state_dir: Path | None = None,
    clean_boot: bool = False,
    managed_workspace_root: Path | None = None,
    **overrides: Any,
) -> BootstrapConfig:
    """Build a BootstrapConfig for repo-local maintenance work."""
    repo_root = repo_root.expanduser().resolve()
    return BootstrapConfig(
        state_dir=(state_dir or (repo_root / ".opencas")).expanduser().resolve(),
        session_id=session_id,
        workspace_root=repo_root,
        managed_workspace_root=managed_workspace_root.expanduser().resolve()
        if managed_workspace_root is not None
        else None,
        clean_boot=clean_boot,
        **overrides,
    )
