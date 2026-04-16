"""Tests for repo-local maintenance script bootstrap defaults."""

from pathlib import Path

from opencas.maintenance import build_repo_local_bootstrap_config

REPO_ROOT = Path(__file__).resolve().parents[1]


def test_build_repo_local_bootstrap_config_defaults_to_managed_workspace() -> None:
    repo_root = REPO_ROOT
    config = build_repo_local_bootstrap_config(
        repo_root,
        session_id="maintenance-test",
    )

    assert config.state_dir == repo_root / ".opencas"
    assert config.primary_workspace_root() == repo_root
    assert config.agent_workspace_root() == repo_root / "workspace"


def test_build_repo_local_bootstrap_config_supports_managed_override() -> None:
    repo_root = REPO_ROOT
    managed_root = repo_root / "workspace" / "agents"

    config = build_repo_local_bootstrap_config(
        repo_root,
        session_id="maintenance-test",
        managed_workspace_root=managed_root,
    )

    assert config.agent_workspace_root() == managed_root
