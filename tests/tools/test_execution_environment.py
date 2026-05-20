from __future__ import annotations

import json
from pathlib import Path

import pytest

from opencas.tools.adapters.google_workspace import (
    GoogleWorkspaceToolAdapter,
    google_workspace_cli_available,
)
from opencas.tools.adapters.shell import ShellToolAdapter


def _install_fake_gws(home: Path, body: str = "echo '{\"ok\":true}'\n") -> Path:
    bin_dir = home / ".npm-global" / "bin"
    bin_dir.mkdir(parents=True)
    script = bin_dir / "gws"
    script.write_text("#!/bin/sh\n" + body, encoding="utf-8")
    script.chmod(0o755)
    return script


def test_shell_adapter_finds_home_npm_global_bin_when_path_omits_it(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    home = tmp_path / "home"
    _install_fake_gws(home, body="echo GWS_HELP_OK\n")
    monkeypatch.setenv("HOME", str(home))
    monkeypatch.setenv("PATH", "/usr/bin")

    result = ShellToolAdapter(cwd=str(tmp_path))(
        "bash_run_command",
        {"command": "gws --help"},
    )

    payload = json.loads(result.output)
    assert result.success is True
    assert payload["ok"] is True
    assert payload["stdout"].strip() == "GWS_HELP_OK"


def test_shell_adapter_missing_bare_command_returns_structured_not_found(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    home = tmp_path / "home"
    home.mkdir()
    monkeypatch.setenv("HOME", str(home))
    monkeypatch.setenv("PATH", "/usr/bin")

    result = ShellToolAdapter(cwd=str(tmp_path))(
        "bash_run_command",
        {"command": "definitely-not-an-opencas-command"},
    )

    payload = json.loads(result.output)
    assert result.success is False
    assert payload["ok"] is False
    assert payload["code"] == 127
    assert "command not found" in payload["stderr"]
    assert result.metadata["missing_command"] is True


def test_google_workspace_cli_available_uses_user_bin_path(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    home = tmp_path / "home"
    _install_fake_gws(home)
    monkeypatch.setenv("HOME", str(home))
    monkeypatch.setenv("PATH", "/usr/bin")

    assert google_workspace_cli_available("gws") is True


@pytest.mark.asyncio
async def test_google_workspace_adapter_runs_resolved_user_bin_gws(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    home = tmp_path / "home"
    fake_gws = _install_fake_gws(home, body="echo '{\"auth\":\"ok\"}'\n")
    monkeypatch.setenv("HOME", str(home))
    monkeypatch.setenv("PATH", "/usr/bin")

    result = await GoogleWorkspaceToolAdapter(command="gws")(
        "google_workspace_auth_status",
        {},
    )

    payload = json.loads(result.output)
    assert result.success is True
    assert payload == {"auth": "ok"}
    assert result.metadata["resolved_command"] == str(fake_gws)
