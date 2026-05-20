from __future__ import annotations

import json
from pathlib import Path

import pytest

from opencas.tools.adapters.cli import CliDiscoveryToolAdapter


def _install_fake_cli(home: Path, name: str = "gws") -> Path:
    bin_dir = home / ".npm-global" / "bin"
    bin_dir.mkdir(parents=True)
    script = bin_dir / name
    script.write_text(
        "#!/bin/sh\n"
        "case \"$1\" in\n"
        "  --help|-h|help) echo 'fake gws help: gmail calendar drive';;\n"
        "  --version|version) echo 'fake gws 1.2.3';;\n"
        "  *) echo 'unexpected args' >&2; exit 2;;\n"
        "esac\n",
        encoding="utf-8",
    )
    script.chmod(0o755)
    return script


def test_cli_discovery_resolves_user_local_command_and_collects_help(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    home = tmp_path / "home"
    fake = _install_fake_cli(home)
    monkeypatch.setenv("HOME", str(home))
    monkeypatch.setenv("PATH", "/usr/bin")

    result = CliDiscoveryToolAdapter(cwd=str(tmp_path))(
        "cli_discover_command",
        {"command": "gws", "probe_args": ["--help", "--version"]},
    )

    payload = json.loads(result.output)
    assert result.success is True
    assert payload["ok"] is True
    assert payload["resolved_path"] == str(fake)
    assert payload["matches"] == [str(fake)]
    assert payload["probes"][0]["argv"] == ["gws", "--help"]
    assert "gmail calendar drive" in payload["probes"][0]["stdout"]
    assert "fake gws 1.2.3" in payload["probes"][1]["stdout"]


def test_cli_discovery_returns_structured_missing_command(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    home = tmp_path / "home"
    home.mkdir()
    monkeypatch.setenv("HOME", str(home))
    monkeypatch.setenv("PATH", "/usr/bin")

    result = CliDiscoveryToolAdapter(cwd=str(tmp_path))(
        "cli_discover_command",
        {"command": "missing-gws"},
    )

    payload = json.loads(result.output)
    assert result.success is False
    assert payload["ok"] is False
    assert payload["code"] == 127
    assert "command not found" in payload["stderr"]
    assert result.metadata["missing_command"] is True


def test_cli_discovery_rejects_command_with_arguments(tmp_path: Path) -> None:
    result = CliDiscoveryToolAdapter(cwd=str(tmp_path))(
        "cli_discover_command",
        {"command": "gws --help"},
    )

    assert result.success is False
    assert "do not include arguments" in result.output
