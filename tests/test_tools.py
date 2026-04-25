"""Tests for the tool registry and adapters."""

from pathlib import Path
import pytest

from opencas.autonomy.models import ActionRiskTier
from opencas.tools import (
    FileSystemToolAdapter,
    ShellToolAdapter,
    ToolRegistry,
    ToolResult,
)
from opencas.tools.validation import create_default_tool_validation_pipeline


@pytest.fixture
def registry():
    return ToolRegistry()


@pytest.fixture
def tmp_dir(tmp_path: Path):
    return tmp_path


def test_register_and_list(registry: ToolRegistry) -> None:
    def dummy_adapter(name: str, args: dict) -> ToolResult:
        return ToolResult(success=True, output="ok", metadata={})

    registry.register("demo", "A demo tool", dummy_adapter, ActionRiskTier.READONLY)
    assert len(registry.list_tools()) == 1
    entry = registry.get("demo")
    assert entry is not None
    assert entry.name == "demo"
    assert entry.risk_tier == ActionRiskTier.READONLY


def test_execute_unknown_tool(registry: ToolRegistry) -> None:
    result = registry.execute("missing", {})
    assert result.success is False
    assert "not found" in result.output


def test_fs_read_file(tmp_dir: Path) -> None:
    adapter = FileSystemToolAdapter(allowed_roots=[str(tmp_dir)])
    test_file = tmp_dir / "hello.txt"
    test_file.write_text("world", encoding="utf-8")

    result = adapter("fs_read_file", {"file_path": str(test_file)})
    assert result.success is True
    assert result.output == "world"


def test_fs_list_dir(tmp_dir: Path) -> None:
    adapter = FileSystemToolAdapter(allowed_roots=[str(tmp_dir)])
    (tmp_dir / "a.txt").write_text("a", encoding="utf-8")
    (tmp_dir / "b_dir").mkdir()

    result = adapter("fs_list_dir", {"dir_path": str(tmp_dir)})
    assert result.success is True
    assert "a.txt" in result.output
    assert "b_dir" in result.output


def test_fs_write_file(tmp_dir: Path) -> None:
    adapter = FileSystemToolAdapter(allowed_roots=[str(tmp_dir)])
    target = tmp_dir / "out.txt"

    result = adapter("fs_write_file", {"file_path": str(target), "content": "data"})
    assert result.success is True
    assert target.read_text(encoding="utf-8") == "data"


def test_fs_path_violation(tmp_dir: Path) -> None:
    adapter = FileSystemToolAdapter(allowed_roots=[str(tmp_dir)])
    outside = tmp_dir.parent / "secret.txt"
    outside.write_text("secret", encoding="utf-8")

    result = adapter("fs_read_file", {"file_path": str(outside)})
    assert result.success is False
    assert "outside allowed" in result.output or "PermissionError" in result.output


def test_shell_echo(tmp_dir: Path) -> None:
    adapter = ShellToolAdapter(cwd=str(tmp_dir), timeout=5.0)
    result = adapter("bash_run_command", {"command": "echo hello"})
    assert result.success is True
    assert "hello" in result.output


def test_shell_block_dangerous(tmp_dir: Path) -> None:
    adapter = ShellToolAdapter(cwd=str(tmp_dir), timeout=5.0)
    result = adapter("bash_run_command", {"command": "rm -rf /"})
    assert result.success is False
    assert "blocked" in result.output.lower()


def test_registry_risk_tier_routing(registry: ToolRegistry, tmp_dir: Path) -> None:
    fs = FileSystemToolAdapter(allowed_roots=[str(tmp_dir)])
    registry.register("fs_read_file", "Read a file", fs, ActionRiskTier.READONLY)
    test_file = tmp_dir / "x.txt"
    test_file.write_text("y", encoding="utf-8")

    result = registry.execute("fs_read_file", {"file_path": str(test_file)})
    assert result.success is True
    assert result.output == "y"


def test_registry_surfaces_validation_metadata(tmp_dir: Path) -> None:
    registry = ToolRegistry(
        validation_pipeline=create_default_tool_validation_pipeline(roots=[str(tmp_dir)])
    )
    shell = ShellToolAdapter(cwd=str(tmp_dir), timeout=5.0)
    registry.register("bash_run_command", "Run shell", shell, ActionRiskTier.SHELL_LOCAL)

    result = registry.execute("bash_run_command", {"command": "pwd"})
    assert result.success is True
    assert result.metadata["command_permission_class"] == "read_only"
    assert result.metadata["command_family"] == "safe"
