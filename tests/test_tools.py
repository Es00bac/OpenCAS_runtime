"""Tests for the tool registry and adapters."""

import asyncio
import json
from pathlib import Path
import pytest

from opencas.autonomy.models import ActionRiskTier
from opencas.tools import (
    FileSystemToolAdapter,
    ShellToolAdapter,
    ToolRegistry,
    ToolResult,
)
from opencas.planning import PlanStore
from opencas.tools.adapters.plan import PlanToolAdapter
from opencas.tools.adapters.search import SearchToolAdapter
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


@pytest.mark.asyncio
async def test_execute_async_passes_audit_only_to_adapter_args() -> None:
    seen = {}

    async def adapter(name: str, args: dict) -> ToolResult:
        seen.update(args)
        return ToolResult(success=True, output="ok", metadata={})

    registry = ToolRegistry()
    registry.register("audit_tool", "Audit tool", adapter, ActionRiskTier.READONLY)

    result = await registry.execute_async("audit_tool", {}, audit_only=True)

    assert result.success is True
    assert seen["_audit_only"] is True


@pytest.mark.asyncio
async def test_execute_async_traces_safe_argument_signature() -> None:
    class Tracer:
        def __init__(self) -> None:
            self.calls = []

        def log(self, kind, message, payload):
            self.calls.append((kind, message, payload))

    async def adapter(name: str, args: dict) -> ToolResult:
        return ToolResult(success=True, output="ok", metadata={})

    tracer = Tracer()
    registry = ToolRegistry(tracer=tracer)
    registry.register("demo", "Demo", adapter, ActionRiskTier.READONLY)

    result = await registry.execute_async(
        "demo",
        {"file_path": "/tmp/example.txt", "token": "secret-token"},
    )

    assert result.success is True
    executing = next(call for call in tracer.calls if call[1] == "ToolRegistry: tool_executing")
    payload = executing[2]
    assert payload["args_keys"] == ["file_path", "token"]
    assert len(payload["args_hash"]) == 64
    assert "secret-token" not in json.dumps(payload)


def test_fs_read_file(tmp_dir: Path) -> None:
    adapter = FileSystemToolAdapter(allowed_roots=[str(tmp_dir)])
    test_file = tmp_dir / "hello.txt"
    test_file.write_text("world", encoding="utf-8")

    result = adapter("fs_read_file", {"file_path": str(test_file)})
    assert result.success is True
    assert result.output == "world"


def test_fs_read_file_paginates_with_concept_metadata(tmp_dir: Path) -> None:
    adapter = FileSystemToolAdapter(allowed_roots=[str(tmp_dir)])
    test_file = tmp_dir / "novel.txt"
    test_file.write_text("abcdef" * 500, encoding="utf-8")

    result = adapter(
        "fs_read_file",
        {
            "file_path": str(test_file),
            "offset": 0,
            "limit": 10,
            "read_session_id": "story-session-1",
            "concept_scope": "chapter",
            "concept_label": "opening",
        },
    )

    assert result.success is True
    payload = json.loads(result.output)
    assert payload["ok"] is True
    assert payload["read_session_id"] == "story-session-1"
    assert payload["concept_scope"] == "chapter"
    assert payload["concept_label"] == "opening"
    assert payload["offset"] == 0
    assert payload["limit"] == 10
    assert payload["returned_count"] == 10
    assert payload["truncated"] is True
    assert payload["next_offset"] == 10
    assert result.metadata["total_chars"] == len("abcdef" * 500)
    assert result.metadata["next_offset"] == 10


def test_fs_list_dir(tmp_dir: Path) -> None:
    adapter = FileSystemToolAdapter(allowed_roots=[str(tmp_dir)])
    (tmp_dir / "a.txt").write_text("a", encoding="utf-8")
    (tmp_dir / "b_dir").mkdir()

    result = adapter("fs_list_dir", {"dir_path": str(tmp_dir)})
    assert result.success is True
    payload = json.loads(result.output)
    assert [entry["name"] for entry in payload["entries"]] == ["a.txt", "b_dir"]
    assert payload["total_count"] == 2
    assert payload["returned_count"] == 2
    assert payload["truncated"] is False
    assert payload["next_offset"] is None
    assert result.metadata["total_count"] == 2
    assert result.metadata["truncated"] is False


def test_fs_list_dir_paginates_without_absence_ambiguity(tmp_dir: Path) -> None:
    adapter = FileSystemToolAdapter(allowed_roots=[str(tmp_dir)])
    for idx in range(5):
        (tmp_dir / f"chapter_{idx}.md").write_text(str(idx), encoding="utf-8")

    result = adapter("fs_list_dir", {"dir_path": str(tmp_dir), "offset": 1, "limit": 2})

    assert result.success is True
    payload = json.loads(result.output)
    assert [entry["name"] for entry in payload["entries"]] == ["chapter_1.md", "chapter_2.md"]
    assert payload["total_count"] == 5
    assert payload["returned_count"] == 2
    assert payload["truncated"] is True
    assert payload["next_offset"] == 3
    assert result.metadata["total_count"] == 5
    assert result.metadata["returned_count"] == 2
    assert result.metadata["truncated"] is True


def test_glob_search_reports_total_and_truncation(tmp_dir: Path) -> None:
    adapter = SearchToolAdapter(allowed_roots=[str(tmp_dir)])
    for idx in range(5):
        (tmp_dir / f"chapter_{idx}.md").write_text(str(idx), encoding="utf-8")

    result = adapter(
        "glob_search",
        {"path": str(tmp_dir), "pattern": "*.md", "offset": 1, "limit": 2},
    )

    assert result.success is True
    payload = json.loads(result.output)
    assert [Path(path).name for path in payload["files"]] == ["chapter_1.md", "chapter_2.md"]
    assert payload["total_count"] == 5
    assert payload["returned_count"] == 2
    assert payload["truncated"] is True
    assert payload["next_offset"] == 3
    assert result.metadata["total_count"] == 5
    assert result.metadata["truncated"] is True


def test_grep_search_reports_total_and_truncation(tmp_dir: Path) -> None:
    adapter = SearchToolAdapter(allowed_roots=[str(tmp_dir)])
    for idx in range(3):
        (tmp_dir / f"note_{idx}.txt").write_text("needle\n", encoding="utf-8")

    result = adapter(
        "grep_search",
        {"path": str(tmp_dir), "pattern": "needle", "offset": 0, "limit": 1},
    )

    assert result.success is True
    payload = json.loads(result.output)
    assert len(payload["matches"]) == 1
    assert payload["total_count"] == 3
    assert payload["returned_count"] == 1
    assert payload["truncated"] is True
    assert payload["next_offset"] == 1
    assert result.metadata["total_count"] == 3
    assert result.metadata["truncated"] is True


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


@pytest.mark.asyncio
async def test_plan_adapter_does_not_block_running_event_loop(tmp_path: Path) -> None:
    store = await PlanStore(tmp_path / "plans.db").connect()
    registry = ToolRegistry()
    registry.register(
        "enter_plan_mode",
        "Enter plan mode",
        PlanToolAdapter(store),
        ActionRiskTier.READONLY,
    )

    try:
        result = await asyncio.wait_for(
            registry.execute_async(
                "enter_plan_mode",
                {"plan_id": "plan-live", "content": "inspect first"},
            ),
            timeout=2.0,
        )
    finally:
        await store.close()

    assert result.success is True
    assert result.metadata["plan_id"] == "plan-live"


@pytest.mark.asyncio
async def test_execute_in_async_context_returns_guidance(registry: ToolRegistry) -> None:
    result = registry.execute("missing", {})
    assert result.success is False
    assert "execute_async" in result.output
    assert result.metadata["suggested_api"] == "execute_async"
    assert result.metadata["error_type"] == "RuntimeError"
