"""Tests for the tool validation pipeline."""

from pathlib import Path

import pytest

from opencas.tools.validation import (
    assess_command,
    CommandSafetyValidator,
    ContentSizeValidator,
    FilesystemPathValidator,
    FilesystemWatchlistValidator,
    ToolValidationContext,
    ToolValidationPipeline,
    create_default_tool_validation_pipeline,
)


class TestCommandSafetyValidator:
    def test_allows_safe_command(self) -> None:
        v = CommandSafetyValidator()
        result = v.validate(
            "bash_run_command", {"command": "echo hello"}, ToolValidationContext()
        )
        assert result is not None
        assert result.allowed is True

    def test_blocks_dangerous_pattern(self) -> None:
        v = CommandSafetyValidator()
        result = v.validate(
            "bash_run_command", {"command": "rm -rf /"}, ToolValidationContext()
        )
        assert result is not None
        assert result.allowed is False
        assert "blocked pattern" in result.reason

    def test_missing_command(self) -> None:
        v = CommandSafetyValidator()
        result = v.validate("bash_run_command", {}, ToolValidationContext())
        assert result is not None
        assert result.allowed is False
        assert "missing" in result.reason

    def test_abstains_for_non_command_tools(self) -> None:
        v = CommandSafetyValidator()
        result = v.validate("fs_read_file", {}, ToolValidationContext())
        assert result is None

    def test_family_classification_safe(self) -> None:
        v = CommandSafetyValidator()
        result = v.validate(
            "bash_run_command", {"command": "git status"}, ToolValidationContext()
        )
        assert result.allowed is True
        assert result.command_family == "safe"
        assert result.command_permission_class == "read_only"

    def test_family_classification_destructive(self) -> None:
        v = CommandSafetyValidator()
        result = v.validate(
            "bash_run_command", {"command": "rm -rf /tmp/foo"}, ToolValidationContext()
        )
        assert result.allowed is False
        assert result.command_family == "filesystem_destructive"

    def test_family_classification_privilege(self) -> None:
        v = CommandSafetyValidator()
        result = v.validate(
            "bash_run_command", {"command": "sudo apt update"}, ToolValidationContext()
        )
        assert result.allowed is False
        assert result.command_family == "privilege_escalation"

    def test_family_classification_network(self) -> None:
        v = CommandSafetyValidator()
        result = v.validate(
            "bash_run_command", {"command": "curl https://example.com"}, ToolValidationContext()
        )
        assert result.allowed is True
        assert result.command_family == "network"
        assert result.command_permission_class == "network"

    def test_blocks_unsafe_shell_indirection(self) -> None:
        v = CommandSafetyValidator()
        result = v.validate(
            "bash_run_command",
            {"command": "bash -c 'echo hello'"},
            ToolValidationContext(),
        )
        assert result.allowed is False
        assert result.command_family == "unsafe_shell"


class TestAssessCommand:
    def test_git_status_is_read_only(self) -> None:
        assessment = assess_command("git status")
        assert assessment.family == "safe"
        assert assessment.permission_class == "read_only"

    def test_git_reset_hard_is_dangerous(self) -> None:
        assessment = assess_command("git reset --hard HEAD~1")
        assert assessment.permission_class == "dangerous"
        assert assessment.family == "filesystem_destructive"

    def test_unknown_command_defaults_bounded_write(self) -> None:
        assessment = assess_command("custom-tool --arg value")
        assert assessment.permission_class == "bounded_write"


class TestFilesystemPathValidator:
    def test_allows_path_within_roots(self, tmp_path: Path) -> None:
        v = FilesystemPathValidator()
        ctx = ToolValidationContext(roots=[str(tmp_path)])
        sub = tmp_path / "sub.txt"
        result = v.validate("fs_read_file", {"file_path": str(sub)}, ctx)
        assert result is not None
        assert result.allowed is True
        assert result.resolved_path is not None

    def test_blocks_path_outside_roots(self, tmp_path: Path) -> None:
        v = FilesystemPathValidator()
        ctx = ToolValidationContext(roots=[str(tmp_path)])
        result = v.validate("fs_read_file", {"file_path": "/etc/passwd"}, ctx)
        assert result is not None
        assert result.allowed is False
        assert "outside allowed roots" in result.reason

    def test_allows_any_read_when_set(self) -> None:
        v = FilesystemPathValidator()
        ctx = ToolValidationContext(allow_any_read=True)
        result = v.validate("fs_read_file", {"file_path": "/etc/passwd"}, ctx)
        assert result is not None
        assert result.allowed is True

    def test_missing_path(self) -> None:
        v = FilesystemPathValidator()
        ctx = ToolValidationContext(roots=[str(Path.cwd())])
        result = v.validate("fs_read_file", {}, ctx)
        assert result is not None
        assert result.allowed is False
        assert "missing path" in result.reason

    def test_abstains_for_non_fs_tools(self) -> None:
        v = FilesystemPathValidator()
        result = v.validate("bash_run_command", {}, ToolValidationContext())
        assert result is None


class TestFilesystemWatchlistValidator:
    def test_blocks_dotenv_write(self) -> None:
        v = FilesystemWatchlistValidator()
        ctx = ToolValidationContext()
        result = v.validate("fs_write_file", {"file_path": "/workspace/.env"}, ctx)
        assert result is not None
        assert result.allowed is False
        assert "watchlist" in result.reason

    def test_allows_read_of_watchlisted_file(self) -> None:
        v = FilesystemWatchlistValidator()
        ctx = ToolValidationContext()
        result = v.validate("fs_read_file", {"file_path": "/workspace/.env"}, ctx)
        assert result is None

    def test_abstains_for_non_fs_tools(self) -> None:
        v = FilesystemWatchlistValidator()
        result = v.validate("bash_run_command", {}, ToolValidationContext())
        assert result is None


class TestContentSizeValidator:
    def test_blocks_oversized_write(self) -> None:
        v = ContentSizeValidator(max_write_bytes=10)
        ctx = ToolValidationContext()
        result = v.validate("fs_write_file", {"content": "x" * 20}, ctx)
        assert result is not None
        assert result.allowed is False
        assert "too large" in result.reason

    def test_allows_small_write(self) -> None:
        v = ContentSizeValidator(max_write_bytes=100)
        ctx = ToolValidationContext()
        result = v.validate("fs_write_file", {"content": "hello"}, ctx)
        assert result is None

    def test_context_limit_overrides_instance(self) -> None:
        v = ContentSizeValidator(max_write_bytes=100)
        ctx = ToolValidationContext(max_write_bytes=5)
        result = v.validate("fs_write_file", {"content": "hello world"}, ctx)
        assert result is not None
        assert result.allowed is False


class TestToolValidationPipeline:
    def test_default_pipeline_blocks_dangerous_command(self) -> None:
        pipeline = create_default_tool_validation_pipeline()
        result = pipeline.validate(
            "bash_run_command",
            {"command": "rm -rf /"},
            ToolValidationContext(),
        )
        assert result.allowed is False
        assert "blocked pattern" in result.reason

    def test_default_pipeline_blocks_path_violation(self, tmp_path: Path) -> None:
        pipeline = create_default_tool_validation_pipeline(roots=[str(tmp_path)])
        result = pipeline.validate(
            "fs_read_file",
            {"file_path": "/etc/passwd"},
            ToolValidationContext(),
        )
        assert result.allowed is False
        assert "outside allowed roots" in result.reason

    def test_default_pipeline_blocks_watchlist(self) -> None:
        pipeline = create_default_tool_validation_pipeline()
        result = pipeline.validate(
            "fs_write_file",
            {"file_path": "/app/.ssh/id_rsa"},
            ToolValidationContext(),
        )
        assert result.allowed is False
        assert "watchlist" in result.reason

    def test_default_pipeline_allows_safe_operations(self, tmp_path: Path) -> None:
        pipeline = create_default_tool_validation_pipeline(roots=[str(tmp_path)])
        result = pipeline.validate(
            "fs_read_file",
            {"file_path": str(tmp_path / "file.txt")},
            ToolValidationContext(),
        )
        assert result.allowed is True

    def test_pipeline_accumulates_warnings(self) -> None:
        class WarnValidator:
            def validate(self, tool_name, args, context):
                from opencas.tools.validation import ToolValidationResult
                return ToolValidationResult(allowed=True, warnings=["warn1"])

        pipeline = ToolValidationPipeline([WarnValidator(), WarnValidator()])
        result = pipeline.validate("x", {}, ToolValidationContext())
        assert result.allowed is True
        assert result.warnings == ["warn1", "warn1"]
