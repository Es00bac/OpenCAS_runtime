"""Filesystem, edit, and search tool registration for AgentRuntime."""

from __future__ import annotations

from typing import Any, Sequence

from opencas.autonomy.models import ActionRiskTier
from opencas.tools import FileSystemToolAdapter
from opencas.tools.adapters.edit import EditToolAdapter
from opencas.tools.adapters.search import SearchToolAdapter

from .tool_registration_specs import ToolRegistrationSpec, register_tool_specs


def register_foundation_fs_tools(runtime: Any, *, roots: Sequence[str]) -> None:
    fs = FileSystemToolAdapter(allowed_roots=roots)
    register_tool_specs(
        runtime,
        fs,
        [
            ToolRegistrationSpec(
                name="fs_read_file",
                description="Read the contents of a file",
                risk_tier=ActionRiskTier.READONLY,
                schema={
                    "type": "object",
                    "properties": {
                        "file_path": {
                            "type": "string",
                            "description": "Absolute path to the file to read.",
                        },
                        "offset": {
                            "type": "integer",
                            "description": "Optional 0-based character offset to start reading.",
                            "minimum": 0,
                        },
                        "limit": {
                            "type": "integer",
                            "description": (
                                "Optional max number of characters to return. "
                                "If omitted, uses the runtime read limit."
                            ),
                            "minimum": 1,
                        },
                        "read_session_id": {
                            "type": "string",
                            "description": (
                                "Optional identifier for a continuous read session, so chunk "
                                "episodes can be linked across multiple reads."
                            ),
                        },
                        "concept_scope": {
                            "type": "string",
                            "description": (
                                "Optional concept bucket for this chunk (e.g., character, "
                                "location, or event)."
                            ),
                        },
                        "concept_label": {
                            "type": "string",
                            "description": "Optional concept label for a chunk-level artifact.",
                        }
                    },
                    "required": ["file_path"],
                },
            ),
            ToolRegistrationSpec(
                name="fs_list_dir",
                description=(
                    "List the contents of a directory. Returns total_count, "
                    "returned_count, truncated, and next_offset so absence is "
                    "never confused with truncation; paginate with offset/limit "
                    "before concluding a file does not exist."
                ),
                risk_tier=ActionRiskTier.READONLY,
                schema={
                    "type": "object",
                    "properties": {
                        "dir_path": {
                            "type": "string",
                            "description": "Absolute path to the directory to list.",
                        },
                        "offset": {
                            "type": "integer",
                            "description": "0-based offset into the sorted listing.",
                            "minimum": 0,
                        },
                        "limit": {
                            "type": "integer",
                            "description": "Max entries to return for this page.",
                            "minimum": 1,
                        },
                    },
                    "required": ["dir_path"],
                },
            ),
            ToolRegistrationSpec(
                name="fs_write_file",
                description="Write content to a file. Overwrites the file if it exists.",
                risk_tier=ActionRiskTier.WORKSPACE_WRITE,
                schema={
                    "type": "object",
                    "properties": {
                        "file_path": {
                            "type": "string",
                            "description": "Absolute path to the file to write.",
                        },
                        "content": {
                            "type": "string",
                            "description": "The text content to write to the file.",
                        },
                    },
                    "required": ["file_path", "content"],
                },
            ),
        ],
    )

    edit = EditToolAdapter(allowed_roots=roots)
    register_tool_specs(
        runtime,
        edit,
        [
            ToolRegistrationSpec(
                name="edit_file",
                description="Precisely edit a file by replacing old_string with new_string. Requires exact match.",
                risk_tier=ActionRiskTier.WORKSPACE_WRITE,
                schema={
                    "type": "object",
                    "properties": {
                        "file_path": {"type": "string", "description": "Absolute path to the file to edit."},
                        "old_string": {"type": "string", "description": "The exact string to replace."},
                        "new_string": {"type": "string", "description": "The new string to insert."},
                        "occurrence_index": {
                            "type": "integer",
                            "description": "Which occurrence to replace (0-based). Required if multiple matches.",
                        },
                    },
                    "required": ["file_path", "old_string", "new_string"],
                },
            )
        ],
    )

    search = SearchToolAdapter(allowed_roots=roots)
    register_tool_specs(
        runtime,
        search,
        [
            ToolRegistrationSpec(
                name="grep_search",
                description=(
                    "Search files for a regex pattern with optional glob filter "
                    "and per-file match cap. Reports total_count, returned_count, "
                    "truncated, and next_offset; treat the first page as a sample "
                    "and paginate with offset/limit before drawing conclusions."
                ),
                risk_tier=ActionRiskTier.READONLY,
                schema={
                    "type": "object",
                    "properties": {
                        "pattern": {"type": "string", "description": "Regex pattern to search for."},
                        "path": {
                            "type": "string",
                            "description": "Directory or file to search (default: workspace root).",
                        },
                        "output_mode": {
                            "type": "string",
                            "enum": ["content", "files_with_matches"],
                            "description": "Return matching lines or just file paths.",
                        },
                        "glob": {
                            "type": "string",
                            "description": "Filename glob filter passed to ripgrep, e.g. '*.py' or '!*.test.ts'.",
                        },
                        "max_count": {
                            "type": "integer",
                            "description": "Max matches per file (ripgrep --max-count). This is per-file, not pagination.",
                        },
                        "offset": {
                            "type": "integer",
                            "description": "0-based offset into the sorted result set for pagination.",
                            "minimum": 0,
                        },
                        "limit": {
                            "type": "integer",
                            "description": "Max items to return for this page.",
                            "minimum": 1,
                        },
                    },
                    "required": ["pattern"],
                },
            ),
            ToolRegistrationSpec(
                name="glob_search",
                description=(
                    "Find files matching a glob pattern. Reports total_count, "
                    "returned_count, truncated, and next_offset; paginate with "
                    "offset/limit before concluding files are missing."
                ),
                risk_tier=ActionRiskTier.READONLY,
                schema={
                    "type": "object",
                    "properties": {
                        "pattern": {
                            "type": "string",
                            "description": "Glob pattern (e.g. '**/*.py').",
                        },
                        "path": {
                            "type": "string",
                            "description": "Directory to search (default: workspace root).",
                        },
                        "offset": {
                            "type": "integer",
                            "description": "0-based offset into the sorted result set for pagination.",
                            "minimum": 0,
                        },
                        "limit": {
                            "type": "integer",
                            "description": "Max files to return for this page.",
                            "minimum": 1,
                        },
                    },
                    "required": ["pattern"],
                },
            ),
        ],
    )
