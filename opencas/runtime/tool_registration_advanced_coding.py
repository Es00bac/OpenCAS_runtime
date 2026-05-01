"""Coding and introspection tool registration helpers for AgentRuntime."""

from __future__ import annotations

from typing import Any

from opencas.autonomy.models import ActionRiskTier
from opencas.tools.adapters.lsp import LspToolAdapter
from opencas.tools.adapters.repl import ReplToolAdapter

from .tool_registration_specs import ToolRegistrationSpec, register_tool_specs


def register_advanced_coding_tools(runtime: Any) -> None:
    repl = ReplToolAdapter()
    lsp = LspToolAdapter()
    register_tool_specs(
        runtime,
        repl,
        [
            ToolRegistrationSpec(
                name="python_repl",
                description="Execute Python code in a persistent REPL session.",
                risk_tier=ActionRiskTier.WORKSPACE_WRITE,
                schema={
                    "type": "object",
                    "properties": {
                        "code": {
                            "type": "string",
                            "description": "Python code to execute.",
                        },
                        "research_session_id": {
                            "type": "string",
                            "description": "Session ID for persistent state across calls.",
                        },
                    },
                    "required": ["code"],
                },
            ),
        ],
    )
    register_tool_specs(
        runtime,
        lsp,
        [
            ToolRegistrationSpec(
                name="lsp_goto_definition",
                description="Go to the definition of a symbol in a file.",
                risk_tier=ActionRiskTier.READONLY,
                schema={
                    "type": "object",
                    "properties": {
                        "file_path": {"type": "string"},
                        "line": {"type": "integer"},
                        "character": {"type": "integer"},
                    },
                    "required": ["file_path", "line", "character"],
                },
            ),
            ToolRegistrationSpec(
                name="lsp_find_references",
                description="Find all references to a symbol in a file.",
                risk_tier=ActionRiskTier.READONLY,
                schema={
                    "type": "object",
                    "properties": {
                        "file_path": {"type": "string"},
                        "line": {"type": "integer"},
                        "character": {"type": "integer"},
                    },
                    "required": ["file_path", "line", "character"],
                },
            ),
            ToolRegistrationSpec(
                name="lsp_hover",
                description="Get type/documentation info for a symbol in a file.",
                risk_tier=ActionRiskTier.READONLY,
                schema={
                    "type": "object",
                    "properties": {
                        "file_path": {"type": "string"},
                        "line": {"type": "integer"},
                        "character": {"type": "integer"},
                    },
                    "required": ["file_path", "line", "character"],
                },
            ),
            ToolRegistrationSpec(
                name="lsp_document_symbols",
                description="List all symbols (functions, classes, variables) in a file.",
                risk_tier=ActionRiskTier.READONLY,
                schema={
                    "type": "object",
                    "properties": {
                        "file_path": {"type": "string"},
                    },
                    "required": ["file_path"],
                },
            ),
            ToolRegistrationSpec(
                name="lsp_diagnostics",
                description="Get syntax errors and diagnostics for a file.",
                risk_tier=ActionRiskTier.READONLY,
                schema={
                    "type": "object",
                    "properties": {
                        "file_path": {"type": "string"},
                    },
                    "required": ["file_path"],
                },
            ),
        ],
    )
