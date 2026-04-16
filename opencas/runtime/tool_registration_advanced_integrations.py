"""Runtime integration and workspace tool registration helpers for AgentRuntime."""

from __future__ import annotations

import asyncio
from typing import Any

from opencas.autonomy.models import ActionRiskTier
from opencas.tools.adapters.agent import AgentToolAdapter
from opencas.workspace.tool_adapter import (
    GetFileGistSchema,
    ListDirectoryGistsSchema,
    RefreshWorkspaceIndexSchema,
    SearchFileGistsSchema,
    WorkspaceIndexerToolAdapter,
)

from .tool_registration_specs import ToolRegistrationSpec, register_tool_specs


def register_advanced_integration_tools(runtime: Any) -> None:
    mcp_registry = getattr(runtime.ctx, "mcp_registry", None)
    if mcp_registry is not None and runtime.ctx.config.mcp_auto_register:
        try:
            tools = asyncio.run_coroutine_threadsafe(
                runtime._discover_and_register_mcp_tools(), asyncio.get_running_loop()
            ).result()
            runtime._trace("mcp_auto_registered", {"tool_count": len(tools)})
        except Exception as exc:
            runtime._trace("mcp_auto_register_failed", {"error": str(exc)})

    register_tool_specs(
        runtime,
        runtime._make_mcp_list_servers_adapter(),
        [
            ToolRegistrationSpec(
                name="mcp_list_servers",
                description="List configured MCP servers and their initialization status.",
                risk_tier=ActionRiskTier.READONLY,
                schema={"type": "object", "properties": {}, "required": []},
            ),
        ],
    )
    register_tool_specs(
        runtime,
        runtime._make_mcp_register_adapter(),
        [
            ToolRegistrationSpec(
                name="mcp_register_server_tools",
                description="Initialize a specific MCP server and register its tools for the current session.",
                risk_tier=ActionRiskTier.READONLY,
                schema={
                    "type": "object",
                    "properties": {
                        "server_name": {
                            "type": "string",
                            "description": "Name of the MCP server to initialize.",
                        },
                    },
                    "required": ["server_name"],
                },
            ),
        ],
    )

    agent = AgentToolAdapter(runtime=runtime)
    register_tool_specs(
        runtime,
        agent,
        [
            ToolRegistrationSpec(
                name="agent",
                description="Spawn a specialized subagent with a separate tool loop to work on a task.",
                risk_tier=ActionRiskTier.READONLY,
                schema={
                    "type": "object",
                    "properties": {
                        "description": {
                            "type": "string",
                            "description": "Short description of the subagent task.",
                        },
                        "agent_type": {
                            "type": "string",
                            "description": "Type of subagent (e.g. explore, plan, general-purpose).",
                        },
                        "prompt": {
                            "type": "string",
                            "description": "Prompt/instructions for the subagent.",
                        },
                    },
                    "required": ["prompt"],
                },
            ),
        ],
    )

    if hasattr(runtime.ctx, "workspace_index"):
        workspace_adapter = WorkspaceIndexerToolAdapter(runtime.ctx.workspace_index)
        register_tool_specs(
            runtime,
            workspace_adapter,
            [
                ToolRegistrationSpec(
                    name="workspace_get_file_gist",
                    description="Get a highly compressed semantic gist of a file without reading its full content.",
                    risk_tier=ActionRiskTier.READONLY,
                    schema=GetFileGistSchema.model_json_schema(),
                ),
                ToolRegistrationSpec(
                    name="workspace_search_file_gists",
                    description="Semantically search the workspace for files related to a query using gist embeddings.",
                    risk_tier=ActionRiskTier.READONLY,
                    schema=SearchFileGistsSchema.model_json_schema(),
                ),
                ToolRegistrationSpec(
                    name="workspace_list_directory_gists",
                    description="List all files in a directory along with their 1-line gists to understand a subsystem.",
                    risk_tier=ActionRiskTier.READONLY,
                    schema=ListDirectoryGistsSchema.model_json_schema(),
                ),
                ToolRegistrationSpec(
                    name="workspace_refresh_index",
                    description="Force a refresh of the workspace semantic index.",
                    risk_tier=ActionRiskTier.WORKSPACE_WRITE,
                    schema=RefreshWorkspaceIndexSchema.model_json_schema(),
                ),
            ],
        )
