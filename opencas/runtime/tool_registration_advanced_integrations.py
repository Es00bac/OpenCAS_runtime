"""Runtime integration and workspace tool registration helpers for AgentRuntime."""

from __future__ import annotations

import asyncio
from typing import Any

from opencas.autonomy.models import ActionRiskTier
from opencas.tools.adapters.agent import AgentToolAdapter
from opencas.tools.adapters.google_workspace import (
    GoogleWorkspaceToolAdapter,
    google_workspace_cli_available,
)
from opencas.tools.adapters.phone import PhoneToolAdapter
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

    phone = PhoneToolAdapter(runtime)
    register_tool_specs(
        runtime,
        phone,
        [
            ToolRegistrationSpec(
                name="phone_get_status",
                description="Inspect whether the Twilio phone bridge is configured and which caller policies are active.",
                risk_tier=ActionRiskTier.READONLY,
                schema={"type": "object", "properties": {}, "required": []},
            ),
            ToolRegistrationSpec(
                name="phone_call_owner",
                description="Place an outbound phone call to the trusted owner number only. Use this when the assistant genuinely needs to reach the operator by voice.",
                risk_tier=ActionRiskTier.EXTERNAL_WRITE,
                schema={
                    "type": "object",
                    "properties": {
                        "message": {
                            "type": "string",
                            "description": "What the assistant should say when the owner answers.",
                        },
                        "reason": {
                            "type": "string",
                            "description": "Short operator-facing reason for why the phone call is needed.",
                        },
                    },
                    "required": [],
                },
            ),
        ],
    )

    if google_workspace_cli_available():
        google_workspace = GoogleWorkspaceToolAdapter()
        register_tool_specs(
            runtime,
            google_workspace,
            [
                ToolRegistrationSpec(
                    name="google_workspace_auth_status",
                    description="Inspect local Google Workspace CLI authentication and enabled API state.",
                    risk_tier=ActionRiskTier.READONLY,
                    schema={"type": "object", "properties": {}, "required": []},
                ),
                ToolRegistrationSpec(
                    name="google_workspace_schema",
                    description="Inspect a Google Workspace CLI schema reference such as gmail.users.messages.list or drive.files.list.",
                    risk_tier=ActionRiskTier.READONLY,
                    schema={
                        "type": "object",
                        "properties": {
                            "schema_ref": {
                                "type": "string",
                                "description": "Schema reference like drive.files.list or gmail.users.messages.get.",
                            },
                            "resolve_refs": {
                                "type": "boolean",
                                "description": "Resolve nested schema references in the output.",
                            },
                            "timeout_seconds": {
                                "type": "integer",
                                "description": "Command timeout in seconds (default 30).",
                            },
                        },
                        "required": ["schema_ref"],
                    },
                ),
                ToolRegistrationSpec(
                    name="google_workspace_readonly_api",
                    description="Call an allowlisted read-only Google Workspace CLI API method for Gmail, Calendar, Drive, Docs, Sheets, Slides, or People data.",
                    risk_tier=ActionRiskTier.READONLY,
                    schema={
                        "type": "object",
                        "properties": {
                            "service": {"type": "string", "description": "Top-level gws service, for example gmail or drive."},
                            "resource": {"type": "string", "description": "Primary resource name, for example users, files, events, or documents."},
                            "sub_resource": {"type": "string", "description": "Optional nested resource such as messages or values."},
                            "method": {"type": "string", "description": "Read-only method name, such as list, get, or batchGet."},
                            "params": {"type": "object", "description": "JSON parameters passed to gws --params."},
                            "api_version": {"type": "string", "description": "Optional API version override."},
                            "page_all": {"type": "boolean", "description": "Enable gws auto-pagination for list calls."},
                            "page_limit": {"type": "integer", "description": "Maximum pages when page_all=true."},
                            "page_delay_ms": {"type": "integer", "description": "Delay between pages in milliseconds when page_all=true."},
                            "timeout_seconds": {"type": "integer", "description": "Command timeout in seconds (default 30)."},
                        },
                        "required": ["service", "resource", "method"],
                    },
                ),
                ToolRegistrationSpec(
                    name="google_workspace_gmail_headlines",
                    description="List recent Gmail message headlines with sender, subject, date, and labels.",
                    risk_tier=ActionRiskTier.READONLY,
                    schema={
                        "type": "object",
                        "properties": {
                            "query": {"type": "string", "description": "Gmail search query. Defaults to in:inbox."},
                            "label_ids": {
                                "type": "array",
                                "items": {"type": "string"},
                                "description": "Optional Gmail label ids to require on returned messages.",
                            },
                            "max_results": {"type": "integer", "description": "Maximum number of messages to return (default 10, max 25)."},
                            "include_snippet": {"type": "boolean", "description": "Include message snippets when available."},
                            "timeout_seconds": {"type": "integer", "description": "Command timeout in seconds (default 30)."},
                        },
                        "required": [],
                    },
                ),
                ToolRegistrationSpec(
                    name="google_workspace_gmail_get_message",
                    description="Fetch a Gmail message by id using metadata, minimal, or full format.",
                    risk_tier=ActionRiskTier.READONLY,
                    schema={
                        "type": "object",
                        "properties": {
                            "message_id": {"type": "string", "description": "Gmail message id."},
                            "format": {
                                "type": "string",
                                "description": "Message format: metadata, minimal, or full (default metadata).",
                            },
                            "timeout_seconds": {"type": "integer", "description": "Command timeout in seconds (default 30)."},
                        },
                        "required": ["message_id"],
                    },
                ),
                ToolRegistrationSpec(
                    name="google_workspace_calendar_schedule",
                    description="List Google Calendar events for a date or explicit time range.",
                    risk_tier=ActionRiskTier.READONLY,
                    schema={
                        "type": "object",
                        "properties": {
                            "calendar_id": {"type": "string", "description": "Calendar id, default primary."},
                            "date": {"type": "string", "description": "Optional YYYY-MM-DD local date shortcut."},
                            "time_min": {"type": "string", "description": "Optional RFC3339/ISO-8601 time lower bound."},
                            "time_max": {"type": "string", "description": "Optional RFC3339/ISO-8601 time upper bound."},
                            "max_results": {"type": "integer", "description": "Maximum number of events to return (default 10)."},
                            "timeout_seconds": {"type": "integer", "description": "Command timeout in seconds (default 30)."},
                        },
                        "required": [],
                    },
                ),
                ToolRegistrationSpec(
                    name="google_workspace_drive_search",
                    description="Search Google Drive files with a Drive query and return metadata for matching files.",
                    risk_tier=ActionRiskTier.READONLY,
                    schema={
                        "type": "object",
                        "properties": {
                            "query": {"type": "string", "description": "Drive query, default trashed=false."},
                            "page_size": {"type": "integer", "description": "Maximum number of files to return (default 10)."},
                            "timeout_seconds": {"type": "integer", "description": "Command timeout in seconds (default 30)."},
                        },
                        "required": [],
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
