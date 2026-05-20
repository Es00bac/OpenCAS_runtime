"""Runtime integration and workspace tool registration helpers for AgentRuntime."""

from __future__ import annotations

import asyncio
import threading
from typing import Any

from opencas.autonomy.models import ActionRiskTier
from opencas.identity.agent_name import resolve_agent_name
from opencas.platform import CapabilityDescriptor, CapabilitySource, CapabilityStatus
from opencas.tools.adapters.agent import AgentToolAdapter
from opencas.tools.adapters.google_workspace import (
    GoogleWorkspaceToolAdapter,
    google_workspace_cli_available,
)
from opencas.tools.adapters.himalaya_email import (
    HimalayaEmailToolAdapter,
    himalaya_cli_available,
)
from opencas.tools.adapters.initiative_contact import InitiativeContactToolAdapter
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
    agent_name = resolve_agent_name(
        runtime=runtime,
        default="the active OpenCAS agent",
    )
    mcp_registry = getattr(runtime.ctx, "mcp_registry", None)
    if mcp_registry is not None and runtime.ctx.config.mcp_auto_register:
        async def _auto_register_mcp_tools() -> None:
            try:
                tools = await runtime._discover_and_register_mcp_tools()
                runtime._trace("mcp_auto_registered", {"tool_count": len(tools)})
            except Exception as exc:  # pragma: no cover - exercised by runtime failure paths
                runtime._trace("mcp_auto_register_failed", {"error": str(exc)})

        try:
            loop = asyncio.get_running_loop()
        except RuntimeError:
            loop = None
        if loop is not None and getattr(loop, "_thread_id", None) == threading.get_ident():
            loop.create_task(_auto_register_mcp_tools())
        elif loop is not None:
            asyncio.run_coroutine_threadsafe(_auto_register_mcp_tools(), loop)
        else:
            # If no running loop, registration happens inline during startup.
            loop = asyncio.new_event_loop()
            try:
                loop.run_until_complete(_auto_register_mcp_tools())
            finally:
                loop.close()

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
                description=(
                    "Place an outbound phone call to the trusted owner number only. "
                    f"Use this when {agent_name} genuinely needs to reach the operator by voice."
                ),
                risk_tier=ActionRiskTier.EXTERNAL_WRITE,
                schema={
                    "type": "object",
                    "properties": {
                        "message": {
                            "type": "string",
                            "description": f"What {agent_name} should say when the owner answers.",
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

    initiative_contact = InitiativeContactToolAdapter(runtime)
    register_tool_specs(
        runtime,
        initiative_contact,
        [
            ToolRegistrationSpec(
                name="initiative_contact_status",
                description="Inspect OpenCAS's policy-limited owner contact state, recent events, and daily contact count.",
                risk_tier=ActionRiskTier.READONLY,
                schema={"type": "object", "properties": {}, "required": []},
            ),
            ToolRegistrationSpec(
                name="initiative_contact_owner",
                description=(
                    "Send a policy-limited owner notification through the trusted "
                    "initiative-contact channel. "
                    f"Use when {agent_name} genuinely wants to reach out or thinks "
                    "the operator should know something."
                ),
                risk_tier=ActionRiskTier.EXTERNAL_WRITE,
                schema={
                    "type": "object",
                    "properties": {
                        "message": {
                            "type": "string",
                            "description": "Message to send to the owner.",
                        },
                        "reason": {
                            "type": "string",
                            "description": "Short reason for the contact request.",
                        },
                        "urgency": {
                            "type": "string",
                            "enum": ["low", "normal", "high", "critical"],
                            "description": "Urgency used by the contact policy.",
                        },
                        "channel": {
                            "type": "string",
                            "enum": ["auto", "telegram", "phone"],
                            "description": "Preferred delivery channel. Auto defaults to Telegram.",
                        },
                    },
                    "required": ["message"],
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
                    name="google_workspace_calendar_dedupe",
                    description=(
                        "Find exact duplicate Google Calendar events and optionally delete "
                        "duplicate copies after a dry-run review."
                    ),
                    risk_tier=ActionRiskTier.EXTERNAL_WRITE,
                    schema={
                        "type": "object",
                        "properties": {
                            "calendar_id": {"type": "string", "description": "Calendar id, default primary."},
                            "time_min": {"type": "string", "description": "Optional RFC3339/ISO-8601 lower bound."},
                            "time_max": {"type": "string", "description": "Optional RFC3339/ISO-8601 upper bound."},
                            "scan_past_days": {
                                "type": "integer",
                                "description": "Default scan lookback when time_min is omitted, max 3650.",
                            },
                            "scan_future_days": {
                                "type": "integer",
                                "description": "Default scan lookahead when time_max is omitted, max 3650.",
                            },
                            "max_results": {
                                "type": "integer",
                                "description": "Maximum events to scan, default/max 2500.",
                            },
                            "max_deletions": {
                                "type": "integer",
                                "description": "Safety cap for apply=true deletions, default 25.",
                            },
                            "apply": {
                                "type": "boolean",
                                "description": "False performs a dry run; true deletes exact duplicate candidates.",
                            },
                            "send_updates": {
                                "type": "string",
                                "enum": ["none", "externalOnly", "all"],
                                "description": "Calendar notification behavior for deletions; default none.",
                            },
                            "timeout_seconds": {
                                "type": "integer",
                                "description": "Command timeout in seconds (default 30).",
                            },
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

    if himalaya_cli_available():
        himalaya_tool_names = [
            "himalaya_email_accounts",
            "himalaya_email_headlines",
            "himalaya_email_read_message",
        ]
        himalaya_email = HimalayaEmailToolAdapter()
        register_tool_specs(
            runtime,
            himalaya_email,
            [
                ToolRegistrationSpec(
                    name="himalaya_email_accounts",
                    description="List locally configured Himalaya email accounts before choosing an email backend.",
                    risk_tier=ActionRiskTier.READONLY,
                    schema={"type": "object", "properties": {"timeout_seconds": {"type": "integer"}}, "required": []},
                ),
                ToolRegistrationSpec(
                    name="himalaya_email_headlines",
                    description=(
                        "List recent email envelopes through the local Himalaya IMAP CLI. "
                        "Use for email triage, especially when Gmail/gws auth is unavailable or the operator asks generally about email."
                    ),
                    risk_tier=ActionRiskTier.READONLY,
                    schema={
                        "type": "object",
                        "properties": {
                            "folder": {"type": "string", "description": "Mail folder, default INBOX."},
                            "query": {"type": "string", "description": "Himalaya envelope query, default 'order by date desc'."},
                            "page_size": {"type": "integer", "description": "Maximum envelopes, default 10, max 50."},
                            "page": {"type": "integer", "description": "Page number, default 1."},
                            "account": {"type": "string", "description": "Optional Himalaya account name."},
                            "timeout_seconds": {"type": "integer", "description": "Command timeout in seconds (default 30)."},
                        },
                        "required": [],
                    },
                ),
                ToolRegistrationSpec(
                    name="himalaya_email_read_message",
                    description="Preview-read an email message by Himalaya envelope id without marking it seen.",
                    risk_tier=ActionRiskTier.READONLY,
                    schema={
                        "type": "object",
                        "properties": {
                            "message_id": {"type": "string", "description": "Himalaya envelope/message id."},
                            "folder": {"type": "string", "description": "Mail folder, default INBOX."},
                            "account": {"type": "string", "description": "Optional Himalaya account name."},
                            "no_headers": {"type": "boolean", "description": "Return only the message body."},
                            "timeout_seconds": {"type": "integer", "description": "Command timeout in seconds (default 30)."},
                        },
                        "required": ["message_id"],
                    },
                ),
            ],
        )
        capability_registry = getattr(runtime, "capability_registry", None) or getattr(runtime.ctx, "capability_registry", None)
        if capability_registry is not None:
            capability_registry.register(
                CapabilityDescriptor(
                    capability_id="core:email.himalaya_readonly",
                    display_name="Himalaya read-only email",
                    kind="tool",
                    source=CapabilitySource.CORE,
                    owner_id="advanced_integrations",
                    status=CapabilityStatus.ENABLED,
                    description=(
                        "Read-only local email inspection through the configured Himalaya CLI; "
                        "used for account listing, inbox headlines, and message previews."
                    ),
                    tool_names=himalaya_tool_names,
                    declared_dependencies=["himalaya"],
                    metadata={"backend": "himalaya", "write_actions": False, "preview_only": True},
                )
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
