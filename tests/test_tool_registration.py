"""Focused tests for runtime default tool registration."""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import pytest

from opencas.platform import CapabilityRegistry, CapabilitySource, CapabilityStatus
from opencas.runtime.runtime_setup import initialize_runtime_execution
from opencas.runtime.tool_registration import register_runtime_default_tools
from opencas.sandbox import SandboxMode
from opencas.tools import ToolRegistry
from opencas.tools.adapters.workflow import WorkflowToolAdapter


class _Config:
    def __init__(self, root: Path) -> None:
        self._root = root

    def primary_workspace_root(self) -> Path:
        return self._root


class _Runtime(SimpleNamespace):
    def _make_mcp_list_servers_adapter(self):
        return lambda name, args: None

    def _make_mcp_register_adapter(self):
        return lambda name, args: None

    def _trace(self, event, payload=None):
        return None


def test_register_runtime_default_tools_exposes_expected_surface(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        "opencas.runtime.tool_registration_advanced_integrations.google_workspace_cli_available",
        lambda: True,
    )
    runtime = _Runtime(
        ctx=SimpleNamespace(
            sandbox=SimpleNamespace(allowed_roots=[tmp_path], mode=SandboxMode.OFF),
            config=_Config(tmp_path),
        ),
        tools=ToolRegistry(),
        process_supervisor=SimpleNamespace(),
        pty_supervisor=SimpleNamespace(),
        browser_supervisor=SimpleNamespace(),
        tracer=object(),
        baa=SimpleNamespace(),
        executive=SimpleNamespace(),
        creative=SimpleNamespace(),
    )

    register_runtime_default_tools(runtime)

    tool_names = {tool.name for tool in runtime.tools.list_tools()}
    assert runtime.tools.validation_pipeline is not None
    assert "fs_read_file" in tool_names
    assert "cli_discover_command" in tool_names
    assert "workflow_supervise_session" in tool_names
    assert "workflow_cancel_project" in tool_names
    assert "workflow_get_schedule" in tool_names
    assert "workflow_cancel_schedule" in tool_names
    assert "workflow_cancel_task" in tool_names
    assert "workflow_list_plans" in tool_names
    assert "workflow_get_plan" in tool_names
    assert "workflow_get_commitment" in tool_names
    assert "workflow_list_tasks" in tool_names
    assert "workflow_get_task" in tool_names
    assert "proof_chain_promise_lookup" in tool_names
    assert "runtime_status" in tool_names
    assert "workflow_status" in tool_names
    assert "thread_registry_query" in tool_names
    assert "thread_registry_create_candidate" in tool_names
    assert "browser_start" in tool_names
    assert "tui_playwright_open" in tool_names
    assert "tui_playwright_input" in tool_names
    assert "tui_playwright_keys" in tool_names
    assert "tui_playwright_resize" in tool_names
    assert "tui_playwright_snapshot" in tool_names
    assert "tui_playwright_close" in tool_names
    assert "google_workspace_auth_status" in tool_names
    assert "phone_get_status" in tool_names
    assert "phone_call_owner" in tool_names
    assert "initiative_contact_owner" in tool_names
    assert "initiative_contact_status" in tool_names
    assert runtime.tools.get("browser_start").risk_tier.value == "readonly"
    assert runtime.tools.get("browser_click").risk_tier.value == "external_write"
    assert runtime.tools.get("tui_playwright_open").risk_tier.value == "shell_local"
    assert runtime.tools.get("tui_playwright_input").risk_tier.value == "shell_local"
    assert runtime.tools.get("tui_playwright_keys").risk_tier.value == "shell_local"
    assert runtime.tools.get("tui_playwright_resize").risk_tier.value == "shell_local"
    assert runtime.tools.get("tui_playwright_snapshot").risk_tier.value == "shell_local"
    assert runtime.tools.get("tui_playwright_close").risk_tier.value == "shell_local"
    assert runtime.tools.get("google_workspace_gmail_headlines").risk_tier.value == "readonly"
    assert runtime.tools.get("google_workspace_calendar_dedupe").risk_tier.value == "external_write"
    assert runtime.tools.get("phone_get_status").risk_tier.value == "readonly"
    assert runtime.tools.get("phone_call_owner").risk_tier.value == "external_write"
    assert runtime.tools.get("initiative_contact_owner").risk_tier.value == "external_write"
    assert runtime.tools.get("initiative_contact_status").risk_tier.value == "readonly"
    assert "Bulma" not in runtime.tools.get("phone_call_owner").description
    assert "Bulma" not in runtime.tools.get("initiative_contact_owner").description
    assert runtime.tools.get("thread_registry_query").risk_tier.value == "readonly"
    assert runtime.tools.get("thread_registry_create_candidate").risk_tier.value == "workspace_write"
    assert runtime.tools.get("proof_chain_promise_lookup").risk_tier.value == "readonly"
    schedule_tool = runtime.tools.get("workflow_create_schedule")
    assert schedule_tool is not None
    assert "gmail_alert" in schedule_tool.parameters["properties"]["action"]["enum"]
    assert "Gmail" in schedule_tool.parameters["properties"]["action"]["description"]
    assert all(tool.description.strip() for tool in runtime.tools.list_tools())

    assert {
        "workflow_create_schedule",
        "workflow_update_schedule",
        "workflow_cancel_schedule",
        "workflow_list_schedules",
        "workflow_get_schedule",
    } <= tool_names
    assert {
        "workflow_create_plan",
        "workflow_update_plan",
        "workflow_list_plans",
        "workflow_get_plan",
    } <= tool_names
    assert {
        "workflow_create_commitment",
        "workflow_update_commitment",
        "workflow_list_commitments",
        "workflow_get_commitment",
    } <= tool_names
    assert {
        "workflow_cancel_task",
        "workflow_list_tasks",
        "workflow_get_task",
    } <= tool_names


@pytest.mark.asyncio
async def test_registered_workflow_tools_have_adapter_handlers(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        "opencas.runtime.tool_registration_advanced_integrations.google_workspace_cli_available",
        lambda: False,
    )
    runtime = _Runtime(
        ctx=SimpleNamespace(
            sandbox=SimpleNamespace(allowed_roots=[tmp_path], mode=SandboxMode.OFF),
            config=_Config(tmp_path),
        ),
        tools=ToolRegistry(),
        process_supervisor=SimpleNamespace(),
        pty_supervisor=SimpleNamespace(),
        browser_supervisor=SimpleNamespace(),
        tracer=object(),
        baa=SimpleNamespace(),
        executive=SimpleNamespace(),
        creative=SimpleNamespace(),
    )

    register_runtime_default_tools(runtime)
    workflow_tools = sorted(
        entry.name
        for entry in runtime.tools.list_tools()
        if isinstance(entry.adapter, WorkflowToolAdapter)
    )
    adapter = WorkflowToolAdapter(runtime=runtime)
    missing_handlers: list[str] = []
    for tool_name in workflow_tools:
        result = await adapter(tool_name, {})
        if not result.success and str(result.output).startswith("Unknown workflow tool:"):
            missing_handlers.append(tool_name)

    assert missing_handlers == []


def test_register_runtime_default_tools_discovers_gws_from_user_bin(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    home = tmp_path / "home"
    bin_dir = home / ".npm-global" / "bin"
    bin_dir.mkdir(parents=True)
    fake_gws = bin_dir / "gws"
    fake_gws.write_text("#!/bin/sh\necho '{}'\n", encoding="utf-8")
    fake_gws.chmod(0o755)
    monkeypatch.setenv("HOME", str(home))
    monkeypatch.setenv("PATH", "/usr/bin")
    runtime = _Runtime(
        ctx=SimpleNamespace(
            sandbox=SimpleNamespace(allowed_roots=[tmp_path], mode=SandboxMode.OFF),
            config=_Config(tmp_path),
        ),
        tools=ToolRegistry(),
        process_supervisor=SimpleNamespace(),
        pty_supervisor=SimpleNamespace(),
        browser_supervisor=SimpleNamespace(),
        tracer=object(),
        baa=SimpleNamespace(),
        executive=SimpleNamespace(),
        creative=SimpleNamespace(),
    )

    register_runtime_default_tools(runtime)

    tool_names = {tool.name for tool in runtime.tools.list_tools()}
    assert "google_workspace_auth_status" in tool_names
    assert "google_workspace_calendar_schedule" in tool_names
    assert "google_workspace_calendar_dedupe" in tool_names


def test_register_runtime_default_tools_emits_core_capabilities(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        "opencas.runtime.tool_registration_advanced_integrations.google_workspace_cli_available",
        lambda: True,
    )
    monkeypatch.setattr(
        "opencas.runtime.tool_registration_advanced_integrations.himalaya_cli_available",
        lambda: True,
    )
    runtime = _Runtime(
        ctx=SimpleNamespace(
            sandbox=SimpleNamespace(allowed_roots=[tmp_path], mode=SandboxMode.OFF),
            config=_Config(tmp_path),
        ),
        tools=ToolRegistry(),
        capability_registry=CapabilityRegistry(),
        process_supervisor=SimpleNamespace(),
        pty_supervisor=SimpleNamespace(),
        browser_supervisor=SimpleNamespace(),
        tracer=object(),
        baa=SimpleNamespace(),
        executive=SimpleNamespace(),
        creative=SimpleNamespace(),
    )

    register_runtime_default_tools(runtime)

    fs_read = runtime.capability_registry.get("core:fs_read_file")
    cli_discover = runtime.capability_registry.get("core:cli_discover_command")
    bash_run = runtime.capability_registry.get("core:bash_run_command")
    gmail_headlines = runtime.capability_registry.get("core:google_workspace_gmail_headlines")
    himalaya_email = runtime.capability_registry.get("core:email.himalaya_readonly")
    phone_status = runtime.capability_registry.get("core:phone_get_status")
    phone_call = runtime.capability_registry.get("core:phone_call_owner")
    contact_status = runtime.capability_registry.get("core:initiative_contact_status")
    contact_owner = runtime.capability_registry.get("core:initiative_contact_owner")

    assert fs_read is not None
    assert fs_read.display_name == "fs_read_file"
    assert fs_read.kind == "tool"
    assert fs_read.owner_id == "core"
    assert fs_read.tool_names == ["fs_read_file"]
    assert fs_read.metadata["risk_tier"] == "readonly"
    assert "offset" in fs_read.config_schema["properties"]
    assert "limit" in fs_read.config_schema["properties"]
    assert "character offset" in fs_read.config_schema["properties"]["offset"]["description"]
    assert "characters" in fs_read.config_schema["properties"]["limit"]["description"]
    assert "read_session_id" in fs_read.config_schema["properties"]
    assert "concept_scope" in fs_read.config_schema["properties"]
    assert "concept_label" in fs_read.config_schema["properties"]

    assert bash_run is not None
    assert bash_run.display_name == "bash_run_command"
    assert bash_run.kind == "tool"
    assert bash_run.owner_id == "core"
    assert bash_run.tool_names == ["bash_run_command"]
    assert bash_run.metadata["risk_tier"] == "shell_local"

    assert cli_discover is not None
    assert cli_discover.tool_names == ["cli_discover_command"]
    assert cli_discover.metadata["risk_tier"] == "readonly"

    assert gmail_headlines is not None
    assert gmail_headlines.source is CapabilitySource.CORE
    assert gmail_headlines.status is CapabilityStatus.ENABLED
    assert gmail_headlines.metadata["risk_tier"] == "readonly"

    assert himalaya_email is not None
    assert himalaya_email.source is CapabilitySource.CORE
    assert himalaya_email.status is CapabilityStatus.ENABLED
    assert himalaya_email.owner_id == "advanced_integrations"
    assert himalaya_email.tool_names == [
        "himalaya_email_accounts",
        "himalaya_email_headlines",
        "himalaya_email_read_message",
    ]
    assert himalaya_email.metadata["write_actions"] is False

    assert phone_status is not None
    assert phone_status.tool_names == ["phone_get_status"]
    assert phone_status.metadata["risk_tier"] == "readonly"

    assert phone_call is not None
    assert phone_call.tool_names == ["phone_call_owner"]
    assert phone_call.metadata["risk_tier"] == "external_write"

    assert contact_status is not None
    assert contact_status.tool_names == ["initiative_contact_status"]
    assert contact_status.metadata["risk_tier"] == "readonly"

    assert contact_owner is not None
    assert contact_owner.tool_names == ["initiative_contact_owner"]
    assert contact_owner.metadata["risk_tier"] == "external_write"


def test_initialize_runtime_execution_attaches_registry_and_emits_capabilities(tmp_path: Path) -> None:
    tracer = SimpleNamespace(log=lambda *args, **kwargs: None)
    shared_registry = CapabilityRegistry()
    stale_registry = CapabilityRegistry()
    plugin_lifecycle = SimpleNamespace(
        tools=ToolRegistry(),
        capability_registry=stale_registry,
    )
    runtime = _Runtime(
        ctx=SimpleNamespace(
            sandbox=SimpleNamespace(allowed_roots=[tmp_path], mode=SandboxMode.OFF),
            config=_Config(tmp_path),
            hook_bus=None,
            capability_registry=shared_registry,
            plugin_lifecycle=plugin_lifecycle,
        ),
        tracer=tracer,
        approval=SimpleNamespace(),
        orchestrator=SimpleNamespace(),
        llm=SimpleNamespace(),
        capability_registry=stale_registry,
        _register_default_tools=lambda: register_runtime_default_tools(runtime),
        _register_skills=lambda: None,
    )
    context = SimpleNamespace(
        tasks=None,
        event_bus=None,
        receipt_store=None,
        memory=None,
        embeddings=None,
        plugin_lifecycle=plugin_lifecycle,
        capability_registry=shared_registry,
    )

    initialize_runtime_execution(runtime, context)

    assert runtime.capability_registry is shared_registry
    assert runtime.plugin_lifecycle.capability_registry is shared_registry

    fs_read = shared_registry.get("core:fs_read_file")
    bash_run = shared_registry.get("core:bash_run_command")

    assert fs_read is not None
    assert fs_read.source is CapabilitySource.CORE
    assert fs_read.status is CapabilityStatus.ENABLED
    assert fs_read.description == "Read the contents of a file"
    assert fs_read.config_schema["properties"]["file_path"]["type"] == "string"

    assert bash_run is not None
    assert bash_run.source is CapabilitySource.CORE
    assert bash_run.status is CapabilityStatus.ENABLED
