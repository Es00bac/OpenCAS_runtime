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
    assert "workflow_supervise_session" in tool_names
    assert "runtime_status" in tool_names
    assert "workflow_status" in tool_names
    assert "browser_start" in tool_names
    assert "google_workspace_auth_status" in tool_names
    assert "phone_get_status" in tool_names
    assert "phone_call_owner" in tool_names
    assert "initiative_contact_owner" in tool_names
    assert "initiative_contact_status" in tool_names
    assert runtime.tools.get("browser_start").risk_tier.value == "readonly"
    assert runtime.tools.get("browser_click").risk_tier.value == "external_write"
    assert runtime.tools.get("google_workspace_gmail_headlines").risk_tier.value == "readonly"
    assert runtime.tools.get("phone_get_status").risk_tier.value == "readonly"
    assert runtime.tools.get("phone_call_owner").risk_tier.value == "external_write"
    assert runtime.tools.get("initiative_contact_owner").risk_tier.value == "external_write"
    assert runtime.tools.get("initiative_contact_status").risk_tier.value == "readonly"


def test_register_runtime_default_tools_emits_core_capabilities(
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
    bash_run = runtime.capability_registry.get("core:bash_run_command")
    gmail_headlines = runtime.capability_registry.get("core:google_workspace_gmail_headlines")
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

    assert bash_run is not None
    assert bash_run.display_name == "bash_run_command"
    assert bash_run.kind == "tool"
    assert bash_run.owner_id == "core"
    assert bash_run.tool_names == ["bash_run_command"]
    assert bash_run.metadata["risk_tier"] == "shell_local"

    assert gmail_headlines is not None
    assert gmail_headlines.source is CapabilitySource.CORE
    assert gmail_headlines.status is CapabilityStatus.ENABLED
    assert gmail_headlines.metadata["risk_tier"] == "readonly"

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
