"""Focused tests for runtime default tool registration."""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

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


def test_register_runtime_default_tools_exposes_expected_surface(tmp_path: Path) -> None:
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
