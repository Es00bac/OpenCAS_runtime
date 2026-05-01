from __future__ import annotations

import asyncio
from types import SimpleNamespace

import pytest

from opencas.runtime.tool_registration_advanced_integrations import (
    register_advanced_integration_tools,
)
from opencas.platform import CapabilityRegistry, CapabilityStatus
from opencas.runtime.tool_runtime import (
    discover_and_register_mcp_tools,
    make_mcp_register_adapter,
    register_mcp_server_tools,
)
from opencas.tools import ToolRegistry


class _FakeMCPRegistry:
    def __init__(self) -> None:
        self._configs: dict[str, SimpleNamespace] = {}
        self._tools: dict[str, dict[str, dict[str, object]]] = {}
        self._initialized: set[str] = set()
        self._ensure_results: dict[str, bool] = {}

    def set_server(
        self,
        server_name: str,
        tools: list[dict[str, object]],
        *,
        initialized: bool,
    ) -> None:
        self._configs[server_name] = SimpleNamespace(command=f"{server_name}-command")
        self._tools[server_name] = {tool["name"]: tool for tool in tools}
        self._ensure_results[server_name] = initialized
        if initialized:
            self._initialized.add(server_name)
        else:
            self._initialized.discard(server_name)

    async def ensure_initialized(self, server_name: str) -> bool:
        ok = self._ensure_results.get(server_name, False)
        if ok:
            self._initialized.add(server_name)
        return ok


def _make_runtime(fake_registry: _FakeMCPRegistry) -> SimpleNamespace:
    return SimpleNamespace(
        tools=ToolRegistry(),
        capability_registry=CapabilityRegistry(),
        ctx=SimpleNamespace(mcp_registry=fake_registry),
    )


@pytest.mark.asyncio
async def test_discover_and_register_mcp_tools_projects_current_tools() -> None:
    fake_registry = _FakeMCPRegistry()
    fake_registry.set_server(
        "filesystem",
        [
            {
                "name": "read_file",
                "description": "Read a file",
                "inputSchema": {"type": "object"},
            }
        ],
        initialized=True,
    )
    runtime = _make_runtime(fake_registry)

    registered = await discover_and_register_mcp_tools(runtime)

    assert registered == ["read_file"]

    descriptor = runtime.capability_registry.get("mcp:filesystem.read_file")
    assert descriptor is not None
    assert descriptor.status is CapabilityStatus.ENABLED
    assert descriptor.owner_id == "mcp:filesystem"
    assert descriptor.tool_names == ["read_file"]


@pytest.mark.asyncio
async def test_register_mcp_server_tools_replaces_failed_validation_on_recovery() -> None:
    fake_registry = _FakeMCPRegistry()
    fake_registry.set_server("filesystem", [], initialized=False)
    runtime = _make_runtime(fake_registry)

    registered = await register_mcp_server_tools(runtime, "filesystem")

    assert registered == []
    failure = runtime.capability_registry.get("mcp:filesystem")
    assert failure is not None
    assert failure.status is CapabilityStatus.FAILED_VALIDATION
    assert failure.validation_errors == ["MCP server initialization failed"]

    fake_registry.set_server(
        "filesystem",
        [
            {
                "name": "read_file",
                "description": "Read a file",
                "inputSchema": {"type": "object"},
            }
        ],
        initialized=True,
    )

    registered = await register_mcp_server_tools(runtime, "filesystem")

    assert registered == ["read_file"]
    assert runtime.capability_registry.get("mcp:filesystem") is None
    descriptor = runtime.capability_registry.get("mcp:filesystem.read_file")
    assert descriptor is not None
    assert descriptor.status is CapabilityStatus.ENABLED
    assert descriptor.owner_id == "mcp:filesystem"


@pytest.mark.asyncio
async def test_register_mcp_server_tools_clears_stale_runtime_tools_on_shrink_and_failure() -> None:
    fake_registry = _FakeMCPRegistry()
    fake_registry.set_server(
        "filesystem",
        [
            {
                "name": "alpha",
                "description": "Alpha tool",
                "inputSchema": {"type": "object"},
            },
            {
                "name": "beta",
                "description": "Beta tool",
                "inputSchema": {"type": "object"},
            },
        ],
        initialized=True,
    )
    runtime = _make_runtime(fake_registry)

    registered = await register_mcp_server_tools(runtime, "filesystem")
    assert registered == ["alpha", "beta"]
    assert runtime.tools.get("alpha") is not None
    assert runtime.tools.get("beta") is not None

    fake_registry.set_server(
        "filesystem",
        [
            {
                "name": "alpha",
                "description": "Alpha tool",
                "inputSchema": {"type": "object"},
            }
        ],
        initialized=True,
    )

    registered = await register_mcp_server_tools(runtime, "filesystem")
    assert registered == ["alpha"]
    assert runtime.tools.get("alpha") is not None
    assert runtime.tools.get("beta") is None

    fake_registry.set_server("filesystem", [], initialized=False)

    registered = await register_mcp_server_tools(runtime, "filesystem")
    assert registered == []
    assert runtime.tools.get("alpha") is None
    failure = runtime.capability_registry.get("mcp:filesystem")
    assert failure is not None
    assert failure.status is CapabilityStatus.FAILED_VALIDATION


@pytest.mark.asyncio
async def test_mcp_register_adapter_returns_failed_result_on_init_failure() -> None:
    fake_registry = _FakeMCPRegistry()
    fake_registry.set_server("filesystem", [], initialized=False)
    runtime = _make_runtime(fake_registry)

    adapter = make_mcp_register_adapter(runtime)
    result = await adapter("mcp_register_server_tools", {"server_name": "filesystem"})

    assert result.success is False
    assert "failed validation" in result.output
    assert result.metadata["server"] == "filesystem"
    assert result.metadata["validation_errors"] == ["MCP server initialization failed"]


@pytest.mark.asyncio
async def test_namespaced_mcp_tool_names_do_not_collapse() -> None:
    fake_registry = _FakeMCPRegistry()
    fake_registry.set_server(
        "filesystem",
        [
            {
                "name": "mcp__filesystem__alpha__read",
                "description": "Read alpha",
                "inputSchema": {"type": "object"},
            },
            {
                "name": "mcp__filesystem__beta__read",
                "description": "Read beta",
                "inputSchema": {"type": "object"},
            },
        ],
        initialized=True,
    )
    runtime = _make_runtime(fake_registry)

    registered = await register_mcp_server_tools(runtime, "filesystem")

    assert registered == [
        "mcp__filesystem__alpha__read",
        "mcp__filesystem__beta__read",
    ]
    assert runtime.capability_registry.get("mcp:filesystem.alpha__read") is not None
    assert runtime.capability_registry.get("mcp:filesystem.beta__read") is not None
    assert runtime.capability_registry.get("mcp:filesystem.read") is None


@pytest.mark.asyncio
async def test_register_advanced_integration_tools_fires_mcp_auto_register_without_blocking(
    monkeypatch,
) -> None:
    seen: list[str] = []
    seen_events: list[tuple[str, dict[str, object]]] = []
    done = asyncio.Event()

    async def discover_and_register() -> list[str]:
        await asyncio.sleep(0)
        seen.append("discovered")
        done.set()
        return ["alpha", "beta"]

    runtime = SimpleNamespace(
        ctx=SimpleNamespace(
            mcp_registry=SimpleNamespace(),
            config=SimpleNamespace(mcp_auto_register=True),
        ),
        _discover_and_register_mcp_tools=discover_and_register,
        _make_mcp_list_servers_adapter=lambda: "list_servers",
        _make_mcp_register_adapter=lambda: "register_server",
        _trace=lambda event, payload: seen_events.append((event, payload)),
    )

    monkeypatch.setattr(
        "opencas.runtime.tool_registration_advanced_integrations.register_tool_specs",
        lambda *_args, **_kwargs: None,
    )
    register_advanced_integration_tools(runtime)
    await asyncio.wait_for(done.wait(), 1)

    assert "discovered" in seen
    assert seen_events == [("mcp_auto_registered", {"tool_count": 2})]
