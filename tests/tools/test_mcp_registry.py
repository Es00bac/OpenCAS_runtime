"""Tests for MCPRegistry lazy loading."""

import pytest

from opencas.tools.mcp_registry import MCPRegistry, MCPServerConfig


def test_list_all_tools_returns_empty_when_no_servers():
    reg = MCPRegistry([])
    assert reg.list_all_tools() == []


def test_list_all_tools_returns_empty_until_initialized():
    reg = MCPRegistry([MCPServerConfig(name="fetch", command="echo", args=["hello"])])
    assert reg.list_all_tools() == []


@pytest.mark.asyncio
async def test_ensure_initialized_fails_for_missing_command():
    reg = MCPRegistry([MCPServerConfig(name="bad", command="/nonexistent/command", args=[])])
    ok = await reg.ensure_initialized("bad")
    assert ok is False


@pytest.mark.asyncio
async def test_call_tool_fails_when_server_not_configured():
    reg = MCPRegistry([])
    result = await reg.call_tool("missing", "tool", {})
    assert result["isError"] is True
    assert "not available" in result["content"]


@pytest.mark.asyncio
async def test_close_is_safe_when_no_sessions():
    reg = MCPRegistry([])
    await reg.close()
