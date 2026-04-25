"""Adapter factory for MCP tools to the OpenCAS ToolRegistry interface."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, Dict

from .models import ToolResult

if TYPE_CHECKING:
    from .mcp_registry import MCPRegistry


def make_mcp_tool_adapter(registry: "MCPRegistry", server_name: str, tool_name: str):
    """Return an async adapter callable for an MCP tool."""

    async def adapter(name: str, args: Dict[str, Any]) -> ToolResult:
        result = await registry.call_tool(server_name, tool_name, args)
        return ToolResult(
            success=not result.get("isError", False),
            output=result.get("content", ""),
            metadata={"server": server_name, "tool": tool_name},
        )

    return adapter
