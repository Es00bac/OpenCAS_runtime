"""Lazy-loading MCP registry for on-demand tool integration."""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)


@dataclass
class MCPServerConfig:
    """Configuration for a single MCP stdio server."""

    name: str
    command: str
    args: List[str]
    env: Optional[Dict[str, str]] = None


class MCPRegistry:
    """Registry that lazily initializes MCP servers and caches discovered tools."""

    def __init__(self, configs: List[MCPServerConfig]) -> None:
        self._configs = {cfg.name: cfg for cfg in configs}
        self._sessions: Dict[str, Any] = {}
        self._tools: Dict[str, Dict[str, Any]] = {}
        self._initialized: set[str] = set()

    def list_all_tools(self) -> List[Dict[str, Any]]:
        """Return metadata for tools from already-initialized servers only."""
        tools: List[Dict[str, Any]] = []
        for server_name in self._initialized:
            for tool_meta in self._tools.get(server_name, {}).values():
                tools.append({**tool_meta, "server": server_name})
        return tools

    async def ensure_initialized(self, server_name: str) -> bool:
        """Start MCP session and discover tools for a specific server."""
        if server_name in self._initialized:
            return True
        cfg = self._configs.get(server_name)
        if cfg is None:
            return False
        try:
            from .mcp_client import MCPClientWrapper
            client = MCPClientWrapper(command=cfg.command, args=cfg.args, env=cfg.env)
            ok = await client.connect()
            if not ok:
                return False
            tools = await client.list_tools()
            self._sessions[server_name] = client
            self._tools[server_name] = {t["name"]: t for t in tools}
            self._initialized.add(server_name)
            return True
        except Exception as exc:
            logger.warning("Failed to initialize MCP server %s: %s", server_name, exc)
            return False

    async def call_tool(self, server_name: str, tool_name: str, args: Dict[str, Any]) -> Dict[str, Any]:
        """Ensure server is initialized, then call the tool."""
        ok = await self.ensure_initialized(server_name)
        if not ok:
            return {"isError": True, "content": f"MCP server {server_name} not available"}
        client = self._sessions.get(server_name)
        if client is None:
            return {"isError": True, "content": f"MCP session for {server_name} missing"}
        return await client.call_tool(tool_name, args)

    async def close(self) -> None:
        for client in list(self._sessions.values()):
            try:
                await client.close()
            except Exception:
                pass
        self._sessions.clear()
