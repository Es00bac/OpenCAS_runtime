"""Thin MCP client wrapper with optional imports."""

from __future__ import annotations

from typing import Any, Dict, List, Optional

try:
    from mcp import ClientSession, StdioServerParameters
    from mcp.client.stdio import stdio_client
except ImportError:  # pragma: no cover
    ClientSession = None  # type: ignore[misc,assignment]
    StdioServerParameters = None  # type: ignore[misc,assignment]
    stdio_client = None  # type: ignore[misc,assignment]


class MCPClientWrapper:
    """Manages a single MCP stdio session."""

    def __init__(self, command: str, args: List[str], env: Optional[Dict[str, str]] = None) -> None:
        self.command = command
        self.args = args
        self.env = env
        self._session: Optional[Any] = None
        self._exit_stack: Optional[Any] = None

    async def connect(self) -> bool:
        if ClientSession is None or stdio_client is None or StdioServerParameters is None:
            return False
        import contextlib
        self._exit_stack = contextlib.AsyncExitStack()
        server_params = StdioServerParameters(
            command=self.command,
            args=self.args,
            env=self.env,
        )
        try:
            read_stream, write_stream = await self._exit_stack.enter_async_context(
                stdio_client(server_params)
            )
            session = await self._exit_stack.enter_async_context(
                ClientSession(read_stream, write_stream)
            )
            await session.initialize()
            self._session = session
            return True
        except Exception:
            await self._exit_stack.aclose()
            self._exit_stack = None
            return False

    async def list_tools(self) -> List[Dict[str, Any]]:
        if self._session is None:
            return []
        result = await self._session.list_tools()
        # result is a TypedDict-like object; convert to plain dicts
        tools: List[Dict[str, Any]] = []
        for tool in getattr(result, "tools", []):
            tools.append(
                {
                    "name": tool.name,
                    "description": getattr(tool, "description", ""),
                    "inputSchema": getattr(tool, "inputSchema", {"type": "object"}),
                }
            )
        return tools

    async def call_tool(self, tool_name: str, args: Dict[str, Any]) -> Dict[str, Any]:
        if self._session is None:
            return {"isError": True, "content": "MCP session not connected"}
        result = await self._session.call_tool(tool_name, args)
        is_error = False
        content_parts: List[str] = []
        for item in getattr(result, "content", []):
            if getattr(item, "type", None) == "text":
                content_parts.append(getattr(item, "text", ""))
        if getattr(result, "isError", False):
            is_error = True
        return {"isError": is_error, "content": "\n".join(content_parts)}

    async def close(self) -> None:
        if self._exit_stack is not None:
            await self._exit_stack.aclose()
            self._exit_stack = None
        self._session = None
