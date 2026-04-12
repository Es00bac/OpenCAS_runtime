"""Runtime state adapter for agent-visible control-plane inspection."""

from __future__ import annotations

import json
from typing import Any, Dict

from ..models import ToolResult


class RuntimeStateToolAdapter:
    """Expose runtime control-plane state to the agent as a readonly tool."""

    def __init__(self, runtime: Any) -> None:
        self.runtime = runtime

    def __call__(self, name: str, args: Dict[str, Any]) -> ToolResult:
        if name != "runtime_status":
            return ToolResult(False, f"Unknown runtime state tool: {name}", {})
        snapshot = self.runtime.control_plane_status()
        return ToolResult(
            success=True,
            output=json.dumps(snapshot),
            metadata={"source": "runtime_control_plane"},
        )
