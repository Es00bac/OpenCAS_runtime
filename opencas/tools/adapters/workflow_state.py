"""Workflow state adapter for higher-level project and writing inspection."""

from __future__ import annotations

import json
from typing import Any, Dict

from ..models import ToolResult


class WorkflowStateToolAdapter:
    """Expose higher-level workflow state from the runtime and stores."""

    def __init__(self, runtime: Any) -> None:
        self.runtime = runtime

    async def __call__(self, name: str, args: Dict[str, Any]) -> ToolResult:
        if name != "workflow_status":
            return ToolResult(False, f"Unknown workflow state tool: {name}", {})
        limit = int(args.get("limit", 10))
        project_id = args.get("project_id")
        payload = await self.runtime.workflow_status(
            limit=limit,
            project_id=str(project_id) if project_id is not None else None,
        )
        return ToolResult(
            success=True,
            output=json.dumps(payload),
            metadata={"source": "runtime_workflow_state", "limit": limit},
        )
