"""Higher-level operator workflow tools.

These composite tools let the LLM operate at task level rather than
choreographing low-level tool calls manually. Each tool wraps underlying
stores (commitments, work objects, plans) and runtime capabilities.
"""

from __future__ import annotations

from typing import Any, Dict

from ..models import ToolResult
from .workflow_supervision import supervise_session
from .workflow_tasking import (
    create_commitment,
    create_plan,
    create_schedule,
    create_writing_task,
    list_commitments,
    list_schedules,
    repo_triage,
    update_commitment,
    update_plan,
    update_schedule,
)


class WorkflowToolAdapter:
    """Composite workflow tools for writing, project management, and supervision."""

    def __init__(self, runtime: Any) -> None:
        self.runtime = runtime

    async def __call__(self, name: str, args: Dict[str, Any]) -> ToolResult:
        handler = {
            "workflow_create_commitment": create_commitment,
            "workflow_update_commitment": update_commitment,
            "workflow_list_commitments": list_commitments,
            "workflow_create_schedule": create_schedule,
            "workflow_update_schedule": update_schedule,
            "workflow_list_schedules": list_schedules,
            "workflow_create_writing_task": create_writing_task,
            "workflow_create_plan": create_plan,
            "workflow_update_plan": update_plan,
            "workflow_repo_triage": repo_triage,
            "workflow_supervise_session": supervise_session,
        }.get(name)
        if handler is None:
            return ToolResult(False, f"Unknown workflow tool: {name}", {})
        try:
            return await handler(self.runtime, args)
        except Exception as exc:
            return ToolResult(False, str(exc), {"error_type": type(exc).__name__})
