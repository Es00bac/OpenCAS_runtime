"""Runtime and workflow-state tool registration for AgentRuntime."""

from __future__ import annotations

from typing import Any

from opencas.autonomy.models import ActionRiskTier
from opencas.tools.adapters.runtime_state import RuntimeStateToolAdapter
from opencas.tools.adapters.workflow_state import WorkflowStateToolAdapter

from .tool_registration_specs import ToolRegistrationSpec, register_tool_specs


def register_workflow_state_tools(runtime: Any) -> None:
    runtime_state = RuntimeStateToolAdapter(runtime=runtime)
    register_tool_specs(
        runtime,
        runtime_state,
        [
            ToolRegistrationSpec(
                name="runtime_status",
                description="Return workspace, sandbox, and execution control-plane state.",
                risk_tier=ActionRiskTier.READONLY,
                schema={"type": "object", "properties": {}, "required": []},
            )
        ],
    )

    workflow_state = WorkflowStateToolAdapter(runtime=runtime)
    register_tool_specs(
        runtime,
        workflow_state,
        [
            ToolRegistrationSpec(
                name="workflow_status",
                description="Return higher-level workflow state including goals, commitments, plans, work objects, and receipts.",
                risk_tier=ActionRiskTier.READONLY,
                schema={
                    "type": "object",
                    "properties": {
                        "limit": {"type": "integer", "description": "Maximum items to include per section."},
                        "project_id": {"type": "string", "description": "Optional project id to focus the workflow summary."},
                    },
                    "required": [],
                },
            )
        ],
    )
