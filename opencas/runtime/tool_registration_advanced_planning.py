"""Interactive and planning tool registration helpers for AgentRuntime."""

from __future__ import annotations

from typing import Any

from opencas.autonomy.models import ActionRiskTier
from opencas.tools.adapters.interactive import InteractiveToolAdapter
from opencas.tools.adapters.plan import PlanToolAdapter

from .tool_registration_specs import ToolRegistrationSpec, register_tool_specs


def register_advanced_planning_tools(runtime: Any) -> None:
    interactive = InteractiveToolAdapter()
    register_tool_specs(
        runtime,
        interactive,
        [
            ToolRegistrationSpec(
                name="ask_user_question",
                description="Ask the user a clarifying question and pause the loop.",
                risk_tier=ActionRiskTier.READONLY,
                schema={
                    "type": "object",
                    "properties": {
                        "question": {
                            "type": "string",
                            "description": "The question to present to the user.",
                        },
                    },
                    "required": ["question"],
                },
            ),
        ],
    )

    plan_store = getattr(runtime.ctx, "plan_store", None)
    plan = PlanToolAdapter(store=plan_store)
    register_tool_specs(
        runtime,
        plan,
        [
            ToolRegistrationSpec(
                name="enter_plan_mode",
                description="Enter a constrained planning phase where only read tools and plan writes are allowed.",
                risk_tier=ActionRiskTier.READONLY,
                schema={
                    "type": "object",
                    "properties": {
                        "plan_id": {
                            "type": "string",
                            "description": "Optional identifier for the plan.",
                        },
                        "content": {
                            "type": "string",
                            "description": "Initial plan content.",
                        },
                        "project_id": {
                            "type": "string",
                            "description": "Optional project association.",
                        },
                        "task_id": {
                            "type": "string",
                            "description": "Optional task association.",
                        },
                    },
                    "required": [],
                },
            ),
            ToolRegistrationSpec(
                name="exit_plan_mode",
                description="Exit planning phase and resume normal tool access.",
                risk_tier=ActionRiskTier.READONLY,
                schema={
                    "type": "object",
                    "properties": {
                        "plan_id": {
                            "type": "string",
                            "description": "Optional identifier for the plan.",
                        },
                        "content": {
                            "type": "string",
                            "description": "Final plan content to save.",
                        },
                    },
                    "required": [],
                },
            ),
        ],
    )
