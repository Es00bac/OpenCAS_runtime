"""Runtime and workflow-state tool registration for AgentRuntime."""

from __future__ import annotations

from typing import Any

from opencas.autonomy.models import ActionRiskTier
from opencas.tools.adapters.proof_chain import ProofChainToolAdapter
from opencas.tools.adapters.runtime_state import RuntimeStateToolAdapter
from opencas.tools.adapters.self_inspection import SelfInspectionToolAdapter
from opencas.tools.adapters.thread_registry import ThreadRegistryToolAdapter
from opencas.tools.adapters.wellbeing import WellbeingToolAdapter
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

    proof_chain = ProofChainToolAdapter(runtime=runtime)
    register_tool_specs(
        runtime,
        proof_chain,
        [
            ToolRegistrationSpec(
                name="proof_chain_promise_lookup",
                description=(
                    "Return promise claims joined to commitments, schedules, BAA tasks, "
                    "execution receipts, and proof status since an optional ISO-8601 time."
                ),
                risk_tier=ActionRiskTier.READONLY,
                schema={
                    "type": "object",
                    "properties": {
                        "since": {
                            "type": "string",
                            "description": "Optional ISO-8601 lower bound for commitment creation time.",
                        },
                        "limit": {
                            "type": "integer",
                            "description": "Maximum promise records to return. Default 50.",
                        },
                    },
                    "required": [],
                },
            )
        ],
    )

    self_inspection = SelfInspectionToolAdapter(runtime=runtime)
    register_tool_specs(
        runtime,
        self_inspection,
        [
            ToolRegistrationSpec(
                name="self_inspection_query",
                description=(
                    "Search OpenCAS self-inspection records: response-shape drift, "
                    "tool-use intent, valence-source tags, and commitment gaps."
                ),
                risk_tier=ActionRiskTier.READONLY,
                schema={
                    "type": "object",
                    "properties": {
                        "query": {"type": "string", "description": "Optional term or phrase to search for."},
                        "session_id": {"type": "string", "description": "Optional session id filter."},
                        "limit": {"type": "integer", "description": "Maximum records to return. Default 10."},
                    },
                    "required": [],
                },
            )
        ],
    )

    wellbeing = WellbeingToolAdapter(runtime=runtime)
    register_tool_specs(
        runtime,
        wellbeing,
        [
            ToolRegistrationSpec(
                name="wellbeing_query",
                description=(
                    "Return OpenCAS operational wellbeing state, recent events, "
                    "maintenance recommendations, and review-required self-modification proposals."
                ),
                risk_tier=ActionRiskTier.READONLY,
                schema={
                    "type": "object",
                    "properties": {
                        "include_recent_events": {
                            "type": "boolean",
                            "description": "Include recent wellbeing events.",
                        },
                        "include_recommendations": {
                            "type": "boolean",
                            "description": "Include recent maintenance recommendations.",
                        },
                        "include_proposals": {
                            "type": "boolean",
                            "description": "Include recent self-modification proposals.",
                        },
                        "include_maintenance_outcomes": {
                            "type": "boolean",
                            "description": "Include recent bounded maintenance outcome receipts and effect metadata.",
                        },
                        "limit": {
                            "type": "integer",
                            "description": "Maximum records to return per section. Default 10.",
                        },
                    },
                    "required": [],
                },
            )
        ],
    )

    thread_registry = ThreadRegistryToolAdapter(runtime=runtime)
    register_tool_specs(
        runtime,
        thread_registry,
        [
            ToolRegistrationSpec(
                name="thread_registry_query",
                description=(
                    "Query peripheral thread anchors and bead records without "
                    "promoting them into active tasks."
                ),
                risk_tier=ActionRiskTier.READONLY,
                schema={
                    "type": "object",
                    "properties": {
                        "thread_anchor_id": {
                            "type": "string",
                            "description": "Optional thread anchor id filter.",
                        },
                        "status": {
                            "type": "string",
                            "description": "Optional bead status filter.",
                        },
                        "source_kind": {
                            "type": "string",
                            "description": "Optional bead source kind filter.",
                        },
                        "thread_status": {
                            "type": "string",
                            "description": "Optional thread status filter.",
                        },
                        "limit": {
                            "type": "integer",
                            "description": "Maximum records to return. Default 10.",
                        },
                    },
                    "required": [],
                },
            ),
            ToolRegistrationSpec(
                name="thread_registry_create_candidate",
                description=(
                    "Create a grounded peripheral bead candidate from an "
                    "autonomous artifact, daydream, or manual observation."
                ),
                risk_tier=ActionRiskTier.WORKSPACE_WRITE,
                schema={
                    "type": "object",
                    "properties": {
                        "thread_anchor_id": {
                            "type": "string",
                            "description": "Existing thread anchor id.",
                        },
                        "thread_title": {
                            "type": "string",
                            "description": "Thread title to create or reuse when anchor id is absent.",
                        },
                        "thread_kind": {
                            "type": "string",
                            "description": "Thread kind for newly created anchors.",
                        },
                        "title": {"type": "string", "description": "Bead title."},
                        "summary": {
                            "type": "string",
                            "description": "Recoverable summary of the bead.",
                        },
                        "source_kind": {
                            "type": "string",
                            "description": "Source kind, such as autonomous_artifact or daydream_reflection.",
                        },
                        "source_ref": {
                            "type": "string",
                            "description": "Grounded source reference for the bead.",
                        },
                        "content": {
                            "type": "string",
                            "description": "Source content used for full-content hashing.",
                        },
                        "user_commissioned": {
                            "type": "boolean",
                            "description": "Whether the user explicitly commissioned this bead.",
                        },
                    },
                    "required": ["title", "summary", "source_ref", "content"],
                },
            ),
        ],
    )
