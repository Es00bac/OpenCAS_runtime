"""Memory tool registration for her Sense of Self."""

from __future__ import annotations

from typing import Any

from opencas.autonomy.models import ActionRiskTier
from opencas.tools.adapters.artifact import ArtifactLookupSchema, ArtifactToolAdapter
from opencas.tools.autobiography_tool import (
    AutobiographyToolAdapter,
    RecallAutobiographySchema,
)
from opencas.tools.cognitive_tools import (
    CognitiveContextQuerySchema,
    CognitiveFocusSchema,
    CognitiveProspectiveMemorySchema,
    CognitiveSkillCreateSchema,
    CognitiveSocialModelSchema,
    CognitiveToolAdapter,
    CognitiveWorkingMemorySchema,
)
from opencas.tools.memory_tools import MemoryToolAdapter

from .tool_registration_specs import ToolRegistrationSpec, register_tool_specs


def register_memory_tools(runtime: Any) -> None:
    """Register high-fidelity memory retrieval tools."""
    adapter = MemoryToolAdapter(runtime)
    artifact_adapter = ArtifactToolAdapter(runtime)
    autobiography_adapter = AutobiographyToolAdapter(runtime)
    cognitive_adapter = CognitiveToolAdapter(runtime)

    register_tool_specs(
        runtime,
        adapter,
        [
            ToolRegistrationSpec(
                name="search_memories",
                description="Search the active agent's high-fidelity semantic memory for specific concepts or events.",
                risk_tier=ActionRiskTier.READONLY,
                schema={
                    "type": "object",
                    "properties": {
                        "query": {
                            "type": "string",
                            "description": "Semantic search query.",
                        },
                        "limit": {
                            "type": "integer",
                            "description": "Max results to return.",
                        }
                    },
                    "required": ["query"],
                },
            ),
            ToolRegistrationSpec(
                name="recall_concepts",
                description="Perform a combined keyword and semantic search for specific entities.",
                risk_tier=ActionRiskTier.READONLY,
                schema={
                    "type": "object",
                    "properties": {
                        "concepts": {
                            "type": "array",
                            "items": {"type": "string"},
                            "description": "List of keywords or entities.",
                        }
                    },
                    "required": ["concepts"],
                },
            ),
        ],
    )
    register_tool_specs(
        runtime,
        artifact_adapter,
        [
            ToolRegistrationSpec(
                name="artifact_lookup",
                description=(
                    "Return a unified timeline for a workspace artifact (file). "
                    "Joins memory episodes, schedules, tasks, receipts, commitments, "
                    "plans, and provenance transitions for the given path or checksum. "
                    "Use this when the operator asks 'did you make this?', 'where did "
                    "this come from?', 'what changed about this file?', or 'what work "
                    "is connected to this path?'. Returns current filesystem metadata, "
                    "sibling paths with the same checksum, and a sorted timeline."
                ),
                risk_tier=ActionRiskTier.READONLY,
                schema=ArtifactLookupSchema.model_json_schema(),
            ),
        ],
    )
    register_tool_specs(
        runtime,
        autobiography_adapter,
        [
            ToolRegistrationSpec(
                name="recall_autobiography",
                description=(
                    "Recall autobiographical memory. Returns a compact reconstruction "
                    "packet: essence summary, ranked strongest evidence with labels "
                    "(autobiographical/artifact/procedural/assistant_turn/mention_only), "
                    "gaps, and next-best lookup suggestions. Use when the operator asks "
                    "'do you remember', 'what did we work on', 'what have we done', or "
                    "any question about past lived experience. Treats retrieved self-"
                    "evidence as autobiographical memory: the agent reconstructs from "
                    "stored records, not from human continuous consciousness."
                ),
                risk_tier=ActionRiskTier.READONLY,
                schema=RecallAutobiographySchema.model_json_schema(),
            ),
        ],
    )
    register_tool_specs(
        runtime,
        cognitive_adapter,
        [
            ToolRegistrationSpec(
                name="cognitive_context_query",
                description=(
                    "Query the agent's cognitive state spine: active attention, "
                    "working memory, prospective memories, learned skills, recent "
                    "feedback-loop records, surprise/counterfactual/narrative evidence, "
                    "and failed-recall recovery records. Use this before denying "
                    "self-knowledge or when deciding what the agent can focus on, "
                    "remember, learn from, or act on."
                ),
                risk_tier=ActionRiskTier.READONLY,
                schema=CognitiveContextQuerySchema.model_json_schema(),
            ),
            ToolRegistrationSpec(
                name="cognitive_focus_set",
                description=(
                    "Set or refresh an internal attention/focus target. This is a "
                    "low-risk OpenCAS cognitive-state update, not a user-facing promise. "
                    "Use it to keep the agent focused on a current goal, artifact, "
                    "thread, or investigation when evidence shows that focus matters."
                ),
                risk_tier=ActionRiskTier.WORKSPACE_WRITE,
                schema=CognitiveFocusSchema.model_json_schema(),
            ),
            ToolRegistrationSpec(
                name="cognitive_working_memory_update",
                description=(
                    "Update a bounded working-memory slot for active assumptions, "
                    "current artifact state, unresolved questions, or next-step evidence. "
                    "Use this to prevent losing the thread across tool calls and turns."
                ),
                risk_tier=ActionRiskTier.WORKSPACE_WRITE,
                schema=CognitiveWorkingMemorySchema.model_json_schema(),
            ),
            ToolRegistrationSpec(
                name="cognitive_prospective_memory_set",
                description=(
                    "Create or refresh a future-intention memory. This complements "
                    "workflow schedules/commitments: include a schedule, task, receipt, "
                    "or commitment proof when follow-through must happen without another prompt."
                ),
                risk_tier=ActionRiskTier.WORKSPACE_WRITE,
                schema=CognitiveProspectiveMemorySchema.model_json_schema(),
            ),
            ToolRegistrationSpec(
                name="cognitive_skill_library_search",
                description=(
                    "Inspect evidence-gated learned procedures extracted from procedural "
                    "memory. Use this before repeating a familiar task so the agent can "
                    "reuse what it has learned from experience."
                ),
                risk_tier=ActionRiskTier.READONLY,
                schema=CognitiveContextQuerySchema.model_json_schema(),
            ),
            ToolRegistrationSpec(
                name="cognitive_skill_create",
                description=(
                    "Create or refresh a durable learned procedure from a successful or "
                    "corrected interaction. Use this when the operator asks to teach/remember "
                    "a workflow, or after a non-trivial task reveals a reusable procedure, so "
                    "future turns can surface and route by the learned skill. Include evidence_refs "
                    "and tool_sequence whenever available."
                ),
                risk_tier=ActionRiskTier.WORKSPACE_WRITE,
                schema=CognitiveSkillCreateSchema.model_json_schema(),
            ),
            ToolRegistrationSpec(
                name="cognitive_social_model_record",
                description=(
                    "Record an evidence-grounded Theory-of-Mind model for a third-party "
                    "AI agent or autonomous system, separate from the operator/user model. "
                    "Use this when another AI/system's limits, intentions, or behavior "
                    "matter for collaboration, delegation, or diagnosis."
                ),
                risk_tier=ActionRiskTier.WORKSPACE_WRITE,
                schema=CognitiveSocialModelSchema.model_json_schema(),
            ),
        ],
    )
