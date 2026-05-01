"""Memory tool registration for her Sense of Self."""

from __future__ import annotations
from typing import Any

from opencas.autonomy.models import ActionRiskTier
from opencas.tools.memory_tools import MemoryToolAdapter
from .tool_registration_specs import ToolRegistrationSpec, register_tool_specs

def register_memory_tools(runtime: Any) -> None:
    """Register high-fidelity memory retrieval tools."""
    adapter = MemoryToolAdapter(runtime)

    register_tool_specs(
        runtime,
        adapter,
        [
            ToolRegistrationSpec(
                name="search_memories",
                description="Search her high-fidelity semantic memory for specific concepts or events.",
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
