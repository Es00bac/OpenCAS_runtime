"""OpenAI-compatible function schema exporter for ToolRegistry."""

from typing import Any, Dict, List

from .models import ToolEntry


def build_tool_schemas(tools: List[ToolEntry]) -> List[Dict[str, Any]]:
    """Build OpenAI function-calling schemas from a list of ToolEntry objects."""
    schemas: List[Dict[str, Any]] = []
    for entry in tools:
        schema: Dict[str, Any] = {
            "type": "function",
            "function": {
                "name": entry.name,
                "description": entry.description,
                "parameters": entry.parameters,
            },
        }
        schemas.append(schema)
    return schemas
