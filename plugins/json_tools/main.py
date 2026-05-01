"""JSON tools plugin for OpenCAS."""

from __future__ import annotations

import json
from typing import Any, Dict

from opencas.autonomy.models import ActionRiskTier
from opencas.plugins.manifest import validate_config_payload
from opencas.tools.models import ToolResult


_MISSING = object()


def _walk_path(data: Any, path: str) -> Any:
    if not path or path == "$":
        return data
    if path.startswith("$."):
        path = path[2:]
    cursor: Any = data
    for raw_segment in path.split("."):
        if not raw_segment:
            continue
        # Support array index notation: foo[0] or foo[0][1]
        segment = raw_segment
        index_parts: list[int] = []
        while "[" in segment and segment.endswith("]"):
            head, _, idx_part = segment.rpartition("[")
            try:
                index_parts.insert(0, int(idx_part[:-1]))
            except ValueError:
                return _MISSING
            segment = head
        if segment:
            if not isinstance(cursor, dict) or segment not in cursor:
                return _MISSING
            cursor = cursor[segment]
        for index in index_parts:
            if not isinstance(cursor, list) or index >= len(cursor) or index < -len(cursor):
                return _MISSING
            cursor = cursor[index]
    return cursor


def _coerce_data(args: Dict[str, Any]) -> tuple[Any | None, str | None]:
    if "data" in args:
        return args["data"], None
    raw = args.get("json")
    if raw is None:
        return None, "either 'data' (object) or 'json' (string) is required"
    try:
        return json.loads(str(raw)), None
    except json.JSONDecodeError as exc:
        return None, f"json parse failed: {exc}"


def _json_query(args: Dict[str, Any]) -> ToolResult:
    data, err = _coerce_data(args)
    if err is not None:
        return ToolResult(success=False, output=err, metadata={})
    path = str(args.get("path", "$"))
    result = _walk_path(data, path)
    if result is _MISSING:
        return ToolResult(success=False, output=f"path not found: {path}", metadata={"path": path})
    rendered = json.dumps(result, indent=2) if not isinstance(result, str) else result
    return ToolResult(success=True, output=rendered, metadata={"path": path, "type": type(result).__name__})


def _json_validate(args: Dict[str, Any]) -> ToolResult:
    data, err = _coerce_data(args)
    if err is not None:
        return ToolResult(success=False, output=err, metadata={})
    schema = args.get("schema")
    if not isinstance(schema, dict):
        return ToolResult(success=False, output="schema must be an object", metadata={})
    errors = validate_config_payload(schema, data)
    if errors:
        return ToolResult(
            success=False,
            output="\n".join(errors),
            metadata={"error_count": len(errors)},
        )
    return ToolResult(success=True, output="ok", metadata={"error_count": 0})


def register_skills(skill_registry, tools) -> None:
    tools.register(
        "json_query",
        "Query JSON with a dotted path (e.g. 'a.b[0].c'). Pass either 'data' (object) or 'json' (string).",
        lambda name, args: _json_query(args),
        ActionRiskTier.READONLY,
        {
            "type": "object",
            "properties": {
                "data": {"description": "JSON-compatible object."},
                "json": {"type": "string", "description": "JSON text to parse."},
                "path": {"type": "string", "description": "Dotted path; '$' returns root."},
            },
        },
    )
    tools.register(
        "json_validate",
        "Validate JSON against a lightweight JSON-schema-like object (type/required/properties/items/enum).",
        lambda name, args: _json_validate(args),
        ActionRiskTier.READONLY,
        {
            "type": "object",
            "properties": {
                "data": {"description": "JSON-compatible value."},
                "json": {"type": "string", "description": "JSON text to parse."},
                "schema": {"type": "object", "description": "Schema object."},
            },
            "required": ["schema"],
        },
    )
