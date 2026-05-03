"""Tool adapter for querying self-inspection records."""

from __future__ import annotations

import json
from typing import Any, Dict

from ..models import ToolResult


class SelfInspectionToolAdapter:
    """Read-only access to response/tool/commitment inspection records."""

    def __init__(self, runtime: Any) -> None:
        self.runtime = runtime

    async def __call__(self, name: str, args: Dict[str, Any]) -> ToolResult:
        if name != "self_inspection_query":
            return ToolResult(False, f"Unknown self-inspection tool: {name}", {})
        store = getattr(self.runtime, "self_inspection_store", None)
        if store is None:
            return ToolResult(False, "Self-inspection store is not available", {})
        query = str(args.get("query") or "").strip()
        limit = int(args.get("limit") or 10)
        session_id = str(args.get("session_id") or "").strip() or None
        if query:
            records = await store.search(query, session_id=session_id, limit=limit)
        else:
            records = await store.list_recent(session_id=session_id, limit=limit)
        items = [record.model_dump(mode="json") for record in records]
        payload = {
            "count": len(items),
            "items": items,
        }
        return ToolResult(True, json.dumps(payload), payload)
