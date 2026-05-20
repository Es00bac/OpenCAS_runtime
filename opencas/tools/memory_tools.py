"""Memory retrieval tool adapter for OpenCAS."""

from __future__ import annotations

import re
from datetime import timezone
from typing import Any, Dict, List, Optional, Tuple

from opencas.tools.models import ToolResult


class MemoryToolAdapter:
    """Adapter for high-fidelity memory retrieval tools."""

    def __init__(self, runtime: Any):
        self.runtime = runtime

    async def __call__(self, name: str, args: Dict[str, Any]) -> ToolResult:
        try:
            if name == "search_memories":
                output, metadata = await self._search_memories(args)
                return ToolResult(success=True, output=output, metadata=metadata)
            if name == "recall_concepts":
                output, metadata = await self._recall_concepts(args)
                return ToolResult(success=True, output=output, metadata=metadata)
            return ToolResult(
                success=False, output=f"Unknown memory tool: {name}", metadata={}
            )
        except Exception as exc:
            return ToolResult(
                success=False,
                output=str(exc),
                metadata={"error_type": type(exc).__name__},
            )

    async def _search_memories(self, args: Dict[str, Any]) -> Tuple[str, Dict[str, Any]]:
        query = args.get("query", "")
        limit = int(args.get("limit", 10))
        offset = int(args.get("offset", 0))

        # Use her internal retriever (Gemma-powered)
        retriever = self.runtime.retriever
        inspection = await retriever.inspect(query, limit=limit, offset=offset)
        results = inspection["results"]
        meta = inspection["meta"]

        total_count = meta.get("total_count", len(results))
        returned_count = len(results)
        truncated = total_count > (offset + returned_count)
        next_offset = (offset + returned_count) if truncated else None

        metadata = {
            "total_count": total_count,
            "returned_count": returned_count,
            "offset": offset,
            "limit": limit,
            "truncated": truncated,
            "next_offset": next_offset,
        }

        if not results:
            return (
                "No matching memories found. Searched: episodes (action, turn, "
                "observation, reflection) and memories. Consider artifact_lookup "
                "for path-anchored questions.",
                metadata,
            )

        output = []
        for i, res in enumerate(results):
            output.append(self._render_result_line(offset + i + 1, res))

        return "\n".join(output), metadata

    async def _recall_concepts(self, args: Dict[str, Any]) -> Tuple[str, Dict[str, Any]]:
        concepts = args.get("concepts", [])
        query = " ".join(concepts)
        return await self._search_memories(
            {"query": query, "limit": args.get("limit", 10)}
        )

    def _render_result_line(self, index: int, result: Any) -> str:
        episode = getattr(result, "episode", None)
        memory = getattr(result, "memory", None)
        created_at = getattr(episode, "created_at", None) or getattr(
            memory, "created_at", None
        )
        timestamp = "unknown-time"
        if created_at is not None:
            try:
                if created_at.tzinfo is None:
                    created_at = created_at.replace(tzinfo=timezone.utc)
                timestamp = created_at.astimezone(timezone.utc).isoformat(
                    timespec="seconds"
                )
            except Exception:
                timestamp = str(created_at)

        kind = "MEMORY"
        if episode is not None:
            raw_kind = getattr(episode, "kind", None)
            kind = str(getattr(raw_kind, "value", raw_kind) or "episode").upper()
        elif getattr(result, "source_type", ""):
            kind = str(getattr(result, "source_type", "")).upper()

        tool_name = self._episode_tool_name(episode)
        tool_suffix = f"[tool={tool_name}]" if kind == "ACTION" and tool_name else ""
        content = " ".join(str(getattr(result, "content", "") or "").split())
        if len(content) > 240:
            content = content[:237].rstrip() + "..."
        score = float(getattr(result, "score", 0.0) or 0.0)
        return (
            f"{index}. {timestamp} [{kind}]{tool_suffix} {content} (score: {score:.2f})"
        )

    @staticmethod
    def _episode_tool_name(episode: Any) -> str:
        if episode is None:
            return ""
        payload = getattr(episode, "payload", {}) or {}
        if isinstance(payload, dict):
            for key in ("tool_name", "name"):
                value = str(payload.get(key) or "").strip()
                if value:
                    return value
        content = str(getattr(episode, "content", "") or "")
        match = re.match(r"^tool\s+([A-Za-z0-9_:-]+)", content)
        return match.group(1).rstrip(":") if match else ""
