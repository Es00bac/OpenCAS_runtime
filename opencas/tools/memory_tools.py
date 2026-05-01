"""Memory retrieval tool adapter for OpenCAS."""

from __future__ import annotations
from typing import Any, Dict, List, Optional
from opencas.tools.models import ToolResult

class MemoryToolAdapter:
    """Adapter for high-fidelity memory retrieval tools."""

    def __init__(self, runtime: Any):
        self.runtime = runtime

    async def __call__(self, name: str, args: Dict[str, Any]) -> ToolResult:
        try:
            if name == "search_memories":
                output = await self._search_memories(args)
                return ToolResult(success=True, output=output, metadata={})
            if name == "recall_concepts":
                output = await self._recall_concepts(args)
                return ToolResult(success=True, output=output, metadata={})
            return ToolResult(success=False, output=f"Unknown memory tool: {name}", metadata={})
        except Exception as exc:
            return ToolResult(success=False, output=str(exc), metadata={"error_type": type(exc).__name__})

    async def _search_memories(self, args: Dict[str, Any]) -> str:
        query = args.get("query", "")
        limit = int(args.get("limit", 10))
        # Use her internal retriever (Gemma-powered)
        retriever = self.runtime.retriever
        results = await retriever.retrieve(query, limit=limit)

        if not results:
            return "No matching memories found."

        output = []
        for i, res in enumerate(results):
            output.append(f"{i+1}. [{res.source_type}] {res.content} (score: {res.score:.2f})")

        return "\n".join(output)

    async def _recall_concepts(self, args: Dict[str, Any]) -> str:
        concepts = args.get("concepts", [])
        query = " ".join(concepts)
        return await self._search_memories({"query": query})
