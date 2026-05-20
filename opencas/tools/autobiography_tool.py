"""Autobiographical recall tool adapter."""

from __future__ import annotations

import json
from datetime import datetime
from typing import Any, Dict, Optional

from pydantic import BaseModel, Field

from opencas.memory.autobiography import AutobiographyReconstructor
from opencas.tools.models import ToolResult


class RecallAutobiographySchema(BaseModel):
    query: str = Field(..., description="Autobiographical recall query.")
    since: Optional[str] = Field(None, description="Optional ISO timestamp lower bound.")
    until: Optional[str] = Field(None, description="Optional ISO timestamp upper bound.")
    max_tokens: int = Field(1500, description="Approximate maximum tokens in the recall packet.")


class AutobiographyToolAdapter:
    """Expose compact autobiographical reconstruction through runtime tools."""

    def __init__(self, runtime: Any) -> None:
        self.runtime = runtime

    async def __call__(self, name: str, arguments: Dict[str, Any]) -> ToolResult:
        if name != "recall_autobiography":
            return ToolResult(success=False, output=f"Unknown autobiography tool: {name}", metadata={})
        try:
            args = RecallAutobiographySchema(**arguments)
            result = await self._reconstructor().recall(
                query=args.query,
                since=self._parse_dt(args.since),
                until=self._parse_dt(args.until),
                max_tokens=args.max_tokens,
                lazy_fill=False,
            )
            payload = result.to_dict()
            return ToolResult(
                success=True,
                output=json.dumps(payload, indent=2, default=str),
                metadata={
                    "confidence": payload.get("confidence"),
                    "evidence_scope": payload.get("evidence_scope"),
                    "evidence_count": len(payload.get("strongest_evidence") or []),
                },
            )
        except Exception as exc:
            return ToolResult(
                success=False,
                output=str(exc),
                metadata={"error_type": type(exc).__name__},
            )

    def _reconstructor(self) -> AutobiographyReconstructor:
        ctx = getattr(self.runtime, "ctx", None)
        reconstructor = getattr(ctx, "autobiography_reconstructor", None)
        if reconstructor is not None:
            return reconstructor
        anchor_store = getattr(ctx, "autobiography_anchor_store", None)
        composer = getattr(ctx, "autobiography_composer", None)
        memory_store = getattr(self.runtime, "memory", None) or getattr(ctx, "memory", None)
        if anchor_store is None or composer is None or memory_store is None:
            raise RuntimeError("autobiographical recall is not wired in this runtime")
        return AutobiographyReconstructor(
            anchor_store=anchor_store,
            composer=composer,
            memory_store=memory_store,
        )

    @staticmethod
    def _parse_dt(value: Optional[str]) -> Optional[datetime]:
        if not value:
            return None
        return datetime.fromisoformat(value)
