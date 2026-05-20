"""Artifact lookup tool adapter."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, Optional

from pydantic import BaseModel, Field, model_validator

from opencas.tools.models import ToolResult
from opencas.workspace.artifact_lookup import ArtifactLookupService


class ArtifactLookupSchema(BaseModel):
    path: Optional[str] = Field(None, description="Absolute or workspace-relative artifact path.")
    checksum: Optional[str] = Field(None, description="SHA-256 checksum to look up.")
    limit: int = Field(50, description="Maximum timeline entries to return.")

    @model_validator(mode="after")
    def require_path_or_checksum(self) -> "ArtifactLookupSchema":
        if not (self.path or self.checksum):
            raise ValueError("Either path or checksum is required.")
        return self


class ArtifactToolAdapter:
    """Expose ArtifactLookupService through the runtime tool registry."""

    def __init__(self, runtime: Any) -> None:
        self.runtime = runtime

    async def __call__(self, name: str, arguments: Dict[str, Any]) -> ToolResult:
        if name != "artifact_lookup":
            return ToolResult(success=False, output=f"Unknown artifact tool: {name}", metadata={})
        try:
            args = ArtifactLookupSchema(**arguments)
            service = self._service()
            result = await service.lookup(path=args.path, checksum=args.checksum, limit=args.limit)
            payload = result.to_dict()
            ordered = {
                "summary_counts": payload["summary_counts"],
                "notes": payload["notes"],
                "query": payload["query"],
                "current": payload["current"],
                "sibling_paths": payload["sibling_paths"],
                "timeline": payload["timeline"],
            }
            return ToolResult(
                success=True,
                output=json.dumps(ordered, indent=2, default=str),
                metadata={
                    "path": args.path,
                    "checksum": args.checksum or payload["current"].get("checksum"),
                    "timeline_count": len(payload["timeline"]),
                },
            )
        except Exception as exc:
            return ToolResult(
                success=False,
                output=str(exc),
                metadata={"error_type": type(exc).__name__},
            )

    def _service(self) -> ArtifactLookupService:
        ctx = getattr(self.runtime, "ctx", None)
        workspace_index = getattr(ctx, "workspace_index", None)
        workspace_store = getattr(workspace_index, "store", None)
        if workspace_store is None:
            workspace_store = getattr(ctx, "workspace_store", None)
        memory_store = getattr(self.runtime, "memory", None) or getattr(ctx, "memory", None)
        config = getattr(ctx, "config", None)
        state_dir = Path(getattr(config, "state_dir", ".")).expanduser()
        return ArtifactLookupService(
            workspace_store=workspace_store,
            memory_store=memory_store,
            schedule_store=getattr(ctx, "schedule_store", None) or getattr(ctx, "schedules", None),
            task_store=getattr(ctx, "tasks", None) or getattr(ctx, "task_store", None),
            receipt_store=getattr(ctx, "receipt_store", None),
            commitment_store=getattr(self.runtime, "commitment_store", None)
            or getattr(ctx, "commitment_store", None),
            plan_store=getattr(ctx, "plan_store", None),
            provenance_path=state_dir / "provenance.transitions.jsonl",
        )
