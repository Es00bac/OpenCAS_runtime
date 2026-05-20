"""Tool adapter for the peripheral thread registry."""

from __future__ import annotations

import json
from typing import Any, Dict

from opencas.thread_registry import BeadSourceKind, BeadStatus, ThreadStatus

from ..models import ToolResult


class ThreadRegistryToolAdapter:
    """Create and query thread/bead records without promoting them into tasks."""

    def __init__(self, runtime: Any) -> None:
        self.runtime = runtime

    async def __call__(self, name: str, args: Dict[str, Any]) -> ToolResult:
        service = getattr(self.runtime, "thread_registry_service", None)
        if service is None:
            return ToolResult(False, "Thread registry service is not available", {})
        if name == "thread_registry_query":
            return await self._query(service, args)
        if name == "thread_registry_create_candidate":
            return await self._create_candidate(service, args)
        return ToolResult(False, f"Unknown thread registry tool: {name}", {})

    async def _query(self, service: Any, args: Dict[str, Any]) -> ToolResult:
        status = args.get("status")
        source_kind = args.get("source_kind")
        if status:
            status = BeadStatus(str(status))
        if source_kind:
            source_kind = BeadSourceKind(str(source_kind))
        limit = max(1, min(50, int(args.get("limit") or 10)))
        beads = await service.list_beads(
            thread_anchor_id=args.get("thread_anchor_id") or None,
            status=status,
            source_kind=source_kind,
            limit=limit,
        )
        threads = await service.list_thread_anchors(
            status=ThreadStatus(str(args["thread_status"])) if args.get("thread_status") else None,
            limit=limit,
        )
        payload = {
            "thread_count": len(threads),
            "threads": [thread.model_dump(mode="json") for thread in threads],
            "bead_count": len(beads),
            "beads": [bead.model_dump(mode="json") for bead in beads],
        }
        return ToolResult(True, json.dumps(payload), payload)

    async def _create_candidate(self, service: Any, args: Dict[str, Any]) -> ToolResult:
        source_kind = BeadSourceKind(str(args.get("source_kind") or BeadSourceKind.MANUAL.value))
        thread_anchor_id = str(args.get("thread_anchor_id") or "").strip()
        thread_title = str(args.get("thread_title") or "").strip()
        if not thread_anchor_id and thread_title:
            anchor = await service.ensure_thread_anchor(
                title=thread_title,
                kind=str(args.get("thread_kind") or "theme"),
                status=ThreadStatus.PERIPHERAL,
            )
            thread_anchor_id = anchor.anchor_id
        if not thread_anchor_id:
            return ToolResult(False, "thread_anchor_id or thread_title is required", {})
        bead = await service.create_candidate_bead(
            thread_anchor_id=thread_anchor_id,
            title=str(args.get("title") or ""),
            summary=str(args.get("summary") or ""),
            source_kind=source_kind,
            source_ref=str(args.get("source_ref") or ""),
            content=str(args.get("content") or ""),
            user_commissioned=bool(args.get("user_commissioned", False)),
        )
        payload = {"bead": bead.model_dump(mode="json")}
        return ToolResult(True, json.dumps(payload), payload)
