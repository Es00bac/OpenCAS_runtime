"""Thread registry API routes for operator observability."""

from __future__ import annotations

from typing import Any, Dict

from fastapi import APIRouter

from opencas.thread_registry import BeadSourceKind, BeadStatus, ThreadStatus


def build_thread_registry_router(runtime: Any) -> APIRouter:
    """Build read-only thread registry routes wired to *runtime*."""

    r = APIRouter(prefix="/api/thread-registry", tags=["thread-registry"])

    @r.get("/threads")
    async def list_threads(
        status: str | None = None,
        limit: int = 50,
    ) -> Dict[str, Any]:
        service = getattr(runtime, "thread_registry_service", None)
        if service is None:
            return {"available": False, "count": 0, "threads": []}
        thread_status = ThreadStatus(status) if status else None
        threads = await service.list_thread_anchors(status=thread_status, limit=limit)
        return {
            "available": True,
            "count": len(threads),
            "threads": [thread.model_dump(mode="json") for thread in threads],
        }

    @r.get("/beads")
    async def list_beads(
        thread_anchor_id: str | None = None,
        status: str | None = None,
        source_kind: str | None = None,
        limit: int = 50,
    ) -> Dict[str, Any]:
        service = getattr(runtime, "thread_registry_service", None)
        if service is None:
            return {"available": False, "count": 0, "beads": []}
        bead_status = BeadStatus(status) if status else None
        bead_source_kind = BeadSourceKind(source_kind) if source_kind else None
        beads = await service.list_beads(
            thread_anchor_id=thread_anchor_id,
            status=bead_status,
            source_kind=bead_source_kind,
            limit=limit,
        )
        return {
            "available": True,
            "count": len(beads),
            "beads": [bead.model_dump(mode="json") for bead in beads],
        }

    return r
