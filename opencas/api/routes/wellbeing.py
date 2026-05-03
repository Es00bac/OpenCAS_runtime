"""Wellbeing API routes for operator observability."""

from __future__ import annotations

from typing import Any, Dict

from fastapi import APIRouter


def build_wellbeing_router(runtime: Any) -> APIRouter:
    """Build wellbeing routes wired to *runtime*."""

    r = APIRouter(prefix="/api/wellbeing", tags=["wellbeing"])

    @r.get("/snapshot")
    async def get_snapshot() -> Dict[str, Any]:
        store = getattr(runtime, "wellbeing_store", None)
        latest = await store.latest_state() if store is not None else None
        if latest is None and getattr(runtime, "wellbeing_engine", None) is not None:
            assessment = await runtime.wellbeing_engine.assess(runtime)
            latest = assessment.state
        recommendations = []
        if store is not None:
            recommendations = await store.list_recommendations(limit=5)
        return {
            "available": latest is not None,
            "state": latest.model_dump(mode="json") if latest is not None else None,
            "grounding_count": len(latest.grounding) if latest is not None else 0,
            "recommendations": [
                recommendation.model_dump(mode="json")
                for recommendation in recommendations
            ],
        }

    @r.get("/events")
    async def list_events(limit: int = 20) -> Dict[str, Any]:
        store = getattr(runtime, "wellbeing_store", None)
        if store is None:
            return {"available": False, "events": [], "count": 0}
        events = await store.list_events(limit=limit)
        return {
            "available": True,
            "count": len(events),
            "events": [event.model_dump(mode="json") for event in events],
        }

    @r.get("/proposals")
    async def list_proposals(limit: int = 20) -> Dict[str, Any]:
        store = getattr(runtime, "wellbeing_store", None)
        if store is None:
            return {"available": False, "proposals": [], "count": 0}
        proposals = await store.list_self_modification_proposals(limit=limit)
        return {
            "available": True,
            "count": len(proposals),
            "proposals": [proposal.model_dump(mode="json") for proposal in proposals],
        }

    return r
