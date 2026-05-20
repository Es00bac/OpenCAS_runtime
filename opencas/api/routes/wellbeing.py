"""Wellbeing API routes for operator observability."""

from __future__ import annotations

import inspect
from typing import Any, Dict

from fastapi import APIRouter

from opencas.wellbeing.drift import pending_drift_drain_count
from opencas.wellbeing.followup import RecordOnlyMaintenanceFollowupService


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
        maintenance_outcomes = []
        followup = {
            "stuck_loop_count": 0,
            "repeat_count": 0,
            "next_followup_action": None,
            "linked_followup_ids": [],
            "linked_followup_refs": [],
        }
        if store is not None:
            followup = (
                await RecordOnlyMaintenanceFollowupService(
                    store,
                    self_inspection_store=getattr(runtime, "self_inspection_store", None),
                ).evaluate()
            ).model_dump()
            recommendations = await store.list_recommendations(limit=5)
            maintenance_outcomes = await store.list_maintenance_outcomes(limit=5)
            pending_outcomes = await store.list_maintenance_outcomes(limit=100)
        else:
            pending_outcomes = []
        latest_meta = (latest.meta or {}) if latest is not None else {}
        drift_drain_pending = pending_drift_drain_count(pending_outcomes)
        if not drift_drain_pending:
            drift_drain_pending = int(latest_meta.get("drift_drain_pending") or 0)
        return {
            "available": latest is not None,
            "state": latest.model_dump(mode="json") if latest is not None else None,
            "grounding_count": len(latest.grounding) if latest is not None else 0,
            "drift_drain_pending": drift_drain_pending,
            "stuck_loop_count": followup["stuck_loop_count"],
            "repeat_count": followup["repeat_count"],
            "next_followup_action": followup["next_followup_action"],
            "linked_followup_ids": followup["linked_followup_ids"],
            "linked_followup_refs": followup["linked_followup_refs"],
            "recommendations": [
                recommendation.model_dump(mode="json")
                for recommendation in recommendations
            ],
            "maintenance_outcome_count": len(maintenance_outcomes),
            "maintenance_outcomes": [
                outcome.model_dump(mode="json") for outcome in maintenance_outcomes
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

    @r.get("/outcomes")
    async def list_outcomes(limit: int = 20, action_type: str | None = None) -> Dict[str, Any]:
        store = getattr(runtime, "wellbeing_store", None)
        if store is None:
            return {"available": False, "items": [], "outcomes": [], "count": 0}
        outcomes = await store.list_maintenance_outcomes(
            action_type=action_type,
            limit=limit,
        )
        items = [outcome.model_dump(mode="json") for outcome in outcomes]
        return {
            "available": True,
            "count": len(outcomes),
            "items": items,
            "outcomes": items,
        }

    @r.post("/maintenance")
    async def run_maintenance(payload: Dict[str, Any] | None = None) -> Dict[str, Any]:
        runner = getattr(runtime, "run_wellbeing_maintenance", None)
        if not callable(runner):
            return {"available": False, "reason": "wellbeing_maintenance_unavailable"}
        request = dict(payload or {})
        result = runner(force_action=request.get("force_action"))
        if inspect.isawaitable(result):
            result = await result
        return dict(result or {})

    return r
