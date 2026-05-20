"""Tool adapter for querying operational wellbeing state."""

from __future__ import annotations

import json
from typing import Any, Dict

from ..models import ToolResult


class WellbeingToolAdapter:
    """Read-only access to wellbeing state, events, recommendations, and proposals."""

    def __init__(self, runtime: Any) -> None:
        self.runtime = runtime

    async def __call__(self, name: str, args: Dict[str, Any]) -> ToolResult:
        if name != "wellbeing_query":
            return ToolResult(False, f"Unknown wellbeing tool: {name}", {})
        store = getattr(self.runtime, "wellbeing_store", None)
        if store is None:
            return ToolResult(False, "Wellbeing store is not available", {})

        limit = max(1, min(50, int(args.get("limit") or 10)))
        latest = await store.latest_state()
        events = []
        recommendations = []
        proposals = []
        maintenance_outcomes = []
        if bool(args.get("include_recent_events", False)):
            events = await store.list_events(limit=limit)
        if bool(args.get("include_recommendations", False)):
            recommendations = await store.list_recommendations(limit=limit)
        if bool(args.get("include_proposals", False)):
            proposals = await store.list_self_modification_proposals(limit=limit)
        if bool(args.get("include_maintenance_outcomes", False)):
            maintenance_outcomes = await store.list_maintenance_outcomes(limit=limit)

        payload = {
            "latest_state": latest.model_dump(mode="json") if latest is not None else None,
            "event_count": len(events),
            "events": [event.model_dump(mode="json") for event in events],
            "recommendation_count": len(recommendations),
            "recommendations": [
                recommendation.model_dump(mode="json")
                for recommendation in recommendations
            ],
            "proposal_count": len(proposals),
            "proposals": [proposal.model_dump(mode="json") for proposal in proposals],
            "maintenance_outcome_count": len(maintenance_outcomes),
            "maintenance_outcomes": [
                outcome.model_dump(mode="json") for outcome in maintenance_outcomes
            ],
        }
        return ToolResult(True, json.dumps(payload), payload)
