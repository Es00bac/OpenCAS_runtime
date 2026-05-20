from __future__ import annotations

from types import SimpleNamespace

import httpx
import pytest
from fastapi import FastAPI

from opencas.api.routes.executive import build_executive_router


@pytest.mark.asyncio
async def test_executive_snapshot_clears_stale_active_work_intention() -> None:
    class _Executive:
        def snapshot(self):
            return {
                "intention": "Return to project: Writing Project 4",
                "intention_source": "active_work",
                "active_goals": [],
                "parked_goal_count": 0,
                "parked_goals": [],
                "parked_goal_metadata": {},
                "archived_parked_goal_count": 0,
                "archived_parked_goals": [],
                "weighted_load": 0.0,
                "capacity_remaining": 5,
                "queue_size": 0,
                "queue_stages": [],
                "queue_metadata": [],
                "recommend_pause": False,
                "timestamp": "2026-05-09T03:40:00+00:00",
            }

    app = FastAPI()
    app.include_router(build_executive_router(SimpleNamespace(ctx=SimpleNamespace(executive=_Executive()))))
    transport = httpx.ASGITransport(app=app)
    async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
        response = await client.get("/api/executive/snapshot")

    assert response.status_code == 200
    payload = response.json()
    assert payload["intention"] is None
    assert payload["intention_source"] == "stale_active_work"
