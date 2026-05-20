"""Cognitive state and feedback-loop API routes."""

from __future__ import annotations

import inspect
from typing import Any

from fastapi import APIRouter


def build_cognition_router(runtime: Any) -> APIRouter:
    """Expose the shared cognitive feedback-loop spine."""

    router = APIRouter(prefix="/api/cognition", tags=["cognition"])

    @router.get("/summary")
    async def summary(query: str = "", limit: int = 8) -> dict[str, Any]:
        store = _store(runtime)
        if store is None:
            return {"available": False, "reason": "cognitive_state_store_unavailable"}
        return await store.query_context(query, limit=limit)

    @router.get("/prompt-block")
    async def prompt_block(query: str = "", session_id: str | None = None) -> dict[str, Any]:
        store = _store(runtime)
        if store is None:
            return {"available": False, "block": ""}
        block = await store.prompt_block(query=query, session_id=session_id, char_budget=1800)
        return {"available": True, "block": block}

    @router.post("/maintenance")
    async def maintenance() -> dict[str, Any]:
        runner = getattr(runtime, "run_cognitive_maintenance", None)
        if not callable(runner):
            return {"available": False, "reason": "cognitive_maintenance_unavailable"}
        result = runner()
        if inspect.isawaitable(result):
            result = await result
        return dict(result or {})

    return router


def _store(runtime: Any) -> Any:
    return getattr(runtime, "cognitive_state_store", None) or getattr(
        getattr(runtime, "ctx", None),
        "cognitive_state_store",
        None,
    )
