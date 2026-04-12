"""Executive API routes for the OpenCAS dashboard."""

from __future__ import annotations

from typing import Any, Dict, List, Optional

from fastapi import APIRouter
from pydantic import BaseModel

router = APIRouter(tags=["executive"])


class ExecutiveSnapshotResponse(BaseModel):
    intention: Optional[str]
    active_goals: List[str]
    capacity_remaining: int
    queue_size: int
    queue_stages: List[str]
    recommend_pause: bool
    timestamp: str


class CommitmentResponse(BaseModel):
    commitment_id: str
    content: str
    status: str
    priority: float
    deadline: Optional[str]
    tags: List[str]
    linked_work_ids: List[str]
    linked_task_ids: List[str]
    created_at: str
    updated_at: str


class PlanResponse(BaseModel):
    plan_id: str
    status: str
    content: str
    project_id: Optional[str]
    task_id: Optional[str]
    created_at: str
    updated_at: str


class ExecutiveSummaryResponse(BaseModel):
    snapshot: ExecutiveSnapshotResponse
    commitments: List[CommitmentResponse]
    plans: List[PlanResponse]


def _commitment_to_dict(c: Any) -> Dict[str, Any]:
    return {
        "commitment_id": str(c.commitment_id),
        "content": c.content,
        "status": c.status.value if hasattr(c.status, "value") else str(c.status),
        "priority": c.priority,
        "deadline": c.deadline.isoformat() if c.deadline else None,
        "tags": c.tags,
        "linked_work_ids": c.linked_work_ids,
        "linked_task_ids": c.linked_task_ids,
        "created_at": c.created_at.isoformat(),
        "updated_at": c.updated_at.isoformat(),
    }


def _plan_to_dict(p: Any) -> Dict[str, Any]:
    return {
        "plan_id": str(p.plan_id),
        "status": p.status,
        "content": p.content,
        "project_id": p.project_id,
        "task_id": p.task_id,
        "created_at": p.created_at.isoformat(),
        "updated_at": p.updated_at.isoformat(),
    }


def build_executive_router(runtime: Any) -> APIRouter:
    """Build executive routes wired to *runtime*."""
    r = APIRouter(prefix="/api/executive", tags=["executive"])

    @r.get("", response_model=ExecutiveSummaryResponse)
    async def get_executive_summary() -> ExecutiveSummaryResponse:
        exec_state = runtime.ctx.executive
        snapshot = exec_state.snapshot()

        commitments: List[CommitmentResponse] = []
        if exec_state.commitment_store is not None:
            try:
                active = await exec_state.commitment_store.list_active(limit=50)
                commitments = [CommitmentResponse(**_commitment_to_dict(c)) for c in active]
            except Exception:
                pass

        plans: List[PlanResponse] = []
        plan_store = getattr(runtime.ctx, "plan_store", None)
        if plan_store is not None:
            try:
                active_plans = await plan_store.list_active(limit=50)
                plans = [PlanResponse(**_plan_to_dict(p)) for p in active_plans]
            except Exception:
                pass

        return ExecutiveSummaryResponse(
            snapshot=ExecutiveSnapshotResponse(
                intention=snapshot.get("intention"),
                active_goals=snapshot.get("active_goals", []),
                capacity_remaining=snapshot.get("capacity_remaining", 0),
                queue_size=snapshot.get("queue_size", 0),
                queue_stages=snapshot.get("queue_stages", []),
                recommend_pause=snapshot.get("recommend_pause", False),
                timestamp=snapshot.get("timestamp", ""),
            ),
            commitments=commitments,
            plans=plans,
        )

    @r.get("/snapshot", response_model=ExecutiveSnapshotResponse)
    async def get_executive_snapshot() -> ExecutiveSnapshotResponse:
        snapshot = runtime.ctx.executive.snapshot()
        return ExecutiveSnapshotResponse(
            intention=snapshot.get("intention"),
            active_goals=snapshot.get("active_goals", []),
            capacity_remaining=snapshot.get("capacity_remaining", 0),
            queue_size=snapshot.get("queue_size", 0),
            queue_stages=snapshot.get("queue_stages", []),
            recommend_pause=snapshot.get("recommend_pause", False),
            timestamp=snapshot.get("timestamp", ""),
        )

    @r.get("/commitments", response_model=List[CommitmentResponse])
    async def list_commitments(status: Optional[str] = "active", limit: int = 50) -> List[CommitmentResponse]:
        store = runtime.ctx.executive.commitment_store
        if store is None:
            return []
        from opencas.autonomy.commitment import CommitmentStatus
        try:
            if status:
                items = await store.list_by_status(CommitmentStatus(status), limit=limit)
            else:
                # no all-in-one list method; fall back to active
                items = await store.list_active(limit=limit)
        except Exception:
            return []
        return [CommitmentResponse(**_commitment_to_dict(c)) for c in items]

    @r.get("/plans", response_model=List[PlanResponse])
    async def list_plans(limit: int = 50) -> List[PlanResponse]:
        plan_store = getattr(runtime.ctx, "plan_store", None)
        if plan_store is None:
            return []
        try:
            items = await plan_store.list_active(limit=limit)
        except Exception:
            return []
        return [PlanResponse(**_plan_to_dict(p)) for p in items]

    @r.get("/events/summary")
    async def get_bulma_event_summary() -> Dict[str, Any]:
        from opencas.legacy.executive_event_index import load_executive_event_summary

        return load_executive_event_summary(runtime.ctx.config.state_dir)

    @r.get("/events/search")
    async def search_bulma_events(
        event_type: Optional[str] = None,
        query: Optional[str] = None,
        limit: int = 50,
    ) -> Dict[str, Any]:
        from opencas.legacy.executive_event_index import search_executive_events

        items = search_executive_events(
            runtime.ctx.config.state_dir,
            event_type=event_type,
            query=query,
            limit=limit,
        )
        return {"count": len(items), "items": items}

    return r
