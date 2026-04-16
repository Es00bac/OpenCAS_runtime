"""Shared work, commitment, and plan helpers for operations routes."""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Any, Callable, Dict, Optional

from opencas.api.operations_models import CommitmentEntry, CommitmentListResponse, CommitmentUpdateRequest, PlanListResponse, PlanSummary, PlanUpdateRequest, WorkItemEntry, WorkListResponse, WorkUpdateRequest
from opencas.autonomy.commitment import CommitmentStatus, commitment_operator_snapshot


def _serialize_work_item(item: Any) -> Dict[str, Any]:
    return {
        "work_id": str(item.work_id),
        "created_at": item.created_at.isoformat(),
        "updated_at": item.updated_at.isoformat(),
        "stage": item.stage.value if hasattr(item.stage, "value") else str(item.stage),
        "content": item.content,
        "project_id": item.project_id,
        "commitment_id": item.commitment_id,
        "portfolio_id": item.portfolio_id,
        "dependency_ids": item.dependency_ids,
        "blocked_by": item.blocked_by,
        "meta": item.meta,
    }


def _serialize_plan(plan: Any) -> Dict[str, Any]:
    return {
        "plan_id": plan.plan_id,
        "status": plan.status,
        "content": plan.content,
        "project_id": getattr(plan, "project_id", None),
        "task_id": getattr(plan, "task_id", None),
        "updated_at": plan.updated_at.isoformat() if getattr(plan, "updated_at", None) else None,
    }


def _serialize_plan_action(action: Any) -> Dict[str, Any]:
    return {
        "tool_name": action.tool_name,
        "success": action.success,
        "result_summary": action.result_summary[:200] if action.result_summary else None,
        "created_at": action.timestamp.isoformat() if getattr(action, "timestamp", None) else None,
    }


class TaskingOperationsService:
    """Collect work, commitment, and plan route behavior behind one seam."""

    def __init__(self, runtime: Any, *, coerce_mapping_payload: Callable[[Any], Dict[str, Any]]) -> None:
        self.runtime = runtime
        self._coerce_mapping_payload = coerce_mapping_payload

    async def list_work(self, *, project_id: Optional[str] = None, limit: int = 50) -> WorkListResponse:
        store = getattr(self.runtime.ctx, "work_store", None)
        if store is None:
            return WorkListResponse(counts={"total": 0, "ready": 0, "blocked": 0}, items=[])

        counts = await store.summary_counts()
        raw_items = await (store.list_by_project(project_id, limit=limit) if project_id else store.list_all(limit=limit))
        items = [
            WorkItemEntry(
                work_id=str(item.work_id),
                title=getattr(item, "title", "") or str(item.content)[:80],
                stage=item.stage.value if hasattr(item.stage, "value") else str(item.stage),
                project_id=getattr(item, "project_id", None),
                blocked_by=getattr(item, "blocked_by", None),
            )
            for item in raw_items
        ]
        return WorkListResponse(counts=counts, items=items)

    async def get_work_item(self, work_id: str) -> Dict[str, Any]:
        store = getattr(self.runtime.ctx, "work_store", None)
        if store is None:
            return {"found": False, "error": "Work store not available"}
        item = await store.get(work_id)
        if item is None:
            return {"found": False}
        return {"found": True, "item": _serialize_work_item(item)}

    async def update_work_item(self, work_id: str, payload: WorkUpdateRequest) -> Dict[str, Any]:
        store = getattr(self.runtime.ctx, "work_store", None)
        if store is None:
            return {"found": False, "error": "Work store not available"}
        item = await store.get(work_id)
        if item is None:
            return {"found": False}

        updated = item.model_copy(deep=True)
        changed = False
        if payload.stage is not None:
            updated.stage = payload.stage
            changed = True
        if payload.content is not None:
            updated.content = payload.content
            changed = True
        if payload.blocked_by is not None:
            updated.blocked_by = payload.blocked_by
            changed = True
        if changed:
            updated.updated_at = datetime.now(timezone.utc)
            await store.save(updated)
        return {"found": True, "item": _serialize_work_item(updated)}

    async def list_commitments(self, *, status: str = "active", limit: int = 50) -> CommitmentListResponse:
        store = getattr(self.runtime, "commitment_store", None)
        if store is None:
            return CommitmentListResponse(count=0, items=[])

        status_map = {
            "active": CommitmentStatus.ACTIVE,
            "blocked": CommitmentStatus.BLOCKED,
            "completed": CommitmentStatus.COMPLETED,
            "abandoned": CommitmentStatus.ABANDONED,
        }
        cs = status_map.get(status, CommitmentStatus.ACTIVE)
        raw_items = await store.list_by_status(cs, limit=limit)
        items = [CommitmentEntry(**commitment_operator_snapshot(item)) for item in raw_items]
        status_counts: Dict[str, int] = {}
        for key, value in status_map.items():
            try:
                status_counts[key] = await store.count_by_status(value)
            except Exception:
                status_counts[key] = 0
        consolidation = {}
        consolidation_status = getattr(self.runtime, "consolidation_status", None)
        if callable(consolidation_status):
            try:
                consolidation = self._coerce_mapping_payload(consolidation_status())
            except Exception:
                consolidation = {}
        return CommitmentListResponse(
            count=len(items),
            items=items,
            summary={"status_counts": status_counts, "last_consolidation": consolidation},
        )

    async def get_commitment(self, commitment_id: str) -> Dict[str, Any]:
        store = getattr(self.runtime, "commitment_store", None)
        if store is None:
            return {"found": False, "error": "Commitment store not available"}
        item = await store.get(commitment_id)
        if item is None:
            return {"found": False}
        return {"found": True, "commitment": commitment_operator_snapshot(item, include_meta=True)}

    async def update_commitment(self, commitment_id: str, payload: CommitmentUpdateRequest) -> Dict[str, Any]:
        store = getattr(self.runtime, "commitment_store", None)
        if store is None:
            return {"found": False, "error": "Commitment store not available"}
        item = await store.get(commitment_id)
        if item is None:
            return {"found": False}

        updated = item.model_copy(deep=True)
        changed = False
        if payload.status is not None:
            updated.status = payload.status
            changed = True
        if payload.content is not None:
            updated.content = payload.content
            changed = True
        if payload.priority is not None:
            updated.priority = payload.priority
            changed = True
        if payload.tags is not None:
            updated.tags = payload.tags
            changed = True
        if changed:
            updated.updated_at = datetime.now(timezone.utc)
            await store.save(updated)
        return {"found": True, "commitment": commitment_operator_snapshot(updated, include_meta=True)}

    async def list_plans(self, *, project_id: Optional[str] = None, limit: int = 20) -> PlanListResponse:
        store = getattr(self.runtime.ctx, "plan_store", None)
        if store is None:
            return PlanListResponse(count=0, items=[])

        plans = await store.list_active(project_id=project_id)
        items = [
            PlanSummary(
                plan_id=plan.plan_id,
                status=plan.status,
                content_preview=plan.content[:200],
                project_id=getattr(plan, "project_id", None),
                updated_at=plan.updated_at.isoformat() if getattr(plan, "updated_at", None) else None,
            )
            for plan in plans[:limit]
        ]
        return PlanListResponse(count=len(items), items=items)

    async def get_plan(self, plan_id: str) -> Dict[str, Any]:
        store = getattr(self.runtime.ctx, "plan_store", None)
        if store is None:
            return {"found": False, "error": "Plan store not available"}
        plan = await store.get_plan(plan_id)
        if plan is None:
            return {"found": False}
        actions = await store.get_actions(plan_id, limit=50)
        return {"found": True, "plan": _serialize_plan(plan), "actions": [_serialize_plan_action(a) for a in actions]}

    async def update_plan(self, plan_id: str, payload: PlanUpdateRequest) -> Dict[str, Any]:
        store = getattr(self.runtime.ctx, "plan_store", None)
        if store is None:
            return {"found": False, "error": "Plan store not available"}
        plan = await store.get_plan(plan_id)
        if plan is None:
            return {"found": False}
        if payload.content is not None:
            await store.update_content(plan_id, payload.content)
        if payload.status is not None:
            await store.set_status(plan_id, str(payload.status))
        updated_plan = await store.get_plan(plan_id)
        if updated_plan is None:
            return {"found": False}
        actions = await store.get_actions(plan_id, limit=50)
        return {"found": True, "plan": _serialize_plan(updated_plan), "actions": [_serialize_plan_action(a) for a in actions]}
