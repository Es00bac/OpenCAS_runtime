"""Shared receipt and background-task helpers for operations routes."""

from __future__ import annotations

from typing import Any, Callable, Dict, List

from opencas.api.operations_models import ReceiptEntry, ReceiptListResponse, TaskEntry, TaskListResponse


class ActivityOperationsService:
    """Collect receipt and background-task route behavior behind one seam."""

    def __init__(
        self,
        runtime: Any,
        *,
        human_title: Callable[[str | None, str], str],
        task_ui_status: Callable[[str, str], str],
    ) -> None:
        self.runtime = runtime
        self._human_title = human_title
        self._task_ui_status = task_ui_status

    async def list_receipts(self, *, limit: int = 50) -> ReceiptListResponse:
        store = getattr(self.runtime.ctx, "receipt_store", None)
        if store is None:
            return ReceiptListResponse(count=0, items=[])

        receipts = await store.list_recent(limit=limit)
        items = [
            ReceiptEntry(
                receipt_id=str(getattr(receipt, "receipt_id", "")),
                task_id=str(getattr(receipt, "task_id", "")),
                status=str(getattr(receipt, "status", "")),
                tool_name=getattr(receipt, "tool_name", None),
                started_at=receipt.started_at.isoformat() if getattr(receipt, "started_at", None) else None,
                finished_at=receipt.finished_at.isoformat() if getattr(receipt, "finished_at", None) else None,
                duration_ms=getattr(receipt, "duration_ms", None),
            )
            for receipt in receipts
        ]
        return ReceiptListResponse(count=len(items), items=items)

    async def get_receipt(self, receipt_id: str) -> Dict[str, Any]:
        store = getattr(self.runtime.ctx, "receipt_store", None)
        if store is None:
            return {"found": False, "error": "Receipt store not available"}
        receipt = await store.get(receipt_id)
        if receipt is None:
            return {"found": False}
        return {"found": True, "receipt": receipt.model_dump(mode="json")}

    async def list_tasks(self, *, limit: int = 50) -> TaskListResponse:
        store = getattr(self.runtime.ctx, "tasks", None)
        if store is None:
            return TaskListResponse(
                counts={"total": 0, "active": 0, "waiting": 0, "completed": 0, "failed": 0},
                items=[],
            )

        sample = await store.list_all(limit=max(limit, 250))
        objective_counts: Dict[str, int] = {}
        for item in sample:
            objective_counts[item.objective] = objective_counts.get(item.objective, 0) + 1

        counts = {"total": len(sample), "active": 0, "waiting": 0, "completed": 0, "failed": 0}
        items: List[TaskEntry] = []
        for item in sample:
            ui_status = self._task_ui_status(
                item.stage.value if hasattr(item.stage, "value") else str(item.stage),
                item.status,
            )
            if ui_status in {"queued", "planning", "executing", "verifying", "recovering"}:
                counts["active"] += 1
            elif ui_status in {"needs approval", "needs clarification"}:
                counts["waiting"] += 1
            elif ui_status == "completed":
                counts["completed"] += 1
            elif ui_status == "failed":
                counts["failed"] += 1
            if len(items) < limit:
                items.append(
                    TaskEntry(
                        task_id=str(item.task_id),
                        title=self._human_title(item.meta.get("title") or item.objective, fallback="Background task"),
                        objective=item.objective,
                        status=ui_status,
                        stage=item.stage.value if hasattr(item.stage, "value") else str(item.stage),
                        source=str(item.meta.get("source", "") or "") or None,
                        project_id=item.project_id,
                        commitment_id=item.commitment_id,
                        updated_at=item.updated_at.isoformat(),
                        duplicate_objective_count=objective_counts.get(item.objective, 1),
                    )
                )
        return TaskListResponse(counts=counts, items=items)

    async def get_task(self, task_id: str) -> Dict[str, Any]:
        store = getattr(self.runtime.ctx, "tasks", None)
        if store is None:
            return {"found": False, "error": "Task store not available"}
        task = await store.get(task_id)
        if task is None:
            return {"found": False}
        result = await store.get_result(task_id)
        lifecycle = await store.list_lifecycle_transitions(task_id, limit=50)
        related = await store.list_all(limit=100)
        duplicate_count = sum(1 for item in related if item.objective == task.objective)
        return {
            "found": True,
            "task": {
                "task_id": str(task.task_id),
                "title": self._human_title(task.meta.get("title") or task.objective, fallback="Background task"),
                "objective": task.objective,
                "status": self._task_ui_status(
                    task.stage.value if hasattr(task.stage, "value") else str(task.stage),
                    task.status,
                ),
                "raw_status": task.status,
                "stage": task.stage.value if hasattr(task.stage, "value") else str(task.stage),
                "created_at": task.created_at.isoformat(),
                "updated_at": task.updated_at.isoformat(),
                "source": str(task.meta.get("source", "") or "") or None,
                "project_id": task.project_id,
                "commitment_id": task.commitment_id,
                "depends_on": task.depends_on,
                "attempt": task.attempt,
                "max_attempts": task.max_attempts,
                "duplicate_objective_count": duplicate_count,
                "meta": task.meta,
                "phases": [phase.model_dump(mode="json") for phase in task.phases],
                "result": result.model_dump(mode="json") if result is not None else None,
            },
            "transitions": [
                {
                    "from_stage": item.get("from_stage"),
                    "to_stage": item.get("to_stage"),
                    "reason": item.get("reason"),
                    "timestamp": item.get("timestamp").isoformat() if item.get("timestamp") is not None else None,
                    "context": item.get("context", {}),
                }
                for item in lifecycle
            ],
        }
