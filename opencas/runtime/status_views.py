"""Status and workflow snapshots for AgentRuntime.

These builders keep monitoring and operator-facing status assembly out of the
main runtime loop so `AgentRuntime` can stay focused on orchestration.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, Dict, List, Optional

from opencas.autonomy.commitment import CommitmentStatus, commitment_operator_snapshot

if TYPE_CHECKING:
    from .agent_loop import AgentRuntime


def build_control_plane_status(runtime: "AgentRuntime") -> Dict[str, Any]:
    """Return a monitoring snapshot of workspace, sandbox, and execution state."""
    configured_workspace_roots = [
        str(root) for root in runtime.ctx.config.all_workspace_roots()
    ]
    sandbox = getattr(runtime.ctx, "sandbox", None)
    sandbox_report = sandbox.report_isolation() if sandbox is not None else {}
    readiness = runtime.readiness.snapshot() if runtime.readiness else {"state": "unknown"}
    return {
        "agent_profile": runtime.agent_profile.model_dump(mode="json"),
        "readiness": readiness,
        "workspace": {
            "session_id": getattr(runtime.ctx.config, "session_id", None),
            "state_dir": str(runtime.ctx.config.state_dir),
            "primary_root": str(runtime.ctx.config.primary_workspace_root()),
            "managed_root": str(runtime.ctx.config.agent_workspace_root()),
            "workspace_roots": configured_workspace_roots,
            "allowed_roots": [
                str(root) for root in getattr(sandbox, "allowed_roots", [])
            ],
        },
        "sandbox": sandbox_report,
        "execution": {
            "processes": runtime.process_supervisor.snapshot(sample_limit=10),
            "pty": runtime.pty_supervisor.snapshot(sample_limit=10),
            "browser": runtime.browser_supervisor.snapshot(sample_limit=10),
        },
        "activity": {
            "current": runtime._activity,
            "since": runtime._activity_since.isoformat(),
        },
        "consolidation": build_consolidation_status(runtime),
        "lanes": runtime.baa.lane_snapshot() if runtime.baa else {},
        "web_trust": (
            runtime.ctx.web_trust.snapshot(limit=10)
            if getattr(runtime.ctx, "web_trust", None) is not None
            else {"available": False, "entries": []}
        ),
        "plugin_trust": (
            runtime.ctx.plugin_trust.snapshot(limit=10)
            if getattr(runtime.ctx, "plugin_trust", None) is not None
            else {"available": False, "entries": []}
        ),
    }


def build_consolidation_status(runtime: "AgentRuntime") -> Dict[str, Any]:
    """Return the latest known nightly consolidation summary."""
    result = runtime._last_consolidation_result
    if not result:
        return {"available": False}
    return {
        "available": True,
        "timestamp": result.get("timestamp"),
        "result_id": result.get("result_id"),
        "clusters_formed": result.get("clusters_formed", 0),
        "memories_created": result.get("memories_created", 0),
        "commitments_consolidated": result.get("commitments_consolidated", 0),
        "commitment_clusters_formed": result.get("commitment_clusters_formed", 0),
        "commitment_work_objects_created": result.get("commitment_work_objects_created", 0),
        "commitments_extracted_from_chat": result.get("commitments_extracted_from_chat", 0),
        "episodes_pruned": result.get("episodes_pruned", 0),
    }


async def build_workflow_status(
    runtime: "AgentRuntime",
    *,
    limit: int = 10,
    project_id: Optional[str] = None,
) -> Dict[str, Any]:
    """Return a summarized view of higher-level workflow state."""
    active_commitments = []
    commitment_count = 0
    commitment_status_counts = {
        CommitmentStatus.ACTIVE.value: 0,
        CommitmentStatus.BLOCKED.value: 0,
        CommitmentStatus.COMPLETED.value: 0,
        CommitmentStatus.ABANDONED.value: 0,
    }
    if runtime.commitment_store:
        active_commitments = await runtime.commitment_store.list_active(limit=limit)
        commitment_count = await runtime.commitment_store.count_by_status(
            CommitmentStatus.ACTIVE
        )
        for status in CommitmentStatus:
            try:
                commitment_status_counts[status.value] = await runtime.commitment_store.count_by_status(
                    status
                )
            except Exception:
                commitment_status_counts[status.value] = 0

    work_counts = {"total": 0, "ready": 0, "blocked": 0}
    work_items = []
    blocked_items = []
    if runtime.ctx.work_store:
        work_counts = await runtime.ctx.work_store.summary_counts()
        if project_id:
            work_items = await runtime.ctx.work_store.list_by_project(
                project_id, limit=limit
            )
            blocked_items = [item for item in work_items if item.blocked_by][:limit]
        else:
            work_items = await runtime.ctx.work_store.list_all(limit=limit)
            blocked_items = await runtime.ctx.work_store.list_blocked(limit=limit)

    active_plans = []
    active_plan_count = 0
    if getattr(runtime.ctx, "plan_store", None) is not None:
        active_plans = await runtime.ctx.plan_store.list_active(project_id=project_id)
        active_plan_count = await runtime.ctx.plan_store.count_active(
            project_id=project_id
        )

    plan_entries = []
    for plan in active_plans[:limit]:
        actions = await runtime.ctx.plan_store.get_actions(plan.plan_id, limit=5)
        plan_entries.append(
            {
                "plan_id": plan.plan_id,
                "status": plan.status,
                "content_preview": plan.content[:240],
                "project_id": plan.project_id,
                "task_id": plan.task_id,
                "updated_at": plan.updated_at.isoformat(),
                "recent_action_count": len(actions),
            }
        )

    recent_receipts = []
    if getattr(runtime.ctx, "receipt_store", None) is not None:
        recent_receipts = await runtime.ctx.receipt_store.list_recent(limit=limit)

    active_projects = []
    seen_projects = set()
    for item in work_items:
        if item.project_id and item.project_id not in seen_projects:
            seen_projects.add(item.project_id)
            active_projects.append(item.project_id)

    return {
        "agent_profile": runtime.agent_profile.model_dump(mode="json"),
        "executive": {
            "intention": runtime.executive.intention,
            "active_goals": list(runtime.executive.active_goals),
            "queued_work_count": len(runtime.executive.task_queue),
            "capacity_remaining": runtime.executive.capacity_remaining,
            "recommend_pause": runtime.executive.recommend_pause(),
        },
        "commitments": {
            "active_count": commitment_count,
            "status_counts": commitment_status_counts,
            "items": [commitment_operator_snapshot(item) for item in active_commitments],
        },
        "work": {
            "counts": work_counts,
            "active_projects": active_projects[:limit],
            "items": [
                {
                    "work_id": str(item.work_id),
                    "content": item.content,
                    "stage": item.stage.value,
                    "project_id": item.project_id,
                    "commitment_id": item.commitment_id,
                    "blocked_by": item.blocked_by,
                    "meta": item.meta,
                }
                for item in work_items
            ],
            "blocked_items": [
                {
                    "work_id": str(item.work_id),
                    "content": item.content,
                    "stage": item.stage.value,
                    "project_id": item.project_id,
                    "blocked_by": item.blocked_by,
                }
                for item in blocked_items
            ],
        },
        "plans": {
            "active_count": active_plan_count,
            "items": plan_entries,
        },
        "receipts": {
            "recent_count": len(recent_receipts),
            "items": [
                {
                    "receipt_id": str(item.receipt_id),
                    "task_id": str(item.task_id),
                    "objective": item.objective,
                    "success": item.success,
                    "created_at": item.created_at.isoformat(),
                    "completed_at": item.completed_at.isoformat()
                    if item.completed_at
                    else None,
                    "checkpoint_commit": item.checkpoint_commit,
                }
                for item in recent_receipts
            ],
        },
        "consolidation": build_consolidation_status(runtime),
    }
