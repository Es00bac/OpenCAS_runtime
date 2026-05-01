"""Status and workflow snapshots for AgentRuntime.

These builders keep monitoring and operator-facing status assembly out of the
main runtime loop so `AgentRuntime` can stay focused on orchestration.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, Dict, List, Optional

from opencas.autonomy.commitment import CommitmentStatus, commitment_operator_snapshot
from opencas.runtime.consolidation_state import load_consolidation_runtime_state
from opencas.runtime.consolidation_worker import load_consolidation_worker_status

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
        config = getattr(getattr(runtime, "ctx", None), "config", None)
        state_dir = getattr(config, "state_dir", None)
        if state_dir is None:
            return {"available": False}
        persisted = load_consolidation_runtime_state(state_dir)
        worker_status = load_consolidation_worker_status(state_dir)
        last_run_at = persisted.get("last_run_at")
        if not last_run_at:
            return {"available": False, "worker": worker_status or None}
        return {
            "available": True,
            "timestamp": last_run_at,
            "result_id": persisted.get("last_result_id"),
            "clusters_formed": 0,
            "memories_created": 0,
            "commitments_consolidated": 0,
            "commitment_clusters_formed": 0,
            "commitment_work_objects_created": 0,
            "commitments_extracted_from_chat": 0,
            "episodes_pruned": 0,
            "persisted_only": True,
            "worker": worker_status or None,
        }
    state_dir = getattr(getattr(getattr(runtime, "ctx", None), "config", None), "state_dir", None)
    worker_status = load_consolidation_worker_status(state_dir) if state_dir is not None else {}
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
        "persisted_only": False,
        "worker": result.get("worker") or worker_status or None,
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

    project_ledger = []
    if getattr(runtime, "project_resume", None) is not None:
        project_ledger = await runtime.project_resume.list_projects(
            limit=limit,
            project_id=project_id,
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

    executive = runtime.executive
    if hasattr(executive, "queue_summary"):
        queue_summary = executive.queue_summary()
    else:
        queue_metadata = executive.queue_metadata()
        queue_summary = {
            "counts": {
                "total": len(queue_metadata),
                "active": sum(1 for item in queue_metadata if item.get("state") == "active"),
                "held": sum(1 for item in queue_metadata if item.get("state") == "held"),
                "ready": sum(1 for item in queue_metadata if item.get("bearing") == "ready"),
                "queued": sum(1 for item in queue_metadata if item.get("bearing") == "queued"),
                "waiting": sum(1 for item in queue_metadata if item.get("bearing") == "waiting"),
            },
            "items": queue_metadata,
        }

    return {
        "agent_profile": runtime.agent_profile.model_dump(mode="json"),
        "executive": {
            "intention": runtime.executive.intention,
            "active_goals": list(runtime.executive.active_goals),
            "parked_goal_count": len(list(getattr(runtime.executive, "parked_goals", []) or [])),
            "parked_goals": list(getattr(runtime.executive, "parked_goals", []) or []),
            "parked_goal_metadata": dict(getattr(runtime.executive, "parked_goal_metadata", {}) or {}),
            "queued_work_count": queue_summary["counts"]["total"],
            "weighted_load": getattr(runtime.executive, "weighted_queue_load", lambda: 0.0)(),
            "capacity_remaining": runtime.executive.capacity_remaining,
            "recommend_pause": runtime.executive.recommend_pause(),
            "queue": {
                "counts": queue_summary["counts"],
                "items": queue_summary["items"],
            },
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
        "project_ledger": {
            "count": len(project_ledger),
            "items": [
                {
                    "signature": item.signature,
                    "display_name": item.display_name,
                    "canonical_artifact_path": item.canonical_artifact_path,
                    "supporting_artifact_paths": item.supporting_artifact_paths,
                    "synopsis": item.synopsis,
                    "source_surfaces": item.source_surfaces,
                    "active_work_count": item.active_work_count,
                    "active_plan_count": item.active_plan_count,
                    "primary_loop_id": item.primary_loop_id,
                    "duplicate_loop_ids": item.duplicate_loop_ids,
                    "matched_project_ids": item.matched_project_ids,
                    "has_live_workstream": item.has_live_workstream,
                    "retry_state": item.retry_state,
                    "best_next_step": item.best_next_step,
                    "latest_salvage_packet_id": item.latest_salvage_packet_id,
                    "last_salvage_outcome": item.last_salvage_outcome,
                }
                for item in project_ledger
            ],
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
