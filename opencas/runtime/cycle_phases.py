"""Creative and executive cycle helpers for AgentRuntime.

These helpers keep `AgentRuntime.run_cycle()` orchestration-shaped by owning the
repeatable phase logic for promotion, workspace intervention, and queue drain.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Dict, List, Optional

from opencas.autonomy import WorkObject, WorkStage
from opencas.autonomy.commitment import CommitmentStatus
from opencas.autonomy.executive import ExecutiveState
from opencas.autonomy.intervention import (
    InterventionDecision,
    InterventionKind,
    InterventionPolicy,
)
from opencas.autonomy.workspace import (
    ExecutiveWorkspace,
    ExecutionMode,
    PortfolioBoost,
    WorkspaceItemKind,
)
from opencas.execution.models import RepairTask

from .continuity_breadcrumbs import current_runtime_focus, record_burst_continuity
from .tom_intention_mirror import mirror_runtime_intention, resolve_runtime_intention

if TYPE_CHECKING:
    from .agent_loop import AgentRuntime


@dataclass
class CycleWorkspaceOutcome:
    """Structured output of the workspace/intervention phase."""

    workspace: ExecutiveWorkspace
    decision: InterventionDecision


async def enqueue_promoted_cycle_work(runtime: "AgentRuntime") -> int:
    """Move newly promoted creative work into the executive path for this cycle."""
    promoted_tasks = 0
    for stage in (WorkStage.MICRO_TASK, WorkStage.PROJECT_SEED, WorkStage.PROJECT):
        for work in runtime.creative.list_by_stage(stage):
            if await _work_already_completed(runtime, work):
                await _retire_completed_cycle_work(
                    runtime,
                    work,
                    reason="completed_before_enqueue",
                )
                continue
            if stage == WorkStage.PROJECT:
                await runtime.orchestrator.decompose(work)
                promoted_tasks += 1
                continue
            if not await _should_enqueue_promoted_work(runtime, work):
                continue
            if runtime.executive.enqueue(work):
                promoted_tasks += 1
    return promoted_tasks


async def evaluate_workspace_intervention(
    runtime: "AgentRuntime",
) -> Optional[CycleWorkspaceOutcome]:
    """Rebuild the workspace, choose an intervention, and apply its side effects."""
    try:
        workspace = await _rebuild_workspace(runtime)
        live_orders = await _list_live_work_orders(runtime)
        affective_pressures = await _list_affective_pressures(runtime)
        decision = InterventionPolicy.evaluate(
            workspace=workspace,
            baa_queue_depth=runtime.baa.queue_size,
            held_count=runtime.baa.held_size,
            somatic_recommends_pause=runtime.ctx.somatic.state.fatigue > 0.7,
            live_work_orders=live_orders,
            affective_pressures=affective_pressures,
        )
        await _apply_intervention_decision(runtime, workspace, decision)
        return CycleWorkspaceOutcome(workspace=workspace, decision=decision)
    except Exception as exc:
        runtime._trace("workspace_intervention_error", {"error": str(exc)})
        return None


async def drain_executive_cycle_queue(runtime: "AgentRuntime") -> int:
    """Submit ready executive work to the bounded assistant for this cycle."""
    if _executive_fatigue_pause_active(runtime.executive):
        return 0

    pause_reason = runtime.executive.pause_reason()
    if pause_reason and pause_reason != "overload":
        return 0

    work = runtime.executive.dequeue()
    if work is None:
        return 0

    if await _work_already_completed(runtime, work):
        await _retire_completed_cycle_work(
            runtime,
            work,
            reason="completed_before_dispatch",
        )
        return 0

    runtime.executive.set_intention_from_work(work)
    await mirror_runtime_intention(
        runtime,
        work.content,
        source="active_work_dispatch",
        work_id=str(getattr(work, "work_id", "")),
        stage=getattr(work.stage, "value", str(work.stage)),
    )
    future = await _dispatch_dequeued_work(runtime, work)
    await _resolve_immediate_dispatch_intention(runtime, work, future)
    return 1


def _executive_fatigue_pause_active(executive: ExecutiveState) -> bool:
    somatic = getattr(executive, "somatic", None)
    state = getattr(somatic, "state", None)
    return bool(state is not None and getattr(state, "fatigue", 0.0) > 0.7)


def _enum_value(value: Any) -> str:
    return str(getattr(value, "value", value) or "")


async def _should_enqueue_promoted_work(
    runtime: "AgentRuntime",
    work: WorkObject,
) -> bool:
    if not work.commitment_id or runtime.commitment_store is None:
        return True
    commitment = await runtime.commitment_store.get(work.commitment_id)
    if commitment and commitment.status in (
        CommitmentStatus.BLOCKED,
        CommitmentStatus.ABANDONED,
    ):
        return False
    if commitment:
        ExecutiveState.apply_commitment_execution_bias(work, commitment)
    return True


async def _rebuild_workspace(runtime: "AgentRuntime") -> ExecutiveWorkspace:
    commitments = await _list_workspace_commitments(runtime)
    work_objects = await _list_ready_work_objects(runtime)
    portfolio_boosts = await _build_portfolio_boosts(runtime)
    return ExecutiveWorkspace.rebuild(
        commitments=commitments,
        work_objects=work_objects,
        portfolio_boosts=portfolio_boosts,
        somatic_modulators=runtime.modulators,
        relational=runtime.ctx.relational,
        tom=runtime.tom,
    )


async def _list_workspace_commitments(runtime: "AgentRuntime") -> List[Any]:
    if runtime.commitment_store is None:
        return []
    commitments = await runtime.commitment_store.list_active(limit=100)
    commitments.extend(
        await runtime.commitment_store.list_by_status(
            CommitmentStatus.BLOCKED,
            limit=50,
        )
    )
    return commitments


async def _list_ready_work_objects(runtime: "AgentRuntime") -> List[WorkObject]:
    if runtime.ctx.work_store is None:
        return []
    return await runtime.ctx.work_store.list_ready(limit=100)


async def _build_portfolio_boosts(runtime: "AgentRuntime") -> Dict[str, PortfolioBoost]:
    if runtime.portfolio_store is None:
        return {}
    clusters = await runtime.portfolio_store.list_all(limit=1000)
    boosts: Dict[str, PortfolioBoost] = {}
    for cluster in clusters:
        boost = min(0.15, cluster.spark_count * 0.02)
        boosts[str(cluster.cluster_id)] = PortfolioBoost(
            portfolio_id=str(cluster.cluster_id),
            spark_count=cluster.spark_count,
            boost=boost,
        )
    return boosts


async def _list_live_work_orders(runtime: "AgentRuntime") -> List[Dict[str, Any]]:
    if runtime.ctx.tasks is None:
        return []
    pending_tasks = await runtime.ctx.tasks.list_pending(limit=100)
    return [
        {
            "task_id": str(task.task_id),
            "stage": task.stage.value,
            "objective": task.objective,
        }
        for task in pending_tasks
    ]


async def _list_affective_pressures(runtime: "AgentRuntime") -> List[Dict[str, Any]]:
    service = getattr(getattr(runtime, "ctx", None), "affective_examinations", None)
    loader = getattr(service, "list_unresolved_pressures", None)
    if not callable(loader):
        return []
    session_id = getattr(getattr(runtime.ctx, "config", None), "session_id", None)
    try:
        records = await loader(session_id=session_id, limit=8)
    except Exception:
        return []
    pressures: list[dict[str, Any]] = []
    for record in records:
        pressure = _enum_value(getattr(record, "action_pressure", ""))
        if pressure in {"", "continue", "archive_only"}:
            continue
        affect = getattr(record, "affect", None)
        pressures.append(
            {
                "examination_id": str(getattr(record, "examination_id", "")),
                "source_type": _enum_value(getattr(record, "source_type", "")),
                "source_id": getattr(record, "source_id", ""),
                "action_pressure": pressure,
                "primary_emotion": _enum_value(getattr(affect, "primary_emotion", "")),
                "intensity": getattr(record, "intensity", 0.0),
                "confidence": getattr(record, "confidence", 0.0),
                "bounded_reason": getattr(record, "bounded_reason", ""),
            }
        )
    return pressures


async def _apply_intervention_decision(
    runtime: "AgentRuntime",
    workspace: ExecutiveWorkspace,
    decision: InterventionDecision,
) -> None:
    # Keep intervention side effects in one place so the runtime can evolve the
    # policy and execution responses without re-splicing them into run_cycle().
    if decision.kind == InterventionKind.LAUNCH_BACKGROUND:
        if workspace.focus and workspace.focus.execution_mode == ExecutionMode.BACKGROUND_AGENT:
            repair_task = RepairTask(
                objective=workspace.focus.content,
                project_id=workspace.focus.project_id,
                commitment_id=workspace.focus.commitment_id,
                meta={"source": "intervention_launch_background"},
            )
            await runtime.baa.submit(repair_task)
        return

    if decision.kind == InterventionKind.RETIRE_OR_DEFER_FOCUS:
        if workspace.focus:
            await _retire_or_defer_focus(runtime, workspace.focus, decision.reason)
        return

    if decision.kind in (
        InterventionKind.SURFACE_CLARIFICATION,
        InterventionKind.SURFACE_APPROVAL,
    ):
        runtime._trace(
            "intervention_surface",
            {
                "kind": decision.kind.value,
                "target": decision.target_item_id,
                "reason": decision.reason,
            },
        )
        return

    if decision.kind == InterventionKind.VERIFY_COMPLETED_WORK:
        runtime._trace(
            "intervention_verify",
            {
                "target": decision.target_item_id,
                "reason": decision.reason,
            },
        )
        return

    if decision.kind == InterventionKind.RECLAIM_TO_FOREGROUND:
        runtime._trace(
            "intervention_reclaim",
            {
                "target": decision.target_item_id,
                "reason": decision.reason,
            },
        )


async def _retire_or_defer_focus(
    runtime: "AgentRuntime",
    focus: Any,
    reason: str,
) -> None:
    if focus.kind == WorkspaceItemKind.TASK:
        focus_id = str(focus.item_id)
        removed = runtime.executive.remove_work(focus_id)
        if not removed and runtime.ctx.work_store:
            await runtime.ctx.work_store.delete(focus_id)
        return

    runtime._trace(
        "intervention_defer_non_task_focus",
        {
            "target": str(focus.item_id),
            "kind": focus.kind.value,
            "reason": reason,
        },
    )


async def _dispatch_dequeued_work(runtime: "AgentRuntime", work: WorkObject) -> Any:
    try:
        work_id = getattr(work, "work_id", None)
        focus = current_runtime_focus(runtime, work.content)
        await record_burst_continuity(
            runtime,
            trigger="work_burst_started",
            phase="start",
            intent=f"Dispatching burst for {work.content[:80]}",
            focus=focus,
            next_step="await BAA completion and goal resolution",
            note=f"work_id={work_id}" if work_id else None,
        )
    except Exception:
        runtime._trace(
            "continuity_breadcrumb_dispatch_error",
            {
                "work_id": str(work.work_id) if hasattr(work, "work_id") else str(work.meta.get("work_id")) if work.meta else None
            },
        )

    if work.stage == WorkStage.PROJECT:
        await runtime.orchestrator.decompose(work)
        return None

    meta: Dict[str, Any] = {}
    if work.meta:
        meta.update(work.meta)
    repair_task = RepairTask(
        objective=work.content,
        project_id=work.project_id,
        commitment_id=work.commitment_id,
        meta=meta,
    )
    return await runtime.baa.submit(repair_task)


async def _resolve_immediate_dispatch_intention(
    runtime: "AgentRuntime",
    work: WorkObject,
    future: Any,
) -> None:
    if future is None or not getattr(future, "done", lambda: False)():
        return
    try:
        result = future.result()
    except Exception as exc:
        runtime._trace(
            "tom_intention_immediate_result_error",
            {"work_id": str(getattr(work, "work_id", "")), "error": str(exc)},
        )
        return
    if not bool(getattr(result, "success", False)):
        return
    await resolve_runtime_intention(runtime, work.content)
    await _retire_completed_cycle_work(
        runtime,
        work,
        reason="immediate_success_result",
    )


async def _work_already_completed(
    runtime: "AgentRuntime",
    work: WorkObject,
    *,
    limit: int = 500,
) -> bool:
    task_store = getattr(getattr(runtime, "ctx", None), "tasks", None)
    task_lister = getattr(task_store, "list_all", None)
    if not callable(task_lister):
        return False
    target = _normalize_cycle_work_text(work.content)
    if not target:
        return False
    try:
        tasks = await task_lister(limit=limit)
    except Exception as exc:
        runtime._trace(
            "completed_work_lookup_error",
            {"work_id": str(getattr(work, "work_id", "")), "error": str(exc)},
        )
        return False
    return any(
        _task_completed_successfully(task)
        and _normalize_cycle_work_text(getattr(task, "objective", "")) == target
        for task in tasks
    )


async def _retire_completed_cycle_work(
    runtime: "AgentRuntime",
    work: WorkObject,
    *,
    reason: str,
) -> None:
    work_id = str(getattr(work, "work_id", "") or "")
    if not work_id:
        return
    creative = getattr(runtime, "creative", None)
    remover = getattr(creative, "remove", None)
    removed_from_creative = False
    if callable(remover):
        try:
            removed_from_creative = bool(remover(work_id))
        except Exception as exc:
            runtime._trace(
                "completed_work_creative_remove_error",
                {"work_id": work_id, "reason": reason, "error": str(exc)},
            )
    work_store = getattr(getattr(runtime, "ctx", None), "work_store", None)
    deleter = getattr(work_store, "delete", None)
    deleted_from_store = False
    if callable(deleter):
        try:
            deleted_from_store = bool(await deleter(work_id))
        except Exception as exc:
            runtime._trace(
                "completed_work_store_delete_error",
                {"work_id": work_id, "reason": reason, "error": str(exc)},
            )
    runtime._trace(
        "completed_cycle_work_retired",
        {
            "work_id": work_id,
            "reason": reason,
            "removed_from_creative": removed_from_creative,
            "deleted_from_store": deleted_from_store,
        },
    )


def _task_completed_successfully(task: Any) -> bool:
    stage = _enum_value(getattr(task, "stage", "")).lower()
    status = str(getattr(task, "status", "") or "").lower()
    return stage == "done" and status == "completed"


def _normalize_cycle_work_text(content: Any) -> str:
    return " ".join(str(content or "").strip().lower().split())
