"""Runtime hooks for operational wellbeing self-maintenance."""

from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from opencas.daydream import DaydreamReflection
from opencas.wellbeing import SelfModificationProposal, WellbeingEvent, WellbeingRecommendation


async def run_runtime_wellbeing_maintenance(
    runtime: Any,
    *,
    force_action: str | None = None,
) -> dict[str, Any]:
    """Assess wellbeing and persist a bounded maintenance recommendation."""

    engine = getattr(runtime, "wellbeing_engine", None)
    planner = getattr(runtime, "maintenance_planner", None)
    store = getattr(runtime, "wellbeing_store", None)
    if engine is None or planner is None:
        return {
            "available": False,
            "reason": "wellbeing_components_unavailable",
            "actions": [],
            "hot_applied": False,
        }

    assessment = await engine.assess(runtime)
    builder = getattr(runtime, "builder", None)
    if builder is not None:
        builder.latest_wellbeing_state = assessment.state
    plan = planner.plan(assessment)
    if force_action:
        plan.meta["force_action"] = force_action

    proposal_path = ""
    if store is not None:
        await store.save_state(assessment.state)
        event = WellbeingEvent(
            event_type="assessment",
            summary="wellbeing maintenance assessment",
            grounding=assessment.grounding,
            meta={
                "concern_dimensions": [dimension.value for dimension in assessment.concern_dimensions],
                "overall_risk": assessment.state.overall_risk,
            },
        )
        await store.append_event(event)
        recommendation = WellbeingRecommendation(
            reason="; ".join(action.reason for action in plan.actions[:3]),
            actions=plan.actions,
            grounding=assessment.grounding,
            meta={
                "background_throttle": plan.background_throttle,
                "surface_to_user": plan.surface_to_user,
                "overall_risk": assessment.state.overall_risk,
            },
        )
        await store.save_recommendation(recommendation)
        if force_action == "self_modification_proposal":
            proposal = _build_self_modification_proposal(runtime, assessment)
            proposal_path = proposal.artifact_path
            await store.save_self_modification_proposal(proposal)

    trace = getattr(runtime, "_trace", None)
    if callable(trace):
        trace(
            "wellbeing_maintenance",
            {
                "overall_risk": assessment.state.overall_risk,
                "actions": [action.action_type.value for action in plan.actions],
                "background_throttle": plan.background_throttle,
            },
        )

    return {
        "available": True,
        "state": assessment.state.model_dump(mode="json"),
        "concern_dimensions": [dimension.value for dimension in assessment.concern_dimensions],
        "actions": [
            {
                "type": action.action_type.value,
                "reason": action.reason,
                "priority": action.priority,
            }
            for action in plan.actions
        ],
        "background_throttle": plan.background_throttle,
        "surface_to_user": plan.surface_to_user,
        "hot_applied": False,
        "proposal_path": proposal_path,
    }


async def record_runtime_daydream_wellbeing(runtime: Any, reflection: DaydreamReflection) -> dict[str, Any]:
    """Record daydream thoughts into the runtime fascination graph."""

    graph = getattr(runtime, "fascination_graph", None)
    if graph is None:
        return {"available": False, "recorded": 0}
    recorded = []
    for thought in list(getattr(reflection, "thoughts", []) or []):
        recorded.append(graph.observe_thought(thought))
    trace = getattr(runtime, "_trace", None)
    if callable(trace) and recorded:
        trace(
            "daydream_wellbeing_recorded",
            {
                "reflection_id": str(getattr(reflection, "reflection_id", "")),
                "fascination_count": len(recorded),
            },
        )
    return {
        "available": True,
        "recorded": len(recorded),
        "active": [node.model_dump(mode="json") for node in graph.active(limit=5)],
    }


def _build_self_modification_proposal(runtime: Any, assessment: Any) -> SelfModificationProposal:
    config = getattr(getattr(runtime, "ctx", None), "config", None)
    if config is not None and callable(getattr(config, "agent_workspace_root", None)):
        workspace_root = config.agent_workspace_root()
    else:
        workspace_root = Path.cwd() / "workspace"
    proposal_dir = Path(workspace_root) / "self_maintenance" / "proposals"
    proposal_dir.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    title = "Review wellbeing-driven self-maintenance adjustment"
    slug = "-".join(title.lower().split()[:6])
    artifact_path = proposal_dir / f"{timestamp}-{slug}.md"
    evidence_ids = [
        str(item.evidence_ids[0])
        for item in assessment.grounding
        if getattr(item, "evidence_ids", None)
    ]
    content = "\n".join(
        [
            "# Self-Maintenance Proposal",
            "",
            f"Title: {title}",
            "",
            "Observed problem:",
            (
                f"Wellbeing assessment risk={assessment.state.overall_risk:.3f}, "
                f"truth_pressure={assessment.state.truth_pressure:.3f}, "
                f"drift_load={assessment.state.drift_load:.3f}, "
                f"promise_load={assessment.state.promise_load:.3f}."
            ),
            "",
            "Proposed change:",
            "Create a reviewed prompt, config, skill, plugin, or code patch candidate only if follow-up evidence confirms the pattern.",
            "",
            "Risk summary:",
            "Proposal-only. No prompt, config, or code changes have been applied.",
            "",
            "Tests required before apply:",
            "- targeted regression for the observed behavior",
            "- ruff/compileall for changed Python files",
            "- live verification if runtime behavior changes",
            "",
            "Review state: proposed",
            "",
        ]
    )
    artifact_path.write_text(content, encoding="utf-8")
    return SelfModificationProposal(
        title=title,
        rationale="Wellbeing maintenance was asked to produce a reviewable self-modification proposal.",
        target_kind="prompt_adjustment",
        evidence_ids=evidence_ids,
        review_required=True,
        artifact_path=str(artifact_path),
        risk_summary="Proposal-only; no hot application.",
    )
