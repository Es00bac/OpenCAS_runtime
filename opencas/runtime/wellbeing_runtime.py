"""Runtime hooks for operational wellbeing self-maintenance."""

from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from opencas.daydream import DaydreamReflection
from opencas.proof_chain import ProofClaimType, ProofEvidenceKind
from opencas.wellbeing import (
    MaintenanceAction,
    MaintenanceActionType,
    MaintenanceOutcome,
    SelfModificationProposal,
    WellbeingEvent,
    WellbeingRecommendation,
)
from opencas.wellbeing.drift import annotate_drift_drain_effect
from opencas.wellbeing.followup import RecordOnlyMaintenanceFollowupService
from opencas.wellbeing.self_modification import SelfModificationProposalGenerator

OBSERVED_LOWER_RISK_MIN_DELTA = -0.02


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
    outcomes: list[MaintenanceOutcome] = []
    assessment_for_result = assessment
    if store is not None:
        previous_state = await store.latest_state()
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
        outcomes, proposal_path = await _execute_maintenance_actions(
            runtime,
            assessment=assessment,
            actions=plan.actions,
            recommendation_id=str(recommendation.recommendation_id),
            previous_state=previous_state,
            force_action=force_action,
        )
        if outcomes:
            recommendation.status = "executed"
            recommendation.meta["outcome_ids"] = [
                str(outcome.outcome_id) for outcome in outcomes
            ]
        await store.save_recommendation(recommendation)
        for outcome in outcomes:
            await annotate_drift_drain_effect(
                getattr(runtime, "self_inspection_store", None),
                outcome,
            )
            await store.save_maintenance_outcome(outcome)
            await _append_maintenance_action_event(
                store,
                outcome,
                recommendation_id=str(recommendation.recommendation_id),
                grounding=assessment.grounding,
            )
            await _record_maintenance_proof(runtime, outcome)
        if outcomes:
            assessment_for_result = await engine.assess(runtime)
            if builder is not None:
                builder.latest_wellbeing_state = assessment_for_result.state
            await store.save_state(assessment_for_result.state)
        followup_service = RecordOnlyMaintenanceFollowupService(
            store,
            self_inspection_store=getattr(runtime, "self_inspection_store", None),
        )
        followup_summary = (await followup_service.evaluate()).model_dump()
        followup_proposal = await SelfModificationProposalGenerator(
            runtime
        ).ensure_for_followup(
            next_followup_action=followup_summary.get("next_followup_action"),
            state=assessment_for_result.state,
        )
        if followup_proposal is not None:
            followup_summary = (
                await RecordOnlyMaintenanceFollowupService(
                    store,
                    self_inspection_store=getattr(runtime, "self_inspection_store", None),
                ).evaluate()
            ).model_dump()
            if not proposal_path:
                proposal_path = followup_proposal.artifact_path
    else:
        followup_summary = {
            "stuck_loop_count": 0,
            "repeat_count": 0,
            "next_followup_action": None,
            "linked_followup_ids": [],
            "linked_followup_refs": [],
        }

    trace = getattr(runtime, "_trace", None)
    if callable(trace):
        trace(
            "wellbeing_maintenance",
            {
                "overall_risk": assessment.state.overall_risk,
                "actions": [action.action_type.value for action in plan.actions],
                "background_throttle": plan.background_throttle,
                "outcomes": len(outcomes),
            },
        )

    return {
        "available": True,
        "state": assessment_for_result.state.model_dump(mode="json"),
        "concern_dimensions": [
            dimension.value for dimension in assessment_for_result.concern_dimensions
        ],
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
        "stuck_loop_count": followup_summary["stuck_loop_count"],
        "repeat_count": followup_summary["repeat_count"],
        "drift_drain_pending": int(
            (assessment_for_result.state.meta or {}).get("drift_drain_pending") or 0
        ),
        "next_followup_action": followup_summary["next_followup_action"],
        "linked_followup_ids": followup_summary["linked_followup_ids"],
        "linked_followup_refs": followup_summary["linked_followup_refs"],
        "maintenance_outcomes_recorded": len(outcomes),
        "maintenance_outcomes": [
            {
                "id": str(outcome.outcome_id),
                "action_type": outcome.action_type,
                "outcome": outcome.outcome,
                "evidence_ids": outcome.evidence_ids,
                "before_risk": outcome.before_risk,
                "after_risk": outcome.after_risk,
                "meta": outcome.meta,
            }
            for outcome in outcomes
        ],
    }


async def _append_maintenance_action_event(
    store: Any,
    outcome: MaintenanceOutcome,
    *,
    recommendation_id: str,
    grounding: list[Any],
) -> None:
    """Mirror maintenance receipts into the realtime wellbeing event stream."""
    meta = {
        "recommendation_id": recommendation_id,
        "outcome_id": str(outcome.outcome_id),
        "action_type": outcome.action_type,
        "outcome": outcome.outcome,
        "before_risk": outcome.before_risk,
        "after_risk": outcome.after_risk,
        "evidence_ids": list(outcome.evidence_ids),
    }
    for key in (
        "effect_basis",
        "effect_direction",
        "risk_delta",
        "followup_required",
        "drift_drain_status",
        "drift_observations_drained",
    ):
        if key in outcome.meta:
            meta[key] = outcome.meta[key]
    await store.append_event(
        WellbeingEvent(
            event_type="maintenance_action",
            summary=f"{outcome.action_type}: {outcome.outcome}",
            grounding=list(grounding or []),
            meta=meta,
        )
    )
    await store.append_event(
        WellbeingEvent(
            event_type="intervention_dispatched",
            summary=f"{outcome.action_type}: {outcome.outcome}",
            grounding=list(grounding or []),
            meta=meta,
        )
    )


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


async def _record_maintenance_proof(runtime: Any, outcome: MaintenanceOutcome) -> None:
    proof_chain = getattr(runtime, "proof_chain", None)
    if proof_chain is None:
        return
    try:
        claim = await proof_chain.record_claim(
            claim_type=ProofClaimType.MAINTENANCE,
            claim=f"Maintenance action {outcome.action_type} recorded outcome {outcome.outcome}.",
            subject=outcome.action_type,
            meta={"outcome_id": str(outcome.outcome_id)},
        )
        await proof_chain.link_evidence(
            claim.claim_id,
            evidence_kind=ProofEvidenceKind.MAINTENANCE_OUTCOME,
            evidence_id=f"maintenance_outcome:{outcome.outcome_id}",
            summary=outcome.outcome,
            supports_claim=True,
            meta={"action_type": outcome.action_type},
        )
    except Exception as exc:
        trace = getattr(runtime, "_trace", None)
        if callable(trace):
            trace(
                "proof_chain_maintenance_outcome_failed",
                {
                    "outcome_id": str(outcome.outcome_id),
                    "error": str(exc),
                },
            )


async def _execute_maintenance_actions(
    runtime: Any,
    *,
    assessment: Any,
    actions: list[MaintenanceAction],
    recommendation_id: str,
    previous_state: Any | None = None,
    force_action: str | None = None,
) -> tuple[list[MaintenanceOutcome], str]:
    """Execute bounded maintenance actions and return durable receipts.

    Execution here means creating inspectable records or proposal artifacts. It
    never hot-applies prompt, config, or code changes.
    """

    store = getattr(runtime, "wellbeing_store", None)
    outcomes: list[MaintenanceOutcome] = []
    proposal_path = ""
    current_risk = float(getattr(assessment.state, "overall_risk", 0.0) or 0.0)
    evidence_ids = _maintenance_evidence_ids(assessment, recommendation_id)
    forced_self_proposal = force_action == MaintenanceActionType.SELF_MODIFICATION_PROPOSAL.value

    for action in actions:
        action_type = action.action_type
        if action_type == MaintenanceActionType.SELF_MODIFICATION_PROPOSAL:
            proposal = _build_self_modification_proposal(runtime, assessment)
            proposal_path = proposal.artifact_path
            if store is not None:
                await store.save_self_modification_proposal(proposal)
            effect = _maintenance_effect(current_risk, action_type)
            outcomes.append(
                MaintenanceOutcome(
                    action_type=action_type.value,
                    before_risk=effect.get("before_risk", current_risk),
                    after_risk=effect["after_risk"],
                    outcome="proposal_created",
                    evidence_ids=evidence_ids + [str(proposal.proposal_id)],
                    meta={
                        "artifact_path": proposal.artifact_path,
                        "review_required": proposal.review_required,
                        "hot_applied": False,
                        **effect["meta"],
                    },
                )
            )
            continue

        if forced_self_proposal:
            continue

        outcome_name = _outcome_name_for_action(action_type)
        effect = _maintenance_effect(
            current_risk,
            action_type,
            previous_state=previous_state,
            current_state=assessment.state,
        )
        meta = {
            "reason": action.reason,
            "priority": action.priority,
            "hot_applied": False,
            **effect["meta"],
        }
        thread_result = await _maybe_record_maintenance_thread(runtime, action, assessment)
        if thread_result:
            meta["thread_registry"] = thread_result
        outcomes.append(
            MaintenanceOutcome(
                action_type=action_type.value,
                before_risk=effect.get("before_risk", current_risk),
                after_risk=effect["after_risk"],
                outcome=outcome_name,
                evidence_ids=evidence_ids,
                meta=meta,
            )
        )

    if forced_self_proposal and not proposal_path:
        proposal = _build_self_modification_proposal(runtime, assessment)
        proposal_path = proposal.artifact_path
        if store is not None:
            await store.save_self_modification_proposal(proposal)
        effect = _maintenance_effect(current_risk, MaintenanceActionType.SELF_MODIFICATION_PROPOSAL)
        outcomes.append(
            MaintenanceOutcome(
                action_type=MaintenanceActionType.SELF_MODIFICATION_PROPOSAL.value,
                before_risk=effect.get("before_risk", current_risk),
                after_risk=effect["after_risk"],
                outcome="proposal_created",
                evidence_ids=evidence_ids + [str(proposal.proposal_id)],
                meta={
                    "artifact_path": proposal.artifact_path,
                    "review_required": proposal.review_required,
                    "hot_applied": False,
                    "forced": True,
                    **effect["meta"],
                },
            )
        )

    return outcomes, proposal_path


def _maintenance_evidence_ids(assessment: Any, recommendation_id: str) -> list[str]:
    evidence_ids = [f"wellbeing_recommendation:{recommendation_id}"]
    for item in getattr(assessment, "grounding", []) or []:
        for evidence_id in getattr(item, "evidence_ids", []) or []:
            text = str(evidence_id or "").strip()
            if text and text not in evidence_ids:
                evidence_ids.append(text)
    return evidence_ids


def _outcome_name_for_action(action_type: MaintenanceActionType) -> str:
    names = {
        MaintenanceActionType.RECOVER: "background_throttle_recorded",
        MaintenanceActionType.PRESERVE_PROMISES: "promise_preservation_receipt",
        MaintenanceActionType.TRUTH_REPAIR: "truth_repair_receipt",
        MaintenanceActionType.BOUNDARY_CHECK: "boundary_check_receipt",
        MaintenanceActionType.DEEP_THINK: "deep_think_seed_recorded",
        MaintenanceActionType.CURIOSITY_INCUBATE: "curiosity_seed_recorded",
        MaintenanceActionType.OPERATOR_SUMMARY: "stable_state_recorded",
    }
    return names.get(action_type, "maintenance_receipt_recorded")


def _maintenance_effect(
    current_risk: float,
    action_type: MaintenanceActionType,
    *,
    previous_state: Any | None = None,
    current_state: Any | None = None,
) -> dict[str, Any]:
    observed = _observed_lower_risk_payload(
        current_risk,
        action_type,
        previous_state=previous_state,
        current_state=current_state,
    )
    if observed is not None:
        return observed

    if action_type in {
        MaintenanceActionType.OPERATOR_SUMMARY,
        MaintenanceActionType.DEEP_THINK,
        MaintenanceActionType.CURIOSITY_INCUBATE,
        MaintenanceActionType.SELF_MODIFICATION_PROPOSAL,
    }:
        return _effect_payload(
            current_risk,
            current_risk,
            basis="record_only",
            direction="unmeasured",
            followup_required=action_type != MaintenanceActionType.OPERATOR_SUMMARY,
        )
    if action_type in {
        MaintenanceActionType.RECOVER,
        MaintenanceActionType.PRESERVE_PROMISES,
        MaintenanceActionType.TRUTH_REPAIR,
        MaintenanceActionType.BOUNDARY_CHECK,
    }:
        return _effect_payload(
            current_risk,
            max(0.0, current_risk - 0.04),
            basis="estimated",
            direction="expected_lower",
            followup_required=True,
        )
    return _effect_payload(
        current_risk,
        current_risk,
        basis="record_only",
        direction="unmeasured",
        followup_required=True,
    )


def _observed_lower_risk_payload(
    current_risk: float,
    action_type: MaintenanceActionType,
    *,
    previous_state: Any | None,
    current_state: Any | None,
) -> dict[str, Any] | None:
    if action_type != MaintenanceActionType.DEEP_THINK:
        return None
    if previous_state is None or current_state is None:
        return None

    previous_risk = float(getattr(previous_state, "overall_risk", 0.0) or 0.0)
    delta = round(current_risk - previous_risk, 3)
    if delta > OBSERVED_LOWER_RISK_MIN_DELTA:
        return None

    payload = _effect_payload(
        previous_risk,
        current_risk,
        basis="observed",
        direction="observed_lower",
        followup_required=False,
    )
    payload["before_risk"] = round(max(0.0, min(1.0, previous_risk)), 3)
    payload["meta"].update(
        {
            "observed_previous_state_id": str(getattr(previous_state, "state_id", "")),
            "observed_current_state_id": str(getattr(current_state, "state_id", "")),
            "observed_previous_risk": round(max(0.0, min(1.0, previous_risk)), 3),
            "observed_current_risk": round(max(0.0, min(1.0, current_risk)), 3),
        }
    )
    return payload


def _effect_payload(
    before_risk: float,
    after_risk: float,
    *,
    basis: str,
    direction: str,
    followup_required: bool,
) -> dict[str, Any]:
    after = round(max(0.0, min(1.0, after_risk)), 3)
    before = round(max(0.0, min(1.0, before_risk)), 3)
    return {
        "after_risk": after,
        "meta": {
            "effect_basis": basis,
            "effect_direction": direction,
            "risk_delta": round(after - before, 3),
            "followup_required": followup_required,
        },
    }


async def _maybe_record_maintenance_thread(
    runtime: Any,
    action: MaintenanceAction,
    assessment: Any,
) -> dict[str, Any]:
    if action.action_type not in {
        MaintenanceActionType.DEEP_THINK,
        MaintenanceActionType.CURIOSITY_INCUBATE,
        MaintenanceActionType.TRUTH_REPAIR,
    }:
        return {}
    service = getattr(runtime, "thread_registry_service", None)
    if service is None:
        return {}
    try:
        bead = await service.ingest_autonomous_artifact(
            source_ref=f"wellbeing:{assessment.assessment_id}:{action.action_type.value}",
            content=(
                f"{action.action_type.value}: {action.reason} "
                f"overall_risk={assessment.state.overall_risk:.3f}"
            ),
            thread_title="Wellbeing self-maintenance",
            title="Maintenance " + action.action_type.value.replace("_", " "),
            summary=action.reason,
            user_commissioned=False,
        )
    except Exception as exc:
        return {"error": str(exc)}
    return {"bead_id": bead.bead_id, "thread_anchor_id": bead.thread_anchor_id}


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
            (
                "Create a reviewed prompt, config, skill, plugin, or code patch candidate "
                "only if follow-up evidence confirms the pattern."
            ),
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
