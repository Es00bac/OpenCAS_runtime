from pathlib import Path

import pytest
import pytest_asyncio

from opencas.autonomy.commitment import Commitment
from opencas.bootstrap import BootstrapConfig, BootstrapPipeline
from opencas.cognition.self_inspection import (
    DriftObservation,
    SelfInspectionPhase,
    SelfInspectionRecord,
    build_post_turn_self_inspection_record,
)
from opencas.daydream.models import (
    DaydreamReflection,
    DaydreamThought,
    DaydreamThoughtKind,
    DaydreamThoughtRoute,
)
from opencas.proof_chain import ProofClaimType, ProofEvidenceKind
from opencas.runtime import AgentRuntime
from opencas.thread_registry import BeadSourceKind
from opencas.wellbeing import MaintenanceActionType, WellbeingState


@pytest_asyncio.fixture
async def runtime(tmp_path: Path):
    config = BootstrapConfig(
        state_dir=tmp_path,
        session_id="wellbeing-runtime-session",
    )
    ctx = await BootstrapPipeline(config).run()
    runtime = AgentRuntime(ctx)
    try:
        yield runtime
    finally:
        await runtime._close_stores()


def test_runtime_setup_exposes_wellbeing_components(runtime: AgentRuntime):
    assert runtime.wellbeing_engine is not None
    assert runtime.wellbeing_store is not None
    assert runtime.maintenance_planner is not None


@pytest.mark.asyncio
async def test_high_recovery_need_suppresses_low_value_background_work(runtime: AgentRuntime):
    runtime.ctx.somatic.state.fatigue = 0.9
    runtime.ctx.somatic.state.tension = 0.8

    result = await runtime.run_wellbeing_maintenance()

    assert result["actions"]
    assert result["state"]["recovery_need"] >= 0.7
    assert result["background_throttle"] is True
    assert result["hot_applied"] is False

    latest = await runtime.wellbeing_store.latest_state()
    recommendations = await runtime.wellbeing_store.list_recommendations(limit=5)
    outcomes = await runtime.wellbeing_store.list_maintenance_outcomes(limit=5)
    events = await runtime.wellbeing_store.list_events(limit=10)
    assert latest is not None
    assert recommendations
    assert recommendations[0].status == "executed"
    assert recommendations[0].actions[0].action_type == MaintenanceActionType.RECOVER
    assert outcomes
    assert outcomes[0].evidence_ids
    assert outcomes[0].meta["effect_basis"] in {"observed", "estimated", "record_only"}
    assert "risk_delta" in outcomes[0].meta
    action_events = [event for event in events if event.event_type == "maintenance_action"]
    assert action_events
    assert action_events[0].meta["outcome_id"] == str(outcomes[0].outcome_id)
    assert action_events[0].meta["action_type"] == outcomes[0].action_type
    assert action_events[0].meta["effect_basis"] == outcomes[0].meta["effect_basis"]
    intervention_events = [
        event for event in events if event.event_type == "intervention_dispatched"
    ]
    assert intervention_events
    assert intervention_events[0].meta["outcome_id"] == str(outcomes[0].outcome_id)
    assert intervention_events[0].meta["action_type"] == outcomes[0].action_type
    assert intervention_events[0].meta["evidence_ids"] == outcomes[0].evidence_ids
    assert result["maintenance_outcomes_recorded"] >= 1
    proof_claims = await runtime.proof_store.list_claims(
        claim_type=ProofClaimType.MAINTENANCE,
        limit=5,
    )
    assert proof_claims
    assert proof_claims[0].evidence_links[0].evidence_kind == ProofEvidenceKind.MAINTENANCE_OUTCOME


@pytest.mark.asyncio
async def test_keeper_daydream_updates_fascination_graph(runtime: AgentRuntime):
    thought = DaydreamThought(
        kind=DaydreamThoughtKind.QUESTION,
        route=DaydreamThoughtRoute.INCUBATE,
        summary="Why do repeated repair loops reduce curiosity?",
        usefulness=0.7,
        novelty=0.8,
        confidence=0.6,
    )
    reflection = DaydreamReflection(
        spark_content="Repeated repair loops may reduce curiosity.",
        thoughts=[thought],
        keeper=True,
    )

    await runtime.record_daydream_wellbeing(reflection)

    active = runtime.fascination_graph.active(limit=5)
    assert active
    assert active[0].recurrence_count >= 1


@pytest.mark.asyncio
async def test_self_modification_proposal_writes_workspace_artifact(runtime: AgentRuntime):
    result = await runtime.run_wellbeing_maintenance(
        force_action="self_modification_proposal"
    )

    proposal_path = result["proposal_path"]
    assert "/workspace/self_maintenance/proposals/" in proposal_path
    assert result["hot_applied"] is False

    proposals = await runtime.wellbeing_store.list_self_modification_proposals(limit=5)
    assert proposals
    assert proposals[0].review_required is True


@pytest.mark.asyncio
async def test_drift_maintenance_records_followup_when_effect_is_not_measured(runtime: AgentRuntime):
    record = SelfInspectionRecord(
        session_id="wellbeing-effect-session",
        phase=SelfInspectionPhase.POST_TURN,
        drift_observations=[
            DriftObservation(reason="Repeated unsupported capability framing."),
            DriftObservation(reason="Repeated promise without linked execution."),
        ],
    )
    await runtime.self_inspection_store.save(record)

    result = await runtime.run_wellbeing_maintenance()
    outcomes = await runtime.wellbeing_store.list_maintenance_outcomes(
        action_type=MaintenanceActionType.DEEP_THINK.value,
        limit=5,
    )

    assert any(action["type"] == MaintenanceActionType.DEEP_THINK.value for action in result["actions"])
    assert outcomes
    assert outcomes[0].meta["effect_basis"] == "record_only"
    assert outcomes[0].meta["effect_direction"] == "unmeasured"
    assert outcomes[0].meta["followup_required"] is True
    assert "error" not in outcomes[0].meta.get("thread_registry", {})
    assert outcomes[0].meta["thread_registry"]["bead_id"]
    beads = await runtime.thread_registry_store.list_beads(
        source_kind=BeadSourceKind.AUTONOMOUS_ARTIFACT,
        limit=5,
    )
    assert any(bead.bead_id == outcomes[0].meta["thread_registry"]["bead_id"] for bead in beads)


@pytest.mark.asyncio
async def test_drift_maintenance_records_observed_lower_risk_when_state_declines(
    runtime: AgentRuntime,
):
    previous_state = WellbeingState(
        coherence=0.2,
        autonomy=0.2,
        curiosity=0.2,
        drift_load=1.0,
        truth_pressure=0.8,
        overall_risk=0.95,
    )
    await runtime.wellbeing_store.save_state(previous_state)
    record = SelfInspectionRecord(
        session_id="wellbeing-observed-drift-session",
        phase=SelfInspectionPhase.POST_TURN,
        drift_observations=[
            DriftObservation(reason="Repeated unsupported capability framing."),
            DriftObservation(reason="Repeated promise without linked execution."),
        ],
    )
    await runtime.self_inspection_store.save(record)

    result = await runtime.run_wellbeing_maintenance()
    outcomes = await runtime.wellbeing_store.list_maintenance_outcomes(
        action_type=MaintenanceActionType.DEEP_THINK.value,
        limit=5,
    )

    assert any(action["type"] == MaintenanceActionType.DEEP_THINK.value for action in result["actions"])
    assert outcomes
    outcome = outcomes[0]
    assert outcome.before_risk == previous_state.overall_risk
    assert outcome.after_risk == outcome.meta["observed_current_risk"]
    assert result["state"]["overall_risk"] <= outcome.after_risk
    assert outcome.meta["effect_basis"] == "observed"
    assert outcome.meta["effect_direction"] == "observed_lower"
    assert outcome.meta["observed_previous_state_id"] == str(previous_state.state_id)
    assert outcome.meta["drift_drain_status"] == "drained"
    assert outcome.meta["drift_observations_drained"] >= 1


@pytest.mark.asyncio
async def test_self_modification_action_records_proposal_without_force(runtime: AgentRuntime):
    runtime.ctx.somatic.state.certainty = 0.25
    record = SelfInspectionRecord(
        session_id="wellbeing-drift-session",
        phase=SelfInspectionPhase.POST_TURN,
        drift_observations=[
            DriftObservation(reason="Repeated unsupported capability framing."),
            DriftObservation(reason="Repeated promise without linked execution."),
        ],
    )
    await runtime.self_inspection_store.save(record)

    result = await runtime.run_wellbeing_maintenance()

    assert any(
        action["type"] == MaintenanceActionType.SELF_MODIFICATION_PROPOSAL.value
        for action in result["actions"]
    )
    assert "/workspace/self_maintenance/proposals/" in result["proposal_path"]
    assert result["hot_applied"] is False

    proposals = await runtime.wellbeing_store.list_self_modification_proposals(limit=5)
    outcomes = await runtime.wellbeing_store.list_maintenance_outcomes(
        action_type=MaintenanceActionType.SELF_MODIFICATION_PROPOSAL.value,
        limit=5,
    )
    assert proposals
    assert proposals[0].review_required is True
    assert outcomes
    assert outcomes[0].outcome == "proposal_created"


@pytest.mark.asyncio
async def test_wellbeing_e2e_pressure_recovery_promise_preservation(runtime: AgentRuntime):
    runtime.ctx.somatic.state.fatigue = 0.88
    runtime.ctx.somatic.state.tension = 0.76
    commitment = Commitment(
        content="follow up on the face rendering fix",
        tags=["user_facing"],
        meta={"source": "test", "source_session_id": "wellbeing-e2e"},
    )
    await runtime.commitment_store.save(commitment)
    record = build_post_turn_self_inspection_record(
        session_id="wellbeing-e2e",
        user_input="The face still is not rendering.",
        assistant_output="I will follow up on the face rendering fix.",
        captured_commitments=[commitment],
    )
    await runtime.self_inspection_store.save(record)

    result = await runtime.run_wellbeing_maintenance()

    assert result["state"]["recovery_need"] >= 0.7
    assert any(action["type"] == "preserve_promises" for action in result["actions"])
    assert result["hot_applied"] is False
