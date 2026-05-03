from pathlib import Path

import pytest
import pytest_asyncio

from opencas.autonomy.commitment import Commitment
from opencas.bootstrap import BootstrapConfig, BootstrapPipeline
from opencas.cognition.self_inspection import build_post_turn_self_inspection_record
from opencas.daydream.models import (
    DaydreamReflection,
    DaydreamThought,
    DaydreamThoughtKind,
    DaydreamThoughtRoute,
)
from opencas.runtime import AgentRuntime
from opencas.wellbeing import MaintenanceActionType


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
    assert latest is not None
    assert recommendations
    assert recommendations[0].actions[0].action_type == MaintenanceActionType.RECOVER


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
