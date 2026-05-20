from pathlib import Path

import pytest
import pytest_asyncio

from opencas.bootstrap import BootstrapConfig, BootstrapPipeline
from opencas.runtime import AgentRuntime
from opencas.thread_registry import BeadSourceKind
from opencas.wellbeing import MaintenanceActionType, MaintenanceOutcome
from opencas.wellbeing.followup import RecordOnlyMaintenanceFollowupService
from opencas.wellbeing.self_modification import (
    PROPOSAL_THREAD_ANCHOR_ID,
    SelfModificationProposalGenerator,
)


@pytest_asyncio.fixture
async def runtime(tmp_path: Path):
    config = BootstrapConfig(
        state_dir=tmp_path / "state",
        workspace_root=tmp_path,
        session_id="self-modification-proposal",
    )
    ctx = await BootstrapPipeline(config).run()
    runtime = AgentRuntime(ctx)
    try:
        yield runtime
    finally:
        await runtime._close_stores()


async def _seed_record_only_loop(runtime: AgentRuntime, repeat_count: int = 3) -> dict:
    for _ in range(repeat_count):
        await runtime.wellbeing_store.save_maintenance_outcome(
            MaintenanceOutcome(
                action_type=MaintenanceActionType.DEEP_THINK.value,
                before_risk=0.4,
                after_risk=0.4,
                outcome="deep_think_seed_recorded",
                meta={
                    "effect_basis": "record_only",
                    "effect_direction": "unmeasured",
                    "followup_required": True,
                    "risk_delta": 0.0,
                },
            )
        )
    summary = await RecordOnlyMaintenanceFollowupService(
        runtime.wellbeing_store,
        self_inspection_store=runtime.self_inspection_store,
    ).evaluate()
    assert summary.next_followup_action is not None
    return summary.next_followup_action


@pytest.mark.asyncio
async def test_stuck_loop_followup_generates_reviewable_proposal_artifact(runtime):
    next_action = await _seed_record_only_loop(runtime)

    proposal = await SelfModificationProposalGenerator(runtime).ensure_for_followup(
        next_followup_action=next_action,
        state=None,
    )

    assert proposal is not None
    assert "/workspace/self/proposals/" in proposal.artifact_path
    artifact = Path(proposal.artifact_path)
    assert artifact.exists()
    content = artifact.read_text(encoding="utf-8")
    for section in [
        "## Detected Stuck-Loop Pattern",
        "## Evidence",
        "## Hypothesis",
        "## Proposed Change",
        "## Operator Instruction",
    ]:
        assert section in content

    proposals = await runtime.wellbeing_store.list_self_modification_proposals(limit=5)
    assert len(proposals) == 1
    assert proposals[0].meta["recommendation_id"] == next_action["recommendation_id"]


@pytest.mark.asyncio
async def test_stuck_loop_proposal_generation_is_idempotent(runtime):
    next_action = await _seed_record_only_loop(runtime)
    generator = SelfModificationProposalGenerator(runtime)

    first = await generator.ensure_for_followup(next_followup_action=next_action, state=None)
    second = await generator.ensure_for_followup(next_followup_action=next_action, state=None)

    assert first is not None
    assert second is not None
    assert first.artifact_path == second.artifact_path
    proposals = await runtime.wellbeing_store.list_self_modification_proposals(limit=20)
    assert len(proposals) == 1


@pytest.mark.asyncio
async def test_stuck_loop_proposal_creates_thread_registry_bead(runtime):
    next_action = await _seed_record_only_loop(runtime)

    proposal = await SelfModificationProposalGenerator(runtime).ensure_for_followup(
        next_followup_action=next_action,
        state=None,
    )

    assert proposal is not None
    assert proposal.meta["thread_registry"]["thread_anchor_id"] == PROPOSAL_THREAD_ANCHOR_ID
    beads = await runtime.thread_registry_store.list_beads(
        source_kind=BeadSourceKind.AUTONOMOUS_ARTIFACT,
        thread_anchor_id=PROPOSAL_THREAD_ANCHOR_ID,
        limit=5,
    )
    assert beads


@pytest.mark.asyncio
async def test_runtime_executes_stuck_loop_followup_and_surfaces_artifact_path(runtime):
    await _seed_record_only_loop(runtime)

    result = await runtime.run_wellbeing_maintenance()

    next_action = result["next_followup_action"]
    assert next_action["type"] == "self_modification_proposal"
    assert next_action["proposal_artifact_path"]
    assert Path(next_action["proposal_artifact_path"]).exists()
