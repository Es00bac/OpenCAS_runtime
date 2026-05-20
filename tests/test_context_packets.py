from __future__ import annotations

from types import SimpleNamespace

import pytest

from opencas.context import (
    ContextLane,
    ContextPacketBuilder,
    ContextProposal,
    ContextProposalStore,
)


class _SnapshotArbiter:
    async def issue_snapshot(self, *, reason: str = ""):
        return SimpleNamespace(
            snapshot_id="truth:7:abc",
            epoch=7,
            executive={
                "intention": "Revise writing project 4246",
                "intention_source": "operator_requested",
                "active_goals": ["Revise Chapter 3"],
                "queue_size": 1,
            },
            baa={"live_task_ids": ["task-1"], "live_task_count": 1},
            schedules={"upcoming": [{"schedule_id": "schedule-1"}]},
            commitments={"status_counts": {"active": 1}},
            receipts={"recent_receipt_ids": ["receipt-1"]},
            runtime_activity={"activity": "working"},
        )


@pytest.mark.asyncio
async def test_executive_packet_uses_live_truth_and_accepted_proposals(tmp_path) -> None:
    proposal_store = await ContextProposalStore(tmp_path / "proposals.db").connect()
    try:
        proposal = ContextProposal(
            source_lane=ContextLane.REFLECTIVE,
            source_snapshot_id="truth:6:old",
            source_epoch=6,
            proposal_kind="project_next_step",
            project_id="writing-project-4246",
            content="Use the daydream as contrast only, not as manuscript fact.",
        )
        await proposal_store.save(proposal)
        await proposal_store.mark_accepted(proposal.proposal_id, accepted_by="truth_arbiter")
        builder = ContextPacketBuilder(
            runtime=SimpleNamespace(),
            truth_arbiter=_SnapshotArbiter(),
            proposal_store=proposal_store,
        )

        packet = await builder.build_executive_packet(
            user_input="Continue Chapter 3.",
            project_id="writing-project-4246",
        )

        rendered = packet.render()
        assert packet.lane == ContextLane.EXECUTIVE
        assert packet.truth_snapshot_id == "truth:7:abc"
        assert packet.truth_epoch == 7
        assert packet.can_write is True
        assert "LIVE FACT" in rendered
        assert "Revise writing project 4246" in rendered
        assert "ACCEPTED SUPPORT" in rendered
        assert "Use the daydream as contrast only" in rendered
    finally:
        await proposal_store.close()


@pytest.mark.asyncio
async def test_reflective_packet_has_no_write_authority_and_includes_rejected_ideas(tmp_path) -> None:
    proposal_store = await ContextProposalStore(tmp_path / "proposals.db").connect()
    try:
        proposal = ContextProposal(
            source_lane=ContextLane.REFLECTIVE,
            source_snapshot_id="truth:4:old",
            source_epoch=4,
            proposal_kind="bad_idea_to_avoid",
            project_id="writing-project-4246",
            content="Do not treat an unverified daydream as canon.",
        )
        await proposal_store.save(proposal)
        await proposal_store.mark_rejected(proposal.proposal_id, reason="No manuscript evidence.")
        builder = ContextPacketBuilder(
            runtime=SimpleNamespace(),
            truth_arbiter=_SnapshotArbiter(),
            proposal_store=proposal_store,
        )

        packet = await builder.build_reflective_packet(
            user_input="Think about Chapter 3.",
            project_id="writing-project-4246",
        )

        rendered = packet.render()
        assert packet.lane == ContextLane.REFLECTIVE
        assert packet.can_write is False
        assert "NO WRITE AUTHORITY" in rendered
        assert "REJECTED IDEA" in rendered
        assert "Do not treat an unverified daydream as canon." in rendered
    finally:
        await proposal_store.close()
