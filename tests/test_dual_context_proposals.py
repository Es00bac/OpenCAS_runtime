from __future__ import annotations

import pytest

from opencas.context import (
    ContextAuthority,
    ContextLane,
    ContextProposal,
    ContextProposalStore,
    ProposalStatus,
)


@pytest.mark.asyncio
async def test_reflective_proposals_persist_with_source_labels(tmp_path) -> None:
    store = await ContextProposalStore(tmp_path / "proposals.db").connect()
    try:
        proposal = ContextProposal(
            source_lane=ContextLane.REFLECTIVE,
            source_snapshot_id="truth:1:abc",
            source_epoch=1,
            proposal_kind="bad_idea_to_avoid",
            project_id="writing-project-4246",
            content="Do not repeat the chapter 3 detour without manuscript evidence.",
            evidence_refs=["daydream_signal:9"],
            confidence=0.8,
        )

        await store.save(proposal)
        loaded = await store.get(proposal.proposal_id)

        assert loaded is not None
        assert loaded.source_lane == ContextLane.REFLECTIVE
        assert loaded.authority == ContextAuthority.PROPOSAL
        assert loaded.status == ProposalStatus.PENDING
        assert loaded.evidence_refs == ["daydream_signal:9"]
    finally:
        await store.close()


@pytest.mark.asyncio
async def test_rejected_proposals_remain_retrievable_by_project(tmp_path) -> None:
    store = await ContextProposalStore(tmp_path / "proposals.db").connect()
    try:
        proposal = ContextProposal(
            source_lane=ContextLane.REFLECTIVE,
            source_snapshot_id="truth:1:abc",
            source_epoch=1,
            proposal_kind="memory_association",
            project_id="writing-project-4246",
            content="Maybe Chapter 3 should open with the archive door.",
        )
        await store.save(proposal)
        rejected = await store.mark_rejected(
            proposal.proposal_id,
            reason="No current manuscript evidence supports this opening.",
        )

        project_items = await store.list_by_project("writing-project-4246", include_terminal=True)

        assert rejected is not None
        assert rejected.status == ProposalStatus.REJECTED
        assert rejected.validation["reason"] == "No current manuscript evidence supports this opening."
        assert [item.proposal_id for item in project_items] == [proposal.proposal_id]

    finally:
        await store.close()


@pytest.mark.asyncio
async def test_accepted_proposals_are_queryable_for_baa_task(tmp_path) -> None:
    store = await ContextProposalStore(tmp_path / "proposals.db").connect()
    try:
        proposal = ContextProposal(
            source_lane=ContextLane.REFLECTIVE,
            source_snapshot_id="truth:2:def",
            source_epoch=2,
            proposal_kind="project_next_step",
            project_id="writing-project-4246",
            task_id="task-1",
            content="Use the accepted daydream as a contrast note, not a plot fact.",
        )
        await store.save(proposal)
        await store.mark_accepted(proposal.proposal_id, accepted_by="truth_arbiter")

        task_items = await store.list_by_task("task-1")

        assert len(task_items) == 1
        assert task_items[0].status == ProposalStatus.ACCEPTED
        assert task_items[0].authority == ContextAuthority.EXECUTIVE_COMMITTED
        assert task_items[0].validation["arbiter_decision_id"] == f"accepted:{proposal.proposal_id}"
        assert task_items[0].validation["proposal_validation_status"] == "accepted"
    finally:
        await store.close()


@pytest.mark.asyncio
async def test_proposal_search_requires_relevant_overlap_and_ranks_matches(tmp_path) -> None:
    store = await ContextProposalStore(tmp_path / "proposals.db").connect()
    try:
        broad = ContextProposal(
            source_lane=ContextLane.REFLECTIVE,
            source_snapshot_id="truth:1:abc",
            source_epoch=1,
            proposal_kind="context_note",
            content="A chapter note without the specific project name.",
        )
        relevant = ContextProposal(
            source_lane=ContextLane.REFLECTIVE,
            source_snapshot_id="truth:2:def",
            source_epoch=2,
            proposal_kind="bad_idea_to_avoid",
            project_id="writing project 4246",
            content="writing project 4246 chapter 3 should not reuse the hallway pacing idea.",
        )
        await store.save(broad)
        await store.save(relevant)

        results = await store.search_text("Return to writing project 4246 chapter 3", limit=5)
        generic = await store.search_text("memory", limit=5)

        assert [item.proposal_id for item in results] == [relevant.proposal_id]
        assert generic == []
    finally:
        await store.close()


@pytest.mark.asyncio
async def test_proposal_store_reports_recent_items_and_status_counts(tmp_path) -> None:
    store = await ContextProposalStore(tmp_path / "proposals.db").connect()
    try:
        pending = ContextProposal(
            source_lane=ContextLane.REFLECTIVE,
            source_snapshot_id="truth:1:abc",
            source_epoch=1,
            proposal_kind="context_note",
            content="Pending idea",
        )
        rejected = ContextProposal(
            source_lane=ContextLane.REFLECTIVE,
            source_snapshot_id="truth:2:def",
            source_epoch=2,
            proposal_kind="bad_idea_to_avoid",
            content="Rejected idea",
        )
        await store.save(pending)
        await store.save(rejected)
        await store.mark_rejected(rejected.proposal_id, reason="not supported")

        counts = await store.count_by_status()
        recent = await store.list_recent(limit=5)

        assert counts == {"pending": 1, "rejected": 1}
        assert {item.proposal_id for item in recent} == {
            pending.proposal_id,
            rejected.proposal_id,
        }
    finally:
        await store.close()
