from __future__ import annotations

from types import SimpleNamespace

import pytest

from opencas.context import ContextAuthority, ContextLane, ContextProposal, ContextProposalStore, ProposalStatus
from opencas.execution import BoundedAssistantAgent, RepairTask
from opencas.execution.executor import RepairExecutor
from opencas.execution.store import TaskStore
from opencas.tools import ToolRegistry


@pytest.mark.asyncio
async def test_baa_holds_uncommitted_reflective_origin_task() -> None:
    baa = BoundedAssistantAgent(tools=ToolRegistry(), max_concurrent=1)
    task = RepairTask(
        objective="Use a daydream idea to revise Chapter 3.",
        meta={
            "origin_context_lane": "reflective",
            "context_truth_epoch": 2,
            "context_truth_snapshot_id": "truth:2:def",
        },
    )

    future = await baa.submit(task)

    assert future is not None
    assert baa.queue_size == 0
    assert str(task.task_id) in baa._held
    assert task.meta["context_guard_status"] == "held"
    assert task.meta["context_guard_reason"] == "reflective_proposal_not_committed"


@pytest.mark.asyncio
async def test_context_guarded_reflective_task_is_not_dependency_released() -> None:
    baa = BoundedAssistantAgent(tools=ToolRegistry(), max_concurrent=1)
    task = RepairTask(
        objective="Let an unaccepted daydream schedule work.",
        meta={
            "origin_context_lane": "reflective",
            "context_truth_epoch": 2,
            "context_truth_snapshot_id": "truth:2:def",
        },
    )

    await baa.submit(task)
    released = await baa.try_release_held()

    assert released == 0
    assert baa.queue_size == 0
    assert str(task.task_id) in baa._held
    assert task.meta["context_guard_status"] == "held"
    assert task.meta["context_guard_reason"] == "reflective_proposal_not_committed"


@pytest.mark.asyncio
async def test_context_guarded_reflective_task_is_not_operator_released() -> None:
    baa = BoundedAssistantAgent(tools=ToolRegistry(), max_concurrent=1)
    task = RepairTask(
        objective="Manually release an unaccepted daydream idea.",
        meta={
            "origin_context_lane": "reflective",
            "context_truth_epoch": 2,
            "context_truth_snapshot_id": "truth:2:def",
        },
    )
    task_id = str(task.task_id)
    await baa.submit(task)

    resolved = await baa.resolve_hold(task_id)

    assert resolved is False
    assert baa.queue_size == 0
    assert task_id in baa._held
    assert task.meta["context_guard_status"] == "held"
    assert task.meta["context_guard_reason"] == "reflective_proposal_not_committed"


@pytest.mark.asyncio
async def test_baa_accepts_reflective_task_after_arbiter_commit() -> None:
    baa = BoundedAssistantAgent(tools=ToolRegistry(), max_concurrent=1)
    task = RepairTask(
        objective="Use accepted contrast note in Chapter 3 revision.",
        meta={
            "origin_context_lane": "reflective",
            "context_truth_epoch": 3,
            "context_truth_snapshot_id": "truth:3:abc",
            "authority": "executive_committed",
            "proposal_validation_status": "accepted",
            "accepted_proposal_ids": ["proposal:1"],
            "arbiter_decision_id": "decision:1",
        },
    )

    await baa.submit(task)

    assert str(task.task_id) not in baa._held
    assert baa.queue_size == 1


@pytest.mark.asyncio
async def test_baa_accepts_reflective_task_from_accepted_proposal_store(tmp_path) -> None:
    proposal_store = await ContextProposalStore(tmp_path / "context_proposals.db").connect()
    proposal = ContextProposal(
        source_lane=ContextLane.REFLECTIVE,
        source_snapshot_id="truth:3:abc",
        source_epoch=3,
        proposal_kind="project_next_step",
        content="Use the Chapter 3 daydream only after executive acceptance.",
        evidence_refs=["memory:daydream-1"],
    )
    await proposal_store.save(proposal)
    accepted = await proposal_store.mark_accepted(proposal.proposal_id, accepted_by="truth_arbiter")
    assert accepted is not None

    runtime = SimpleNamespace(
        context_proposals=proposal_store,
        ctx=SimpleNamespace(context_proposal_store=proposal_store),
    )
    baa = BoundedAssistantAgent(tools=ToolRegistry(), runtime=runtime, max_concurrent=1)
    task = RepairTask(
        objective="Use accepted daydream support in Chapter 3 revision.",
        meta={
            "origin_context_lane": "reflective",
            "accepted_proposal_ids": [proposal.proposal_id],
        },
    )

    try:
        await baa.submit(task)
    finally:
        await proposal_store.close()

    assert str(task.task_id) not in baa._held
    assert baa.queue_size == 1
    assert task.meta["authority"] == "executive_committed"
    assert task.meta["proposal_validation_status"] == "accepted"
    assert task.meta["arbiter_decision_id"].startswith("accepted_proposals:")


@pytest.mark.asyncio
async def test_baa_releases_previously_held_reflective_task_after_arbiter_commit() -> None:
    baa = BoundedAssistantAgent(tools=ToolRegistry(), max_concurrent=1)
    task = RepairTask(
        objective="Use accepted contrast note in Chapter 3 revision.",
        meta={
            "origin_context_lane": "reflective",
            "context_truth_epoch": 3,
            "context_truth_snapshot_id": "truth:3:abc",
        },
    )
    task_id = str(task.task_id)
    await baa.submit(task)

    task.meta.update(
        {
            "authority": "executive_committed",
            "proposal_validation_status": "accepted",
            "accepted_proposal_ids": ["proposal:1"],
            "arbiter_decision_id": "decision:1",
        }
    )
    resolved = await baa.resolve_hold(task_id)

    assert resolved is True
    assert task_id not in baa._held
    assert baa.queue_size == 1
    assert task.meta["context_guard_status"] == "released"
    assert "context_guard_reason" not in task.meta


@pytest.mark.asyncio
async def test_baa_restores_context_guarded_pending_task_as_held(tmp_path, monkeypatch) -> None:
    store = await TaskStore(tmp_path / "tasks.db").connect()
    task = RepairTask(
        objective="Restore an unaccepted daydream task.",
        meta={
            "origin_context_lane": "reflective",
            "context_truth_epoch": 2,
            "context_truth_snapshot_id": "truth:2:def",
        },
    )
    await store.save(task)

    baa = BoundedAssistantAgent(tools=ToolRegistry(), store=store, max_concurrent=1)
    monkeypatch.setattr(baa._lanes, "start", lambda *, worker_factory: None)

    try:
        await baa.start()

        assert str(task.task_id) in baa._held
        assert baa.queue_size == 0
        restored = await store.get(str(task.task_id))
        assert restored is not None
        assert restored.meta["context_guard_status"] == "held"
        assert restored.meta["context_guard_reason"] == "reflective_proposal_not_committed"
    finally:
        await store.close()


@pytest.mark.asyncio
async def test_baa_accepts_ordinary_executive_task_without_proposal_metadata() -> None:
    baa = BoundedAssistantAgent(tools=ToolRegistry(), max_concurrent=1)
    task = RepairTask(objective="Run the ordinary background repair.")

    await baa.submit(task)

    assert str(task.task_id) not in baa._held
    assert baa.queue_size == 1


@pytest.mark.asyncio
async def test_baa_executor_injects_accepted_proposal_context(tmp_path) -> None:
    proposal_store = await ContextProposalStore(tmp_path / "context_proposals.db").connect()
    proposal = ContextProposal(
        source_lane=ContextLane.REFLECTIVE,
        source_snapshot_id="truth:3:abc",
        source_epoch=3,
        proposal_kind="project_next_step",
        content="Use the quiet Chapter 3 contrast note as accepted support.",
        evidence_refs=["daydream_reflection:chapter-3"],
        status=ProposalStatus.ACCEPTED,
        authority=ContextAuthority.EXECUTIVE_COMMITTED,
    )
    await proposal_store.save(proposal)

    class _ToolLoop:
        def __init__(self) -> None:
            self.messages = []

        async def run(self, *, messages, **_kwargs):
            self.messages = messages
            return SimpleNamespace(
                guard_fired=False,
                guard_reason=None,
                final_output="used context",
                tool_calls=[],
            )

    tool_loop = _ToolLoop()
    runtime = SimpleNamespace(
        tool_loop=tool_loop,
        context_proposals=proposal_store,
        scheduler=None,
    )
    executor = RepairExecutor(tools=ToolRegistry(), runtime=runtime)
    task = RepairTask(
        objective="Revise Chapter 3 from accepted proposal.",
        meta={
            "origin_context_lane": "reflective",
            "authority": "executive_committed",
            "proposal_validation_status": "accepted",
            "arbiter_decision_id": "decision:1",
            "accepted_proposal_ids": [proposal.proposal_id],
            "context_truth_snapshot_id": "truth:3:abc",
            "context_truth_epoch": 3,
        },
    )

    try:
        output = await executor._execute_plan(task, "Use the accepted support.")
    finally:
        await proposal_store.close()

    assert output == "used context"
    system_content = tool_loop.messages[0]["content"]
    assert "Dual-context execution context:" in system_content
    assert "origin_context_lane=reflective" in system_content
    assert "Accepted proposal support (project_next_step)" in system_content
    assert "quiet Chapter 3 contrast note" in system_content
    assert proposal.proposal_id in system_content
