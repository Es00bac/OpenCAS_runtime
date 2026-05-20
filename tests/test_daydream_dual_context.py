from __future__ import annotations

from types import SimpleNamespace

import pytest

from opencas.context import ContextLane, ContextProposalStore, ProposalStatus
from opencas.daydream import DaydreamReflection, DaydreamThought, DaydreamThoughtRoute
from opencas.runtime.reflection_runtime import _persist_daydream_context_proposal


class _TruthArbiter:
    async def issue_snapshot(self, *, reason: str = ""):
        return SimpleNamespace(snapshot_id="truth:9:abc", epoch=9)


@pytest.mark.asyncio
async def test_daydream_reflection_persists_reflective_proposal(tmp_path) -> None:
    store = await ContextProposalStore(tmp_path / "proposals.db").connect()
    try:
        runtime = SimpleNamespace(
            context_proposals=store,
            truth_arbiter=_TruthArbiter(),
            _trace=lambda *_args, **_kwargs: None,
        )
        reflection = DaydreamReflection(
            spark_content="The Chapter 3 archive-door idea might be a tempting but unsupported opening.",
            synthesis="Use this as a caution when revising writing project 4246.",
            fascination_thread="writing-project-4246",
            thoughts=[
                DaydreamThought(
                    route=DaydreamThoughtRoute.DISCARD,
                    summary="Do not treat the archive-door image as canon without manuscript evidence.",
                    risk=0.75,
                    confidence=0.8,
                )
            ],
        )

        proposal = await _persist_daydream_context_proposal(
            runtime,
            reflection,
            association_memory_id="memory:daydream-1",
        )

        assert proposal is not None
        loaded = await store.get(proposal.proposal_id)
        assert loaded is not None
        assert loaded.source_lane == ContextLane.REFLECTIVE
        assert loaded.source_snapshot_id == "truth:9:abc"
        assert loaded.source_epoch == 9
        assert loaded.proposal_kind == "bad_idea_to_avoid"
        assert loaded.project_id == "writing-project-4246"
        assert loaded.status == ProposalStatus.PENDING
        assert "archive-door" in loaded.content
        assert "memory:daydream-1" in loaded.evidence_refs
        assert reflection.experience_context["context_proposal_id"] == proposal.proposal_id
    finally:
        await store.close()


@pytest.mark.asyncio
async def test_daydream_reflection_save_failure_does_not_abort_reflection() -> None:
    class _FailingProposalStore:
        async def save(self, _proposal):
            raise RuntimeError("database locked")

    traces = []
    runtime = SimpleNamespace(
        context_proposals=_FailingProposalStore(),
        truth_arbiter=_TruthArbiter(),
        _trace=lambda name, payload: traces.append((name, payload)),
    )
    reflection = DaydreamReflection(
        spark_content="A fragile daydream should still survive store hiccups.",
        synthesis="Keep the reflection, skip only the proposal persistence.",
        fascination_thread="writing-project-4246",
        thoughts=[
            DaydreamThought(
                route=DaydreamThoughtRoute.INCUBATE,
                summary="Chapter 3 store hiccup test.",
                risk=0.1,
                confidence=0.8,
            )
        ],
    )

    proposal = await _persist_daydream_context_proposal(
        runtime,
        reflection,
        association_memory_id="daydream-1",
    )

    assert proposal is None
    assert traces[0][0] == "daydream_context_proposal_save_failed"
