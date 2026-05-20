from __future__ import annotations

from types import SimpleNamespace

import pytest

from opencas.context import ContextLane, ContextProposal, ProposalStatus
from opencas.runtime.context_proposal_intake import intake_context_proposals


class _ProposalStore:
    def __init__(self, proposals: list[ContextProposal]) -> None:
        self.proposals = proposals
        self.saved: list[ContextProposal] = []

    async def list_recent(self, *, status=None, limit=20):
        status_value = getattr(status, "value", status)
        items = [
            proposal
            for proposal in self.proposals
            if status_value is None or proposal.status.value == status_value
        ]
        return items[:limit]

    async def save(self, proposal: ContextProposal) -> None:
        self.saved.append(proposal)


class _CognitiveStore:
    def __init__(self) -> None:
        self.attention: list[tuple[str, dict]] = []
        self.working_memory: list[tuple[str, str, dict]] = []
        self.events: list[tuple[object, str, dict]] = []

    async def upsert_attention(self, label: str, **kwargs):
        self.attention.append((label, kwargs))
        return SimpleNamespace(target_id=f"attention-{len(self.attention)}")

    async def upsert_working_memory(self, slot: str, content: str, **kwargs):
        self.working_memory.append((slot, content, kwargs))
        return SimpleNamespace(item_id=f"memory-{len(self.working_memory)}")

    async def record_event(self, kind, summary: str, **kwargs):
        self.events.append((kind, summary, kwargs))
        return SimpleNamespace(event_id=f"event-{len(self.events)}")


class _SignalStore:
    def __init__(self) -> None:
        self.saved = []

    async def save_signal(self, signal) -> None:
        self.saved.append(signal)


class _Promotion:
    def __init__(self) -> None:
        self.signal_store = _SignalStore()
        self.routed = []

    async def route_signal(self, signal):
        self.routed.append(signal)
        return {"status": "routed", "route": signal.suggested_route.value, "signal_id": signal.signal_id}


def _proposal(*, content: str, validation: dict | None = None, confidence: float = 0.86) -> ContextProposal:
    return ContextProposal(
        source_lane=ContextLane.REFLECTIVE,
        source_snapshot_id="truth:1:test",
        source_epoch=1,
        proposal_kind="observed_context_relevance_followup",
        content=content,
        evidence_refs=["youtube_transcript:video.txt"],
        confidence=confidence,
        validation=validation or {},
    )


@pytest.mark.asyncio
async def test_context_proposal_intake_routes_actionable_next_step() -> None:
    proposal = _proposal(
        content=(
            "Observed context relevance follow-up.\n"
            "Connection: The media connects autonomy claims to evidence receipts.\n"
            "Agent viewpoint: Autonomy claims should show proof, not only phrasing.\n"
            "Next step: Create an autonomy-change checklist for future OpenCAS feature work."
        ),
        validation={"source": "desktop_context", "work_relevant": True, "salience": 0.82},
    )
    proposal_store = _ProposalStore([proposal])
    cognitive_store = _CognitiveStore()
    promotion = _Promotion()
    runtime = SimpleNamespace(context_proposals=proposal_store, daydream_promotion=promotion)

    result = await intake_context_proposals(runtime, cognitive_store)

    assert result["eligible"] == 1
    assert result["attention_updates"] == 1
    assert result["working_memory_updates"] == 1
    assert result["self_work_signals"] == 1
    assert cognitive_store.working_memory[0][1].startswith("Create an autonomy-change checklist")
    assert promotion.routed[0].source_mode == "context_proposal_intake"
    assert promotion.routed[0].source_reflection_id == proposal.proposal_id
    assert proposal.validation["context_proposal_intake"]["actionable"] is True
    assert proposal.validation["context_proposal_intake"]["signal_status"] == "routed"


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "next_step",
    [
        "Keep listening for concrete claims before preserving a deeper follow-up.",
        "Wait for playback or a fuller transcript segment before commenting.",
    ],
)
async def test_context_proposal_intake_keeps_monitoring_notes_non_executable(next_step: str) -> None:
    proposal = _proposal(
        content=(
            "Observed context relevance follow-up.\n"
            "Agent viewpoint: This may become useful if a later concrete claim appears.\n"
            f"Next step: {next_step}"
        ),
        validation={"source": "desktop_context", "work_relevant": True, "salience": 0.78},
    )
    proposal_store = _ProposalStore([proposal])
    cognitive_store = _CognitiveStore()
    promotion = _Promotion()
    runtime = SimpleNamespace(context_proposals=proposal_store, daydream_promotion=promotion)

    result = await intake_context_proposals(runtime, cognitive_store)

    assert result["eligible"] == 1
    assert result["attention_updates"] == 1
    assert result["working_memory_updates"] == 0
    assert result["self_work_signals"] == 0
    assert promotion.routed == []
    assert proposal.validation["context_proposal_intake"]["actionable"] is False


@pytest.mark.asyncio
async def test_context_proposal_intake_does_not_duplicate_ingested_proposal() -> None:
    proposal = _proposal(
        content="Observed context relevance follow-up. Next step: Review the prior artifact.",
        validation={"context_proposal_intake": {"ingested_at": "already"}},
    )
    proposal.status = ProposalStatus.PENDING
    proposal_store = _ProposalStore([proposal])
    cognitive_store = _CognitiveStore()
    promotion = _Promotion()
    runtime = SimpleNamespace(context_proposals=proposal_store, daydream_promotion=promotion)

    result = await intake_context_proposals(runtime, cognitive_store)

    assert result["already_ingested"] == 1
    assert result["eligible"] == 0
    assert cognitive_store.attention == []
    assert proposal_store.saved == []


@pytest.mark.asyncio
async def test_context_proposal_intake_caps_recurring_self_work() -> None:
    proposals = [
        _proposal(
            content=(
                "Observed context relevance follow-up.\n"
                f"Agent viewpoint: Proposal {index} is relevant.\n"
                f"Next step: Create follow-up note {index}."
            ),
            validation={"source": "desktop_context", "work_relevant": True, "salience": 0.85},
        )
        for index in range(5)
    ]
    proposal_store = _ProposalStore(proposals)
    cognitive_store = _CognitiveStore()
    promotion = _Promotion()
    runtime = SimpleNamespace(context_proposals=proposal_store, daydream_promotion=promotion)

    result = await intake_context_proposals(
        runtime,
        cognitive_store,
        max_intakes=4,
        max_self_work_signals=2,
    )

    assert result["eligible"] == 4
    assert result["attention_updates"] == 4
    assert result["working_memory_updates"] == 4
    assert result["self_work_signals"] == 2
    assert len(promotion.routed) == 2
    assert proposal_store.saved[2].validation["context_proposal_intake"]["signal_status"] == (
        "deferred_by_intake_cap"
    )
