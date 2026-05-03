"""Tests for OpenCAS self-inspection records."""

from pathlib import Path

import pytest
import pytest_asyncio

from opencas.bootstrap import BootstrapConfig, BootstrapPipeline
from opencas.cognition import (
    CommitmentGap,
    CommitmentGapStatus,
    DriftObservation,
    ResponseShapeSignature,
    SelfInspectionPhase,
    SelfInspectionRecord,
    SelfInspectionStore,
    ToolUseInspection,
    ValenceSourceTag,
    build_post_turn_self_inspection_record,
    build_response_shape_signature,
)
from opencas.cognition.grounding import CognitionGrounding, GroundingKind, GroundingSource
from opencas.runtime import AgentRuntime
from opencas.tools.models import ToolResult


@pytest.mark.asyncio
async def test_self_inspection_store_round_trips_records(tmp_path: Path) -> None:
    store = await SelfInspectionStore(tmp_path / "self_inspection.db").connect()
    try:
        record = SelfInspectionRecord(
            session_id="session-a",
            phase=SelfInspectionPhase.POST_TURN,
            response_signature=ResponseShapeSignature(
                structure="paragraph",
                word_count=8,
                paragraph_count=1,
                bullet_count=0,
                question_count=1,
                promise_count=1,
                uncertainty_count=0,
                source_claim_count=1,
                warmth_count=0,
                shape_tags=["promise_language", "source_grounding"],
            ),
            drift_observations=[
                DriftObservation(
                    reason="Similar promise-heavy response shape appeared repeatedly.",
                    severity="notice",
                    route="observe",
                    grounding=[
                        CognitionGrounding(
                            kind=GroundingKind.DERIVED,
                            source=GroundingSource.RUNTIME,
                            claim="Repeated response shape detected from stored signatures.",
                            evidence_ids=["record-1", "record-2"],
                            confidence=0.7,
                        )
                    ],
                )
            ],
            valence_sources=[
                ValenceSourceTag(
                    source="commitment_pressure",
                    confidence=0.68,
                    reason="The response created a new promise.",
                )
            ],
            tool_use_inspections=[
                ToolUseInspection(
                    tool_name="memory_search",
                    call_id="tool-1",
                    reason="Needed stored evidence before answering.",
                    objective="answer from memory evidence",
                )
            ],
            commitment_gaps=[
                CommitmentGap(
                    commitment_id="commitment-1",
                    promised="Return to the dashboard proof.",
                    gap_type="captured_without_execution_link",
                    status=CommitmentGapStatus.OPEN,
                    likely_cause="new_conversation_commitment",
                )
            ],
        )

        await store.save(record)
        recent = await store.list_recent(session_id="session-a", limit=5)
        unresolved = await store.list_unresolved_commitment_gaps(limit=5)
        search_hits = await store.search("memory_search", limit=5)

        assert len(recent) == 1
        assert recent[0].phase == SelfInspectionPhase.POST_TURN
        assert recent[0].response_signature is not None
        assert recent[0].response_signature.promise_count == 1
        assert recent[0].drift_observations[0].grounding[0].claim.startswith("Repeated")
        assert recent[0].tool_use_inspections[0].tool_name == "memory_search"
        assert unresolved[0].commitment_gaps[0].commitment_id == "commitment-1"
        assert search_hits[0].record_id == recent[0].record_id
    finally:
        await store.close()


def test_response_shape_signature_tracks_structure_without_rewriting() -> None:
    signature = build_response_shape_signature(
        "I do not know that from memory.\n\n- I will check the stored record.\n- I will show the evidence."
    )

    assert signature.structure == "bullet_list"
    assert signature.paragraph_count == 2
    assert signature.bullet_count == 2
    assert signature.promise_count == 2
    assert signature.uncertainty_count >= 1
    assert "bullet_list" in signature.shape_tags
    assert "promise_language" in signature.shape_tags
    assert "uncertainty_language" in signature.shape_tags


def test_post_turn_self_inspection_flags_borrowed_human_interface() -> None:
    record = build_post_turn_self_inspection_record(
        session_id="session-a",
        user_input="I use Firefox with a keyboard and mouse.",
        assistant_output="I would use arrow keys to step through a replay and click any word.",
    )

    reasons = [observation.reason for observation in record.drift_observations]
    assert any("operator interface" in reason for reason in reasons)


@pytest_asyncio.fixture
async def runtime(tmp_path: Path):
    config = BootstrapConfig(
        state_dir=tmp_path,
        session_id="self-inspection-session",
    )
    ctx = await BootstrapPipeline(config).run()
    runtime = AgentRuntime(ctx)
    try:
        yield runtime
    finally:
        await runtime._close_stores()


@pytest.mark.asyncio
async def test_conversation_turn_records_pre_and_post_self_inspection(
    runtime: AgentRuntime,
) -> None:
    runtime.llm.chat_completion = _mock_chat_completion("I promise. I will return to the evidence check.")

    await runtime.converse("Please inspect this before answering.")

    records = await runtime.self_inspection_store.list_recent(
        session_id="self-inspection-session",
        limit=10,
    )

    assert [record.phase for record in records] == [
        SelfInspectionPhase.PRE_TURN,
        SelfInspectionPhase.POST_TURN,
    ]
    assert records[0].meta["user_input_excerpt"] == "Please inspect this before answering."
    assert records[1].response_signature is not None
    assert records[1].response_signature.promise_count >= 1
    assert records[1].commitment_gaps
    assert records[1].commitment_gaps[0].gap_type == "captured_without_execution_link"
    assert records[1].valence_sources


@pytest.mark.asyncio
async def test_self_inspection_query_tool_returns_searchable_records(
    runtime: AgentRuntime,
) -> None:
    record = SelfInspectionRecord(
        session_id="self-inspection-session",
        phase=SelfInspectionPhase.POST_TURN,
        tool_use_inspections=[
            ToolUseInspection(
                tool_name="web_search",
                call_id="tool-web",
                reason="Needed current external evidence.",
                objective="verify a source",
            )
        ],
    )
    await runtime.self_inspection_store.save(record)

    result = await runtime.tools.execute_async(
        "self_inspection_query",
        {"query": "web_search", "limit": 5},
    )

    assert isinstance(result, ToolResult)
    assert result.success is True
    assert result.metadata["count"] == 1
    assert result.metadata["items"][0]["tool_use_inspections"][0]["tool_name"] == "web_search"


def _mock_chat_completion(response_text: str):
    async def _mock(*args, **kwargs):
        if kwargs.get("source") in {"response_integrity", "response_integrity_retry"}:
            return {
                "choices": [
                    {
                        "message": {
                            "content": (
                                '{"needs_revision":false,'
                                '"reasons":["response shape is acceptable"],'
                                '"revised_response":""}'
                            )
                        }
                    }
                ]
            }
        if kwargs.get("source") == "values_alignment":
            return {"choices": [{"message": {"content": '{"violations":[]}'}}]}
        return {"choices": [{"message": {"content": response_text}}]}

    return _mock
