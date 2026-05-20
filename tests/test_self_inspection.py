"""Tests for OpenCAS self-inspection records."""

from pathlib import Path

import pytest
import pytest_asyncio

from opencas.autonomy.models import ActionRiskTier
from opencas.bootstrap import BootstrapConfig, BootstrapPipeline
from opencas.cognition import (
    CommitmentGap,
    CommitmentGapStatus,
    DriftObservation,
    ResponseShapeSignature,
    SelfInspectionPhase,
    SelfInspectionRecord,
    SelfInspectionStore,
    ToolCallTransit,
    ToolChainTransitSummary,
    ToolUseInspection,
    ValenceSourceTag,
    build_post_turn_self_inspection_record,
    build_response_shape_signature,
    build_tool_chain_transit_summary,
)
from opencas.cognition.grounding import CognitionGrounding, GroundingKind, GroundingSource
from opencas.runtime import AgentRuntime
from opencas.thread_registry import BeadSourceKind
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
            tool_call_transits=[
                ToolCallTransit(
                    chain_id="chain-1",
                    call_id="tool-1",
                    tool_name="memory_search",
                    entry_intent="Check stored evidence before answering.",
                    trust_context="operator_requested",
                    success=True,
                    duration_ms=12,
                    result_shape={"tags": ["nonempty"], "output_chars": 42},
                    certainty_delta=0.05,
                    somatic_delta={},
                )
            ],
            tool_chain_summary=ToolChainTransitSummary(
                chain_id="chain-1",
                objective="answer from memory evidence",
                entry_intent="Check stored evidence before answering.",
                trust_context="operator_requested",
                call_ids=["tool-1"],
                call_count=1,
                success_count=1,
                failure_count=0,
                result_shape_counts={"nonempty": 1},
                certainty_delta=0.05,
                load_type="cognitive",
                undercoupled_somatic=False,
            ),
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
        assert recent[0].tool_call_transits[0].trust_context == "operator_requested"
        assert recent[0].tool_chain_summary is not None
        assert recent[0].tool_chain_summary.call_ids == ["tool-1"]
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


def test_tool_chain_summary_is_derived_from_call_transits() -> None:
    calls = [
        ToolCallTransit(
            chain_id="chain-7",
            call_id="call-a",
            tool_name="workflow_status",
            entry_intent="Verify the current schedule state.",
            trust_context="operator_requested",
            success=True,
            duration_ms=35,
            result_shape={"tags": ["nonempty"], "output_chars": 120},
            certainty_delta=0.04,
            somatic_delta={},
        ),
        ToolCallTransit(
            chain_id="chain-7",
            call_id="call-b",
            tool_name="workflow_list_schedules",
            entry_intent="Verify the current schedule state.",
            trust_context="operator_requested",
            success=False,
            duration_ms=50,
            result_shape={"tags": ["failed", "blocked"], "output_chars": 22},
            certainty_delta=-0.08,
            somatic_delta={},
        ),
    ]

    summary = build_tool_chain_transit_summary(
        objective="Check whether the health schedule exists.",
        chain_id="chain-7",
        call_transits=calls,
    )

    assert summary.call_ids == ["call-a", "call-b"]
    assert summary.trust_context == "operator_requested"
    assert summary.call_count == 2
    assert summary.failure_count == 1
    assert summary.result_shape_counts["blocked"] == 1
    assert summary.certainty_delta == -0.04
    assert summary.undercoupled_somatic is True
    assert summary.meta["source"] == "tool_call_transits"


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


@pytest.mark.asyncio
async def test_conversation_turn_persists_tool_transit_records(
    runtime: AgentRuntime,
) -> None:
    runtime.tools.register(
        "transit_probe",
        "Probe transit persistence.",
        lambda _name, _args: ToolResult(True, "probe ok", {"certainty_delta": 0.1}),
        ActionRiskTier.READONLY,
    )
    runtime.llm.chat_completion = _mock_tool_then_answer()

    await runtime.converse("Please use the transit probe before answering.")

    records = await runtime.self_inspection_store.search("transit_probe", limit=5)
    post_records = [record for record in records if record.phase == SelfInspectionPhase.POST_TURN]

    assert post_records
    record = post_records[-1]
    assert record.tool_call_transits
    assert record.tool_call_transits[0].tool_name == "transit_probe"
    assert record.tool_call_transits[0].trust_context == "operator_requested"
    assert record.tool_chain_summary is not None
    assert record.tool_chain_summary.call_ids == [record.tool_call_transits[0].call_id]

    query_result = await runtime.tools.execute_async(
        "self_inspection_query",
        {
            "query": "transit_probe",
            "session_id": "self-inspection-session",
            "limit": 5,
        },
    )
    query_post_records = [
        item
        for item in query_result.metadata["items"]
        if item["phase"] == SelfInspectionPhase.POST_TURN.value
    ]
    assert query_post_records
    assert query_post_records[-1]["tool_call_transits"][0]["tool_name"] == "transit_probe"
    assert query_post_records[-1]["tool_chain_summary"]["call_ids"] == ["probe-call"]


@pytest.mark.asyncio
async def test_conversation_turn_records_failed_tool_transit_thread_bead(
    runtime: AgentRuntime,
) -> None:
    runtime.tools.register(
        "transit_probe",
        "Probe failed transit persistence.",
        lambda _name, _args: ToolResult(False, "probe blocked", {"reason": "blocked"}),
        ActionRiskTier.READONLY,
    )
    runtime.llm.chat_completion = _mock_tool_then_answer()

    await runtime.converse("Please use the transit probe before answering.")

    records = await runtime.self_inspection_store.search("transit_probe", limit=5)
    post_records = [record for record in records if record.phase == SelfInspectionPhase.POST_TURN]
    assert post_records
    assert post_records[-1].tool_call_transits
    assert post_records[-1].tool_call_transits[0].success is False

    beads = await runtime.thread_registry_store.list_beads(
        source_kind=BeadSourceKind.TOOL_CALL_TRANSIT,
        limit=5,
    )
    assert beads
    chain_id = post_records[-1].tool_call_transits[0].chain_id
    assert beads[0].source_ref == (
        f"tool_call_transit:self-inspection-session:{chain_id}:probe-call"
    )
    assert "transit_probe" in beads[0].summary

    manifest = await runtime.builder.build(
        "Why did the transit probe fail or get blocked?",
        session_id="self-inspection-session",
    )
    assert manifest.system is not None
    assert "Thread registry continuity cues:" in manifest.system.content
    assert "tool_call_transit" in manifest.system.content
    assert "transit_probe" in manifest.system.content


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


def _mock_tool_then_answer():
    state = {"tool_loop_calls": 0}

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
        if kwargs.get("source") == "tool_use_loop":
            state["tool_loop_calls"] += 1
            if state["tool_loop_calls"] == 1:
                return {
                    "choices": [
                        {
                            "message": {
                                "content": "I need a runtime probe before answering.",
                                "tool_calls": [
                                    {
                                        "id": "probe-call",
                                        "function": {
                                            "name": "transit_probe",
                                            "arguments": "{}",
                                        },
                                    }
                                ],
                            }
                        }
                    ]
                }
        return {"choices": [{"message": {"content": "Probe complete."}}]}

    return _mock
