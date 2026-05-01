"""Tests for evidence-linked affective examination records."""

import hashlib

import pytest

from opencas.affective import (
    AffectiveActionPressure,
    AffectiveExaminationService,
    AffectiveExaminationStore,
    AffectiveSourceType,
)
from opencas.somatic.models import AffectState, PrimaryEmotion


@pytest.mark.asyncio
async def test_tool_result_examination_roundtrips_and_deduplicates(tmp_path):
    store = AffectiveExaminationStore(tmp_path / "affective.db")
    await store.connect()
    service = AffectiveExaminationService(store)

    first = await service.examine_tool_result(
        session_id="session-1",
        source_id="tool-call-1",
        tool_name="web_fetch",
        success=False,
        output="timeout while fetching the evidence; result is uncertain",
    )
    duplicate = await service.examine_tool_result(
        session_id="session-1",
        source_id="tool-call-1",
        tool_name="web_fetch",
        success=False,
        output="timeout while fetching the evidence; result is uncertain",
    )

    recent = await store.list_recent(limit=10)
    unresolved = await store.list_unresolved_pressures(session_id="session-1")
    await store.close()

    assert duplicate.examination_id == first.examination_id
    assert len(recent) == 1
    assert recent[0].source_type == AffectiveSourceType.TOOL_RESULT
    assert recent[0].action_pressure == AffectiveActionPressure.VERIFY
    assert recent[0].source_excerpt == "timeout while fetching the evidence; result is uncertain"
    assert recent[0].source_hash
    assert unresolved[0].action_pressure == AffectiveActionPressure.VERIFY


@pytest.mark.asyncio
async def test_repeated_tool_pressure_becomes_already_recognized_question(tmp_path):
    store = AffectiveExaminationStore(tmp_path / "affective.db")
    await store.connect()
    service = AffectiveExaminationService(store, repeated_pressure_limit=2)

    await service.examine_tool_result(
        session_id="session-1",
        source_id="tool-call-1",
        tool_name="web_fetch",
        success=False,
        output="timeout while fetching evidence",
    )
    repeated = await service.examine_tool_result(
        session_id="session-1",
        source_id="tool-call-2",
        tool_name="web_fetch",
        success=False,
        output="timeout while fetching evidence",
    )
    summary = await service.recent_pressure_summary(session_id="session-1")
    await store.close()

    assert repeated.action_pressure == AffectiveActionPressure.ASK_CLARIFYING_QUESTION
    assert repeated.meta["already_recognized"] is True
    assert "already recognized" in repeated.bounded_reason
    assert summary["already_recognized"] is True
    assert summary["latest"]["action_pressure"] == "ask_clarifying_question"


@pytest.mark.asyncio
async def test_large_tool_output_is_truncated_and_hashed(tmp_path):
    store = AffectiveExaminationStore(tmp_path / "affective.db")
    await store.connect()
    service = AffectiveExaminationService(store, max_excerpt_chars=64)
    output = "warning: " + ("large evidence " * 80)

    record = await service.examine_tool_result(
        session_id="session-1",
        source_id="tool-call-large",
        tool_name="fs_read_file",
        success=True,
        output=output,
    )
    await store.close()

    assert len(record.source_excerpt) <= 64
    assert record.source_hash == hashlib.sha256(output.encode("utf-8")).hexdigest()
    assert record.meta["output_truncated"] is True


@pytest.mark.asyncio
async def test_empty_successful_tool_result_is_archive_only(tmp_path):
    store = AffectiveExaminationStore(tmp_path / "affective.db")
    await store.connect()
    service = AffectiveExaminationService(store)

    record = await service.examine_tool_result(
        session_id="session-1",
        source_id="tool-call-empty",
        tool_name="workspace_search_file_gists",
        success=True,
        output='{"results": []}',
    )
    summary = await service.recent_pressure_summary(session_id="session-1")
    await store.close()

    assert record.action_pressure == AffectiveActionPressure.ARCHIVE_ONLY
    assert record.meta["tool_evidence_quality"] == "empty_result"
    assert summary["available"] is False


@pytest.mark.asyncio
async def test_truncated_successful_tool_result_requires_verification_not_continue(tmp_path):
    store = AffectiveExaminationStore(tmp_path / "affective.db")
    await store.connect()
    service = AffectiveExaminationService(store, max_excerpt_chars=64)
    output = "retrieved concepts:\n" + ("dense memory evidence " * 80)

    record = await service.examine_tool_result(
        session_id="session-1",
        source_id="tool-call-broad",
        tool_name="recall_concepts",
        success=True,
        output=output,
    )
    await store.close()

    assert record.action_pressure == AffectiveActionPressure.VERIFY
    assert record.meta["tool_evidence_quality"] == "truncated"
    assert "truncated" in record.bounded_reason


@pytest.mark.asyncio
async def test_continue_tool_pressure_stays_out_of_recent_prompt_summary(tmp_path):
    store = AffectiveExaminationStore(tmp_path / "affective.db")
    await store.connect()
    service = AffectiveExaminationService(store)

    record = await service.examine_tool_result(
        session_id="session-1",
        source_id="tool-call-continue",
        tool_name="fs_read_file",
        success=True,
        output="Loaded the requested source file and found the expected function.",
    )
    summary = await service.recent_pressure_summary(session_id="session-1")
    await store.close()

    assert record.action_pressure == AffectiveActionPressure.CONTINUE
    assert summary["available"] is False
    assert summary["prompt_block"] == ""


@pytest.mark.asyncio
async def test_approval_block_is_archived_without_bypassing_policy(tmp_path):
    store = AffectiveExaminationStore(tmp_path / "affective.db")
    await store.connect()
    service = AffectiveExaminationService(store)

    record = await service.examine_tool_result(
        session_id="session-1",
        source_id="tool-call-approval-block",
        tool_name="bash_run_command",
        success=False,
        output="Tool execution blocked: approval required before running this command",
    )
    await store.close()

    assert record.action_pressure == AffectiveActionPressure.ARCHIVE_ONLY
    assert "without bypassing" in record.bounded_reason


@pytest.mark.asyncio
async def test_memory_retrieval_examination_roundtrips_and_filters(tmp_path):
    store = AffectiveExaminationStore(tmp_path / "affective.db")
    await store.connect()
    service = AffectiveExaminationService(store)

    record = await service.examine_memory_retrieval(
        session_id="session-1",
        source_type="episode",
        source_id="episode-1",
        content="We promised to resume the dashboard verification path.",
        affect=AffectState(
            primary_emotion=PrimaryEmotion.CONCERNED,
            valence=-0.3,
            arousal=0.6,
            intensity=0.7,
            certainty=0.8,
        ),
    )
    filtered = await service.list_recent(
        session_id="session-1",
        source_type=AffectiveSourceType.RETRIEVED_MEMORY,
        primary_emotion="concerned",
        action_pressure=AffectiveActionPressure.RESUME_COMMITMENT,
        decay_state="active",
    )
    await store.close()

    assert record.source_id == "episode:episode-1"
    assert record.action_pressure == AffectiveActionPressure.RESUME_COMMITMENT
    assert len(filtered) == 1
    assert filtered[0].meta["memory_source_id"] == "episode-1"
