"""Bounded qualification proof for the promise-continuity lifecycle."""

from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest
import pytest_asyncio

from opencas.autonomy import WorkObject, WorkStage
from opencas.autonomy.commitment import Commitment, CommitmentStatus
from opencas.bootstrap import BootstrapConfig, BootstrapPipeline
from opencas.context.models import MessageRole
from opencas.memory import EpisodeKind
from opencas.runtime.agent_loop import AgentRuntime
from opencas.tom import BeliefSubject, IntentionStatus


class _FakeEmbeddings:
    async def embed(self, content: str, task_type: str = "retrieval_query"):
        return SimpleNamespace(vector=[1.0, 0.0], source_hash=f"hash:{content}")


@pytest_asyncio.fixture
async def runtime(tmp_path: Path):
    config = BootstrapConfig(
        state_dir=tmp_path,
        session_id="promise-phase7",
    )
    ctx = await BootstrapPipeline(config).run()
    runtime = AgentRuntime(ctx)
    try:
        yield runtime
    finally:
        await runtime._close_stores()


@pytest.mark.asyncio
async def test_contextual_self_commitments_persist_to_commitments_and_tom(
    runtime: AgentRuntime,
) -> None:
    session_id = runtime.ctx.config.session_id or "promise-phase7"

    content = (
        "Got it. I'm logging this as a future capability you want to build for me: "
        "voice mode via Edge TTS or similar, so we can converse naturally. "
        "I'll remind you if it drifts.\n\n"
        "**Tomorrow 10:30:**\n"
        "- Screenshot body double session\n"
        "- I'll hold your task list, watch your screen, chime in when you drift\n"
        "- We'll calibrate \"too far\" live"
    )

    commitments = await runtime._capture_self_commitments(content, session_id)

    assert {item.content for item in commitments} == {
        "Remind user about voice mode via Edge TTS or similar",
        "Support screenshot body double session",
    }
    assert all(item.status == CommitmentStatus.ACTIVE for item in commitments)
    assert all(item.meta["source"] == "assistant_response" for item in commitments)

    active_intentions = runtime.tom.list_intentions(
        actor=BeliefSubject.SELF,
        status=IntentionStatus.ACTIVE,
    )
    intention_contents = {item.content for item in active_intentions}
    assert "remind user about voice mode via edge tts or similar" in intention_contents
    assert "support screenshot body double session" in intention_contents


@pytest.mark.asyncio
async def test_assistant_role_acceptance_persists_capability_audited_commitment(
    runtime: AgentRuntime,
) -> None:
    session_id = runtime.ctx.config.session_id or "promise-phase7"
    user_turn = (
        "I need to develop a productive routine, do things to get some money "
        "flowing in one way or another, preferably not employed by anyone else. "
        "Can you be my assistant, figuring out a business model that I can "
        "realistically pull off?"
    )
    assistant_turn = (
        "Yes. I can be that assistant.\n\n"
        "I've got this tracked as an active working-memory item now."
    )

    commitments = await runtime._capture_self_commitments(
        assistant_turn,
        session_id,
        user_input=user_turn,
    )

    assert len(commitments) == 1
    saved = commitments[0]
    assert saved.status == CommitmentStatus.ACTIVE
    assert saved.content == (
        "Support the operator with developing a productive routine and realistic "
        "non-employee income/business model"
    )
    assert saved.meta["normalization_source"] == "contextual_assistant_acceptance"
    assert saved.meta["capability_audit"]["status"] == "supported"
    assert "workflow_create_commitment" in saved.meta["capability_audit"]["available_tools"]
    assert "workflow_create_schedule" in saved.meta["capability_audit"]["available_tools"]
    assert "cognitive_focus_set" in saved.meta["capability_audit"]["available_tools"]
    assert "mcp_register_server_tools" in saved.meta["capability_audit"]["available_tools"]
    attention = await runtime.ctx.cognitive_state_store.list_attention(limit=8)
    prospective = await runtime.ctx.cognitive_state_store.list_prospective_memories(limit=8)
    assert any("income/business model" in item.label for item in attention)
    assert any("Proactively follow through" in item.action for item in prospective)


@pytest.mark.asyncio
async def test_broader_help_acceptance_persists_as_ongoing_income_mission(
    runtime: AgentRuntime,
) -> None:
    session_id = runtime.ctx.config.session_id or "promise-phase7"
    user_turn = (
        "Will you help me start making income? This is a high priority mission, "
        "not a one-off task. I need proactive support on a complex multi-step project."
    )
    assistant_turn = "Yes. I'll help, and I will treat it as ongoing support."

    commitments = await runtime._capture_self_commitments(
        assistant_turn,
        session_id,
        user_input=user_turn,
    )

    assert len(commitments) == 1
    saved = commitments[0]
    assert saved.status == CommitmentStatus.ACTIVE
    assert "start making income" in saved.content.lower()
    attention = await runtime.ctx.cognitive_state_store.list_attention(limit=8)
    working = await runtime.ctx.cognitive_state_store.list_working_memory(limit=8)
    prospective = await runtime.ctx.cognitive_state_store.list_prospective_memories(limit=8)
    assert any("Mission support:" in item.label for item in attention)
    assert any(str(saved.commitment_id)[:8] in item.slot for item in working)
    assert any(str(saved.commitment_id) in " ".join(item.evidence_refs) for item in prospective)


@pytest.mark.asyncio
async def test_cognitive_maintenance_backfills_existing_income_support_commitment(
    runtime: AgentRuntime,
) -> None:
    existing = Commitment(
        content=(
            "Support Jarrod in developing a productive routine and a realistic "
            "non-employee income/business model"
        ),
        status=CommitmentStatus.ACTIVE,
        priority=8.0,
        tags=[
            "jarrod_support",
            "routine",
            "income",
            "business_model",
            "non_employee",
            "follow_through",
        ],
        meta={"source": "workflow_create_commitment"},
    )
    await runtime.commitment_store.save(existing)

    result = await runtime.run_cognitive_maintenance()

    assert result["commitment_followthrough_seeds"] >= 1
    attention = await runtime.ctx.cognitive_state_store.list_attention(limit=8)
    working = await runtime.ctx.cognitive_state_store.list_working_memory(limit=8)
    prospective = await runtime.ctx.cognitive_state_store.list_prospective_memories(limit=8)
    assert any("income/business model" in item.label for item in attention)
    assert any(str(existing.commitment_id)[:8] in item.slot for item in working)
    assert any(str(existing.commitment_id) in " ".join(item.evidence_refs) for item in prospective)


@pytest.mark.asyncio
async def test_explicit_source_grounding_promise_persists(runtime: AgentRuntime) -> None:
    session_id = runtime.ctx.config.session_id or "promise-phase7"

    commitments = await runtime._capture_self_commitments(
        "I promise. No making things up. Research if I want a perspective, or say I don't know. Straight with you.",
        session_id,
    )

    assert len(commitments) == 1
    saved = commitments[0]
    assert saved.status == CommitmentStatus.ACTIVE
    assert saved.meta["trigger"] == "promise"
    assert saved.meta["normalization_source"] == "explicit_promise"
    assert "source-grounding promise" in saved.content.lower()
    assert "no making things up" in saved.content.lower()
    assert "research" in saved.content.lower()


@pytest.mark.asyncio
async def test_promise_lifecycle_qualification_scenario(runtime: AgentRuntime) -> None:
    from opencas.api.routes.chat import build_chat_router
    from opencas.api.routes.operations import build_operations_router

    session_id = runtime.ctx.config.session_id or "promise-phase7"

    runtime.ctx.somatic.set_fatigue(0.84)
    user_turn = "Please come back to the memory atlas overhaul after you rest."
    assistant_turn = "I'll come back to the memory atlas overhaul after I rest."
    await runtime._record_episode(
        user_turn,
        EpisodeKind.TURN,
        session_id=session_id,
        role="user",
    )
    await runtime.ctx.context_store.append(session_id, MessageRole.USER, user_turn)
    await runtime._record_episode(
        assistant_turn,
        EpisodeKind.TURN,
        session_id=session_id,
        role="assistant",
    )
    await runtime.ctx.context_store.append(
        session_id, MessageRole.ASSISTANT, assistant_turn
    )
    commitments = await runtime._capture_self_commitments(assistant_turn, session_id)
    assert len(commitments) == 1

    blocked_commitments = await runtime.commitment_store.list_by_status(
        CommitmentStatus.BLOCKED
    )
    assert len(blocked_commitments) == 1
    primary = blocked_commitments[0]
    assert primary.meta["source"] == "assistant_response"
    assert primary.meta["blocked_reason"] == "executive_fatigue"
    assert primary.meta["resume_policy"] == "auto_on_executive_recovery"
    assert "memory atlas overhaul" in primary.content.lower()

    linked_work = WorkObject(
        content=primary.content,
        stage=WorkStage.MICRO_TASK,
        commitment_id=str(primary.commitment_id),
        project_id="promise-project",
    )
    await runtime.ctx.work_store.save(linked_work)
    await runtime.commitment_store.link_work(
        str(primary.commitment_id), str(linked_work.work_id)
    )

    runtime.ctx.somatic.set_fatigue(0.2)
    resume_result = await runtime.executive.resume_deferred_work()
    assert resume_result["unblocked_commitments"] == 1
    assert resume_result["restored_work"] >= 1

    resumed = await runtime.commitment_store.get(str(primary.commitment_id))
    assert resumed is not None
    assert resumed.status == CommitmentStatus.ACTIVE
    assert resumed.meta["resume_reason"] == "executive_recovery"
    assert resumed.meta["previous_blocked_reason"] == "executive_fatigue"
    assert str(linked_work.work_id) in {
        str(item.work_id) for item in runtime.executive.task_queue
    }

    duplicate_a = Commitment(
        content="Return to the daily review ritual",
        status=CommitmentStatus.BLOCKED,
        meta={"source": "assistant_response", "blocked_reason": "executive_fatigue"},
    )
    duplicate_b = Commitment(
        content="Return to the daily review ritual",
        status=CommitmentStatus.BLOCKED,
        meta={"source": "assistant_response", "blocked_reason": "executive_fatigue"},
    )
    await runtime.commitment_store.save(duplicate_a)
    await runtime.commitment_store.save(duplicate_b)
    runtime.consolidation.embeddings = _FakeEmbeddings()

    merge_result = await runtime.consolidation._consolidate_commitments(
        similarity_threshold=0.1
    )
    assert merge_result["commitments_merged"] >= 1
    assert merge_result["work_objects_created"] == 0

    blocked_after_merge = await runtime.commitment_store.list_by_status(
        CommitmentStatus.BLOCKED
    )
    blocked_daily_review = [
        item
        for item in blocked_after_merge
        if item.content == "Return to the daily review ritual"
    ]
    assert len(blocked_daily_review) == 1
    assert (
        blocked_daily_review[0].meta["consolidation_merge_rationale"]
        == "heuristic_exact_duplicate"
    )

    recovery_session = "promise-recovery-window"
    await runtime._record_episode(
        "Please come back to the scheduler resume path after you rest.",
        EpisodeKind.TURN,
        session_id=recovery_session,
        role="user",
    )
    await runtime._record_episode(
        "I'll come back to the scheduler resume path when I'm ready.",
        EpisodeKind.TURN,
        session_id=recovery_session,
        role=None,
    )
    runtime.consolidation.llm.chat_completion = AsyncMock(
        return_value={
            "choices": [
                {
                    "message": {
                        "content": '[{"candidate_id":"1","content":"Return to the scheduler resume path","inferred_status":"active","reason":"user-facing follow-up promise"}]'
                    }
                }
            ]
        }
    )

    extracted_count = await runtime.consolidation._extract_commitments_from_chat_logs()
    assert extracted_count == 1

    extracted_active = await runtime.commitment_store.list_by_status(
        CommitmentStatus.ACTIVE
    )
    extracted = next(
        item
        for item in extracted_active
        if item.meta.get("source") == "nightly_consolidation"
        and item.content == "Return to the scheduler resume path"
    )
    assert extracted.meta["previous_user_turn"] == (
        "Please come back to the scheduler resume path after you rest."
    )
    assert extracted.meta["role_source"] == "roleless_fallback"
    assert extracted.linked_work_ids

    runtime._last_consolidation_result = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "commitments_consolidated": merge_result["commitments_merged"],
        "commitment_clusters_formed": merge_result["clusters_formed"],
        "commitment_work_objects_created": merge_result["work_objects_created"],
        "commitments_extracted_from_chat": extracted_count,
    }

    workflow = await runtime.workflow_status(project_id="promise-project")
    assert workflow["commitments"]["status_counts"]["active"] >= 2
    assert workflow["commitments"]["status_counts"]["blocked"] >= 1
    assert workflow["consolidation"]["commitments_extracted_from_chat"] == 1
    assert "promise-project" in workflow["work"]["active_projects"]

    operations_router = build_operations_router(runtime)
    list_commitments = next(
        route.endpoint
        for route in operations_router.routes
        if getattr(route, "path", None) == "/api/operations/commitments"
    )
    get_commitment = next(
        route.endpoint
        for route in operations_router.routes
        if getattr(route, "path", None)
        == "/api/operations/commitments/{commitment_id}"
    )
    active_payload = await list_commitments(status="active", limit=20)
    blocked_payload = await list_commitments(status="blocked", limit=20)
    extracted_detail = await get_commitment(str(extracted.commitment_id))

    assert any(
        item.lifecycle.get("resume_reason") == "executive_recovery"
        for item in active_payload.items
    )
    assert any(
        item.lifecycle.get("merge_rationale") == "heuristic_exact_duplicate"
        for item in blocked_payload.items
    )
    assert (
        extracted_detail["commitment"]["lifecycle"]["previous_user_turn"]
        == "Please come back to the scheduler resume path after you rest."
    )

    chat_router = build_chat_router(runtime)
    context_summary = next(
        route.endpoint
        for route in chat_router.routes
        if getattr(route, "path", None) == "/api/chat/context-summary"
    )
    summary = await context_summary(session_id=session_id, task_limit=6, include_details=True)
    assert summary.consolidation["commitments_extracted_from_chat"] == 1
    assert summary.executive["recommend_pause"] is False
