"""Tests for cognitive state spine and prompt integration."""

from __future__ import annotations

import asyncio
from datetime import datetime, timedelta, timezone
from types import SimpleNamespace

import pytest
import pytest_asyncio

from opencas.affective_registry import AffectiveRegistryWriter, ExecutionPhase
from opencas.cognition import (
    CognitiveEventKind,
    CognitiveStateStore,
    LearnedSkill,
    SelfInspectionStore,
)
from opencas.cognition.self_inspection import (
    CommitmentGap,
    CommitmentGapStatus,
    DriftObservation,
    SelfInspectionPhase,
    SelfInspectionRecord,
)
from opencas.context import ContextBuilder, MemoryRetriever, SessionContextStore
from opencas.embeddings import EmbeddingCache, EmbeddingService
from opencas.execution.models import ExecutionReceipt
from opencas.execution.receipt_store import ExecutionReceiptStore
from opencas.identity import IdentityManager, IdentityStore
from opencas.memory import Episode, EpisodeKind, MemoryStore
from opencas.runtime.cognitive_runtime import run_runtime_cognitive_maintenance
from opencas.runtime.cognitive_runtime import _record_narrative_summary
from opencas.identity.registry import SelfKnowledgeRegistry
from opencas.scheduling import ScheduleService, ScheduleStore
from opencas.somatic import SomaticState
from opencas.thread_registry import (
    BeadSourceKind,
    ThreadRegistryService,
    ThreadRegistryStore,
)
from opencas.tom import BeliefSubject, ToMEngine
from opencas.tools.cognitive_tools import CognitiveToolAdapter


@pytest_asyncio.fixture
async def cognitive_store(tmp_path):
    store = CognitiveStateStore(tmp_path / "cognitive.db")
    await store.connect()
    yield store
    await store.close()


@pytest.mark.asyncio
async def test_cognitive_skill_create_tool_persists_reusable_procedure(cognitive_store):
    adapter = CognitiveToolAdapter(SimpleNamespace(cognitive_state_store=cognitive_store))

    result = await adapter(
        "cognitive_skill_create",
        {
            "name": "create reusable skill after corrected workflow",
            "description": "After a corrected multi-step workflow succeeds, save the trigger, cautions, and exact tools.",
            "tool_sequence": ["cognitive_skill_library_search", "cognitive_skill_create"],
            "preconditions": ["operator asks to teach Bulma a reusable procedure"],
            "evidence_refs": ["conversation:skill-request"],
            "success_count": 4,
            "risk_tier": "workspace_write",
        },
    )

    skills = await cognitive_store.list_learned_skills(limit=5)
    assert result.success is True
    assert skills[0].activation_status == "evidence_gated_auto_use"
    assert skills[0].tool_sequence == ["cognitive_skill_library_search", "cognitive_skill_create"]
    assert "conversation:skill-request" in skills[0].evidence_refs


@pytest.mark.asyncio
async def test_prompt_block_renders_query_relevant_learned_skill_not_unrelated_high_confidence(cognitive_store):
    await cognitive_store.upsert_learned_skill(
        LearnedSkill(
            name="generic youtube note checkpoint",
            description="Use media state and transcripts to append one narrow watch note.",
            tool_sequence=["fs_read_file", "bash_run_command"],
            preconditions=["youtube watch notes"],
            success_count=10,
            failure_count=0,
            risk_tier="workspace_write",
            evidence_refs=["episode:youtube"],
        )
    )
    await cognitive_store.upsert_learned_skill(
        LearnedSkill(
            name="triage email inbox for important items",
            description="Search local and Google email backends, read promising messages, and summarize actionable mail.",
            tool_sequence=["google_workspace_gmail_headlines", "himalaya_email_headlines", "himalaya_email_read_message"],
            preconditions=["operator asks what is important in email"],
            success_count=2,
            failure_count=0,
            risk_tier="readonly",
            evidence_refs=["episode:email-triage"],
        )
    )

    skills = await cognitive_store.list_learned_skills(query="anything important in my email inbox", limit=1)
    block = await cognitive_store.prompt_block(query="anything important in my email inbox")

    assert skills[0].name == "triage email inbox for important items"
    assert "triage email inbox for important items" in block
    assert "generic youtube note checkpoint" not in block


@pytest.mark.asyncio
async def test_cognitive_state_prompt_block_uses_retrievable_written_state(cognitive_store):
    await cognitive_store.upsert_attention(
        "finish artifact spine follow-up",
        strength=0.8,
        source="test",
        evidence_refs=["task:TASK-1"],
    )
    await cognitive_store.upsert_working_memory(
        "current_artifact",
        "The current artifact is the cognitive state spine.",
        priority=0.9,
        source="test",
    )
    await cognitive_store.record_event(
        CognitiveEventKind.SURPRISE,
        "Unexpected duplicate schedule should trigger dedupe review",
        source="test",
        salience=1.8,
    )

    block = await cognitive_store.prompt_block(query="artifact duplicate schedule")

    assert "Cognitive state evidence" in block
    assert "finish artifact spine follow-up" in block
    assert "current_artifact" in block
    assert "duplicate schedule" in block
    assert "not a response script" in block


@pytest.mark.asyncio
async def test_context_builder_injects_cognitive_state(cognitive_store, tmp_path):
    ctx_store = SessionContextStore(tmp_path / "context.db")
    await ctx_store.connect()
    mem_store = MemoryStore(tmp_path / "memory.db")
    await mem_store.connect()
    cache = EmbeddingCache(tmp_path / "embeddings.db")
    await cache.connect()
    retriever = MemoryRetriever(
        memory=mem_store,
        embeddings=EmbeddingService(cache=cache, model_id="local-fallback"),
    )
    await cognitive_store.upsert_working_memory(
        "active_test",
        "Use cognitive evidence in the prompt.",
        priority=0.8,
    )
    builder = ContextBuilder(
        store=ctx_store,
        retriever=retriever,
        cognitive_state_store=cognitive_store,
    )

    manifest = await builder.build("What is active?", session_id="s1")

    assert manifest.system is not None
    assert "Cognitive state evidence" in manifest.system.content
    assert "active_test" in manifest.system.content

    await ctx_store.close()
    await mem_store.close()
    await cache.close()


@pytest.mark.asyncio
async def test_context_builder_records_proactive_channel_audit_without_prompt_change(cognitive_store, tmp_path):
    ctx_store = SessionContextStore(tmp_path / "context.db")
    await ctx_store.connect()
    mem_store = MemoryStore(tmp_path / "memory.db")
    await mem_store.connect()
    cache = EmbeddingCache(tmp_path / "embeddings.db")
    await cache.connect()
    retriever = MemoryRetriever(
        memory=mem_store,
        embeddings=EmbeddingService(cache=cache, model_id="local-fallback"),
    )
    await cognitive_store.upsert_attention(
        "review scheduler proof gaps",
        strength=0.8,
        source="test",
        evidence_refs=["attention:proof-gaps"],
    )
    await cognitive_store.upsert_working_memory(
        "current_gap",
        "The current gap is active-turn proactive surfacing.",
        priority=0.85,
        source="test",
        evidence_refs=["wm:current-gap"],
    )
    await cognitive_store.upsert_prospective_memory(
        "Ask whether the proactive cue helped.",
        condition="after an active-session observation",
        evidence_refs=["prospective:ask-helped"],
    )
    await cognitive_store.upsert_learned_skill(
        LearnedSkill(
            name="inspect prompt telemetry",
            description="Use telemetry traces to inspect whether prompt channels rendered.",
            tool_sequence=["telemetry_query"],
            preconditions=["prompt audit"],
            success_count=4,
            failure_count=0,
            risk_tier="readonly",
            evidence_refs=["skill:telemetry-query"],
        )
    )
    builder = ContextBuilder(
        store=ctx_store,
        retriever=retriever,
        cognitive_state_store=cognitive_store,
    )

    manifest = await builder.build("What is open right now?", session_id="s1")

    audit = manifest.system.meta["proactive_channel_audit"]
    assert "Proactive observation" not in manifest.system.content
    assert audit["channels"]["attention"]["produced_count"] == 1
    assert audit["channels"]["attention"]["rendered_count"] == 1
    assert audit["channels"]["working_memory"]["rendered_count"] == 1
    assert audit["channels"]["prospective_memory"]["rendered_count"] == 1
    assert audit["channels"]["learned_skills"]["rendered_count"] == 1
    assert "wm:current-gap" in audit["channels"]["working_memory"]["evidence_ids"]
    assert audit["channels"]["working_memory"]["novel_observation_count"] == 1

    await ctx_store.close()
    await mem_store.close()
    await cache.close()


@pytest.mark.asyncio
async def test_prospective_memory_lists_recent_unscheduled_followthrough(cognitive_store):
    for index in range(6):
        await cognitive_store.upsert_prospective_memory(
            f"Older unscheduled follow-through {index}",
            condition="when useful",
            confidence=0.5,
        )
    await asyncio.sleep(0.01)
    await cognitive_store.upsert_prospective_memory(
        "Proactively follow through on commitment: income mission",
        condition="when reflection or daydreaming can advance it",
        confidence=0.5,
        evidence_refs=["commitment:income"],
    )

    items = await cognitive_store.list_prospective_memories(limit=3)

    assert any("income mission" in item.action for item in items)


@pytest.mark.asyncio
async def test_cognitive_maintenance_promotes_recurring_life_priority_across_phrasings(tmp_path):
    cognitive = CognitiveStateStore(tmp_path / "cognitive.db")
    await cognitive.connect()
    memory = MemoryStore(tmp_path / "memory.db")
    await memory.connect()
    base = datetime(2026, 5, 1, tzinfo=timezone.utc)
    messages = [
        "I do not have a job or income right now and need help figuring this out.",
        "I am behind on rent and need help with my options.",
        "I need to work on things that could lead to me getting money.",
        "Can you check my email for anything useful for unemployment or rent assistance?",
        "I need a productive routine and a business model to get money flowing.",
    ]
    try:
        for index, content in enumerate(messages):
            await memory.save_episode(
                Episode(
                    kind=EpisodeKind.TURN,
                    session_id="priority-pattern",
                    content=content,
                    created_at=base + timedelta(days=index),
                    payload={"role": "user"},
                )
            )
        runtime = SimpleNamespace(
            cognitive_state_store=cognitive,
            memory=memory,
            executive=SimpleNamespace(intention="", active_goals=[]),
            _activity="idle",
            _activity_since=None,
        )

        result = await run_runtime_cognitive_maintenance(runtime)

        assert result["priority_patterns"]["promoted"] == 1
        attention = await cognitive.list_attention(limit=10)
        assert any("financial stability" in item.label.lower() for item in attention)
        working_memory = await cognitive.list_working_memory(limit=10)
        assert any(item.slot == "operator_priority:financial_stability" for item in working_memory)
        prospective = await cognitive.list_prospective_memories(limit=10)
        assert any("financial stability" in item.action.lower() and item.trigger_at for item in prospective)
    finally:
        await cognitive.close()
        await memory.close()


@pytest.mark.asyncio
async def test_cognitive_maintenance_resolves_life_priority_after_new_user_context(tmp_path):
    cognitive = CognitiveStateStore(tmp_path / "cognitive.db")
    await cognitive.connect()
    memory = MemoryStore(tmp_path / "memory.db")
    await memory.connect()
    base = datetime(2026, 5, 1, tzinfo=timezone.utc)
    old_messages = [
        "I do not have a job or income right now and need help figuring this out.",
        "I am behind on rent and need help with my options.",
        "I need to work on things that could lead to me getting money.",
        "Can you check my email for anything useful for unemployment or rent assistance?",
    ]
    try:
        for index, content in enumerate(old_messages):
            await memory.save_episode(
                Episode(
                    kind=EpisodeKind.TURN,
                    session_id="priority-resolution",
                    content=content,
                    created_at=base + timedelta(days=index),
                    payload={"role": "user"},
                )
            )
        runtime = SimpleNamespace(
            cognitive_state_store=cognitive,
            memory=memory,
            executive=SimpleNamespace(intention="", active_goals=[]),
            _activity="idle",
            _activity_since=None,
        )
        await run_runtime_cognitive_maintenance(runtime)
        await memory.save_episode(
            Episode(
                kind=EpisodeKind.TURN,
                session_id="priority-resolution",
                content=(
                    "I got a full-time job, my rent is caught up, and I do not need "
                    "rental assistance anymore."
                ),
                created_at=base + timedelta(days=10),
                payload={"role": "user"},
            )
        )

        result = await run_runtime_cognitive_maintenance(runtime)

        assert result["priority_patterns"]["resolved"] == 1
        attention = await cognitive.list_attention(active_only=False, limit=20)
        assert any(
            "financial stability" in item.label.lower() and item.status == "resolved"
            for item in attention
        )
        working_memory = await cognitive.list_working_memory(active_only=False, limit=20)
        assert any(
            item.slot == "operator_priority:financial_stability" and item.status == "resolved"
            for item in working_memory
        )
        prospective = await cognitive.list_prospective_memories(active_only=False, limit=20)
        assert any(
            "financial stability" in item.action.lower() and item.status == "resolved"
            for item in prospective
        )
    finally:
        await cognitive.close()
        await memory.close()


@pytest.mark.asyncio
async def test_cognitive_maintenance_promotes_explicit_current_priority_without_domain_script(tmp_path):
    cognitive = CognitiveStateStore(tmp_path / "cognitive.db")
    await cognitive.connect()
    memory = MemoryStore(tmp_path / "memory.db")
    await memory.connect()
    try:
        await memory.save_episode(
            Episode(
                kind=EpisodeKind.TURN,
                session_id="explicit-priority",
                content="Right now my priority is writing the WingLab client proposal and delivery plan.",
                created_at=datetime(2026, 6, 1, tzinfo=timezone.utc),
                payload={"role": "user"},
            )
        )
        runtime = SimpleNamespace(
            cognitive_state_store=cognitive,
            memory=memory,
            executive=SimpleNamespace(intention="", active_goals=[]),
            _activity="idle",
            _activity_since=None,
        )

        result = await run_runtime_cognitive_maintenance(runtime)

        assert result["priority_patterns"]["dynamic_promoted"] == 1
        working_memory = await cognitive.list_working_memory(limit=12)
        assert any(
            item.slot.startswith("operator_priority:explicit:")
            and "WingLab client proposal" in item.content
            for item in working_memory
        )
    finally:
        await cognitive.close()
        await memory.close()


@pytest.mark.asyncio
async def test_cognitive_maintenance_resolves_explicit_priority_when_user_says_it_is_over(tmp_path):
    cognitive = CognitiveStateStore(tmp_path / "cognitive.db")
    await cognitive.connect()
    memory = MemoryStore(tmp_path / "memory.db")
    await memory.connect()
    try:
        await memory.save_episode(
            Episode(
                kind=EpisodeKind.TURN,
                session_id="explicit-priority-resolution",
                content="Right now my priority is writing the WingLab client proposal and delivery plan.",
                created_at=datetime(2026, 6, 1, tzinfo=timezone.utc),
                payload={"role": "user"},
            )
        )
        runtime = SimpleNamespace(
            cognitive_state_store=cognitive,
            memory=memory,
            executive=SimpleNamespace(intention="", active_goals=[]),
            _activity="idle",
            _activity_since=None,
        )
        await run_runtime_cognitive_maintenance(runtime)
        await memory.save_episode(
            Episode(
                kind=EpisodeKind.TURN,
                session_id="explicit-priority-resolution",
                content="The WingLab proposal priority is done and over now.",
                created_at=datetime(2026, 6, 3, tzinfo=timezone.utc),
                payload={"role": "user"},
            )
        )

        result = await run_runtime_cognitive_maintenance(runtime)

        assert result["priority_patterns"]["dynamic_resolved"] == 1
        working_memory = await cognitive.list_working_memory(active_only=False, limit=12)
        assert any(
            item.slot.startswith("operator_priority:explicit:")
            and item.status == "resolved"
            and "WingLab" in item.content
            for item in working_memory
        )
    finally:
        await cognitive.close()
        await memory.close()


@pytest.mark.asyncio
async def test_cognitive_tools_expose_state_without_canned_language(cognitive_store):
    adapter = CognitiveToolAdapter(SimpleNamespace(cognitive_state_store=cognitive_store))

    focus = await adapter(
        "cognitive_focus_set",
        {"label": "verify cognitive tools", "strength": 0.74, "source": "test"},
    )
    query = await adapter(
        "cognitive_context_query",
        {"query": "verify cognitive tools", "limit": 5},
    )

    assert focus.success is True
    assert query.success is True
    assert "verify cognitive tools" in query.output
    assert "response script" not in query.output.lower()


@pytest.mark.asyncio
async def test_cognitive_social_model_tool_records_third_party_agent(cognitive_store, tmp_path):
    identity = IdentityManager(IdentityStore(tmp_path / "identity"))
    identity.load()
    tom = ToMEngine(identity=identity)
    adapter = CognitiveToolAdapter(SimpleNamespace(cognitive_state_store=cognitive_store, tom=tom))

    result = await adapter(
        "cognitive_social_model_record",
        {
            "entity_id": "opus",
            "entity_label": "Opus",
            "belief": "noticed a stability gap",
            "intention": "audit memory behavior",
            "confidence": 0.73,
            "evidence_refs": ["chat:1"],
        },
    )

    assert result.success is True
    assert tom.list_beliefs(subject=BeliefSubject.AGENT)[0].meta["entity_id"] == "opus"
    events = await cognitive_store.list_recent_events(kind=CognitiveEventKind.SOCIAL_MODEL, limit=5)
    assert any("Opus" in event.summary for event in events)


@pytest.mark.asyncio
async def test_cognitive_maintenance_extracts_procedural_skills_and_commitment_gaps(tmp_path):
    cognitive = CognitiveStateStore(tmp_path / "cognitive.db")
    await cognitive.connect()
    memory = MemoryStore(tmp_path / "memory.db")
    await memory.connect()
    self_inspection = SelfInspectionStore(tmp_path / "self_inspection.db")
    await self_inspection.connect()

    await memory.save_episode(
        Episode(
            kind=EpisodeKind.PROCEDURAL,
            session_id="task-1",
            content=(
                "Objective: run tests and fix failures\n"
                "Tool sequence:\n"
                "- tool bash_run_command\n"
                "- tool fs_read_file\n"
                "Outcome: success"
            ),
        )
    )
    await self_inspection.save(
        SelfInspectionRecord(
            session_id="s1",
            phase=SelfInspectionPhase.POST_TURN,
            commitment_gaps=[
                CommitmentGap(
                    commitment_id="c1",
                    promised="verify the schedule exists",
                    gap_type="missing_proof",
                    status=CommitmentGapStatus.OPEN,
                )
            ],
        )
    )

    runtime = SimpleNamespace(
        cognitive_state_store=cognitive,
        memory=memory,
        self_inspection_store=self_inspection,
        executive=SimpleNamespace(intention="finish cognitive completeness", active_goals=[]),
        _activity="idle",
        _activity_since=None,
    )

    result = await run_runtime_cognitive_maintenance(runtime)
    skills = await cognitive.list_learned_skills(limit=10)
    block = await cognitive.prompt_block(query="schedule proof tests")

    assert result["available"] is True
    assert result["skills"]["skills_created_or_updated"] == 1
    assert skills[0].activation_status == "evidence_gated_auto_use"
    assert "Unresolved commitment gap" in block
    assert "Evidence-gated learned procedures" in block

    await cognitive.close()
    await memory.close()
    await self_inspection.close()


@pytest.mark.asyncio
async def test_cognitive_maintenance_promotes_repeated_drift_to_thread_registry_context(tmp_path):
    cognitive = CognitiveStateStore(tmp_path / "cognitive.db")
    await cognitive.connect()
    self_inspection = SelfInspectionStore(tmp_path / "self_inspection.db")
    await self_inspection.connect()
    thread_store = await ThreadRegistryStore(tmp_path / "thread_registry.db").connect()
    thread_service = ThreadRegistryService(
        store=thread_store,
        workspace_root=tmp_path / "workspace",
    )
    ctx_store = SessionContextStore(tmp_path / "context.db")
    await ctx_store.connect()
    mem_store = MemoryStore(tmp_path / "memory.db")
    await mem_store.connect()
    cache = EmbeddingCache(tmp_path / "embeddings.db")
    await cache.connect()
    try:
        drift_reason = "Repeated unsupported capability framing."
        for idx in range(2):
            await self_inspection.save(
                SelfInspectionRecord(
                    session_id=f"drift-{idx}",
                    phase=SelfInspectionPhase.POST_TURN,
                    drift_observations=[DriftObservation(reason=drift_reason)],
                )
            )
        runtime = SimpleNamespace(
            cognitive_state_store=cognitive,
            self_inspection_store=self_inspection,
            thread_registry_service=thread_service,
            executive=SimpleNamespace(intention="", active_goals=[]),
            _activity="idle",
            _activity_since=None,
        )

        await run_runtime_cognitive_maintenance(runtime)
        await run_runtime_cognitive_maintenance(runtime)

        beads = await thread_store.list_beads(
            source_kind=BeadSourceKind.SELF_INSPECTION_DRIFT,
            limit=10,
        )
        drift_beads = [
            bead
            for bead in beads
            if bead.source_ref.startswith("self_inspection:drift:")
        ]
        assert len(drift_beads) == 1
        assert drift_reason in drift_beads[0].summary

        retriever = MemoryRetriever(
            memory=mem_store,
            embeddings=EmbeddingService(cache=cache, model_id="local-fallback"),
        )
        builder = ContextBuilder(
            store=ctx_store,
            retriever=retriever,
            thread_registry_store=thread_store,
        )
        manifest = await builder.build(
            "Stop the unsupported capability framing drift.",
            session_id="s1",
        )

        assert manifest.system is not None
        assert "Thread registry continuity cues:" in manifest.system.content
        assert drift_reason in manifest.system.content
        assert "self_inspection:drift:" in manifest.system.content
    finally:
        await cognitive.close()
        await self_inspection.close()
        await thread_store.close()
        await ctx_store.close()
        await mem_store.close()
        await cache.close()


@pytest.mark.asyncio
async def test_cognitive_maintenance_ignores_audit_only_self_inspection(tmp_path):
    cognitive = CognitiveStateStore(tmp_path / "cognitive.db")
    await cognitive.connect()
    self_inspection = SelfInspectionStore(tmp_path / "self_inspection.db")
    await self_inspection.connect()
    thread_store = await ThreadRegistryStore(tmp_path / "thread_registry.db").connect()
    thread_service = ThreadRegistryService(
        store=thread_store,
        workspace_root=tmp_path / "workspace",
    )
    try:
        drift_reason = "Audit-only unsupported capability framing."
        for idx in range(2):
            await self_inspection.save(
                SelfInspectionRecord(
                    session_id=f"audit-{idx}",
                    phase=SelfInspectionPhase.POST_TURN,
                    drift_observations=[DriftObservation(reason=drift_reason)],
                    meta={"audit_only": True},
                )
            )
        runtime = SimpleNamespace(
            cognitive_state_store=cognitive,
            self_inspection_store=self_inspection,
            thread_registry_service=thread_service,
            executive=SimpleNamespace(intention="", active_goals=[]),
            _activity="idle",
            _activity_since=None,
        )

        result = await run_runtime_cognitive_maintenance(runtime)
        beads = await thread_store.list_beads(
            source_kind=BeadSourceKind.SELF_INSPECTION_DRIFT,
            limit=10,
        )
        events = await cognitive.list_recent_events(limit=10)

        assert result["available"] is True
        assert beads == []
        assert not any(drift_reason in event.summary for event in events)
    finally:
        await cognitive.close()
        await self_inspection.close()
        await thread_store.close()


@pytest.mark.asyncio
async def test_cognitive_maintenance_promotes_repeated_gap_to_schedule(tmp_path):
    cognitive = CognitiveStateStore(tmp_path / "cognitive.db")
    await cognitive.connect()
    self_inspection = SelfInspectionStore(tmp_path / "self_inspection.db")
    await self_inspection.connect()
    schedule_store = ScheduleStore(tmp_path / "schedules.db")
    await schedule_store.connect()
    schedule_service = ScheduleService(schedule_store)
    try:
        for idx in range(2):
            await self_inspection.save(
                SelfInspectionRecord(
                    session_id=f"s{idx}",
                    phase=SelfInspectionPhase.POST_TURN,
                    commitment_gaps=[
                        CommitmentGap(
                            commitment_id=f"c{idx}",
                            promised="verify the daily news schedule",
                            gap_type="missing_proof",
                            status=CommitmentGapStatus.OPEN,
                        )
                    ],
                )
            )
        runtime = SimpleNamespace(
            cognitive_state_store=cognitive,
            self_inspection_store=self_inspection,
            schedule_service=schedule_service,
            executive=SimpleNamespace(intention="", active_goals=[]),
            _activity="idle",
            _activity_since=None,
        )

        result = await run_runtime_cognitive_maintenance(runtime)
        schedules = await schedule_store.list_items(limit=20)

        assert result["promotions"]["self_inspection_promoted"] == 1
        assert len(schedules) == 1
        assert schedules[0].meta["source"] == "cognitive_maintenance"
        assert schedules[0].meta["promotion_kind"] == "self_inspection_gap"
    finally:
        await cognitive.close()
        await self_inspection.close()
        await schedule_store.close()


@pytest.mark.asyncio
async def test_cognitive_maintenance_bridges_affective_registry_to_telemetry(tmp_path):
    cognitive = CognitiveStateStore(tmp_path / "cognitive.db")
    await cognitive.connect()
    receipt_store = ExecutionReceiptStore(tmp_path / "receipts.db")
    await receipt_store.connect()
    writer = AffectiveRegistryWriter(tmp_path / "affective_registry" / "events.jsonl")
    writer.append_from_somatic_state(
        SomaticState(tension=0.2, fatigue=0.1, valence=0.1, certainty=0.8),
        phase=ExecutionPhase.TURN_END,
        session_id="s1",
    )
    writer.append_from_somatic_state(
        SomaticState(tension=0.7, fatigue=0.3, valence=-0.2, certainty=0.4),
        phase=ExecutionPhase.TURN_END,
        session_id="s1",
    )
    await receipt_store.save_direct(
        ExecutionReceipt(
            task_id="11111111-1111-1111-1111-111111111111",
            objective="verify cognitive telemetry",
            success=True,
            output="verified",
        )
    )
    runtime = SimpleNamespace(
        cognitive_state_store=cognitive,
        ctx=SimpleNamespace(
            config=SimpleNamespace(state_dir=tmp_path),
            affective_registry_writer=writer,
            receipt_store=receipt_store,
        ),
        executive=SimpleNamespace(intention="", active_goals=[]),
        _activity="idle",
        _activity_since=None,
    )
    try:
        result = await run_runtime_cognitive_maintenance(runtime)
        events = await cognitive.list_recent_events(kind=CognitiveEventKind.TELEMETRY_AFFECT, limit=5)

        assert result["telemetry_affect"]["available"] is True
        assert result["telemetry_affect"]["registry_sync"]["snapshots_created"] == 2
        assert result["telemetry_affect"]["registry_sync"]["trajectories_created"] == 1
        assert result["telemetry_affect"]["quality_sync"]["signals_created"] == 1
        assert result["telemetry_affect"]["graph"]["edges_created"] == 1
        assert any("Telemetry affect processed" in event.summary for event in events)
    finally:
        await cognitive.close()
        await receipt_store.close()


@pytest.mark.asyncio
async def test_cognitive_maintenance_promotes_high_curiosity_to_creative_ladder(tmp_path):
    cognitive = CognitiveStateStore(tmp_path / "cognitive.db")
    await cognitive.connect()
    added = []

    node = SimpleNamespace(
        key="novel-widget",
        summary="Imagine a small widget that compares schedule promises with proof.",
        source="test",
        route="act_now",
        salience=0.84,
        novelty=0.61,
        recurrence_count=2,
        should_force_work=True,
    )
    runtime = SimpleNamespace(
        cognitive_state_store=cognitive,
        fascination_graph=SimpleNamespace(active=lambda limit=5: [node]),
        creative=SimpleNamespace(add=added.append, _ladder=[]),
        executive=SimpleNamespace(intention="", active_goals=[]),
        _activity="idle",
        _activity_since=None,
    )
    try:
        result = await run_runtime_cognitive_maintenance(runtime)
        events = await cognitive.list_recent_events(kind=CognitiveEventKind.CURIOSITY, limit=10)

        assert result["curiosity_promotions"] == 1
        assert len(added) == 1
        assert added[0].meta["origin"] == "cognitive_curiosity"
        assert any("promoted into creative ladder" in event.summary for event in events)
    finally:
        await cognitive.close()


@pytest.mark.asyncio
async def test_cognitive_maintenance_records_scored_counterfactual_review(tmp_path):
    cognitive = CognitiveStateStore(tmp_path / "cognitive.db")
    await cognitive.connect()
    memory = MemoryStore(tmp_path / "memory.db")
    await memory.connect()
    try:
        for idx, content in enumerate(
            [
                "tool workflow_list_schedules failed: missing schedule record",
                "tool workflow_list_schedules error: not found after retry",
            ],
            start=1,
        ):
            await memory.save_episode(
                Episode(
                    kind=EpisodeKind.ACTION,
                    session_id=f"s{idx}",
                    content=content,
                )
            )
        runtime = SimpleNamespace(
            cognitive_state_store=cognitive,
            memory=memory,
            executive=SimpleNamespace(intention="verify schedule proof", active_goals=[]),
            _activity="idle",
            _activity_since=None,
        )

        result = await run_runtime_cognitive_maintenance(runtime)
        events = await cognitive.list_recent_events(kind=CognitiveEventKind.COUNTERFACTUAL, limit=5)
        working = await cognitive.list_working_memory(limit=10)

        assert result["events_recorded"] >= 1
        assert events
        assert events[0].payload["counterfactual"]["recommended"]["strategy"] == "verify_prerequisite"
        assert any(
            item.slot == "counterfactual_review"
            and item.payload["counterfactual"]["recommended"]["strategy"] == "verify_prerequisite"
            for item in working
        )
    finally:
        await cognitive.close()
        await memory.close()


@pytest.mark.asyncio
async def test_cognitive_maintenance_narrative_updates_identity_self_knowledge(tmp_path):
    cognitive = CognitiveStateStore(tmp_path / "cognitive.db")
    await cognitive.connect()
    identity = IdentityManager(IdentityStore(tmp_path / "identity"))
    identity.load()
    try:
        await cognitive.record_event(
            CognitiveEventKind.SURPRISE,
            "User correction contradicted a capability denial",
            source="test",
            salience=1.8,
        )
        await cognitive.record_event(
            CognitiveEventKind.UNCERTAINTY_SEEKING,
            "Evidence check required before final answer",
            source="test",
            salience=1.5,
        )
        runtime = SimpleNamespace(
            cognitive_state_store=cognitive,
            ctx=SimpleNamespace(identity=identity),
            executive=SimpleNamespace(intention="", active_goals=[]),
            _activity="idle",
            _activity_since=None,
        )

        await run_runtime_cognitive_maintenance(runtime)

        assert identity.self_model.self_beliefs["cognitive_narrative.recent_arc"]["theme_counts"]
        assert {"term": "surprise", "count": 1} in identity.self_model.recent_themes
    finally:
        await cognitive.close()


@pytest.mark.asyncio
async def test_duplicate_recent_arc_emits_narrator_redundancy_event(tmp_path):
    registry = SelfKnowledgeRegistry(tmp_path / "self_knowledge.jsonl")
    identity = IdentityManager(IdentityStore(tmp_path / "identity"), registry=registry)
    identity.load()
    event = SimpleNamespace(
        event_id="event-1",
        kind=SimpleNamespace(value="procedural_skill"),
        summary="Learned procedure evidence_gated_auto_use",
        salience=1.0,
        created_at="2026-05-08T00:00:00+00:00",
    )
    recent_arc = {
        "summary": "Recent cognitive arc: procedural_skill: Learned procedure evidence_gated_auto_use",
        "theme_counts": {"procedural_skill": 1},
        "evidence_refs": ["cognitive_event:event-1"],
    }
    registry.record(
        "cognitive_narrative",
        "recent_arc",
        recent_arc,
        meta={"source": "test"},
    )
    published = []

    class FakeBus:
        async def publish(self, bus_event):
            published.append(bus_event)

    async def _record_event(*args, **kwargs):
        return None

    async def _list_recent_events(limit=12):
        return [event]

    store = SimpleNamespace(
        list_recent_events=_list_recent_events,
        record_event=_record_event,
    )
    runtime = SimpleNamespace(
        ctx=SimpleNamespace(identity=identity),
        cognition_bus=FakeBus(),
    )

    recorded = await _record_narrative_summary(runtime, store)

    assert recorded == 1
    assert [event.kind for event in published] == ["narrator.redundancy_observed"]
    assert published[0].payload["skipped"] is True
    assert published[0].payload["content_hash"]
    assert len(registry.list_by_domain("cognitive_narrative")) == 1
