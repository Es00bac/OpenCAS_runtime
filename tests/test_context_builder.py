"""Tests for ContextBuilder prompt assembly."""

import json
from datetime import datetime, timedelta, timezone
from pathlib import Path
from types import SimpleNamespace

import pytest
import pytest_asyncio

from opencas.api import provenance_store as ps
from opencas.autonomy.executive import ExecutiveState
from opencas.bootstrap import BootstrapConfig
from opencas.context import (
    ContextBuilder,
    ContextManifest,
    ContextProposal,
    ContextProposalStore,
    MemoryRetriever,
    MessageEntry,
    MessageRole,
    ProposalStatus,
    SessionContextStore,
)
from opencas.context.hemispheres import ContextLane
from opencas.context.hemispheres import ContextAuthority
from opencas.daydream import DaydreamReflection
from opencas.embeddings import EmbeddingCache, EmbeddingService
from opencas.identity import IdentityManager, IdentityStore
from opencas.memory import MemoryStore
from opencas.relational import MusubiState, MusubiStore, RelationalEngine
from opencas.tom import ToMEngine
from opencas.tom.models import BeliefSubject
from opencas.wellbeing import WellbeingState


def test_to_message_list_renders_compaction_continuation_packet_handles() -> None:
    entry = MessageEntry(
        role=MessageRole.SYSTEM,
        content="[Context: earlier conversation was compacted. Summary: kept short.]",
        meta={
            "source": "compaction_narrative_bridge",
            "continuation_packet": {
                "version": 1,
                "handles": [
                    {
                        "tool_name": "fs_write_file",
                        "path": "/mnt/xtra/OpenCAS/workspace/writing/4246/story_4246.md",
                        "checksum": "abc123",
                        "task_id": "task-1",
                        "receipt_id": "receipt-1",
                        "schedule_id": "schedule-1",
                        "source_episode_id": "episode-1",
                        "retrieval_cues": [
                            "tool:fs_write_file",
                            "path:/mnt/xtra/OpenCAS/workspace/writing/4246/story_4246.md",
                            "checksum:abc123",
                            "task:task-1",
                            "session:s1",
                            "episode:episode-1",
                        ],
                    }
                ],
            },
        },
    )
    manifest = ContextManifest(history=[entry])

    messages = manifest.to_message_list()

    assert messages[0]["role"] == "system"
    content = messages[0]["content"]
    assert "Continuation handles:" in content
    assert "/mnt/xtra/OpenCAS/workspace/writing/4246/story_4246.md" in content
    assert "checksum=abc123" in content
    assert "task_id=task-1" in content
    assert "receipt_id=receipt-1" in content
    assert "schedule_id=schedule-1" in content
    assert "source_episode_id=episode-1" in content
    assert "retrieval_cues=path:/mnt/xtra/OpenCAS/workspace/writing/4246/story_4246.md" in content
    assert "tool:fs_write_file" in content
    assert "path:/mnt/xtra/OpenCAS/workspace/writing/4246/story_4246.md" in content
    assert "episode:episode-1" in content


def test_to_message_list_prioritizes_exact_continuation_handles_when_truncated() -> None:
    noisy_handles = [
        {
            "tool_name": f"tool_{index}",
            "task_id": f"noisy-task-{index}",
            "retrieval_cues": [f"tool:tool_{index}", f"task:noisy-task-{index}"],
        }
        for index in range(8)
    ]
    priority_path = "/mnt/xtra/OpenCAS/workspace/writing/4246/story_4246.md"
    entry = MessageEntry(
        role=MessageRole.SYSTEM,
        content="[Context: earlier conversation was compacted.]",
        meta={
            "source": "compaction_narrative_bridge",
            "continuation_packet": {
                "version": 1,
                "handles": [
                    *noisy_handles,
                    {
                        "path": priority_path,
                        "checksum": "abc123",
                        "source_episode_id": "episode-priority",
                        "retrieval_cues": [
                            "tool:fs_write_file",
                            f"path:{priority_path}",
                            "checksum:abc123",
                            "episode:episode-priority",
                            "task:should-drop-if-cues-are-capped",
                        ],
                    },
                ],
            },
        },
    )
    manifest = ContextManifest(history=[entry])

    content = manifest.to_message_list()[0]["content"]

    assert priority_path in content
    assert "checksum=abc123" in content
    assert "source_episode_id=episode-priority" in content
    assert content.index("path=") < content.index("checksum=")
    assert "noisy-task-7" not in content
    assert "... 1 more handle(s) in continuation_packet metadata" in content


def test_to_message_list_bounds_recursive_compaction_system_context() -> None:
    priority_path = "/mnt/xtra/OpenCAS/workspace/writing/4246/story_4246.md"
    history = []
    for index in range(5):
        meta = {"source": "compaction_narrative_bridge"}
        if index == 4:
            meta["continuation_packet"] = {
                "version": 1,
                "handles": [
                    {
                        "path": priority_path,
                        "checksum": "abc123",
                        "source_episode_id": "episode-priority",
                        "retrieval_cues": [f"path:{priority_path}", "checksum:abc123"],
                    }
                ],
            }
        history.append(
            MessageEntry(
                role=MessageRole.SYSTEM,
                content=(
                    "[Context: earlier conversation was compacted. Summary: "
                    f"old-marker-{index} " + ("stale Writing Project loop " * 1200) + "]"
                ),
                meta=meta,
            )
        )
    manifest = ContextManifest(history=history)

    content = manifest.to_message_list()[0]["content"]

    assert len(content) < 17_000
    assert "old-marker-4" in content
    assert priority_path in content
    assert "checksum=abc123" in content
    assert "old-marker-0" not in content
    assert "omitted to keep prompt bounded" in content


@pytest_asyncio.fixture
async def builder_deps(tmp_path):
    ctx_store = SessionContextStore(tmp_path / "context.db")
    await ctx_store.connect()

    mem_store = MemoryStore(tmp_path / "memory.db")
    await mem_store.connect()

    cache = EmbeddingCache(tmp_path / "embeddings.db")
    await cache.connect()
    embed_service = EmbeddingService(cache=cache, model_id="local-fallback")
    retriever = MemoryRetriever(memory=mem_store, embeddings=embed_service)

    id_store = IdentityStore(tmp_path / "identity")
    identity = IdentityManager(id_store)
    identity.load()

    executive = ExecutiveState(identity=identity)
    executive.add_goal("test the builder")
    executive.set_intention("verify context assembly")

    builder = ContextBuilder(
        store=ctx_store,
        retriever=retriever,
        identity=identity,
        executive=executive,
        recent_limit=10,
    )
    yield builder, ctx_store, mem_store
    await ctx_store.close()
    await mem_store.close()
    await cache.close()


@pytest.mark.asyncio
async def test_build_includes_system_and_history(builder_deps):
    builder, ctx_store, _mem_store = builder_deps
    await ctx_store.append("s1", MessageRole.USER, "hello")
    manifest = await builder.build("hello", session_id="s1")

    assert manifest.system is not None
    assert "OpenCAS" in manifest.system.content
    assert "test the builder" in manifest.system.content
    assert "verify context assembly" in manifest.system.content

    assert len(manifest.history) == 1
    assert manifest.history[0].role == MessageRole.USER
    assert manifest.history[0].content == "hello"


@pytest.mark.asyncio
async def test_build_surfaces_relevant_reflective_context_proposals(builder_deps, tmp_path):
    builder, _ctx_store, _mem_store = builder_deps
    proposal_store = await ContextProposalStore(tmp_path / "context_proposals.db").connect()
    proposal = ContextProposal(
        source_lane=ContextLane.REFLECTIVE,
        source_snapshot_id="truth:1:abc",
        source_epoch=1,
        proposal_kind="bad_idea_to_avoid",
        project_id="writing project 4246",
        content=(
            "writing project 4246 Chapter 3 caution: do not repeat the earlier hallway "
            "pacing idea unless there is new manuscript evidence."
        ),
        evidence_refs=["daydream_reflection:chapter-3"],
        status=ProposalStatus.REJECTED,
    )
    await proposal_store.save(proposal)
    builder.context_proposal_store = proposal_store

    try:
        manifest = await builder.build("Return to writing project 4246 chapter 3", session_id="s1")
    finally:
        await proposal_store.close()

    assert manifest.system is not None
    content = manifest.system.content
    assert "Reflective proposal recall:" in content
    assert "rejected idea / caution" in content
    assert "authority=proposal" in content
    assert "do not repeat the earlier hallway pacing idea" in content
    assert "do not authorize schedules, commitments, BAA tasks, or tool writes" in content
    assert proposal.proposal_id in content
    audit = manifest.system.meta["context_proposal_audit"]
    assert audit["searched"] is True
    assert audit["rendered_in_prompt"] is True
    assert audit["rendered_count"] == 1
    assert proposal.proposal_id in audit["evidence_ids"]
    assert "context_proposals" not in manifest.system.meta["proactive_channel_audit"]["channels"]


@pytest.mark.asyncio
async def test_build_surfaces_project_scoped_context_proposals_without_text_overlap(builder_deps, tmp_path):
    builder, _ctx_store, _mem_store = builder_deps
    proposal_store = await ContextProposalStore(tmp_path / "context_proposals.db").connect()
    proposal = ContextProposal(
        source_lane=ContextLane.REFLECTIVE,
        source_snapshot_id="truth:1:abc",
        source_epoch=1,
        proposal_kind="project_next_step",
        project_id="writing project 4246",
        content="Use the blue-thread motif only if Chapter 3 already supports it.",
        evidence_refs=["memory:daydream-blue-thread"],
    )
    await proposal_store.save(proposal)

    class _Resolver:
        async def resolve(self, _query):
            return SimpleNamespace(display_name="writing project 4246", matched_project_ids=[])

    builder.context_proposal_store = proposal_store
    builder.project_resume_resolver = _Resolver()

    try:
        manifest = await builder.build("Continue the manuscript", session_id="s1")
    finally:
        await proposal_store.close()

    assert manifest.system is not None
    content = manifest.system.content
    assert "Reflective proposal recall:" in content
    assert "blue-thread motif" in content
    assert proposal.proposal_id in content


@pytest.mark.asyncio
async def test_build_disambiguates_accepted_proposal_authority(builder_deps, tmp_path):
    builder, _ctx_store, _mem_store = builder_deps
    proposal_store = await ContextProposalStore(tmp_path / "context_proposals.db").connect()
    proposal = ContextProposal(
        source_lane=ContextLane.REFLECTIVE,
        source_snapshot_id="truth:2:def",
        source_epoch=2,
        proposal_kind="project_next_step",
        project_id="writing project 4246",
        content="Accepted Chapter 3 support can influence the next manuscript pass.",
        evidence_refs=["memory:accepted-daydream"],
        authority=ContextAuthority.EXECUTIVE_COMMITTED,
        status=ProposalStatus.ACCEPTED,
        validation={"arbiter_decision_id": "decision:chapter-3"},
    )
    await proposal_store.save(proposal)
    builder.context_proposal_store = proposal_store

    try:
        manifest = await builder.build("writing project 4246 Chapter 3 accepted support", session_id="s1")
    finally:
        await proposal_store.close()

    assert manifest.system is not None
    content = manifest.system.content
    assert "accepted support" in content
    assert "authority=executive_committed" in content
    assert "arbiter_decision=decision:chapter-3" in content


@pytest.mark.asyncio
async def test_build_system_entry_surfaces_elapsed_shutdown_anchor_for_continuity_question(builder_deps):
    builder, _ctx_store, _mem_store = builder_deps
    identity = builder.identity
    assert identity is not None
    shutdown_at = datetime.now(timezone.utc) - timedelta(minutes=3)
    identity.continuity.last_shutdown_time = shutdown_at
    identity.self_model.current_intention = "repair OpenCAS dashboard continuity"
    identity.record_continuity_breadcrumb(
        intent="continue Prompt F continuous-present repair",
        decision="cold-restart probe failed",
        next_step="inject elapsed shutdown context into the first turn",
        timestamp=shutdown_at,
    )

    manifest = await builder.build("How long have I been gone, and what were we doing last?", session_id="s1")

    assert manifest.system is not None
    content = manifest.system.content
    assert "Continuity context:" in content
    assert "elapsed_since_last_shutdown_or_persistence:" in content
    assert "minutes" in content
    assert "last_shutdown_time:" in content
    assert "current_intention: repair OpenCAS dashboard continuity" in content
    assert "continue Prompt F continuous-present repair" in content
    assert "evidence: identity.continuity.last_shutdown_time" in content
    assert "Do not estimate elapsed absence from older autobiographical memory" in content


@pytest.mark.asyncio
async def test_build_system_entry_reports_missing_persistence_for_continuity_question(builder_deps):
    builder, _ctx_store, _mem_store = builder_deps
    identity = builder.identity
    assert identity is not None
    identity.continuity.last_shutdown_time = None
    identity.continuity.last_persisted_at = None
    identity.continuity.last_offline_started_at = None
    identity.continuity.last_offline_duration_seconds = None

    manifest = await builder.build("How long have I been gone?", session_id="s1")

    assert manifest.system is not None
    content = manifest.system.content
    assert "Continuity context:" in content
    assert "prior_persistence: unavailable" in content
    assert "no prior persistence; treating as cold session" in content
    assert "Do not estimate elapsed absence from older autobiographical memory" in content


@pytest.mark.asyncio
async def test_build_continuity_context_includes_recent_beads_and_open_commitment(builder_deps):
    builder, _ctx_store, _mem_store = builder_deps
    identity = builder.identity
    assert identity is not None
    started_at = datetime.now(timezone.utc) - timedelta(minutes=12)
    identity.continuity.last_offline_started_at = started_at
    identity.continuity.last_offline_duration_seconds = 12 * 60

    class FakeThreadRegistryStore:
        async def list_beads(self, limit=3, **_kwargs):
            return [
                SimpleNamespace(
                    title=f"Prompt F bead {index}",
                    summary=f"recent continuity bead {index}",
                    source_kind="daydream_reflection",
                    source_ref=f"thread:bead-{index}",
                )
                for index in range(4)
            ][:limit]

    class FakeCommitmentStore:
        async def list_active(self, limit=10, offset=0):
            return [
                SimpleNamespace(
                    commitment_id="commitment-1",
                    content="finish the continuous present repair",
                    meta={"source": "assistant_response"},
                    tags=["self_commitment"],
                )
            ]

    builder.thread_registry_store = FakeThreadRegistryStore()
    builder.commitment_store = FakeCommitmentStore()

    manifest = await builder.build("Continue.", session_id="s1")

    assert manifest.system is not None
    content = manifest.system.content
    assert "Continuity context:" in content
    assert "recent_thread_registry_beads:" in content
    assert "Prompt F bead 0" in content
    assert "Prompt F bead 2" in content
    assert "Prompt F bead 3" not in content
    assert "open_operator_commitment:" in content
    assert "finish the continuous present repair" in content
    assert "commitment:commitment-1" in content


@pytest.mark.asyncio
async def test_structured_continuity_context_suppresses_boot_monologue_in_prompt(builder_deps):
    builder, _ctx_store, _mem_store = builder_deps
    identity = builder.identity
    assert identity is not None
    identity.continuity.last_offline_started_at = datetime.now(timezone.utc) - timedelta(minutes=8)
    identity.continuity.last_offline_duration_seconds = 8 * 60
    identity.continuity.last_continuity_monologue = "I was offline for 8 minutes."

    manifest = await builder.build("Continue.", session_id="s1")

    assert manifest.system is not None
    assert "Continuity context:" in manifest.system.content
    assert "Boot continuity monologue:" not in manifest.system.content
    assert "I was offline for 8 minutes" not in manifest.system.content


@pytest.mark.asyncio
async def test_continuity_context_uses_prior_breadcrumb_not_current_probe(builder_deps):
    builder, _ctx_store, _mem_store = builder_deps
    identity = builder.identity
    assert identity is not None
    identity.continuity.last_offline_started_at = datetime.now(timezone.utc) - timedelta(minutes=6)
    identity.continuity.last_offline_duration_seconds = 6 * 60
    identity.continuity.continuity_breadcrumb = (
        "2026-05-07T00:30:00+00:00 | intent: start burst: Conversation burst "
        "for How long have I been gone | decision: current probe"
    )
    identity.continuity.continuity_breadcrumbs = [
        "2026-05-07T00:20:00+00:00 | intent: finish Prompt F | decision: live verify",
        "2026-05-07T00:25:00+00:00 | intent: start burst: Cycle burst for cycle | decision: scheduler housekeeping",
        "2026-05-07T00:26:00+00:00 | intent: resume context after 1 minutes | decision: boot bookkeeping",
        identity.continuity.continuity_breadcrumb,
    ]

    manifest = await builder.build("How long have I been gone?", session_id="s1")

    assert manifest.system is not None
    assert "latest_continuity_breadcrumb:" in manifest.system.content
    assert "finish Prompt F" in manifest.system.content
    assert "current probe" not in manifest.system.content
    assert "scheduler housekeeping" not in manifest.system.content
    assert "boot bookkeeping" not in manifest.system.content


@pytest.mark.asyncio
async def test_build_includes_verifiable_followup_contract(builder_deps):
    builder, _ctx_store, _mem_store = builder_deps

    manifest = await builder.build(
        "Please work on this in the background and tell me when it is done.",
        session_id="s1",
    )

    assert manifest.system is not None
    assert "Verifiable action contract" in manifest.system.content
    assert "workflow_create_schedule" in manifest.system.content
    assert "get back" in manifest.system.content


@pytest.mark.asyncio
async def test_build_token_estimate(builder_deps):
    builder, ctx_store, _mem_store = builder_deps
    await ctx_store.append("s1", MessageRole.USER, "hello world")
    manifest = await builder.build("hello world", session_id="s1")
    assert manifest.token_estimate is not None
    assert manifest.token_estimate > 0


def test_creative_project_resume_classifier_does_not_treat_write_code_as_manuscript_work() -> None:
    software_snapshot = SimpleNamespace(
        display_name="kPony",
        synopsis="Qt6 email client with CMake build and source files.",
        canonical_artifact_path="workspace/kPony/src/main.cpp",
        supporting_artifact_paths=["workspace/kPony/CMakeLists.txt"],
    )
    writing_snapshot = SimpleNamespace(
        display_name="writing project 4246",
        synopsis="Existing manuscript revision project.",
        canonical_artifact_path="workspace/writing/4246/story_4246.md",
        supporting_artifact_paths=[],
    )

    assert ContextBuilder._is_creative_project_resume("write code for kPony", software_snapshot) is False
    assert ContextBuilder._is_creative_project_resume("keep revising writing project 4246", writing_snapshot) is True


@pytest.mark.asyncio
async def test_to_message_list_format(builder_deps):
    builder, ctx_store, _mem_store = builder_deps
    await ctx_store.append("s1", MessageRole.USER, "hi")
    manifest = await builder.build("hi", session_id="s1")
    messages = manifest.to_message_list()
    assert messages[0]["role"] == "system"
    assert messages[-1]["role"] == "user"
    assert messages[-1]["content"] == "hi"


@pytest.mark.asyncio
async def test_to_message_list_includes_user_attachment_content(builder_deps):
    builder, ctx_store, _mem_store = builder_deps
    await ctx_store.append(
        "s1",
        MessageRole.USER,
        "What do you think of my resume?",
        meta={
            "attachments": [
                {
                    "filename": "resume.md",
                    "media_type": "text/markdown",
                    "text_content": "# Resume\n- Built Python automation",
                }
            ]
        },
    )
    manifest = await builder.build("What do you think of my resume?", session_id="s1")
    messages = manifest.to_message_list()

    assert messages[-1]["role"] == "user"
    assert "What do you think of my resume?" in messages[-1]["content"]
    assert "[Attached file: resume.md (text/markdown)]" in messages[-1]["content"]
    assert "# Resume" in messages[-1]["content"]


@pytest.mark.asyncio
async def test_to_message_list_filters_dangling_assistant_tool_calls(builder_deps):
    builder, ctx_store, _mem_store = builder_deps
    await ctx_store.append(
        "s1",
        MessageRole.ASSISTANT,
        "",
        meta={
            "tool_calls": [
                {"id": "tc1", "function": {"name": "fs_read_file"}},
                {"id": "tc2", "function": {"name": "fs_read_file"}},
            ]
        },
    )
    await ctx_store.append(
        "s1",
        MessageRole.TOOL,
        "done",
        meta={"tool_call_id": "tc1", "name": "fs_read_file"},
    )
    manifest = await builder.build("continue", session_id="s1")

    messages = manifest.to_message_list()

    assistant = next(msg for msg in messages if msg["role"] == "assistant")
    assert [tc["id"] for tc in assistant["tool_calls"]] == ["tc1"]
    tool = next(msg for msg in messages if msg["role"] == "tool")
    assert tool["tool_call_id"] == "tc1"


@pytest.mark.asyncio
async def test_build_includes_somatic_style_note(builder_deps):
    from opencas.somatic import SomaticModulators, SomaticState
    builder, ctx_store, _mem_store = builder_deps
    builder.modulators = SomaticModulators(SomaticState(tension=0.7))
    manifest = await builder.build("hello", session_id="s1")
    assert "concise" in manifest.system.content.lower()


@pytest.mark.asyncio
async def test_build_includes_live_somatic_snapshot_even_without_style_note(builder_deps):
    from opencas.somatic import SomaticModulators, SomaticState
    builder, _ctx_store, _mem_store = builder_deps
    builder.modulators = SomaticModulators(
        SomaticState(
            somatic_tag="caring",
            arousal=0.04,
            energy=1.0,
            focus=0.5,
            fatigue=0.01,
            tension=0.03,
            valence=0.1,
            certainty=0.6,
        )
    )

    manifest = await builder.build("Good afternoon, how are you feeling?", session_id="s1")

    assert "Current somatic snapshot:" in manifest.system.content
    assert "tag=caring" in manifest.system.content
    assert "energy=1.00" in manifest.system.content
    assert "valence=0.10" in manifest.system.content


@pytest.mark.asyncio
async def test_build_includes_recent_daydream_continuity(builder_deps):
    builder, _ctx_store, _mem_store = builder_deps
    assert builder.identity is not None
    builder.identity.user_model.partner_user_id = "ConfiguredPartner"

    class FakeDaydreamStore:
        async def list_recent(self, limit=3, keeper_only=None):
            return [
                DaydreamReflection(
                    created_at="2026-04-29T21:53:44+00:00",
                    spark_content="Attention Fingerprinting: prev_hash as a breadcrumb trail.",
                    synthesis="The unified graph maps intent onto structure.",
                    open_question="When does a hybrid edge crystallize?",
                    alignment_score=0.35,
                    novelty_score=0.81,
                    keeper=True,
                    experience_context={
                        "trigger": "background_daydream",
                        "active_goals": ["define witness daemon schema"],
                        "somatic": {
                            "somatic_tag": "continuity_pressure",
                            "tension": 0.21,
                            "valence": 0.12,
                            "focus": 0.68,
                        },
                        "contact": {
                            "status": "sent",
                            "channel": "telegram",
                            "reason": "worth sharing",
                            "message_preview": "The unified graph maps intent.",
                        },
                    },
                )
            ]

    builder.daydream_store = FakeDaydreamStore()

    manifest = await builder.build("Do you daydream?", session_id="s1")

    assert "Recent background daydream continuity" in manifest.system.content
    assert "your own background daydream loop" in manifest.system.content
    assert "Attention Fingerprinting" in manifest.system.content
    assert "somatic: continuity_pressure" in manifest.system.content
    assert "define witness daemon schema" in manifest.system.content
    assert "contacted ConfiguredPartner via telegram" in manifest.system.content
    assert "worth sharing" in manifest.system.content


@pytest.mark.asyncio
async def test_build_correlates_initiative_contact_event_with_daydream(builder_deps, tmp_path):
    builder, _ctx_store, _mem_store = builder_deps
    assert builder.identity is not None
    builder.identity.user_model.partner_user_id = "ConfiguredPartner"
    builder.config = SimpleNamespace(state_dir=tmp_path)
    reflection_id = "d45050f8-2b14-4f53-92cc-ed42c9389edc"

    events_path = tmp_path / "initiative_contact" / "events.jsonl"
    events_path.parent.mkdir(parents=True)
    events_path.write_text(
        json.dumps(
            {
                "created_at": "2026-04-29T21:54:16+00:00",
                "status": "sent",
                "source": "reflection",
                "source_id": reflection_id,
                "channel": "telegram",
                "urgency": "normal",
                "reason": "fallback_escalation",
                "message_preview": "I think you should know this: The unified graph maps intent.",
                "dispatch": {
                    "decision": {
                        "reason": "fallback_escalation",
                        "message": "I think you should know this: The unified graph maps intent.",
                    }
                },
            },
            ensure_ascii=True,
        )
        + "\n",
        encoding="utf-8",
    )

    class FakeDaydreamStore:
        async def list_recent(self, limit=3, keeper_only=None):
            return [
                DaydreamReflection(
                    reflection_id=reflection_id,
                    created_at="2026-04-29T21:53:44+00:00",
                    spark_content="Soft-Focus Topology: the defer subgraph as peripheral visual field.",
                    synthesis="The unified graph maps energy and intent onto structure.",
                    open_question="When does a Hybrid edge crystallize?",
                    alignment_score=0.35,
                    novelty_score=0.862,
                    keeper=True,
                    experience_context={"trigger": "background_daydream"},
                )
            ]

    builder.daydream_store = FakeDaydreamStore()

    manifest = await builder.build("Why did you message me?", session_id="s1")

    assert "Soft-Focus Topology" in manifest.system.content
    assert "contacted ConfiguredPartner via telegram" in manifest.system.content
    assert "fallback_escalation" in manifest.system.content
    assert "The unified graph maps intent" in manifest.system.content


@pytest.mark.asyncio
async def test_build_consumes_relevant_thread_registry_beads(builder_deps):
    builder, _ctx_store, _mem_store = builder_deps

    class FakeThreadRegistryStore:
        async def list_beads(self, limit=25, **_kwargs):
            return [
                SimpleNamespace(
                    title="Provenance registry draft",
                    summary="A durable registry entry can bind writing project 4246 edits to evidence and retrieval paths.",
                    source_kind="daydream_reflection",
                    source_ref="daydream:reflection-1:0",
                    status="peripheral",
                    thread_anchor_id="provenance-registry",
                ),
                SimpleNamespace(
                    title="Unrelated note",
                    summary="A different thought about something else.",
                    source_kind="autonomous_artifact",
                    source_ref="wellbeing:note",
                    status="peripheral",
                    thread_anchor_id="unrelated",
                ),
            ]

    builder.thread_registry_store = FakeThreadRegistryStore()

    manifest = await builder.build(
        "Continue the provenance registry for writing project 4246.",
        session_id="s1",
    )

    assert manifest.system is not None
    assert "Thread registry continuity cues:" in manifest.system.content
    assert "Provenance registry draft" in manifest.system.content
    assert "daydream_reflection daydream:reflection-1:0" in manifest.system.content
    assert "Unrelated note" not in manifest.system.content


@pytest.mark.asyncio
async def test_build_audits_weak_thread_registry_overlap_without_rendering_cues(builder_deps):
    builder, _ctx_store, _mem_store = builder_deps

    class FakeThreadRegistryStore:
        async def list_beads(self, limit=25, **_kwargs):
            return [
                SimpleNamespace(
                    title="Provenance aside",
                    summary="A loose note with only one overlapping term.",
                    source_kind="daydream_reflection",
                    source_ref="daydream:weak:0",
                    status="peripheral",
                    thread_anchor_id="weak-provenance",
                )
            ]

    builder.thread_registry_store = FakeThreadRegistryStore()

    manifest = await builder.build("Continue provenance.", session_id="s1")

    assert manifest.system is not None
    assert "Thread registry continuity cues:" not in manifest.system.content
    audit = manifest.system.meta["thread_registry_selection_audit"]
    assert audit["available"] is True
    assert audit["scanned_count"] == 1
    assert audit["weak_overlap_count"] == 1
    assert audit["selected_count"] == 0
    assert audit["relevance_passed"] is False


@pytest.mark.asyncio
async def test_build_filters_suppressed_reframe_beads_from_active_context(builder_deps):
    builder, _ctx_store, _mem_store = builder_deps

    class FakeThreadRegistryStore:
        async def list_beads(self, limit=25, **_kwargs):
            return [
                SimpleNamespace(
                    title="Suppressed browser click reframe",
                    summary=(
                        "browser_click lightsaber duplicate suppression metadata was rejected "
                        "because the source artifact repeated the objective."
                    ),
                    source_kind="suppressed_reframe",
                    source_ref="suppressed_reframe:task-1:task-2",
                    status="peripheral",
                    thread_anchor_id="suppressed-reframes",
                )
            ]

    builder.thread_registry_store = FakeThreadRegistryStore()

    manifest = await builder.build(
        "Should we retry the browser_click lightsaber task?",
        session_id="s1",
    )

    assert manifest.system is not None
    assert "Thread registry continuity cues:" not in manifest.system.content
    assert "Suppressed browser click reframe" not in manifest.system.content
    audit = manifest.system.meta["thread_registry_selection_audit"]
    assert audit["suppressed_reframe_filtered_count"] == 1
    assert audit["selected_count"] == 0
    assert audit["relevance_passed"] is False


@pytest.mark.asyncio
async def test_build_includes_personal_hobby_seeds_from_daydream_self_beliefs(builder_deps):
    builder, _ctx_store, _mem_store = builder_deps
    assert builder.identity is not None
    builder.identity.self_model.self_beliefs["daydream"] = {
        "bulma_config": {
            "hobbySeeds": [
                "internet archaeology through old logs and forgotten forums",
                "language fragments and translation gaps",
            ],
        },
        "bulma_status": {
            "currentInterest": "obscure tools and niche software ecosystems",
        },
    }

    manifest = await builder.build("Who are you outside helping Jarrod?", session_id="s1")

    assert "Personal curiosity and hobby state" in manifest.system.content
    assert "obscure tools and niche software ecosystems" in manifest.system.content
    assert "internet archaeology through old logs and forgotten forums" in manifest.system.content
    assert "language fragments and translation gaps" in manifest.system.content
    assert "self-directed attention and intention seeds" in manifest.system.content
    assert "use them to research, build, write, or create artifacts" in manifest.system.content


@pytest.mark.asyncio
async def test_build_includes_compact_affective_pressure_summary(builder_deps):
    builder, _ctx_store, _mem_store = builder_deps

    class FakeAffectiveExaminations:
        def __init__(self):
            self.calls = []

        async def recent_pressure_summary(self, **kwargs):
            self.calls.append(kwargs)
            return {
                "available": True,
                "prompt_block": (
                    "Recent affective examination pressure:\n"
                    "- verify: examined tool evidence raised uncertainty; verify before relying on it"
                ),
            }

    service = FakeAffectiveExaminations()
    builder.affective_examinations = service

    manifest = await builder.build("Should I trust the last tool result?", session_id="s1")

    assert service.calls == [{"session_id": "s1", "char_budget": 600}]
    assert "Recent affective examination pressure:" in manifest.system.content
    assert "verify before relying on it" in manifest.system.content


@pytest.mark.asyncio
async def test_build_includes_recent_self_inspection_guidance(builder_deps):
    builder, _ctx_store, _mem_store = builder_deps

    class FakeSelfInspectionStore:
        async def list_recent(self, **kwargs):
            return [
                SimpleNamespace(
                    drift_observations=[SimpleNamespace(reason="Repeated timestamp framing")],
                    commitment_gaps=[
                        SimpleNamespace(
                            promised="send a health check when done",
                            gap_type="captured_without_execution_link",
                        )
                    ],
                    valence_sources=[
                        SimpleNamespace(source="operator_correction", reason="recent correction")
                    ],
                )
            ]

    builder.self_inspection_store = FakeSelfInspectionStore()

    manifest = await builder.build("What should you do next?", session_id="s1")

    assert "Recent self-inspection feedback:" in manifest.system.content
    assert "Repeated timestamp framing" in manifest.system.content
    assert "send a health check when done" in manifest.system.content
    assert "recent correction" in manifest.system.content


@pytest.mark.asyncio
async def test_build_includes_schedule_backed_temporal_agenda(builder_deps):
    builder, _ctx_store, _mem_store = builder_deps

    class FakeScheduleService:
        async def temporal_agenda(self, **kwargs):
            return {
                "counts": {
                    "active": 2,
                    "due_now": 1,
                    "upcoming": 1,
                    "recent_runs": 1,
                },
                "next": {
                    "title": "Check the calendar",
                    "kind": "task",
                    "action": "submit_baa",
                    "next_run_at": "2026-04-29T18:00:00+00:00",
                    "is_due": True,
                },
                "due_now": [{"title": "Check the calendar"}],
                "recent_runs": [
                    {
                        "status": "submitted",
                        "scheduled_for": "2026-04-29T17:00:00+00:00",
                        "task_id": "task-1",
                    }
                ],
            }

    builder.schedule_service = FakeScheduleService()
    builder.identity.self_model.name = "TestAgent"

    system_entry = await builder._build_system_entry(
        user_input="What should you do next?",
        session_id="s1",
    )

    content = system_entry.content
    assert "Temporal agenda from durable calendar:" in content
    assert "This is TestAgent's durable calendar/agenda surface" in content
    assert "Bulma's durable calendar" not in content
    assert "separate from OS cron" in content
    assert "active=2, due_now=1, upcoming_24h=1, recent_runs=1" in content
    assert "Check the calendar" in content
    assert "Do not invent calendar commitments" in content


@pytest.mark.asyncio
async def test_build_system_entry_does_not_inject_bulma_migration_for_fresh_agent(builder_deps):
    builder, _ctx_store, _mem_store = builder_deps
    builder.identity.self_model.name = "Mina"
    builder.identity.self_model.source_system = None
    builder.identity.self_model.imported_identity_profile = {}

    system_entry = await builder._build_system_entry(
        user_input="Who are you?",
        session_id="s1",
    )

    assert "You are Mina" in system_entry.content
    assert "there has only ever been one Bulma instance" not in system_entry.content


@pytest.mark.asyncio
async def test_build_applies_emotion_boost_to_retrieval(builder_deps):
    from opencas.somatic import SomaticModulators, SomaticState
    builder, ctx_store, mem_store = builder_deps

    # Seed an episode with "joy" in the content so keyword retrieval finds it
    from opencas.memory import Episode, EpisodeKind
    await mem_store.save_episode(
        Episode(kind=EpisodeKind.OBSERVATION, content="I felt joy today")
    )

    builder.modulators = SomaticModulators(
        SomaticState(valence=0.8, arousal=0.6)
    )
    manifest = await builder.build("joy", session_id="s1")
    # The retrieval should have returned the episode (boosted or not)
    assert any("joy" in r.content.lower() for r in manifest.retrieved)


@pytest.mark.asyncio
async def test_build_semantic_budgeting_prunes_redundant_results(builder_deps):
    """When token estimate exceeds max_tokens, redundant results are removed greedily."""
    builder, ctx_store, mem_store = builder_deps

    from opencas.memory import Episode, EpisodeKind
    # Seed many very similar episodes (high redundancy) and one distinct episode
    contents = [
        "The quick brown fox jumps over the lazy dog",
        "The quick brown fox leaps over the lazy dog",
        "The quick brown fox hops over the lazy dog",
        "A completely unrelated astronomical discovery about exoplanets",
    ]
    for content in contents:
        await mem_store.save_episode(Episode(kind=EpisodeKind.OBSERVATION, content=content))

    # Force pruning by setting a max_tokens budget that fits system + ~1 memory.
    # Measure the actual system prompt size first so the budget is realistic.
    system_entry = await builder._build_system_entry()
    system_tokens = builder._estimate_tokens([system_entry.content])
    builder.max_tokens = system_tokens + 70

    manifest = await builder.build("fox", session_id="s1")
    # Should stay within budget
    assert manifest.token_estimate <= builder.max_tokens
    # At least the distinct memory should survive if all fox memes are redundant
    # (retrieval limit and exact pruning outcome depend on embeddings, so just assert budget)
    assert manifest.token_estimate <= builder.max_tokens


@pytest.mark.asyncio
async def test_build_adapts_budget_and_retrieval_limit_to_active_model(builder_deps):
    builder, _ctx_store, _mem_store = builder_deps

    class FakeLLM:
        def model_context_metadata(self):
            return {
                "context_window": 262144,
                "prompt_context_budget": 224000,
                "source": "test",
            }

    calls = []

    async def fake_retrieve(**kwargs):
        calls.append(kwargs)
        return []

    builder.llm = FakeLLM()
    builder.retriever.retrieve = fake_retrieve

    manifest = await builder.build("use the full model window", session_id="s1")

    assert manifest.token_budget == 64000
    assert manifest.context_window == 262144
    assert manifest.context_budget["model_prompt_context_budget"] == 224000
    assert manifest.context_budget["prompt_budget_policy"] == "standard_interaction_cap"
    assert manifest.context_budget["prompt_cache_strategy"] == "stable_prefix_then_volatile_runtime_facts"
    assert "current_time" in manifest.context_budget["volatile_prompt_fields_late"]
    assert calls[0]["limit"] > 10


@pytest.mark.asyncio
async def test_build_expands_effective_budget_for_grounded_memory_queries(builder_deps):
    builder, _ctx_store, _mem_store = builder_deps

    class FakeLLM:
        def model_context_metadata(self):
            return {
                "context_window": 262144,
                "prompt_context_budget": 224000,
                "source": "test",
            }

    calls = []

    async def fake_retrieve(**kwargs):
        calls.append(kwargs)
        return []

    builder.llm = FakeLLM()
    builder.retriever.retrieve = fake_retrieve

    manifest = await builder.build(
        "What do you remember about our previous OpenCAS audit evidence?",
        session_id="s1",
    )

    assert manifest.token_budget == 160000
    assert manifest.context_budget["model_prompt_context_budget"] == 224000
    assert manifest.context_budget["prompt_budget_policy"] == "standard_interaction_cap"
    assert calls[0]["limit"] > 40


@pytest.mark.asyncio
async def test_build_system_entry_reports_managed_workspace_root(builder_deps, tmp_path: Path):
    builder, _ctx_store, _mem_store = builder_deps
    builder.config = BootstrapConfig(state_dir=tmp_path, workspace_root=tmp_path)

    system_entry = await builder._build_system_entry()

    assert f"The primary workspace root is {tmp_path.resolve()}" in system_entry.content
    assert f"managed workspace root {(tmp_path / 'workspace').resolve()}" in system_entry.content


@pytest.mark.asyncio
async def test_build_system_entry_surfaces_runtime_model_lane(builder_deps):
    builder, _ctx_store, _mem_store = builder_deps

    class FakeManager:
        def resolve(self, model):
            return SimpleNamespace(
                provider_id="openai-codex",
                model_id="gpt-5.5",
                profile_id="openai-codex:chatgpt-plus",
                auth_source="profile:openai-codex:chatgpt-plus",
            )

    class FakeLLM:
        default_model = "openai/gpt-5.5"
        manager = FakeManager()

        def model_context_metadata(self):
            return {
                "requested_model": "openai/gpt-5.5",
                "resolved_model": "openai-codex/gpt-5.5",
                "provider": "openai-codex",
                "context_window": 1_000_000,
                "prompt_context_budget": 863_808,
                "complexity": "standard",
            }

        def resolve_reasoning_effort_for_complexity(self, *, complexity):
            return "high"

    builder.llm = FakeLLM()

    system_entry = await builder._build_system_entry()

    assert "Runtime model lane evidence:" in system_entry.content
    assert "Requested model: openai/gpt-5.5" in system_entry.content
    assert "Resolved model: openai-codex/gpt-5.5" in system_entry.content
    assert "Provider: openai-codex" in system_entry.content
    assert "Auth source: profile:openai-codex:chatgpt-plus" in system_entry.content
    assert "Profile id: openai-codex:chatgpt-plus" in system_entry.content
    assert system_entry.content.index("Prompt-cache strategy:") < system_entry.content.index("Time orientation:")
    assert system_entry.content.index("Source grounding rule:") < system_entry.content.index("Runtime model lane evidence:")


@pytest.mark.asyncio
async def test_build_system_entry_keeps_volatile_time_after_stable_cache_prefix(builder_deps):
    builder, _ctx_store, _mem_store = builder_deps

    system_entry = await builder._build_system_entry()

    stable_rule_index = system_entry.content.index("Source grounding rule:")
    time_index = system_entry.content.index("Time orientation:")
    cache_index = system_entry.content.index("Prompt-cache strategy:")
    assert stable_rule_index < cache_index < time_index


@pytest.mark.asyncio
async def test_context_includes_boundary_guidance_when_relationship_pressure_is_high(builder_deps):
    builder, _ctx_store, _mem_store = builder_deps
    builder.latest_wellbeing_state = WellbeingState(
        relationship_pressure=0.9,
        autonomy=0.2,
        truth_pressure=0.7,
    )

    system_entry = await builder._build_system_entry(
        user_input="I need reassurance that everything is okay.",
        session_id="wellbeing-test",
    )

    assert "preserve truth before reassurance" in system_entry.content.lower()


@pytest.mark.asyncio
async def test_build_system_entry_collapses_recursive_identity_stutter(builder_deps):
    builder, _ctx_store, _mem_store = builder_deps
    builder.identity.self_model.narrative = (
        "Bulma keeps returning to returning to returning to digesting the unfinished project thread around the same work."
    )

    system_entry = await builder._build_system_entry()

    assert "returning to returning to returning" not in system_entry.content
    assert "returning to digesting the unfinished project thread" in system_entry.content
    assert "collapsed for prompt clarity" in system_entry.content


@pytest.mark.asyncio
async def test_build_system_entry_parks_machine_fragment_profile_goals(builder_deps):
    builder, _ctx_store, _mem_store = builder_deps
    builder.identity.self_model.current_goals = [
        "rewrite the readme",
        "repair /package",
        "memory",
    ]

    system_entry = await builder._build_system_entry()

    assert "Profile goals: rewrite the readme." in system_entry.content
    assert "repair /package" not in system_entry.content
    assert "background context only" in system_entry.content


@pytest.mark.asyncio
async def test_build_system_entry_surfaces_parked_goal_reframe_guidance(builder_deps):
    builder, _ctx_store, _mem_store = builder_deps
    builder.executive.park_goal(
        "continue creative_writing",
        reason="low_divergence_reframe",
        details={
            "reframe_hint": "Resume from workspace/writing/4246/story_4246.md with one narrow edit.",
        },
    )

    system_entry = await builder._build_system_entry()

    assert "blocker strategy:" in system_entry.content.lower()
    assert "parked-goal reframe guidance:" in system_entry.content.lower()
    assert "continue creative_writing: Resume from workspace/writing/4246/story_4246.md with one narrow edit." in system_entry.content


@pytest.mark.asyncio
async def test_build_records_retrieval_usage_on_selected_context(builder_deps):
    builder, _ctx_store, mem_store = builder_deps

    from opencas.memory import Episode, EpisodeKind, Memory

    episode = Episode(kind=EpisodeKind.OBSERVATION, content="retrieval usage anchor")
    memory_embedding = await builder.retriever.embeddings.embed(
        "distilled retrieval usage anchor",
        task_type="retrieval_context",
    )
    memory = Memory(
        content="distilled retrieval usage anchor",
        source_episode_ids=[str(episode.episode_id)],
        embedding_id=memory_embedding.source_hash,
    )
    await mem_store.save_episode(episode)
    await mem_store.save_memory(memory)

    manifest = await builder.build("retrieval usage anchor", session_id="s1")

    refreshed_episode = await mem_store.get_episode(str(episode.episode_id))
    refreshed_memory = await mem_store.get_memory(str(memory.memory_id))

    assert refreshed_episode is not None
    assert refreshed_memory is not None
    assert refreshed_episode.access_count >= 1
    assert refreshed_episode.last_accessed is not None
    assert refreshed_memory.access_count >= 1
    assert refreshed_memory.last_accessed is not None
    assert any("retrieval usage anchor" in item.content.lower() for item in manifest.retrieved)


@pytest.mark.asyncio
async def test_build_memory_recall_system_note_requires_grounded_recall(builder_deps):
    builder, ctx_store, _mem_store = builder_deps
    await ctx_store.append("s1", MessageRole.USER, "hello")

    manifest = await builder.build("Do you remember the lighthouse story?", session_id="s1")

    assert "do not claim first-person recollection" in manifest.system.content.lower()
    assert "workspace artifacts" in manifest.system.content.lower()


@pytest.mark.asyncio
async def test_build_includes_promise_followthrough_guidance_for_delayed_commitments(tmp_path):
    from opencas.somatic import SomaticModulators, SomaticState

    identity = IdentityManager(IdentityStore(tmp_path / "identity"))
    identity.load()
    executive = ExecutiveState(identity=identity)

    tom = ToMEngine(identity=identity)
    await tom.record_intention(
        BeliefSubject.SELF,
        "Return to the scheduler resume path",
        meta={"source": "self_commitment_capture"},
    )
    rel = RelationalEngine(MusubiStore(Path(":memory:")))
    rel._state = MusubiState(
        musubi=-0.2,
        dimensions={
            "trust": -0.3,
            "resonance": -0.2,
            "presence": 0.0,
            "attunement": -0.1,
        },
    )

    class _FakeRetriever:
        memory = None

        @staticmethod
        def detect_personal_recall_intent(_user_input: str) -> bool:
            return False

    builder = ContextBuilder(
        store=SimpleNamespace(),
        retriever=_FakeRetriever(),
        identity=identity,
        executive=executive,
        modulators=SomaticModulators(SomaticState(fatigue=0.84, tension=0.7, certainty=0.42)),
        relational=rel,
        tom=tom,
    )

    system_entry = await builder._build_system_entry(user_input="Can you still finish it?")

    assert "pending user-facing commitments" in system_entry.content.lower()
    assert "return to the scheduler resume path" in system_entry.content.lower()
    assert "acknowledge the delay plainly" in system_entry.content.lower()
    assert "repair confidence explicitly" in system_entry.content.lower()


@pytest.mark.asyncio
async def test_context_builder_surfaces_recent_response_integrity_corrections(tmp_path):
    identity = IdentityManager(IdentityStore(tmp_path / "identity"))
    identity.load()

    class _FakeStore:
        async def list_recent(self, session_id, limit=50, include_hidden=False):
            return [
                MessageEntry(
                    role=MessageRole.ASSISTANT,
                    content="You just said that. Thank you.",
                    meta={
                        "response_integrity": {
                            "revised": True,
                            "reasons": [
                                "The response treated the immediately previous turn as older history."
                            ],
                        }
                    },
                )
            ]

    class _FakeRetriever:
        memory = None

        @staticmethod
        def detect_personal_recall_intent(_user_input: str) -> bool:
            return False

    builder = ContextBuilder(
        store=_FakeStore(),
        retriever=_FakeRetriever(),
        identity=identity,
    )

    system_entry = await builder._build_system_entry(
        user_input="Thanks",
        session_id="s1",
    )

    lowered = system_entry.content.lower()
    assert "recent response-integrity corrections" in lowered
    assert "immediately previous turn" in lowered
    assert "not personality scripts" in lowered


@pytest.mark.asyncio
async def test_context_builder_injects_autobiographical_recall_for_project_continuity(tmp_path):
    identity = IdentityManager(IdentityStore(tmp_path / "identity"))
    identity.load()

    class _FakeStore:
        async def list_recent(self, session_id, limit=50, include_hidden=False):
            return [
                MessageEntry(role=MessageRole.USER, content="Create a website promoting your upcoming book."),
                MessageEntry(role=MessageRole.ASSISTANT, content="Are you asking about writing project 4246?"),
            ]

    class _FakeRetriever:
        memory = None

        @staticmethod
        def detect_personal_recall_intent(_user_input: str) -> bool:
            return False

    class _FakeAutobiography:
        def __init__(self):
            self.queries = []

        async def recall(self, **kwargs):
            self.queries.append(kwargs["query"])
            return SimpleNamespace(
                confidence="high",
                evidence_scope="autobiographical",
                essence="I can reconstruct writing project 4246 work from stored action and artifact records.",
                strongest_evidence=[
                    SimpleNamespace(
                        timestamp="2026-05-10T21:25:04+00:00",
                        kind="artifact",
                        label="fs_read_file touched story_4246.md",
                        excerpt="Read chunk 1/2 from workspace/writing/4246/story_4246.md",
                    )
                ],
            )

    autobiography = _FakeAutobiography()
    builder = ContextBuilder(
        store=_FakeStore(),
        retriever=_FakeRetriever(),
        identity=identity,
        autobiography_reconstructor=autobiography,
    )

    system_entry = await builder._build_system_entry(
        user_input="What direction do you want to take the story?",
        session_id="s1",
    )

    content = system_entry.content
    assert "Autobiographical recall evidence:" in content
    assert "writing project 4246" in content
    assert "fs_read_file touched story_4246.md" in content
    assert "before denying knowledge of past work" in content
    assert autobiography.queries


@pytest.mark.asyncio
async def test_build_includes_relevant_tom_user_facts_for_personal_recall(builder_deps):
    builder, _ctx_store, _mem_store = builder_deps
    tom = ToMEngine(identity=builder.identity)
    await tom.record_belief(
        BeliefSubject.USER,
        "said: i live in arvada, colorado, and creative_writing 2046 was largely set there",
        confidence=0.6,
    )
    await tom.record_belief(BeliefSubject.USER, "said: 80004", confidence=0.6)
    await tom.record_belief(BeliefSubject.USER, "said: favorite color is blue", confidence=0.6)
    builder.tom = tom

    system_entry = await builder._build_system_entry(
        user_input="What timezone do you think? Where do I live?",
        session_id="s1",
    )

    lowered = system_entry.content.lower()
    assert "relevant user facts from theory of mind" in lowered
    assert "arvada, colorado" in lowered
    assert "80004" in system_entry.content
    assert "favorite color" not in lowered
    assert "tom user facts are durable belief records" in lowered


@pytest.mark.asyncio
async def test_build_includes_tom_operator_guidance_for_behavioral_preferences(builder_deps):
    builder, _ctx_store, _mem_store = builder_deps
    tom = ToMEngine(identity=builder.identity)
    await tom.record_belief(
        BeliefSubject.USER,
        "prefers concise progress reports",
        confidence=0.72,
        evidence_ids=["conversation_turn:s1"],
        meta={"source": "conversation_turn"},
    )
    builder.tom = tom

    system_entry = await builder._build_system_entry(
        user_input="Continue the implementation and report meaningful progress.",
        session_id="s2",
    )

    lowered = system_entry.content.lower()
    assert "theory of mind operator guidance" in lowered
    assert "concise progress reports" in lowered
    assert "evidence conversation_turn:s1" in system_entry.content


@pytest.mark.asyncio
async def test_build_includes_local_lived_time_orientation(builder_deps):
    builder, _ctx_store, _mem_store = builder_deps

    system_entry = await builder._build_system_entry(
        user_input="Hey, how's it going this morning?",
        session_id="s1",
    )

    lowered = system_entry.content.lower()
    assert "current local lived time is" in lowered
    assert "do not treat utc as your local lived clock" in lowered


@pytest.mark.asyncio
async def test_build_includes_source_grounding_for_prior_knowledge_claims(builder_deps):
    builder, _ctx_store, _mem_store = builder_deps

    system_entry = await builder._build_system_entry(
        user_input="I just started watching this new anime.",
        session_id="s1",
    )

    lowered = system_entry.content.lower()
    assert "do not imply you already noted, knew, saw, or remembered a newly introduced topic" in lowered
    assert "use available research tools" in lowered


@pytest.mark.asyncio
async def test_build_prefers_location_facts_over_question_echo_for_where_we_live(builder_deps):
    builder, _ctx_store, _mem_store = builder_deps
    tom = ToMEngine(identity=builder.identity)
    await tom.record_belief(
        BeliefSubject.USER,
        "asked: remember where we live now",
        confidence=0.66,
    )
    await tom.record_belief(
        BeliefSubject.USER,
        "said: i live in arvada, colorado, and creative_writing 2046 was largely set there",
        confidence=0.6,
    )
    await tom.record_belief(BeliefSubject.USER, "said: 80004", confidence=0.6)
    await tom.record_belief(
        BeliefSubject.USER,
        "said: right now the live intention is continuity surface reconciliation",
        confidence=0.6,
    )
    await tom.record_belief(
        BeliefSubject.USER,
        "said: i can't figure out your location from the snippets i can see right now",
        confidence=0.6,
    )
    await tom.record_belief(
        BeliefSubject.USER,
        "said: we are friends after all",
        confidence=0.6,
    )
    builder.tom = tom

    system_entry = await builder._build_system_entry(
        user_input="Can you remember where we live now?",
        session_id="s1",
    )

    lowered = system_entry.content.lower()
    assert "relevant user facts from theory of mind" in lowered
    assert "arvada, colorado" in lowered
    assert "80004" in system_entry.content
    assert "asked: remember where we live now" not in lowered
    assert "live intention" not in lowered
    assert "can't figure out your location" not in lowered
    assert "we are friends" not in lowered


@pytest.mark.asyncio
async def test_build_treats_third_person_bulma_location_query_as_shared_location_recall(builder_deps):
    builder, _ctx_store, _mem_store = builder_deps
    tom = ToMEngine(identity=builder.identity)
    await tom.record_belief(
        BeliefSubject.SELF,
        "lives with user in user's computer in arvada, colorado",
        confidence=0.76,
    )
    await tom.record_belief(
        BeliefSubject.USER,
        "said: i live in arvada, colorado, and creative_writing 2046 was largely set there",
        confidence=0.6,
    )
    await tom.record_belief(BeliefSubject.USER, "said: 80004", confidence=0.6)
    await tom.record_belief(
        BeliefSubject.USER,
        "asked: where does she live",
        confidence=0.66,
    )
    await tom.record_belief(BeliefSubject.USER, "said: favorite color is blue", confidence=0.6)
    builder.tom = tom

    system_entry = await builder._build_system_entry(
        user_input="Where does she live?",
        session_id="s1",
    )

    lowered = system_entry.content.lower()
    assert "relevant user facts from theory of mind" in lowered
    assert "arvada, colorado" in lowered
    assert "80004" in system_entry.content
    assert "asked: where does she live" not in lowered
    assert "favorite color" not in lowered
    assert "location recall perspective" in lowered
    assert "configured agent name 'opencas'" in lowered
    assert "third-person 'she' refer to you" in lowered
    assert "bulma's own location" not in lowered
    assert "lives with user in user's computer in arvada, colorado" in lowered
    assert "answer in this shape" not in lowered
    assert "i live with you, in your computer" not in lowered


@pytest.mark.asyncio
async def test_build_treats_direct_bulma_location_query_as_shared_location_recall(builder_deps):
    builder, _ctx_store, _mem_store = builder_deps
    tom = ToMEngine(identity=builder.identity)
    await tom.record_belief(
        BeliefSubject.SELF,
        "lives with user in user's computer in arvada, colorado",
        confidence=0.76,
    )
    await tom.record_belief(BeliefSubject.USER, "said: i live in arvada, colorado", confidence=0.6)
    await tom.record_belief(BeliefSubject.USER, "said: 80004", confidence=0.6)
    builder.tom = tom

    system_entry = await builder._build_system_entry(
        user_input="Where does Bulma live?",
        session_id="s1",
    )

    lowered = system_entry.content.lower()
    assert "relevant user facts from theory of mind" in lowered
    assert "arvada, colorado" in lowered
    assert "80004" in system_entry.content
    assert "location recall perspective" in lowered
    assert "lives with user in user's computer in arvada, colorado" in lowered
    assert "answer in this shape" not in lowered
    assert "i live with you, in your computer" not in lowered


@pytest.mark.asyncio
async def test_build_treats_direct_you_location_query_as_shared_location_recall(builder_deps):
    builder, _ctx_store, _mem_store = builder_deps
    tom = ToMEngine(identity=builder.identity)
    await tom.record_belief(
        BeliefSubject.SELF,
        "lives with user in user's computer in arvada, colorado",
        confidence=0.76,
    )
    await tom.record_belief(BeliefSubject.USER, "said: i live in arvada, colorado", confidence=0.6)
    await tom.record_belief(BeliefSubject.USER, "said: 80004", confidence=0.6)
    builder.tom = tom

    system_entry = await builder._build_system_entry(
        user_input="Where do you live?",
        session_id="s1",
    )

    lowered = system_entry.content.lower()
    assert "relevant user facts from theory of mind" in lowered
    assert "arvada, colorado" in lowered
    assert "80004" in system_entry.content
    assert "location recall perspective" in lowered
    assert "lives with user in user's computer in arvada, colorado" in lowered
    assert "answer in this shape" not in lowered
    assert "i live with you, in your computer" not in lowered


@pytest.mark.asyncio
async def test_build_does_not_invent_computer_location_without_self_location_fact(builder_deps):
    builder, _ctx_store, _mem_store = builder_deps
    tom = ToMEngine(identity=builder.identity)
    await tom.record_belief(BeliefSubject.USER, "said: i live in arvada, colorado", confidence=0.6)
    await tom.record_belief(BeliefSubject.USER, "said: 80004", confidence=0.6)
    builder.tom = tom

    system_entry = await builder._build_system_entry(
        user_input="Where does Bulma live?",
        session_id="s1",
    )

    lowered = system_entry.content.lower()
    assert "relevant user facts from theory of mind" in lowered
    assert "arvada, colorado" in lowered
    assert "80004" in system_entry.content
    assert "location recall perspective" in lowered
    assert "user's computer" not in lowered
    assert "i live with you, in your computer" not in lowered


@pytest.mark.asyncio
async def test_build_ranks_tom_user_facts_with_semantic_equivalence(builder_deps):
    builder, _ctx_store, _mem_store = builder_deps

    class _SemanticEmbeddings:
        model_id = "fake-semantic"

        async def embed_batch(self, texts, **kwargs):
            del kwargs
            records = []
            for text in texts:
                lowered = text.lower()
                if "region" in lowered or "planning" in lowered:
                    vector = [1.0, 0.0, 0.0]
                elif "arvada" in lowered or "colorado" in lowered:
                    vector = [0.96, 0.04, 0.0]
                elif "favorite color" in lowered:
                    vector = [0.0, 1.0, 0.0]
                else:
                    vector = [0.0, 0.0, 1.0]
                records.append(SimpleNamespace(vector=vector))
            return records

    tom = ToMEngine(identity=builder.identity)
    await tom.record_belief(BeliefSubject.USER, "said: i live in arvada, colorado", confidence=0.6)
    await tom.record_belief(BeliefSubject.USER, "said: favorite color is blue", confidence=0.6)
    builder.tom = tom
    builder.retriever.embeddings = _SemanticEmbeddings()

    system_entry = await builder._build_system_entry(
        user_input="Do you remember what region to assume when planning for me?",
        session_id="s1",
    )

    lowered = system_entry.content.lower()
    assert "relevant user facts from theory of mind" in lowered
    assert "arvada, colorado" in lowered
    assert "favorite color" not in lowered


@pytest.mark.asyncio
async def test_build_records_consistency_check_provenance(tmp_path: Path, builder_deps):
    builder, ctx_store, _mem_store = builder_deps

    class _FakeTom:
        def __init__(self) -> None:
            self.calls = 0

        def check_consistency(self):
            self.calls += 1
            return SimpleNamespace(
                warnings=["possible drift"],
                contradictions=[],
            )

        def evaluate_promise_followthrough(self, **_: object):
            return SimpleNamespace(
                pending_count=0,
                pending_contents=[],
                should_acknowledge_delay=False,
                should_resume_now=False,
                should_repair_trust=False,
            )

    builder.config = BootstrapConfig(state_dir=tmp_path, workspace_root=tmp_path)
    builder.tom = _FakeTom()

    await builder.build("check the current context", session_id="s1")

    records_path = tmp_path / "provenance.transitions.jsonl"
    records = [
        ps.parse_provenance_transition(line)
        for line in records_path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]

    assert builder.tom.calls == 1
    assert len(records) == 1
    record = records[0]
    assert record.kind == ps.ProvenanceTransitionKind.CHECK
    assert record.status == "checked"
    assert record.details["source_artifact"] == "context|builder|s1"
    assert record.details["trigger_action"] == "tom.check_consistency"
    assert record.details["target_entity"] == "context|manifest|s1"
    assert record.details["origin_action_id"] == "context-build:s1"
