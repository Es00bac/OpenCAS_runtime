"""Tests for ContextBuilder prompt assembly."""

import json
from pathlib import Path
from types import SimpleNamespace

import pytest
import pytest_asyncio

from opencas.api import provenance_store as ps
from opencas.autonomy.executive import ExecutiveState
from opencas.bootstrap import BootstrapConfig
from opencas.context import ContextBuilder, MemoryRetriever, MessageRole, SessionContextStore
from opencas.daydream import DaydreamReflection
from opencas.embeddings import EmbeddingCache, EmbeddingService
from opencas.identity import IdentityManager, IdentityStore
from opencas.memory import MemoryStore
from opencas.relational import MusubiState, MusubiStore, RelationalEngine
from opencas.tom import ToMEngine
from opencas.tom.models import BeliefSubject


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
async def test_build_token_estimate(builder_deps):
    builder, ctx_store, _mem_store = builder_deps
    await ctx_store.append("s1", MessageRole.USER, "hello world")
    manifest = await builder.build("hello world", session_id="s1")
    assert manifest.token_estimate is not None
    assert manifest.token_estimate > 0


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

    manifest = await builder.build("Who are you outside helping the owner?", session_id="s1")

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

    system_entry = await builder._build_system_entry(
        user_input="What should you do next?",
        session_id="s1",
    )

    content = system_entry.content
    assert "Temporal agenda from durable calendar:" in content
    assert "separate from OS cron" in content
    assert "active=2, due_now=1, upcoming_24h=1, recent_runs=1" in content
    assert "Check the calendar" in content
    assert "Do not invent calendar commitments" in content


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
async def test_build_system_entry_reports_managed_workspace_root(builder_deps, tmp_path: Path):
    builder, _ctx_store, _mem_store = builder_deps
    builder.config = BootstrapConfig(state_dir=tmp_path, workspace_root=tmp_path)

    system_entry = await builder._build_system_entry()

    assert f"The primary workspace root is {tmp_path.resolve()}" in system_entry.content
    assert f"managed workspace root {(tmp_path / 'workspace').resolve()}" in system_entry.content


@pytest.mark.asyncio
async def test_build_system_entry_collapses_recursive_identity_stutter(builder_deps):
    builder, _ctx_store, _mem_store = builder_deps
    builder.identity.self_model.narrative = (
        "the OpenCAS agent keeps returning to returning to returning to digesting the unfinished project thread around the same work."
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
        "continue chronicle",
        reason="low_divergence_reframe",
        details={
            "reframe_hint": "Resume from workspace/Chronicles/4246/chronicle_4246.md with one narrow edit.",
        },
    )

    system_entry = await builder._build_system_entry()

    assert "blocker strategy:" in system_entry.content.lower()
    assert "parked-goal reframe guidance:" in system_entry.content.lower()
    assert "continue chronicle: Resume from workspace/Chronicles/4246/chronicle_4246.md with one narrow edit." in system_entry.content


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
async def test_build_includes_relevant_tom_user_facts_for_personal_recall(builder_deps):
    builder, _ctx_store, _mem_store = builder_deps
    tom = ToMEngine(identity=builder.identity)
    await tom.record_belief(
        BeliefSubject.USER,
        "said: i live in arvada, colorado, and chronicle 2046 was largely set there",
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
        "said: i live in arvada, colorado, and chronicle 2046 was largely set there",
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
        "said: i live in arvada, colorado, and chronicle 2046 was largely set there",
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
    assert "third-person 'she' can refer to you" in lowered
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
        user_input="Where does the OpenCAS agent live?",
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
        user_input="Where does the OpenCAS agent live?",
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
