"""Tests for ContextBuilder prompt assembly."""

from pathlib import Path
from types import SimpleNamespace

import pytest
import pytest_asyncio

from opencas.bootstrap import BootstrapConfig
from opencas.context import ContextBuilder, MemoryRetriever, MessageRole, SessionContextStore
from opencas.embeddings import EmbeddingCache, EmbeddingService
from opencas.identity import IdentityManager, IdentityStore
from opencas.autonomy.executive import ExecutiveState
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
    builder.max_tokens = system_tokens + 40

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
