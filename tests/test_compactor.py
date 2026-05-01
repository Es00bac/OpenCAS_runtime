"""Tests for ConversationCompactor."""

from types import SimpleNamespace

import pytest
import pytest_asyncio
from unittest.mock import AsyncMock, MagicMock

from opencas.compaction import ConversationCompactor
from opencas.identity import IdentityManager, IdentityStore
from opencas.memory import Episode, EpisodeKind, MemoryStore


@pytest_asyncio.fixture
async def store(tmp_path):
    s = MemoryStore(tmp_path / "memory.db")
    await s.connect()
    yield s
    await s.close()


@pytest.fixture
def mock_llm():
    mgr = MagicMock()
    resolved = MagicMock()
    resolved.provider_id = "test-provider"
    resolved.model_id = "test-model"
    resolved.provider = MagicMock()
    resolved.provider.chat_completion = AsyncMock(
        return_value={"choices": [{"message": {"content": "Summary of old episodes"}}]}
    )
    mgr.resolve.return_value = resolved
    from opencas.api import LLMClient
    return LLMClient(mgr, default_model="test/model")


@pytest.mark.asyncio
async def test_compact_session_no_op_when_few_episodes(store, mock_llm):
    compactor = ConversationCompactor(memory=store, llm=mock_llm)
    ep = Episode(kind=EpisodeKind.TURN, session_id="s1", content="hello")
    await store.save_episode(ep)
    record = await compactor.compact_session("s1", tail_size=10)
    assert record is None


@pytest.mark.asyncio
async def test_compact_session_summarizes_and_marks_compacted(store, mock_llm, tmp_path):
    from opencas.context import SessionContextStore
    ctx_store = SessionContextStore(tmp_path / "context.db")
    await ctx_store.connect()
    compactor = ConversationCompactor(memory=store, llm=mock_llm, context_store=ctx_store)
    for i in range(12):
        ep = Episode(kind=EpisodeKind.TURN, session_id="s1", content=f"turn {i}")
        await store.save_episode(ep)

    record = await compactor.compact_session("s1", tail_size=10)
    assert record is not None
    assert record.removed_count == 2

    # Episodes should be marked compacted
    compacted = await store.list_episodes(session_id="s1", compacted=True)
    assert len(compacted) == 2

    # A memory should be created
    memories = await store.list_memories(limit=10)
    assert len(memories) == 1
    assert memories[0].content == "Summary of old episodes"
    assert "compaction" in memories[0].tags

    # A compaction record should be persisted
    # (record_compaction writes to compactions table; verification via raw query)
    cursor = await store._execute("SELECT COUNT(*) FROM compactions")
    row = await cursor.fetchone()
    assert row[0] == 1

    # A synthetic continuation message should be injected
    messages = await ctx_store.list_recent(session_id="s1", limit=5)
    assert any("compacted" in m.content and "Summary of old episodes" in m.content for m in messages)
    await ctx_store.close()


@pytest.mark.asyncio
async def test_compact_session_embeds_summary_memory_when_embeddings_available(store, mock_llm):
    embeddings = SimpleNamespace(
        embed=AsyncMock(return_value=SimpleNamespace(source_hash="summary-embedding-hash"))
    )
    compactor = ConversationCompactor(memory=store, llm=mock_llm, embeddings=embeddings)
    for i in range(12):
        ep = Episode(kind=EpisodeKind.TURN, session_id="s1", content=f"turn {i}")
        await store.save_episode(ep)

    record = await compactor.compact_session("s1", tail_size=10)

    assert record is not None
    memories = await store.list_memories(limit=10)
    assert memories[0].embedding_id == "summary-embedding-hash"
    embeddings.embed.assert_awaited_once_with(
        "Summary of old episodes",
        task_type="memory_compaction",
        meta={
            "source": "compaction",
            "session_id": "s1",
            "removed_count": 2,
        },
    )


@pytest.mark.asyncio
async def test_compact_session_records_identity_continuity(store, mock_llm, tmp_path):
    identity_store = IdentityStore(tmp_path / "identity")
    identity = IdentityManager(identity_store)
    identity.load()
    before = identity.continuity.compaction_count

    compactor = ConversationCompactor(memory=store, llm=mock_llm, identity=identity)
    for i in range(12):
        ep = Episode(kind=EpisodeKind.TURN, session_id="s1", content=f"turn {i}")
        await store.save_episode(ep)

    record = await compactor.compact_session("s1", tail_size=10)

    assert record is not None
    assert identity.continuity.compaction_count == before + 1
    assert identity.continuity.last_session_id == "s1"

    fresh_identity = IdentityManager(identity_store)
    fresh_identity.load()
    assert fresh_identity.continuity.compaction_count == before + 1
    assert fresh_identity.continuity.last_session_id == "s1"


@pytest.mark.asyncio
async def test_compact_session_skips_when_removed_count_is_below_minimum(store, mock_llm):
    compactor = ConversationCompactor(memory=store, llm=mock_llm)
    for i in range(12):
        ep = Episode(kind=EpisodeKind.TURN, session_id="s1", content=f"turn {i}")
        await store.save_episode(ep)

    record = await compactor.compact_session("s1", tail_size=10, min_removed_count=3)

    assert record is None
    memories = await store.list_memories(limit=10)
    assert memories == []
