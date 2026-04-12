"""Tests for memory edge confidence tracking (boost, decay, prune)."""

import pytest
import pytest_asyncio

from opencas.memory import EdgeKind, Episode, EpisodeEdge, EpisodeKind, MemoryStore


@pytest_asyncio.fixture
async def store(tmp_path):
    s = MemoryStore(tmp_path / "memory.db")
    await s.connect()
    yield s
    await s.close()


@pytest.mark.asyncio
async def test_boost_edge_confidence(store):
    ep1 = Episode(kind=EpisodeKind.TURN, content="a")
    ep2 = Episode(kind=EpisodeKind.TURN, content="b")
    await store.save_episode(ep1)
    await store.save_episode(ep2)

    edge = EpisodeEdge(source_id=str(ep1.episode_id), target_id=str(ep2.episode_id), confidence=0.5)
    await store.save_edge(edge)

    await store.boost_edge_confidence(str(ep1.episode_id), boost=0.1)
    edges = await store.get_edges_for(str(ep1.episode_id))
    assert len(edges) == 1
    assert edges[0].confidence == pytest.approx(0.6)


@pytest.mark.asyncio
async def test_decay_all_edges(store):
    ep1 = Episode(kind=EpisodeKind.TURN, content="a")
    ep2 = Episode(kind=EpisodeKind.TURN, content="b")
    await store.save_episode(ep1)
    await store.save_episode(ep2)

    edge = EpisodeEdge(source_id=str(ep1.episode_id), target_id=str(ep2.episode_id), confidence=0.5)
    await store.save_edge(edge)

    await store.decay_all_edges(decay=0.9)
    edges = await store.get_edges_for(str(ep1.episode_id))
    assert len(edges) == 1
    assert edges[0].confidence == pytest.approx(0.45)


@pytest.mark.asyncio
async def test_prune_weak_edges(store):
    ep1 = Episode(kind=EpisodeKind.TURN, content="a")
    ep2 = Episode(kind=EpisodeKind.TURN, content="b")
    await store.save_episode(ep1)
    await store.save_episode(ep2)

    edge = EpisodeEdge(source_id=str(ep1.episode_id), target_id=str(ep2.episode_id), confidence=0.02)
    await store.save_edge(edge)

    pruned = await store.prune_weak_edges(min_confidence=0.05)
    assert pruned == 1
    edges = await store.get_edges_for(str(ep1.episode_id))
    assert len(edges) == 0


@pytest.mark.asyncio
async def test_rebuild_edges_decays_and_boosts_existing(store, tmp_path):
    from datetime import datetime, timezone, timedelta
    from opencas.consolidation import NightlyConsolidationEngine
    from opencas.embeddings import EmbeddingCache, EmbeddingService
    from opencas.api import LLMClient
    from opencas.identity import IdentityManager, IdentityStore

    cache = EmbeddingCache(":memory:")
    await cache.connect()
    embeddings = EmbeddingService(cache=cache, model_id="local-fallback")
    id_store = IdentityStore(tmp_path / "identity")
    identity = IdentityManager(id_store)
    identity.load()
    llm = LLMClient(provider_manager=object())  # type: ignore
    engine = NightlyConsolidationEngine(memory=store, embeddings=embeddings, llm=llm, identity=identity)

    now = datetime.now(timezone.utc)
    ep1 = Episode(kind=EpisodeKind.TURN, content="rust basics", created_at=now - timedelta(hours=1))
    ep2 = Episode(kind=EpisodeKind.TURN, content="rust ownership", created_at=now)

    # Embed episodes so the indexer can find candidates via Qdrant/SQLite fallback
    for ep in [ep1, ep2]:
        rec = await embeddings.embed(ep.content)
        ep.embedding_id = rec.source_hash
        await store.save_episode(ep)

    # Pre-create an edge with high confidence
    await store.save_edge(EpisodeEdge(source_id=str(ep1.episode_id), target_id=str(ep2.episode_id), confidence=0.9))

    edges_before = await store.get_edges_for(str(ep1.episode_id))
    assert edges_before[0].confidence == pytest.approx(0.9)

    count = await engine.fabric_builder.rebuild([ep1, ep2], decay=0.95, existing_boost=0.03)
    assert count == 1

    edges_after = await store.get_edges_for(str(ep1.episode_id))
    # Decay then boost: 0.9 * 0.95 + 0.03 = 0.885
    assert edges_after[0].confidence == pytest.approx(0.885, abs=1e-3)

    await cache.close()
