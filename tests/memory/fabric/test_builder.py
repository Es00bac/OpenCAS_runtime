"""Tests for FabricBuilder."""

from datetime import datetime, timezone, timedelta
from unittest.mock import AsyncMock

import pytest
import pytest_asyncio
from pathlib import Path

from opencas.embeddings import EmbeddingCache, EmbeddingService
from opencas.memory import EdgeKind, Episode, EpisodeEdge, EpisodeKind, MemoryStore
from opencas.memory.fabric.builder import FabricBuilder
from opencas.memory.fabric.indexer import Candidate, MemoryIndexer
from opencas.memory.fabric.weigher import ContextProfile, EdgeWeigher


@pytest_asyncio.fixture
async def builder_deps(tmp_path: Path):
    store = MemoryStore(tmp_path / "memory.db")
    await store.connect()
    cache = EmbeddingCache(":memory:")
    await cache.connect()
    embeddings = EmbeddingService(cache=cache, model_id="local-fallback")
    indexer = MemoryIndexer(embeddings=embeddings, top_k=5)
    weigher = EdgeWeigher(profile=ContextProfile.CONSOLIDATION)
    builder = FabricBuilder(store=store, indexer=indexer, weigher=weigher)
    yield store, builder
    await store.close()
    await cache.close()


@pytest.mark.asyncio
async def test_rebuild_creates_typed_edges(builder_deps) -> None:
    store, builder = builder_deps
    now = datetime.now(timezone.utc)
    ep1 = Episode(
        kind=EpisodeKind.TURN,
        content="rust basics",
        created_at=now - timedelta(minutes=5),
    )
    ep2 = Episode(
        kind=EpisodeKind.TURN,
        content="rust ownership",
        created_at=now,
    )
    for ep in [ep1, ep2]:
        rec = await builder.indexer.embeddings.embed(ep.content)
        ep.embedding_id = rec.source_hash
        await store.save_episode(ep)

    count = await builder.rebuild([ep1, ep2], min_confidence=0.01)
    assert count >= 1

    edges = await store.get_edges_for(str(ep1.episode_id))
    assert len(edges) >= 1
    assert edges[0].kind is not None


@pytest.mark.asyncio
async def test_rebuild_boosts_existing_edges(builder_deps) -> None:
    store, builder = builder_deps
    now = datetime.now(timezone.utc)
    ep1 = Episode(kind=EpisodeKind.TURN, content="a", created_at=now)
    ep2 = Episode(kind=EpisodeKind.TURN, content="b", created_at=now)
    for ep in [ep1, ep2]:
        rec = await builder.indexer.embeddings.embed(ep.content)
        ep.embedding_id = rec.source_hash
        await store.save_episode(ep)

    await store.save_edge(
        EpisodeEdge(
            source_id=str(ep1.episode_id),
            target_id=str(ep2.episode_id),
            kind=EdgeKind.SEMANTIC,
            confidence=0.9,
        )
    )

    count = await builder.rebuild([ep1, ep2], decay=0.95, existing_boost=0.03, min_confidence=0.01)
    assert count == 1

    edges = await store.get_edges_for(str(ep1.episode_id))
    # Decay then boost: 0.9 * 0.95 + 0.03 = 0.885
    assert edges[0].confidence == pytest.approx(0.885, abs=1e-3)


@pytest.mark.asyncio
async def test_rebuild_prunes_weak_edges(builder_deps) -> None:
    store, builder = builder_deps
    now = datetime.now(timezone.utc)
    ep1 = Episode(kind=EpisodeKind.TURN, content="x", created_at=now)
    ep2 = Episode(kind=EpisodeKind.TURN, content="y", created_at=now)
    for ep in [ep1, ep2]:
        rec = await builder.indexer.embeddings.embed(ep.content)
        ep.embedding_id = rec.source_hash
        await store.save_episode(ep)

    await store.save_edge(
        EpisodeEdge(
            source_id=str(ep1.episode_id),
            target_id=str(ep2.episode_id),
            kind=EdgeKind.SEMANTIC,
            confidence=0.02,
        )
    )

    count = await builder.rebuild([ep1, ep2], decay=0.95, min_confidence=0.01, prune_threshold=0.05)
    # The old weak edge is decayed to 0.019 and then pruned below 0.05
    edges = await store.get_edges_for(str(ep1.episode_id))
    assert len(edges) == 0


@pytest.mark.asyncio
async def test_rebuild_with_mocked_indexer(builder_deps) -> None:
    store, builder = builder_deps
    now = datetime.now(timezone.utc)
    ep1 = Episode(kind=EpisodeKind.TURN, content="mock a", created_at=now)
    ep2 = Episode(kind=EpisodeKind.TURN, content="mock b", created_at=now)
    ep2.embedding_id = "mock-embed-id"
    for ep in [ep1, ep2]:
        await store.save_episode(ep)

    builder.indexer.candidates = AsyncMock(return_value=[Candidate(episode_id="mock-embed-id", score=0.9)])

    count = await builder.rebuild([ep1, ep2], min_confidence=0.01)
    assert count >= 1
    builder.indexer.candidates.assert_awaited()
