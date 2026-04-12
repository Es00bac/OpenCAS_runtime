"""Tests for graph expansion in MemoryRetriever."""

import pytest
import pytest_asyncio
from pathlib import Path

from opencas.context.retriever import MemoryRetriever
from opencas.embeddings import EmbeddingCache, EmbeddingService
from opencas.memory import Episode, EpisodeEdge, EpisodeKind, MemoryStore
from opencas.memory.fabric.graph import EpisodeGraph


@pytest_asyncio.fixture
async def store(tmp_path: Path):
    s = MemoryStore(tmp_path / "memory.db")
    await s.connect()
    yield s
    await s.close()


@pytest_asyncio.fixture
async def retriever(store: MemoryStore):
    cache = EmbeddingCache(":memory:")
    await cache.connect()
    embeddings = EmbeddingService(cache=cache, model_id="local-fallback")
    return MemoryRetriever(memory=store, embeddings=embeddings)


@pytest_asyncio.fixture
async def retriever_with_graph(store: MemoryStore):
    cache = EmbeddingCache(":memory:")
    await cache.connect()
    embeddings = EmbeddingService(cache=cache, model_id="local-fallback")
    graph = EpisodeGraph(store=store)
    return MemoryRetriever(memory=store, embeddings=embeddings, episode_graph=graph)


@pytest.mark.asyncio
async def test_retriever_graph_expansion_brings_neighbor(store: MemoryStore, retriever: MemoryRetriever) -> None:
    seed = Episode(kind=EpisodeKind.TURN, content="seed episode about rust programming")
    neighbor = Episode(kind=EpisodeKind.TURN, content="neighbor episode about rust libraries")
    await store.save_episode(seed)
    await store.save_episode(neighbor)

    edge = EpisodeEdge(
        source_id=str(seed.episode_id),
        target_id=str(neighbor.episode_id),
        confidence=0.9,
    )
    await store.save_edge(edge)

    # Use a query that would match the seed via keyword search
    results = await retriever.retrieve("rust programming", limit=5, expand_graph=True, min_confidence=0.1)
    source_ids = {r.source_id for r in results}
    assert str(seed.episode_id) in source_ids
    assert str(neighbor.episode_id) in source_ids


@pytest.mark.asyncio
async def test_retriever_no_expansion_when_disabled(store: MemoryStore, retriever: MemoryRetriever) -> None:
    seed = Episode(kind=EpisodeKind.TURN, content="seed")
    neighbor = Episode(kind=EpisodeKind.TURN, content="neighbor")
    await store.save_episode(seed)
    await store.save_episode(neighbor)
    await store.save_edge(
        EpisodeEdge(source_id=str(seed.episode_id), target_id=str(neighbor.episode_id), confidence=0.9)
    )

    results = await retriever.retrieve("seed", limit=5, expand_graph=False)
    source_ids = {r.source_id for r in results}
    assert str(seed.episode_id) in source_ids
    assert str(neighbor.episode_id) not in source_ids


@pytest.mark.asyncio
async def test_retriever_graph_expansion_with_episode_graph(store: MemoryStore, retriever_with_graph: MemoryRetriever) -> None:
    seed = Episode(kind=EpisodeKind.TURN, content="seed episode about python async")
    neighbor = Episode(kind=EpisodeKind.TURN, content="neighbor episode about asyncio")
    await store.save_episode(seed)
    await store.save_episode(neighbor)

    await store.save_edge(
        EpisodeEdge(
            source_id=str(seed.episode_id),
            target_id=str(neighbor.episode_id),
            confidence=0.85,
        )
    )

    results = await retriever_with_graph.retrieve("python async", limit=5, expand_graph=True, min_confidence=0.1)
    source_ids = {r.source_id for r in results}
    assert str(seed.episode_id) in source_ids
    assert str(neighbor.episode_id) in source_ids
