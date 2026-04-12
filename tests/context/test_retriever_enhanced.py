"""Integration tests for enhanced MemoryRetriever with resonance, somatic, and relational signals."""

from datetime import datetime, timedelta, timezone
from pathlib import Path

import pytest
import pytest_asyncio

from opencas.context.retriever import MemoryRetriever
from opencas.embeddings import EmbeddingCache, EmbeddingService
from opencas.memory import Episode, EpisodeEdge, EpisodeKind, Memory, MemoryStore
from opencas.memory.fabric.graph import EpisodeGraph
from opencas.relational import RelationalEngine
from opencas.relational.store import MusubiStore
from opencas.somatic import SomaticManager
from opencas.somatic.models import AffectState, PrimaryEmotion, SomaticState


@pytest_asyncio.fixture
async def base_stores(tmp_path: Path):
    mem_store = MemoryStore(tmp_path / "memory.db")
    await mem_store.connect()
    cache = EmbeddingCache(tmp_path / "embeddings.db")
    await cache.connect()
    embed_service = EmbeddingService(cache=cache, model_id="local-fallback")
    yield mem_store, embed_service
    await mem_store.close()
    await cache.close()


@pytest_asyncio.fixture
async def retriever(base_stores):
    mem_store, embed_service = base_stores
    return MemoryRetriever(memory=mem_store, embeddings=embed_service)


@pytest_asyncio.fixture
async def retriever_with_graph(base_stores):
    mem_store, embed_service = base_stores
    graph = EpisodeGraph(store=mem_store)
    return MemoryRetriever(
        memory=mem_store,
        embeddings=embed_service,
        episode_graph=graph,
    )


@pytest_asyncio.fixture
async def retriever_with_somatic(base_stores, tmp_path: Path):
    mem_store, embed_service = base_stores
    state_path = tmp_path / "somatic.json"
    somatic = SomaticManager(state_path=state_path)
    somatic._state = SomaticState(arousal=0.8, tension=0.6, valence=0.5, focus=0.8)
    return MemoryRetriever(
        memory=mem_store,
        embeddings=embed_service,
        somatic_manager=somatic,
    )


@pytest_asyncio.fixture
async def retriever_with_relational(base_stores, tmp_path: Path):
    mem_store, embed_service = base_stores
    musubi_store = MusubiStore(tmp_path / "musubi.db")
    await musubi_store.connect()
    relational = RelationalEngine(store=musubi_store)
    await relational.connect()
    relational.state.musubi = 0.5
    yield MemoryRetriever(
        memory=mem_store,
        embeddings=embed_service,
        relational_engine=relational,
    )
    await relational.close()


@pytest.mark.asyncio
async def test_retrieve_usage_feedback_boosts_reliability(base_stores, retriever: MemoryRetriever) -> None:
    mem_store, embed_service = base_stores
    ep_good = Episode(
        kind=EpisodeKind.OBSERVATION,
        content="reliable episode about python testing",
        used_successfully=10,
        used_unsuccessfully=0,
    )
    ep_bad = Episode(
        kind=EpisodeKind.OBSERVATION,
        content="unreliable episode about python testing",
        used_successfully=0,
        used_unsuccessfully=10,
    )
    await mem_store.save_episode(ep_good)
    await mem_store.save_episode(ep_bad)

    results = await retriever.retrieve("python testing", limit=5, min_confidence=0.1)
    good_result = next((r for r in results if r.source_id == str(ep_good.episode_id)), None)
    bad_result = next((r for r in results if r.source_id == str(ep_bad.episode_id)), None)

    assert good_result is not None
    assert bad_result is not None
    assert good_result.score > bad_result.score


@pytest.mark.asyncio
async def test_retrieve_temporal_echo_same_weekday(base_stores, retriever: MemoryRetriever) -> None:
    mem_store, _ = base_stores
    now = datetime.now(timezone.utc)
    # Create an episode from exactly one week ago (same weekday)
    past = now - timedelta(days=7)

    ep = Episode(
        kind=EpisodeKind.OBSERVATION,
        content="weekly retrospective about project planning",
        created_at=past,
    )
    await mem_store.save_episode(ep)

    results = await retriever.retrieve("project planning", limit=5, min_confidence=0.1)
    match = next((r for r in results if r.source_id == str(ep.episode_id)), None)
    assert match is not None


@pytest.mark.asyncio
async def test_retrieve_graph_uses_full_edge_strength(base_stores, retriever_with_graph: MemoryRetriever) -> None:
    mem_store, _ = base_stores
    seed = Episode(kind=EpisodeKind.TURN, content="seed about machine learning")
    neighbor = Episode(kind=EpisodeKind.TURN, content="neighbor about neural networks")
    await mem_store.save_episode(seed)
    await mem_store.save_episode(neighbor)

    edge = EpisodeEdge(
        source_id=str(seed.episode_id),
        target_id=str(neighbor.episode_id),
        confidence=0.5,
        semantic_weight=1.0,
        emotional_weight=1.0,
        recency_weight=1.0,
        structural_weight=1.0,
    )
    await mem_store.save_edge(edge)

    results = await retriever_with_graph.retrieve("machine learning", limit=5, expand_graph=True, min_confidence=0.1)
    source_ids = {r.source_id for r in results}
    assert str(seed.episode_id) in source_ids
    assert str(neighbor.episode_id) in source_ids


@pytest.mark.asyncio
async def test_retrieve_somatic_adjustments_applied(base_stores, retriever_with_somatic: MemoryRetriever) -> None:
    mem_store, _ = base_stores
    ep = Episode(kind=EpisodeKind.OBSERVATION, content="aroused topic about creative writing")
    await mem_store.save_episode(ep)

    results = await retriever_with_somatic.retrieve("creative writing", limit=5, min_confidence=0.1)
    assert any(r.source_id == str(ep.episode_id) for r in results)


@pytest.mark.asyncio
async def test_retrieve_relational_boosts_collab_tag(base_stores, retriever_with_relational: MemoryRetriever) -> None:
    mem_store, embed_service = base_stores
    # Use a Memory (has tags field) for collab and an Episode (different family) for baseline
    # to avoid diversity-penalty deduplication while keeping keyword match parity.
    base_content = "memory about pair programming"

    mem_collab = Memory(content=base_content, tags=["collab"])
    embed_record = await embed_service.embed(base_content)
    mem_collab.embedding_id = embed_record.source_hash
    await mem_store.save_memory(mem_collab)

    ep_solo = Episode(kind=EpisodeKind.OBSERVATION, content=base_content)
    await mem_store.save_episode(ep_solo)

    results = await retriever_with_relational.retrieve("programming", limit=5, min_confidence=0.1)
    collab_result = next((r for r in results if r.source_id == str(mem_collab.memory_id)), None)
    solo_result = next((r for r in results if r.source_id == str(ep_solo.episode_id)), None)

    assert collab_result is not None
    assert solo_result is not None
    # High musubi should boost collab-tagged memory above the solo one
    assert collab_result.score > solo_result.score


@pytest.mark.asyncio
async def test_retrieve_emotional_resonance_ranking(base_stores, retriever: MemoryRetriever) -> None:
    mem_store, embed_service = base_stores
    ep_match = Episode(
        kind=EpisodeKind.OBSERVATION,
        content="joyful memory about a sunny day",
        affect=AffectState(primary_emotion=PrimaryEmotion.JOY, valence=0.8, arousal=0.7, intensity=0.9),
    )
    ep_mismatch = Episode(
        kind=EpisodeKind.OBSERVATION,
        content="sad memory about a rainy day",
        affect=AffectState(primary_emotion=PrimaryEmotion.SADNESS, valence=-0.8, arousal=0.2, intensity=0.9),
    )
    await mem_store.save_episode(ep_match)
    await mem_store.save_episode(ep_mismatch)

    query_affect = AffectState(primary_emotion=PrimaryEmotion.JOY, valence=0.8, arousal=0.7, intensity=0.9)
    results = await retriever.retrieve("day", limit=5, min_confidence=0.1, affect_query=query_affect)
    match_result = next((r for r in results if r.source_id == str(ep_match.episode_id)), None)
    mismatch_result = next((r for r in results if r.source_id == str(ep_mismatch.episode_id)), None)

    assert match_result is not None
    assert mismatch_result is not None
    assert match_result.score > mismatch_result.score
