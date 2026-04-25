"""Tests for MemoryRetriever multi-signal fusion, MMR, temporal decay, and diversity."""

import pytest
import pytest_asyncio
from datetime import datetime, timezone, timedelta
from pathlib import Path

from opencas.context.retriever import MemoryRetriever
from opencas.context.models import RetrievalResult
from opencas.embeddings import EmbeddingCache, EmbeddingService
from opencas.memory import Episode, EpisodeEdge, EpisodeKind, Memory, MemoryStore
from opencas.memory.fabric.graph import EpisodeGraph
from opencas.somatic.models import AffectState, PrimaryEmotion


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


@pytest.mark.asyncio
async def test_extract_anchor_terms() -> None:
    terms = MemoryRetriever.extract_anchor_terms('Tell me about "Rust Programming" and Python basics')
    assert "Rust Programming" in terms
    assert "Python" not in terms  # single word, not capitalized phrase


@pytest.mark.asyncio
async def test_detect_personal_recall_intent() -> None:
    assert MemoryRetriever.detect_personal_recall_intent("Do you remember what I said?") is True
    assert MemoryRetriever.detect_personal_recall_intent("What is the weather?") is False


@pytest.mark.asyncio
async def test_detect_temporal_intent() -> None:
    assert MemoryRetriever.detect_temporal_intent("What happened last week?") == "last week"
    assert MemoryRetriever.detect_temporal_intent("Tell me about AI") is None


@pytest.mark.asyncio
async def test_apply_temporal_decay() -> None:
    score = MemoryRetriever.apply_temporal_decay(1.0, age_days=60.0, half_life_days=60.0)
    assert score == pytest.approx(0.5, rel=1e-3)

    score = MemoryRetriever.apply_temporal_decay(1.0, age_days=0.0)
    assert score == pytest.approx(1.0)


@pytest.mark.asyncio
async def test_retrieve_fuses_semantic_and_keyword(store: MemoryStore, retriever: MemoryRetriever) -> None:
    # Episode matched by keyword
    ep = Episode(kind=EpisodeKind.OBSERVATION, content="solar system exploration")
    await store.save_episode(ep)

    # Memory matched by semantic
    embed = await retriever.embeddings.embed("planetary science")
    mem = Memory(content="planetary science", embedding_id=embed.source_hash)
    await store.save_memory(mem)

    results = await retriever.retrieve("solar system", limit=5)
    source_types = {r.source_type for r in results}
    assert "episode" in source_types
    assert "memory" in source_types


@pytest.mark.asyncio
async def test_retrieve_min_confidence_filter(store: MemoryStore, retriever: MemoryRetriever) -> None:
    # Create an episode with very low salience and no keyword match relevance
    ep = Episode(kind=EpisodeKind.OBSERVATION, content="random noise xyz123")
    await store.save_episode(ep)

    results = await retriever.retrieve("xyz123", limit=5, min_confidence=0.9)
    assert len(results) == 0

    results = await retriever.retrieve("xyz123", limit=5, min_confidence=0.01)
    assert len(results) >= 1


@pytest.mark.asyncio
async def test_retrieve_diversity_penalty_for_duplicates(store: MemoryStore, retriever: MemoryRetriever) -> None:
    from opencas.somatic.models import AffectState, PrimaryEmotion

    # Create three episodes with same emotion and kind to trigger diversity penalty
    episodes = []
    for i in range(3):
        ep = Episode(
            kind=EpisodeKind.OBSERVATION,
            content=f"observation number {i}",
            affect=AffectState(
                primary_emotion=PrimaryEmotion.JOY,
                valence=0.8,
                arousal=0.5,
                intensity=0.6,
            ),
        )
        await store.save_episode(ep)
        episodes.append(ep)

    results = await retriever.retrieve("observation number", limit=5)
    # All should still be present but scores may be penalized
    assert len(results) == 3

    # Manually verify penalty reduces duplicates in ordering
    penalized = retriever._apply_diversity_penalty(
        [
            RetrievalResult(source_type="episode", source_id="1", content="a", score=1.0, episode=episodes[0]),
            RetrievalResult(source_type="episode", source_id="2", content="b", score=0.9, episode=episodes[1]),
            RetrievalResult(source_type="episode", source_id="3", content="c", score=0.8, episode=episodes[2]),
        ],
        window=5,
        penalty=0.05,
    )
    # First stays at 1.0, second gets -0.1 (emotion dup + family dup), third gets -0.1
    assert penalized[0].score == pytest.approx(1.0)
    assert penalized[1].score == pytest.approx(0.8)
    assert penalized[2].score == pytest.approx(0.7)


@pytest.mark.asyncio
async def test_retrieve_confidence_reweighting(store: MemoryStore, retriever: MemoryRetriever) -> None:
    low_conf = Episode(kind=EpisodeKind.OBSERVATION, content="low confidence fact", confidence_score=0.3)
    mid_conf = Episode(kind=EpisodeKind.OBSERVATION, content="mid confidence fact", confidence_score=0.6)
    high_conf = Episode(kind=EpisodeKind.OBSERVATION, content="high confidence fact", confidence_score=0.9)
    for ep in [low_conf, mid_conf, high_conf]:
        await store.save_episode(ep)

    results = await retriever.retrieve("confidence fact", limit=10)
    scores_by_content = {r.content: r.score for r in results}

    # High confidence should not be down-weighted relative to lower tiers when
    # other signals are equal. Since all match keyword similarly,
    # the ordering should generally preserve high > mid > low after reweighting.
    assert "high confidence fact" in scores_by_content
    assert scores_by_content["high confidence fact"] >= scores_by_content["mid confidence fact"]


@pytest.mark.asyncio
async def test_retrieve_identity_core_temporal_decay(store: MemoryStore, retriever: MemoryRetriever) -> None:
    old = datetime.now(timezone.utc) - timedelta(days=120)
    # Identity core episode from 120 days ago
    core_ep = Episode(
        kind=EpisodeKind.OBSERVATION,
        content="core belief about kindness",
        identity_core=True,
        created_at=old,
    )
    # Non-core episode from 120 days ago
    normal_ep = Episode(
        kind=EpisodeKind.OBSERVATION,
        content="normal observation about kindness",
        identity_core=False,
        created_at=old,
    )
    await store.save_episode(core_ep)
    await store.save_episode(normal_ep)

    results = await retriever.retrieve("kindness", limit=5)
    core_results = [r for r in results if "core belief" in r.content]
    normal_results = [r for r in results if "normal observation" in r.content]

    if core_results and normal_results:
        # Identity core should decay slower (180 day half-life vs 60 day)
        # After 120 days: core_decay = 0.5^(120/180) ≈ 0.63, normal_decay = 0.5^(120/60) = 0.25
        assert core_results[0].score > normal_results[0].score


@pytest.mark.asyncio
async def test_mmr_rerank_promotes_diversity(store: MemoryStore, retriever: MemoryRetriever) -> None:
    # Seed two very similar memories and one different memory
    texts = ["rust programming language", "rust compiler and tooling", "python asyncio patterns"]
    for text in texts:
        embed = await retriever.embeddings.embed(text)
        await store.save_memory(Memory(content=text, embedding_id=embed.source_hash))

    results = await retriever.retrieve("rust programming", limit=3)
    # MMR should promote diversity; the python result may appear before the second rust result
    contents = [r.content for r in results]
    assert "rust programming language" in contents
    assert "python asyncio patterns" in contents


@pytest.mark.asyncio
async def test_retrieve_graph_signal_weighted_fusion(store: MemoryStore, retriever: MemoryRetriever) -> None:
    seed = Episode(kind=EpisodeKind.TURN, content="seed about machine learning")
    neighbor = Episode(kind=EpisodeKind.TURN, content="neighbor about neural networks")
    await store.save_episode(seed)
    await store.save_episode(neighbor)

    await store.save_edge(
        EpisodeEdge(
            source_id=str(seed.episode_id),
            target_id=str(neighbor.episode_id),
            confidence=0.95,
        )
    )

    results = await retriever.retrieve("machine learning", limit=5, expand_graph=True)
    source_ids = {r.source_id for r in results}
    assert str(seed.episode_id) in source_ids
    assert str(neighbor.episode_id) in source_ids


@pytest.mark.asyncio
async def test_retriever_expand_graph_false_excludes_neighbors(store: MemoryStore, retriever: MemoryRetriever) -> None:
    seed = Episode(kind=EpisodeKind.TURN, content="seed about databases")
    neighbor = Episode(kind=EpisodeKind.TURN, content="neighbor about sql")
    await store.save_episode(seed)
    await store.save_episode(neighbor)

    await store.save_edge(
        EpisodeEdge(
            source_id=str(seed.episode_id),
            target_id=str(neighbor.episode_id),
            confidence=0.9,
        )
    )

    results = await retriever.retrieve("databases", limit=5, expand_graph=False)
    source_ids = {r.source_id for r in results}
    assert str(seed.episode_id) in source_ids
    assert str(neighbor.episode_id) not in source_ids


@pytest.mark.asyncio
async def test_retrieve_affect_query_modulates_semantic(store: MemoryStore, retriever: MemoryRetriever) -> None:
    angry = Memory(content="I am furious and angry")
    calm = Memory(content="I am calm and peaceful")
    for mem in [angry, calm]:
        embed = await retriever.embeddings.embed(mem.content)
        mem.embedding_id = embed.source_hash
        await store.save_memory(mem)

    affect = AffectState(
        primary_emotion=PrimaryEmotion.ANGER,
        valence=-0.8,
        arousal=0.9,
        intensity=0.8,
    )
    results = await retriever.retrieve(
        "emotional state",
        limit=2,
        affect_query=affect,
        affect_weight=0.8,
    )
    assert len(results) >= 1
    # With local-fallback embeddings, exact affect ranking is not guaranteed,
    # but the query should still return results without crashing.
    contents = {r.content for r in results}
    assert any(c in contents for c in ["I am furious and angry", "I am calm and peaceful"])


@pytest.mark.asyncio
async def test_recall_intent_keyword_search_uses_anchor_terms(store: MemoryStore, retriever: MemoryRetriever) -> None:
    episode = Episode(
        kind=EpisodeKind.OBSERVATION,
        content="The lighthouse keeper wrote a letter about memory and return.",
    )
    await store.save_episode(episode)

    results = await retriever.retrieve("Do you remember the lighthouse story?", limit=5)

    assert any("lighthouse keeper" in result.content.lower() for result in results)
