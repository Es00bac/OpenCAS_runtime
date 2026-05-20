"""Tests for MemoryRetriever hybrid retrieval."""

import pytest
import pytest_asyncio

from opencas.context import MemoryRetriever
from opencas.context.models import RetrievalResult
from opencas.embeddings import EmbeddingCache, EmbeddingService
from opencas.memory import Memory, MemoryStore


@pytest_asyncio.fixture
async def stores(tmp_path):
    mem_store = MemoryStore(tmp_path / "memory.db")
    await mem_store.connect()
    cache = EmbeddingCache(tmp_path / "embeddings.db")
    await cache.connect()
    embed_service = EmbeddingService(
        cache=cache,
        model_id="local-fallback",
    )
    retriever = MemoryRetriever(
        memory=mem_store,
        embeddings=embed_service,
    )
    yield mem_store, embed_service, retriever
    await mem_store.close()
    await cache.close()


@pytest.mark.asyncio
async def test_retrieve_empty(stores):
    _mem_store, _embed_service, retriever = stores
    results = await retriever.retrieve("nonexistent query")
    assert results == []


@pytest.mark.asyncio
async def test_retrieve_keyword_search(stores):
    mem_store, _embed_service, retriever = stores
    # Save an episode that matches a keyword query
    from opencas.memory import Episode, EpisodeKind
    ep = Episode(kind=EpisodeKind.OBSERVATION, content="the quick brown fox")
    await mem_store.save_episode(ep)
    results = await retriever.retrieve("fox", limit=5)
    assert len(results) >= 1
    assert any(r.source_type == "episode" and "fox" in r.content for r in results)


@pytest.mark.asyncio
async def test_retrieve_emits_memory_activation_events(tmp_path):
    from opencas.memory import Episode, EpisodeKind

    class RecordingTracer:
        def __init__(self):
            self.activations = []

        def activate_memory_node(self, **payload):
            self.activations.append(payload)

    mem_store = MemoryStore(tmp_path / "memory.db")
    await mem_store.connect()
    cache = EmbeddingCache(tmp_path / "embeddings.db")
    await cache.connect()
    embed_service = EmbeddingService(cache=cache, model_id="local-fallback")
    tracer = RecordingTracer()
    retriever = MemoryRetriever(
        memory=mem_store,
        embeddings=embed_service,
        tracer=tracer,
    )
    ep = Episode(
        kind=EpisodeKind.ARTIFACT,
        content="atlas pulse memory",
        payload={"artifact": {"path": "workspace/atlas.md"}},
    )
    await mem_store.save_episode(ep)

    await retriever.retrieve("atlas pulse", limit=3)

    assert tracer.activations
    assert tracer.activations[0]["node_id"] == f"episode:{ep.episode_id}"
    assert tracer.activations[0]["activation_source"] == "retriever"
    assert tracer.activations[0]["query"] == "atlas pulse"
    assert tracer.activations[0]["extra"]["source_lane"] == "reflective"
    assert tracer.activations[0]["extra"]["context_material"] == "artifact"

    await mem_store.close()
    await cache.close()


@pytest.mark.asyncio
async def test_retrieve_semantic_search(stores):
    mem_store, embed_service, retriever = stores
    # Create a memory and embed it so the cache has a vector
    memory = Memory(content="planetary exploration")
    embed_record = await embed_service.embed("planetary exploration")
    memory.embedding_id = embed_record.source_hash
    await mem_store.save_memory(memory)

    results = await retriever.retrieve("planetary exploration", limit=5)
    assert any(r.source_type == "memory" and "planetary" in r.content for r in results)


@pytest.mark.asyncio
async def test_retrieve_rrf_fusion(stores):
    mem_store, embed_service, retriever = stores
    from opencas.memory import Episode, EpisodeKind

    # Keyword match
    ep = Episode(kind=EpisodeKind.OBSERVATION, content="solar system facts")
    await mem_store.save_episode(ep)

    # Semantic match
    memory = Memory(content="solar system facts")
    embed_record = await embed_service.embed("solar system facts")
    memory.embedding_id = embed_record.source_hash
    await mem_store.save_memory(memory)

    results = await retriever.retrieve("solar system", limit=5)
    # Both keyword and semantic may return the same content conceptually,
    # but deduplication keeps unique source_type+source_id pairs.
    assert len(results) >= 1


@pytest.mark.asyncio
async def test_retrieve_seeds_exact_path_handles_from_episode_payload(stores):
    mem_store, _embed_service, retriever = stores
    from opencas.memory import Episode, EpisodeKind

    artifact_path = "/mnt/xtra/OpenCAS/workspace/writing/4246/story_4246.md"
    ep = Episode(
        kind=EpisodeKind.ACTION,
        content="tool fs_write_file completed successfully",
        payload={
            "tool_name": "fs_write_file",
            "args": {"path": artifact_path},
            "result_metadata": {"checksum": "abc123"},
        },
    )
    await mem_store.save_episode(ep)

    results = await retriever.retrieve(f"Did you make {artifact_path}?", limit=5)

    assert results
    assert results[0].source_type == "episode"
    assert results[0].source_id == str(ep.episode_id)
    assert artifact_path in results[0].content
    assert "exact_handle_match" in results[0].content


@pytest.mark.asyncio
async def test_retrieve_rejects_prose_hex_as_exact_handle_without_checksum_cue(stores):
    mem_store, _embed_service, retriever = stores
    from opencas.memory import Episode, EpisodeKind

    ep = Episode(
        kind=EpisodeKind.OBSERVATION,
        content="The cafebabecafebabe phrase was a metaphor in the operator's prose.",
    )
    await mem_store.save_episode(ep)

    inspection = await retriever.inspect(
        "What did cafebabecafebabe mean in that metaphor?",
        limit=5,
    )

    assert inspection["meta"]["exact_handle_seed_count"] == 0
    assert not any(
        "exact_handle_match" in candidate["content"]
        for candidate in inspection["candidates"]
    )


@pytest.mark.asyncio
async def test_retrieve_rejects_url_as_exact_filesystem_handle(stores):
    mem_store, _embed_service, retriever = stores
    from opencas.memory import Episode, EpisodeKind

    ep = Episode(
        kind=EpisodeKind.OBSERVATION,
        content="The reference URL was https://example.com/writing/4246.",
    )
    await mem_store.save_episode(ep)

    inspection = await retriever.inspect(
        "What did https://example.com/writing/4246 refer to?",
        limit=5,
    )

    assert inspection["meta"]["exact_handle_seed_count"] == 0
    assert not any(
        "exact_handle_match" in candidate["content"]
        for candidate in inspection["candidates"]
    )


@pytest.mark.asyncio
async def test_retrieve_does_not_let_partial_path_phrase_displace_semantic_match(tmp_path):
    mem_store = MemoryStore(tmp_path / "memory.db")
    await mem_store.connect()
    cache = EmbeddingCache(tmp_path / "embeddings.db")
    await cache.connect()

    async def embed_fn(text: str):
        if "writing/4246" in text and "story project" in text:
            return [1.0, 0.0, 0.0]
        return [0.0, 1.0, 0.0]

    embed_service = EmbeddingService(
        cache=cache,
        embed_fn=embed_fn,
        model_id="test-semantic",
    )
    retriever = MemoryRetriever(memory=mem_store, embeddings=embed_service)
    from opencas.memory import Episode, EpisodeKind

    semantic_memory = Memory(
        content=(
            "writing/4246 story project origin context: the operator wanted "
            "continuity around the Writing Project draft and its authorship evidence."
        ),
        salience=10.0,
    )
    semantic_record = await embed_service.embed(semantic_memory.content)
    semantic_memory.embedding_id = semantic_record.source_hash
    await mem_store.save_memory(semantic_memory)

    artifact_path = "/mnt/xtra/OpenCAS/workspace/writing/4246/story_4246.md"
    incidental_episode = Episode(
        kind=EpisodeKind.ACTION,
        content="tool fs_write_file completed successfully",
        payload={
            "tool_name": "fs_write_file",
            "args": {"path": artifact_path},
        },
        salience=10.0,
        confidence_score=1.0,
    )
    await mem_store.save_episode(incidental_episode)

    inspection = await retriever.inspect(
        "Tell me about writing/4246 as a story project.",
        limit=3,
        expand_graph=False,
    )

    assert inspection["meta"]["exact_handle_seed_count"] == 0
    assert inspection["results"]
    assert inspection["results"][0].source_type == "memory"
    assert inspection["results"][0].source_id == str(semantic_memory.memory_id)
    assert "story project origin context" in inspection["results"][0].content

    await mem_store.close()
    await cache.close()


def test_apply_temporal_decay_identity_core():
    from opencas.context.retriever import MemoryRetriever

    score = 1.0
    age_days = 180.0
    decayed = MemoryRetriever.apply_temporal_decay(score, age_days, half_life_days=180.0)
    assert decayed == pytest.approx(0.5, abs=0.01)


def test_apply_temporal_decay_non_identity_faster():
    from opencas.context.retriever import MemoryRetriever
    score = 1.0
    age_days = 60.0
    decayed = MemoryRetriever.apply_temporal_decay(score, age_days, half_life_days=30.0)
    # After 60 days with 30-day half-life, score should be 0.25
    assert decayed == pytest.approx(0.25, abs=0.01)


@pytest.mark.asyncio
async def test_mmr_rerank_promotes_diversity(stores):
    _mem_store, embed_service, retriever = stores
    # Create three records: two very similar, one distinct
    from opencas.memory import Memory
    m1 = Memory(content="planetary exploration missions to mars")
    r1 = await embed_service.embed(m1.content)
    m1.embedding_id = r1.source_hash
    await _mem_store.save_memory(m1)

    m2 = Memory(content="planetary exploration missions to jupiter")
    r2 = await embed_service.embed(m2.content)
    m2.embedding_id = r2.source_hash
    await _mem_store.save_memory(m2)

    m3 = Memory(content="completely unrelated topic about baking bread")
    r3 = await embed_service.embed(m3.content)
    m3.embedding_id = r3.source_hash
    await _mem_store.save_memory(m3)

    # Give m1 and m2 identical high scores, m3 a slightly lower score
    results = [
        RetrievalResult(source_type="memory", source_id=str(m1.memory_id), content=m1.content, score=1.0, memory=m1, embedding=r1.vector),
        RetrievalResult(source_type="memory", source_id=str(m2.memory_id), content=m2.content, score=1.0, memory=m2, embedding=r2.vector),
        RetrievalResult(source_type="memory", source_id=str(m3.memory_id), content=m3.content, score=0.9, memory=m3, embedding=r3.vector),
    ]

    reranked = await retriever._mmr_rerank(results, lambda_param=0.5, limit=2)
    # MMR should pick the highest relevance first, then the most diverse
    assert len(reranked) == 2
    ids = {r.source_id for r in reranked}
    # The two planetary ones are very similar; MMR should drop one in favor of the distinct bread topic
    assert str(m3.memory_id) in ids
