"""Tests for the embeddings module."""

import pytest
import pytest_asyncio
from pathlib import Path
from opencas.embeddings import EmbeddingCache, EmbeddingService
from opencas.embeddings.backfill import EmbeddingBackfill
from opencas.memory import MemoryStore
from opencas.memory.models import Episode, EpisodeKind


@pytest_asyncio.fixture
async def embedding_cache(tmp_path: Path):
    cache = EmbeddingCache(tmp_path / "embeddings.db")
    await cache.connect()
    yield cache
    await cache.close()


@pytest_asyncio.fixture
async def memory_store(tmp_path: Path):
    store = MemoryStore(tmp_path / "memory.db")
    await store.connect()
    yield store
    await store.close()


@pytest.mark.asyncio
async def test_embed_and_cache(embedding_cache: EmbeddingCache) -> None:
    svc = EmbeddingService(embedding_cache)
    rec1 = await svc.embed("hello world")
    rec2 = await svc.embed("hello world")

    assert rec1.source_hash == rec2.source_hash
    assert rec1.embedding_id == rec2.embedding_id  # cache hit
    assert rec1.dimension == len(rec1.vector)
    assert rec1.dimension > 0

    health = await svc.health()
    assert health.total_records == 1
    assert health.cache_hit_rate_1h == 0.5


@pytest.mark.asyncio
async def test_different_texts(embedding_cache: EmbeddingCache) -> None:
    svc = EmbeddingService(embedding_cache)
    rec1 = await svc.embed("hello world")
    rec2 = await svc.embed("goodbye world")

    assert rec1.source_hash != rec2.source_hash
    health = await svc.health()
    assert health.total_records == 2


@pytest.mark.asyncio
async def test_fallback_vector_properties(embedding_cache: EmbeddingCache) -> None:
    svc = EmbeddingService(embedding_cache)
    rec = await svc.embed("test")
    vec = rec.vector
    norm = sum(x * x for x in vec) ** 0.5
    assert pytest.approx(norm, 0.01) == 1.0


@pytest.mark.asyncio
async def test_search_hit_path_tracked(embedding_cache: EmbeddingCache) -> None:
    svc = EmbeddingService(embedding_cache)
    await svc.embed("hello world")
    # Use a vector for an unrelated text that is NOT stored in the cache
    unrelated_vector = list(await svc._fallback_embed("xyz123abc_unrelated"))
    await svc.cache.search_similar(unrelated_vector, limit=1)

    health = await svc.health()
    assert health.semantic_success_count_1h == 0
    assert health.lexical_fallback_count_1h == 0
    assert health.avg_latency_ms_1h is not None
    assert health.avg_latency_ms_1h >= 0

    # Lexical fallback path when query_text is provided and no vector hits
    await svc.cache.search_similar(unrelated_vector, limit=1, query_text="hello world")
    health2 = await svc.health()
    assert health2.lexical_fallback_count_1h >= 1


@pytest.mark.asyncio
async def test_health_ready_ratio(embedding_cache: EmbeddingCache, memory_store) -> None:
    svc = EmbeddingService(embedding_cache, store=memory_store)
    # no episodes yet
    health = await svc.health()
    assert health.ready_ratio == 1.0

    ep = Episode(content="test", kind=EpisodeKind.OBSERVATION)
    await memory_store.save_episode(ep)

    health = await svc.health()
    assert health.ready_ratio == 0.0


@pytest.mark.asyncio
async def test_backfill_missing_embeddings(embedding_cache: EmbeddingCache, memory_store) -> None:
    svc = EmbeddingService(embedding_cache, store=memory_store)
    ep1 = Episode(content="hello world", kind=EpisodeKind.OBSERVATION)
    ep2 = Episode(content="foo bar", kind=EpisodeKind.OBSERVATION)
    await memory_store.save_episodes_batch([ep1, ep2])

    backfill = EmbeddingBackfill(svc, memory_store)
    count = await backfill.backfill_missing_embeddings([ep1, ep2])
    assert count == 2

    # Verify episodes updated
    fetched = await memory_store.get_episode(str(ep1.episode_id))
    assert fetched.embedding_id is not None
    cached = await embedding_cache.get(fetched.embedding_id)
    assert cached is not None
    assert fetched.embedding_id == cached.source_hash

    # Re-running should backfill nothing
    count2 = await backfill.backfill_missing_embeddings([ep1, ep2])
    assert count2 == 0


@pytest.mark.asyncio
async def test_cache_isolation_by_model_and_task_type(embedding_cache: EmbeddingCache) -> None:
    svc_a = EmbeddingService(embedding_cache, model_id="model-a")
    svc_b = EmbeddingService(embedding_cache, model_id="model-b")

    rec_a = await svc_a.embed("shared text", task_type="general")
    rec_b = await svc_b.embed("shared text", task_type="general")
    rec_c = await svc_a.embed("shared text", task_type="retrieval_query")

    assert rec_a.source_hash != rec_b.source_hash
    assert rec_a.source_hash != rec_c.source_hash
    assert rec_b.source_hash != rec_c.source_hash

    health = await svc_a.health()
    assert health.total_records == 3


@pytest.mark.asyncio
async def test_remote_embedding_failure_degrades_to_local_fallback(
    embedding_cache: EmbeddingCache,
) -> None:
    async def _boom(_text: str):
        raise RuntimeError("rate limited")

    svc = EmbeddingService(
        embedding_cache,
        embed_fn=_boom,
        model_id="remote-model",
    )
    record = await svc.embed("graceful degradation")

    assert record.dimension > 0
    assert record.meta["embedding_degraded"] is True
    assert "rate limited" in record.meta["embedding_degraded_reason"]


@pytest.mark.asyncio
async def test_search_similar_skips_dimension_mismatch_records(
    embedding_cache: EmbeddingCache,
) -> None:
    degraded = EmbeddingService(embedding_cache, model_id="mixed-model")
    remote = EmbeddingService(
        embedding_cache,
        model_id="mixed-model",
        embed_fn=lambda _text: _fixed_vector(3072),
    )

    await degraded.embed("fallback sized")
    query_record = await remote.embed("remote sized")

    results = await embedding_cache.search_similar(
        query_record.vector,
        limit=5,
        model_id="mixed-model",
    )

    assert results
    assert all(record.dimension == query_record.dimension for record, _ in results)


async def _fixed_vector(size: int) -> list[float]:
    return [1.0 / size] * size
