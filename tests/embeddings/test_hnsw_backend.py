"""Tests for HnswVectorBackend."""

import pytest

from opencas.embeddings.hnsw_backend import HnswVectorBackend
from opencas.embeddings.models import EmbeddingRecord


@pytest.fixture
def backend():
    b = HnswVectorBackend()
    b.connect()
    yield b
    b.clear()


@pytest.mark.asyncio
async def test_upsert_and_search(backend):
    record = EmbeddingRecord(
        source_hash="hash-1",
        model_id="test-model",
        dimension=3,
        vector=[1.0, 0.0, 0.0],
    )
    assert await backend.upsert(record) is True
    results = await backend.search([1.0, 0.0, 0.0], limit=1)
    assert results == ["hash-1"]


@pytest.mark.asyncio
async def test_search_returns_empty_when_no_match(backend):
    results = await backend.search([1.0, 0.0, 0.0], limit=1)
    assert results == []


@pytest.mark.asyncio
async def test_model_id_post_filtering(backend):
    r1 = EmbeddingRecord(
        source_hash="hash-a",
        model_id="model-a",
        dimension=2,
        vector=[1.0, 0.0],
    )
    r2 = EmbeddingRecord(
        source_hash="hash-b",
        model_id="model-b",
        dimension=2,
        vector=[1.0, 0.0],
    )
    await backend.upsert(r1)
    await backend.upsert(r2)
    results = await backend.search([1.0, 0.0], limit=1, model_id="model-a")
    assert results == ["hash-a"]


@pytest.mark.asyncio
async def test_project_id_post_filtering(backend):
    r1 = EmbeddingRecord(
        source_hash="hash-x",
        model_id="m",
        dimension=2,
        vector=[1.0, 0.0],
        meta={"project_id": "proj-1"},
    )
    r2 = EmbeddingRecord(
        source_hash="hash-y",
        model_id="m",
        dimension=2,
        vector=[1.0, 0.0],
        meta={"project_id": "proj-2"},
    )
    await backend.upsert(r1)
    await backend.upsert(r2)
    results = await backend.search([1.0, 0.0], limit=1, project_id="proj-2")
    assert results == ["hash-y"]


@pytest.mark.asyncio
async def test_search_with_scores(backend):
    record = EmbeddingRecord(
        source_hash="hash-score",
        model_id="test-model",
        dimension=3,
        vector=[1.0, 0.0, 0.0],
    )
    await backend.upsert(record)
    results = await backend.search([1.0, 0.0, 0.0], limit=1, with_scores=True)
    assert len(results) == 1
    source_hash, score = results[0]
    assert source_hash == "hash-score"
    assert score == pytest.approx(1.0)


@pytest.mark.asyncio
async def test_count_and_clear(backend):
    assert backend.count() == 0
    record = EmbeddingRecord(
        source_hash="hash-1",
        model_id="test",
        dimension=2,
        vector=[1.0, 0.0],
    )
    await backend.upsert(record)
    assert backend.count() == 1
    backend.clear()
    assert backend.count() == 0


@pytest.mark.asyncio
async def test_dimension_mismatch_returns_false(backend):
    r1 = EmbeddingRecord(
        source_hash="hash-1",
        model_id="test",
        dimension=2,
        vector=[1.0, 0.0],
    )
    await backend.upsert(r1)
    r2 = EmbeddingRecord(
        source_hash="hash-2",
        model_id="test",
        dimension=3,
        vector=[1.0, 0.0, 0.0],
    )
    assert await backend.upsert(r2) is False
