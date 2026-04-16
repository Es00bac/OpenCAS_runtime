"""Tests for embedding reconciler."""

from pathlib import Path

import pytest
import pytest_asyncio

from opencas.embeddings import EmbeddingCache, EmbeddingService
from opencas.legacy.embedding_reconciler import import_qdrant_collection


@pytest_asyncio.fixture
async def embedding_service(tmp_path: Path):
    cache = EmbeddingCache(tmp_path / "embeddings.db")
    await cache.connect()
    service = EmbeddingService(cache, model_id="local-fallback")
    yield service
    await cache.close()


@pytest.mark.asyncio
async def test_import_qdrant_collection_skips_when_missing(
    embedding_service: EmbeddingService,
    tmp_path: Path,
) -> None:
    count = await import_qdrant_collection(
        tmp_path / "no_qdrant",
        "episodes_embed_v1",
        "openbulma-v4/episodes_embed_v1",
        embedding_service.cache,
    )
    assert count == 0


@pytest.mark.asyncio
async def test_import_qdrant_collection_requires_client(
    embedding_service: EmbeddingService,
    tmp_path: Path,
) -> None:
    # Without a real Qdrant collection on disk, the importer gracefully
    # returns zero rather than crashing.
    count = await import_qdrant_collection(
        tmp_path,
        "episodes_embed_v1",
        "openbulma-v4/episodes_embed_v1",
        embedding_service.cache,
    )
    assert count == 0
