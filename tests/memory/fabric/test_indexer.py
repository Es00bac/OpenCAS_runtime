"""Tests for MemoryIndexer."""

import pytest
import pytest_asyncio
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock

from opencas.embeddings import EmbeddingCache, EmbeddingService
from opencas.memory import Episode, EpisodeKind
from opencas.memory.fabric.indexer import Candidate, MemoryIndexer


@pytest_asyncio.fixture
async def indexer(tmp_path: Path):
    cache = EmbeddingCache(":memory:")
    await cache.connect()
    embeddings = EmbeddingService(cache=cache, model_id="local-fallback")
    idx = MemoryIndexer(embeddings=embeddings, top_k=5)
    yield idx
    await cache.close()


@pytest.mark.asyncio
async def test_candidates_returns_empty_when_no_embedding_id(indexer: MemoryIndexer) -> None:
    ep = Episode(kind=EpisodeKind.TURN, content="no embedding")
    assert ep.embedding_id is None
    result = await indexer.candidates(ep)
    assert result == []


@pytest.mark.asyncio
async def test_candidates_falls_back_to_search_similar_when_qdrant_unavailable(indexer: MemoryIndexer) -> None:
    ep1 = Episode(kind=EpisodeKind.TURN, content="rust programming basics")
    ep2 = Episode(kind=EpisodeKind.TURN, content="rust ownership concepts")

    rec1 = await indexer.embeddings.embed(ep1.content)
    rec2 = await indexer.embeddings.embed(ep2.content)
    ep1.embedding_id = rec1.source_hash
    ep2.embedding_id = rec2.source_hash

    # Qdrant backend is None, so fallback path should be exercised
    result = await indexer.candidates(ep1)
    assert len(result) >= 1
    assert any(c.episode_id == rec2.source_hash for c in result)


@pytest.mark.asyncio
async def test_candidates_excludes_self(indexer: MemoryIndexer) -> None:
    ep = Episode(kind=EpisodeKind.TURN, content="unique content here")
    rec = await indexer.embeddings.embed(ep.content)
    ep.embedding_id = rec.source_hash

    result = await indexer.candidates(ep)
    assert not any(c.episode_id == rec.source_hash for c in result)


@pytest.mark.asyncio
async def test_candidates_uses_qdrant_when_available(indexer: MemoryIndexer) -> None:
    ep = Episode(kind=EpisodeKind.TURN, content="python async patterns")
    rec = await indexer.embeddings.embed(ep.content)
    ep.embedding_id = rec.source_hash

    mock_backend = MagicMock()
    mock_backend.search = AsyncMock(return_value=["other-hash-1", rec.source_hash, "other-hash-2"])
    indexer.embeddings.cache.vector_backend = mock_backend

    result = await indexer.candidates(ep)
    assert len(result) == 2
    ids = {c.episode_id for c in result}
    assert ids == {"other-hash-1", "other-hash-2"}
    assert all(c.score <= 1.0 for c in result)
    mock_backend.search.assert_awaited_once()
