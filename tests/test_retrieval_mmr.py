import pytest

from opencas.context.models import RetrievalResult
from opencas.context.retrieval_mmr import rerank_mmr
from opencas.embeddings.models import EmbeddingRecord


class _Memory:
    def __init__(self, embedding_id: str) -> None:
        self.embedding_id = embedding_id


class _BatchCache:
    def __init__(self, records: dict[str, EmbeddingRecord]) -> None:
        self.records = records
        self.get_many_calls: list[list[str]] = []
        self.get_calls: list[str] = []

    async def get_many(self, identifiers: list[str]) -> dict[str, EmbeddingRecord]:
        self.get_many_calls.append(list(identifiers))
        return {identifier: self.records[identifier] for identifier in identifiers if identifier in self.records}

    async def get(self, identifier: str) -> EmbeddingRecord | None:
        self.get_calls.append(identifier)
        return self.records.get(identifier)


class _Embeddings:
    def __init__(self, cache: _BatchCache) -> None:
        self.cache = cache


@pytest.mark.asyncio
async def test_mmr_batches_missing_embedding_vector_lookups() -> None:
    records = {
        "embed-a": EmbeddingRecord(
            embedding_id="embed-a",
            source_hash="source-a",
            model_id="test-model",
            dimension=2,
            vector=[1.0, 0.0],
        ),
        "embed-b": EmbeddingRecord(
            embedding_id="embed-b",
            source_hash="source-b",
            model_id="test-model",
            dimension=2,
            vector=[0.0, 1.0],
        ),
    }
    cache = _BatchCache(records)
    results = [
        RetrievalResult(
            source_type="memory",
            source_id="a",
            content="alpha",
            score=0.9,
            memory=_Memory("embed-a"),
        ),
        RetrievalResult(
            source_type="memory",
            source_id="b",
            content="bravo",
            score=0.8,
            memory=_Memory("embed-b"),
        ),
    ]

    reranked = await rerank_mmr(_Embeddings(cache), results, limit=2)

    assert [item.source_id for item in reranked] == ["a", "b"]
    assert cache.get_many_calls == [["embed-a", "embed-b"]]
    assert cache.get_calls == []
