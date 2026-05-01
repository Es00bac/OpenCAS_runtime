from __future__ import annotations

from types import SimpleNamespace
from uuid import UUID

import pytest

from opencas.embeddings.models import EmbeddingRecord
from opencas.embeddings.qdrant_backend import QdrantVectorBackend, qdrant_point_id


def test_qdrant_point_id_is_valid_deterministic_uuid_for_source_hash() -> None:
    source_hash = "a" * 64

    first = qdrant_point_id(source_hash)
    second = qdrant_point_id(source_hash)

    assert first == second
    assert str(UUID(first)) == first
    assert first != source_hash


@pytest.mark.asyncio
async def test_qdrant_upsert_uses_uuid_point_id_and_source_payload() -> None:
    captured: dict[str, object] = {}

    class FakeClient:
        async def upsert(self, *, collection_name, points):
            captured["collection_name"] = collection_name
            captured["points"] = points

    backend = QdrantVectorBackend(url="http://127.0.0.1:6333", collection="test")
    backend._client = FakeClient()
    backend._available = True
    record = EmbeddingRecord(
        source_hash="b" * 64,
        model_id="google/embeddinggemma-300m",
        dimension=3,
        vector=[1.0, 0.0, 0.0],
        meta={"project_id": "audit-test"},
    )

    assert await backend.upsert(record) is True

    points = captured["points"]
    assert isinstance(points, list)
    assert len(points) == 1
    point = points[0]
    assert str(UUID(str(point.id))) == qdrant_point_id(record.source_hash)
    assert point.payload["source_hash"] == record.source_hash
    assert point.payload["model_id"] == "google/embeddinggemma-300m"
    assert point.payload["project_id"] == "audit-test"


@pytest.mark.asyncio
async def test_qdrant_upsert_creates_missing_collection_when_dimension_known() -> None:
    calls: list[str] = []

    class FakeClient:
        async def get_collections(self):
            return SimpleNamespace(collections=[])

        async def create_collection(self, **_kwargs):
            calls.append("create_collection")

        async def upsert(self, **_kwargs):
            calls.append("upsert")

    backend = QdrantVectorBackend(
        url="http://127.0.0.1:6333",
        collection="new_collection",
        dimension=3,
    )
    backend._client = FakeClient()
    backend._available = False
    record = EmbeddingRecord(
        source_hash="d" * 64,
        model_id="google/embeddinggemma-300m",
        dimension=3,
        vector=[1.0, 0.0, 0.0],
    )

    assert await backend.upsert(record) is True
    assert calls == ["create_collection", "upsert"]


@pytest.mark.asyncio
async def test_qdrant_search_ignores_points_without_source_hash_payload() -> None:
    class FakeClient:
        async def search(self, **_kwargs):
            return [
                SimpleNamespace(payload={"episodeId": "legacy"}, score=0.99),
                SimpleNamespace(payload={"source_hash": "c" * 64}, score=0.88),
            ]

    backend = QdrantVectorBackend(url="http://127.0.0.1:6333", collection="test")
    backend._client = FakeClient()
    backend._available = True

    hits = await backend.search([1.0, 0.0, 0.0], with_scores=True)

    assert hits == [("c" * 64, 0.88)]


@pytest.mark.asyncio
async def test_qdrant_search_supports_query_points_client_api() -> None:
    captured: dict[str, object] = {}

    class FakeClient:
        async def query_points(self, **kwargs):
            captured.update(kwargs)
            return SimpleNamespace(
                points=[
                    SimpleNamespace(payload={"source_hash": "e" * 64}, score=0.91),
                ]
            )

    backend = QdrantVectorBackend(url="http://127.0.0.1:6333", collection="test")
    backend._client = FakeClient()
    backend._available = True

    hits = await backend.search(
        [1.0, 0.0, 0.0],
        limit=3,
        model_id="google/embeddinggemma-300m",
        project_id="audit-test",
        with_scores=True,
    )

    assert hits == [("e" * 64, 0.91)]
    assert captured["collection_name"] == "test"
    assert captured["query"] == [1.0, 0.0, 0.0]
    assert captured["limit"] == 3
    assert captured["with_payload"] is True
    assert captured["query_filter"] is not None
