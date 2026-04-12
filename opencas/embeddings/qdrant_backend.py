"""Optional Qdrant vector backend for EmbeddingCache acceleration."""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Sequence, Tuple

from opencas.embeddings.models import EmbeddingRecord


class QdrantVectorBackend:
    """Write-through Qdrant acceleration layer for embedding search."""

    def __init__(
        self,
        url: str,
        collection: str = "opencas_embeddings",
        api_key: Optional[str] = None,
        dimension: Optional[int] = None,
    ) -> None:
        self.url = url
        self.collection = collection
        self.api_key = api_key
        self.dimension = dimension
        self._client: Any = None
        self._available = False

    async def connect(self) -> "QdrantVectorBackend":
        try:
            from qdrant_client import AsyncQdrantClient
        except Exception:  # pragma: no cover
            self._available = False
            return self

        try:
            headers = {}
            if self.api_key:
                headers["api-key"] = self.api_key
            self._client = AsyncQdrantClient(url=self.url, headers=headers or None)
            collections = await self._client.get_collections()
            exists = any(c.name == self.collection for c in collections.collections)
            self._available = exists or (self.dimension is not None)
        except Exception:
            self._available = False
        return self

    async def _ensure_collection(self, dimension: int) -> bool:
        if not self._client:
            return False
        try:
            from qdrant_client.models import Distance, VectorParams

            collections = await self._client.get_collections()
            exists = any(c.name == self.collection for c in collections.collections)
            if not exists:
                await self._client.create_collection(
                    collection_name=self.collection,
                    vectors_config=VectorParams(
                        size=dimension,
                        distance=Distance.COSINE,
                    ),
                )
            return True
        except Exception:
            return False

    async def health(self) -> bool:
        if not self._client:
            return False
        try:
            await self._client.get_collections()
            return True
        except Exception:
            return False

    async def close(self) -> None:
        """Release the async Qdrant client."""
        if self._client is None:
            return
        close = getattr(self._client, "close", None)
        if callable(close):
            await close()
        self._client = None
        self._available = False

    async def upsert(self, record: EmbeddingRecord) -> bool:
        if self._client is None:
            return False
        if not self._available:
            ok = await self._ensure_collection(len(record.vector))
            if not ok:
                return False
            self._available = True
        try:
            from qdrant_client.models import PointStruct

            payload: Dict[str, Any] = {
                "source_hash": record.source_hash,
                "model_id": record.model_id,
            }
            if record.meta:
                payload.update(record.meta)
            await self._client.upsert(
                collection_name=self.collection,
                points=[
                    PointStruct(
                        id=record.source_hash,
                        vector=record.vector,
                        payload=payload,
                    )
                ],
            )
            return True
        except Exception:
            return False

    async def search(
        self,
        vector: Sequence[float],
        limit: int = 10,
        model_id: Optional[str] = None,
        project_id: Optional[str] = None,
        with_scores: bool = False,
    ) -> List[str] | List[Tuple[str, float]]:
        """Return ordered list of source_hash hits from Qdrant.

        Falls back to empty list on any error.
        """
        if not self._available or self._client is None:
            return []
        try:
            from qdrant_client.models import Filter, FieldCondition, MatchValue

            conditions = []
            if model_id:
                conditions.append(
                    FieldCondition(
                        key="model_id",
                        match=MatchValue(value=model_id),
                    )
                )
            if project_id:
                conditions.append(
                    FieldCondition(
                        key="project_id",
                        match=MatchValue(value=project_id),
                    )
                )
            query_filter = Filter(must=conditions) if conditions else None

            results = await self._client.search(
                collection_name=self.collection,
                query_vector=list(vector),
                query_filter=query_filter,
                limit=limit,
                with_payload=True,
            )
            hits: List[str] | List[Tuple[str, float]] = []
            for r in results:
                payload = r.payload or {}
                source_hash = payload.get("source_hash")
                if source_hash:
                    if with_scores:
                        hits.append((str(source_hash), float(getattr(r, "score", 0.0))))
                    else:
                        hits.append(str(source_hash))
            return hits
        except Exception:
            return []
