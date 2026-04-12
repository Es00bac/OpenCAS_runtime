"""Local HNSW vector backend using hnswlib."""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from .models import EmbeddingRecord

logger = logging.getLogger(__name__)


class HnswVectorBackend:
    """Lightweight local ANN index with incremental insertion."""

    def __init__(
        self,
        space: str = "cosine",
        M: int = 16,
        ef_construction: int = 200,
    ) -> None:
        self.space = space
        self.M = M
        self.ef_construction = ef_construction
        self._index: Optional[Any] = None
        self._id_map: Dict[int, str] = {}
        self._reverse_map: Dict[str, int] = {}
        self._dimension: Optional[int] = None
        self._metadata: Dict[str, Dict[str, Any]] = {}
        self._next_id = 0

    def connect(self) -> "HnswVectorBackend":
        """No-op for API symmetry; index is lazily initialized."""
        return self

    async def upsert(self, record: EmbeddingRecord) -> bool:
        """Add or update a vector in the HNSW index."""
        try:
            import hnswlib
        except ImportError:
            logger.warning("hnswlib not installed; cannot upsert to HNSW backend")
            return False

        vector = np.array(record.vector, dtype=np.float32)
        dim = len(vector)

        if self._dimension is None:
            self._dimension = dim
            # hnswlib cosine space requires normalized vectors
            self._index = hnswlib.Index(space=self.space, dim=dim)
            # Initial max_elements; we will resize if needed
            self._index.init_index(
                max_elements=max(10000, self._next_id + 1),
                ef_construction=self.ef_construction,
                M=self.M,
            )
            self._index.set_ef(10)

        if self._index is None:
            return False

        if dim != self._dimension:
            logger.warning(
                "HNSW dimension mismatch: expected %s, got %s",
                self._dimension,
                dim,
            )
            return False

        source_hash = record.source_hash

        # Normalize for cosine space
        norm = np.linalg.norm(vector)
        if norm > 0:
            vector = vector / norm

        if source_hash in self._reverse_map:
            internal_id = self._reverse_map[source_hash]
            # hnswlib does not support true updates; we replace by re-adding
            # at the same internal id. This is an approximation but works for
            # most practical purposes because add_items overwrites the vector
            # associated with the label.
            self._index.add_items(vector.reshape(1, -1), np.array([internal_id]))
        else:
            internal_id = self._next_id
            self._next_id += 1
            current_max = self._index.get_max_elements()
            if internal_id >= current_max:
                new_max = max(current_max * 2, internal_id + 1)
                self._index.resize_index(new_max)
            self._index.add_items(vector.reshape(1, -1), np.array([internal_id]))
            self._id_map[internal_id] = source_hash
            self._reverse_map[source_hash] = internal_id

        self._metadata[source_hash] = {
            "model_id": record.model_id,
            "project_id": record.meta.get("project_id") if record.meta else None,
        }
        return True

    async def search(
        self,
        vector: List[float],
        limit: int = 10,
        model_id: Optional[str] = None,
        project_id: Optional[str] = None,
        with_scores: bool = False,
    ) -> List[str] | List[Tuple[str, float]]:
        """Return source_hashes ordered by similarity, with metadata post-filtering."""
        if self._index is None or self._dimension is None:
            return []

        try:
            import hnswlib
        except ImportError:
            return []

        query = np.array(vector, dtype=np.float32)
        norm = np.linalg.norm(query)
        if norm > 0:
            query = query / norm

        ef = limit * 10
        max_ef = 400
        oversample = 4

        while ef <= max_ef:
            self._index.set_ef(ef)
            k = min(limit * oversample, self._index.get_current_count())
            if k == 0:
                return []

            labels, distances = self._index.knn_query(query.reshape(1, -1), k=k)
            labels = labels[0]
            # For cosine space, distance = 1 - cosine_similarity
            similarities = 1.0 - distances[0]

            results: List[str] | List[Tuple[str, float]] = []
            for label, sim in zip(labels, similarities):
                source_hash = self._id_map.get(int(label))
                if source_hash is None:
                    continue
                meta = self._metadata.get(source_hash, {})
                if model_id is not None and meta.get("model_id") != model_id:
                    continue
                if project_id is not None and meta.get("project_id") != project_id:
                    continue
                if with_scores:
                    results.append((source_hash, float(sim)))
                else:
                    results.append(source_hash)
                if len(results) >= limit:
                    return results

            # If we didn't get enough results, increase ef and retry
            if len(results) < limit:
                ef = int(ef * 1.5)
                continue
            return results

        return results[:limit]

    def clear(self) -> None:
        """Reset the index and all maps."""
        self._index = None
        self._id_map.clear()
        self._reverse_map.clear()
        self._metadata.clear()
        self._dimension = None
        self._next_id = 0

    def close(self) -> None:
        """Release the native index and internal maps."""
        self.clear()

    def count(self) -> int:
        """Return current index element count."""
        if self._index is None:
            return 0
        return self._index.get_current_count()
