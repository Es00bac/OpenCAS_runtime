"""MMR reranking helpers for ``MemoryRetriever``."""

from __future__ import annotations

from typing import Dict, List, Optional

import numpy as np

from opencas.context.models import RetrievalResult
from opencas.embeddings import EmbeddingService


async def rerank_mmr(
    embeddings: EmbeddingService,
    results: List[RetrievalResult],
    *,
    lambda_param: float = 0.5,
    limit: int = 10,
) -> List[RetrievalResult]:
    """Rerank results using Maximal Marginal Relevance."""
    if not results:
        return results
    candidate_pool_limit = max(limit, min(len(results), max(limit * 2, limit + 64)))
    if len(results) > candidate_pool_limit:
        results = results[:candidate_pool_limit]

    vectors: List[Optional[np.ndarray]] = []
    missing_embedding_ids: list[str] = []
    missing_by_index: dict[int, str] = {}
    for result in results:
        vector = None
        if result.embedding is not None:
            vector = np.array(result.embedding, dtype=np.float32)
        else:
            episode = getattr(result, "episode", None)
            memory = getattr(result, "memory", None)
            embedding_id = None
            if episode is not None:
                embedding_id = getattr(episode, "embedding_id", None)
            elif memory is not None:
                embedding_id = getattr(memory, "embedding_id", None)
            if embedding_id is not None:
                embedding_id = str(embedding_id)
                missing_by_index[len(vectors)] = embedding_id
                missing_embedding_ids.append(embedding_id)
        vectors.append(vector)

    if missing_embedding_ids:
        records: Dict[str, object] = {}
        get_many = getattr(embeddings.cache, "get_many", None)
        if callable(get_many):
            records = await get_many(missing_embedding_ids)
        for index, embedding_id in missing_by_index.items():
            record = records.get(embedding_id)
            if record is None:
                record = await embeddings.cache.get(embedding_id)
            if record is not None and getattr(record, "vector", None):
                vectors[index] = np.array(record.vector, dtype=np.float32)

    def similarity(i: int, j: int) -> float:
        left = vectors[i]
        right = vectors[j]
        if left is None or right is None:
            return 0.0
        if left.shape != right.shape:
            return 0.0
        left_norm = float(np.linalg.norm(left))
        right_norm = float(np.linalg.norm(right))
        if left_norm == 0.0 or right_norm == 0.0:
            return 0.0
        return float(np.dot(left, right) / (left_norm * right_norm))

    selected_indices: List[int] = []
    remaining = set(range(len(results)))

    while remaining and len(selected_indices) < limit:
        best_idx: Optional[int] = None
        best_score = -float("inf")
        for idx in remaining:
            relevance = results[idx].score
            max_sim = 0.0
            for selected in selected_indices:
                max_sim = max(max_sim, similarity(idx, selected))
            mmr_score = lambda_param * relevance - (1.0 - lambda_param) * max_sim
            if mmr_score > best_score:
                best_score = mmr_score
                best_idx = idx
        if best_idx is None:
            break
        selected_indices.append(best_idx)
        remaining.remove(best_idx)

    return [results[index] for index in selected_indices]
