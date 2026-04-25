"""Conceptual (embedding cosine similarity) scorer."""

from __future__ import annotations

from typing import Any, Optional

import numpy as np

from opencas.embeddings import EmbeddingService
from opencas.memory import Episode


class ConceptualScorer:
    """Score conceptual similarity via embedding cosine similarity."""

    def __init__(self, embeddings: EmbeddingService) -> None:
        self.embeddings = embeddings

    async def score(
        self,
        ep_a: Episode,
        ep_b: Episode,
        context: Optional[Any] = None,
    ) -> float:
        if not ep_a.embedding_id or not ep_b.embedding_id:
            return 0.0
        rec_a = await self.embeddings.cache.get(ep_a.embedding_id)
        rec_b = await self.embeddings.cache.get(ep_b.embedding_id)
        if rec_a is None or rec_b is None:
            return 0.0
        va = np.array(rec_a.vector, dtype=np.float32)
        vb = np.array(rec_b.vector, dtype=np.float32)
        na = float(np.linalg.norm(va))
        nb = float(np.linalg.norm(vb))
        if na == 0 or nb == 0:
            return 0.0
        return float(np.dot(va, vb) / (na * nb))
