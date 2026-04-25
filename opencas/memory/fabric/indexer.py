"""Embedding-first indexer for memory fabric candidate generation."""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional

from opencas.embeddings import EmbeddingService
from opencas.memory import Episode


@dataclass
class Candidate:
    """Nearest-neighbor candidate episode with similarity score."""

    episode_id: str
    score: float


class MemoryIndexer:
    """Generate candidate episode pairs using Qdrant (with SQLite fallback)."""

    def __init__(
        self,
        embeddings: EmbeddingService,
        top_k: int = 24,
    ) -> None:
        self.embeddings = embeddings
        self.top_k = top_k

    async def candidates(self, episode: Episode) -> List[Candidate]:
        """Return nearest-neighbor episode IDs for *episode* via Qdrant search.

        Falls back to a small local brute-force scan over the cache if Qdrant
        is unavailable or returns empty results.
        """
        if not episode.embedding_id:
            return []

        record = await self.embeddings.cache.get(episode.embedding_id)
        if record is None:
            return []

        # Try Qdrant first
        hits: List[str] = []
        if self.embeddings.cache.vector_backend is not None:
            try:
                hits = await self.embeddings.cache.vector_backend.search(
                    record.vector,
                    limit=self.top_k * 2,
                    model_id=self.embeddings.model_id,
                )
            except Exception:
                hits = []

        if hits:
            return [
                Candidate(
                    episode_id=h,
                    score=max(0.0, 1.0 - (rank * 0.0001)),
                )
                for rank, h in enumerate(hits)
                if h != episode.embedding_id
            ][: self.top_k]

        # Graceful fallback: scan local cache
        scored = await self.embeddings.cache.search_similar(
            record.vector,
            limit=self.top_k,
            model_id=self.embeddings.model_id,
        )
        return [
            Candidate(
                episode_id=rec.source_hash,
                score=sim,
            )
            for rec, sim in scored
            if rec.source_hash != episode.embedding_id
        ]
