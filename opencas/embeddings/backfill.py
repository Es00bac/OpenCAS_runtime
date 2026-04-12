"""Embedding backfill for missing or stale embeddings."""

from __future__ import annotations

from typing import TYPE_CHECKING, List

from opencas.embeddings.service import EmbeddingService

if TYPE_CHECKING:
    from opencas.memory import MemoryStore
    from opencas.memory.models import Episode, Memory


class EmbeddingBackfill:
    """Backfills missing embeddings and migrates stale ones onto the active model."""

    def __init__(self, embeddings: EmbeddingService, store: "MemoryStore") -> None:
        self.embeddings = embeddings
        self.store = store

    async def backfill_missing_embeddings(self, episodes: List[Episode]) -> int:
        """Backward-compatible alias for episode alignment."""
        return await self.align_episode_embeddings(episodes)

    async def align_episode_embeddings(self, episodes: List["Episode"]) -> int:
        """Ensure episodes point at embeddings for the active model."""
        stale = await self._stale_episodes(episodes)
        if not stale:
            return 0

        updated = 0
        batch_size = 32
        for i in range(0, len(stale), batch_size):
            batch = stale[i : i + batch_size]
            records = await self._embed_batch(batch)
            for ep, record in zip(batch, records):
                ep.embedding_id = record.source_hash
            await self.store.save_episodes_batch(batch)
            updated += len(batch)
        return updated

    async def align_memory_embeddings(self, memories: List["Memory"]) -> int:
        """Ensure distilled memories point at embeddings for the active model."""
        stale = await self._stale_memories(memories)
        if not stale:
            return 0

        updated = 0
        batch_size = 32
        for i in range(0, len(stale), batch_size):
            batch = stale[i : i + batch_size]
            records = await self._embed_memories_batch(batch)
            for memory, record in zip(batch, records):
                memory.embedding_id = record.source_hash
            for memory in batch:
                await self.store.save_memory(memory)
            updated += len(batch)
        return updated

    async def _embed_batch(self, episodes: List[Episode]) -> List:
        """Compute EmbeddingRecords for a list of episodes."""
        results = []
        for ep in episodes:
            record = await self.embeddings.embed(
                ep.content,
                task_type="memory_episode",
            )
            results.append(record)
        return results

    async def _embed_memories_batch(self, memories: List["Memory"]) -> List:
        """Compute EmbeddingRecords for a list of distilled memories."""
        results = []
        for memory in memories:
            record = await self.embeddings.embed(
                memory.content,
                task_type="memory_distilled",
            )
            results.append(record)
        return results

    async def _stale_episodes(self, episodes: List["Episode"]) -> List["Episode"]:
        return [
            ep for ep in episodes
            if await self._needs_refresh(ep.embedding_id)
        ]

    async def _stale_memories(self, memories: List["Memory"]) -> List["Memory"]:
        return [
            memory for memory in memories
            if await self._needs_refresh(memory.embedding_id)
        ]

    async def _needs_refresh(self, embedding_id: str | None) -> bool:
        if not embedding_id:
            return True
        record = await self.embeddings.cache.get(embedding_id)
        if record is None:
            return True
        return record.model_id != self.embeddings.model_id
