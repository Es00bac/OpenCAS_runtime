"""Tests for MemoryRetriever affect-conditioned retrieval."""

import pytest

from opencas.context.retriever import MemoryRetriever
from opencas.embeddings import EmbeddingCache, EmbeddingService
from opencas.memory import MemoryStore
from opencas.memory.models import Memory
from opencas.somatic.models import AffectState, PrimaryEmotion


@pytest.mark.asyncio
async def test_retriever_blends_affect_query(tmp_path) -> None:
    memory = MemoryStore(tmp_path / "memory.db")
    await memory.connect()

    cache = EmbeddingCache(tmp_path / "embeddings.db")
    await cache.connect()
    embeddings = EmbeddingService(cache, model_id="local-fallback")

    retriever = MemoryRetriever(memory, embeddings)

    # Seed memories with distinct affective content
    for text in ["I am furious and angry", "I am calm and peaceful"]:
        embed = await embeddings.embed(text)
        await memory.save_memory(
            Memory(
                content=text,
                embedding_id=embed.source_hash,
            )
        )

    # Query with angry affect should boost the angry memory
    affect = AffectState(
        primary_emotion=PrimaryEmotion.ANGER,
        valence=-0.8,
        arousal=0.9,
        intensity=0.8,
    )
    results = await retriever.retrieve(
        "emotional state",
        limit=2,
        affect_query=affect,
        affect_weight=0.5,
    )
    assert len(results) >= 1
    top = results[0]
    assert "furious" in top.content or "calm" in top.content

    await memory.close()
    await cache.close()
