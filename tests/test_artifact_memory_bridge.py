"""Tests for syncing authored artifacts into episodic memory."""

from pathlib import Path

import pytest

from opencas.context.retriever import MemoryRetriever
from opencas.embeddings import EmbeddingCache, EmbeddingService
from opencas.memory import ArtifactMemoryBridge, MemoryStore


@pytest.mark.asyncio
async def test_artifact_bridge_syncs_text_artifact_into_memory(tmp_path: Path) -> None:
    state_dir = tmp_path / ".opencas"
    plans_dir = state_dir / "plans" / "story"
    plans_dir.mkdir(parents=True)
    artifact = plans_dir / "story.md"
    artifact.write_text(
        "# The Lighthouse Keeper's Letter\n\nSome lighthouses are built to bring the keeper home.\n",
        encoding="utf-8",
    )

    memory = MemoryStore(state_dir / "memory.db")
    await memory.connect()
    cache = EmbeddingCache(state_dir / "embeddings.db")
    await cache.connect()
    embeddings = EmbeddingService(cache=cache, model_id="local-fallback")
    bridge = ArtifactMemoryBridge(state_dir=state_dir, memory=memory, embeddings=embeddings)

    result = await bridge.sync()
    episodes = await memory.list_artifact_episodes(".opencas/plans/story/story.md")
    memories = await memory.list_memories(limit=10)

    assert result["artifacts"] == 1
    assert len(episodes) == 1
    assert episodes[0].kind.value == "artifact"
    assert episodes[0].payload["artifact"]["title"] == "The Lighthouse Keeper's Letter"
    assert "bring the keeper home" in episodes[0].content
    assert any("artifact_path:.opencas/plans/story/story.md" in item.tags for item in memories)

    await memory.close()
    await cache.close()


@pytest.mark.asyncio
async def test_artifact_bridge_updates_changed_artifact_without_leaving_stale_chunks(tmp_path: Path) -> None:
    state_dir = tmp_path / ".opencas"
    plans_dir = state_dir / "plans"
    plans_dir.mkdir(parents=True)
    artifact = plans_dir / "story.md"
    artifact.write_text(("alpha " * 900), encoding="utf-8")

    memory = MemoryStore(state_dir / "memory.db")
    await memory.connect()
    cache = EmbeddingCache(state_dir / "embeddings.db")
    await cache.connect()
    embeddings = EmbeddingService(cache=cache, model_id="local-fallback")
    bridge = ArtifactMemoryBridge(state_dir=state_dir, memory=memory, embeddings=embeddings)

    first = await bridge.sync()
    initial = await memory.list_artifact_episodes(".opencas/plans/story.md")
    artifact.write_text("short artifact now\n", encoding="utf-8")
    second = await bridge.sync()
    updated = await memory.list_artifact_episodes(".opencas/plans/story.md")

    assert first["episodes_created"] >= 2
    assert len(initial) >= 2
    assert second["episodes_updated"] >= 1
    assert second["episodes_deleted"] >= 1
    assert len(updated) == 1
    assert "short artifact now" in updated[0].content

    await memory.close()
    await cache.close()


@pytest.mark.asyncio
async def test_retriever_can_surface_artifact_memory_for_recall_query(tmp_path: Path) -> None:
    state_dir = tmp_path / ".opencas"
    plans_dir = state_dir / "plans" / "story"
    plans_dir.mkdir(parents=True)
    artifact = plans_dir / "story.md"
    artifact.write_text(
        "# The Lighthouse Keeper's Letter\n\nThe lighthouse keeper wrote a letter about memory, return, and homecoming.\n",
        encoding="utf-8",
    )

    memory = MemoryStore(state_dir / "memory.db")
    await memory.connect()
    cache = EmbeddingCache(state_dir / "embeddings.db")
    await cache.connect()
    embeddings = EmbeddingService(cache=cache, model_id="local-fallback")
    bridge = ArtifactMemoryBridge(state_dir=state_dir, memory=memory, embeddings=embeddings)
    await bridge.sync()

    retriever = MemoryRetriever(memory=memory, embeddings=embeddings)
    results = await retriever.retrieve("Do you remember the lighthouse story?", limit=10)

    assert any("lighthouse keeper" in item.content.lower() for item in results)

    await memory.close()
    await cache.close()
