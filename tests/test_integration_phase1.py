"""Integration tests for OpenCAS Phase 1: Core Substrate."""

import pytest
from pathlib import Path

from opencas.bootstrap import BootstrapConfig, BootstrapPipeline
from opencas.diagnostics import Doctor
from opencas.embeddings import EmbeddingService
from opencas.embeddings.backfill import EmbeddingBackfill
from opencas.embeddings.service import EmbeddingCache
from opencas.identity import IdentityManager, IdentityStore
from opencas.memory import Episode, EpisodeKind, Memory
from opencas.somatic import SomaticManager
from opencas.telemetry import EventKind, TelemetryStore, Tracer


@pytest.mark.asyncio
async def test_full_boot_cycle(tmp_path: Path) -> None:
    """Boot the agent, verify all substrates are online, and shut down cleanly."""
    config = BootstrapConfig(state_dir=tmp_path, session_id="integration-1")
    ctx = await BootstrapPipeline(config).run()

    assert ctx.identity.continuity.boot_count == 1
    assert ctx.memory is not None
    assert ctx.embeddings is not None
    assert ctx.somatic is not None
    assert ctx.llm is not None

    # LLM gateway is wired
    models = ctx.llm.list_available_models()
    assert len(models) > 0
    assert any("claude" in m for m in models)

    # Memory round-trip
    ep = Episode(kind=EpisodeKind.TURN, session_id="integration-1", content="Hello")
    await ctx.memory.save_episode(ep)
    fetched = await ctx.memory.get_episode(str(ep.episode_id))
    assert fetched is not None
    assert fetched.content == "Hello"

    # Embedding round-trip
    rec = await ctx.embeddings.embed("Hello")
    assert rec.dimension == len(rec.vector)
    cached = await ctx.embeddings.embed("Hello")
    assert cached.embedding_id == rec.embedding_id

    # Identity mutation + persistence
    ctx.identity.update_self_belief("test_key", "test_value")

    # Somatic mutation
    ctx.somatic.set_arousal(0.7)

    # Telemetry captured boot
    events = ctx.tracer.store.query(kinds=[EventKind.BOOTSTRAP_STAGE])
    assert len(events) > 0

    # Doctor
    doctor = Doctor(ctx)
    report = await doctor.run_all()
    assert report.overall.value in ("pass", "warn")

    await ctx.close()


@pytest.mark.asyncio
async def test_continuity_across_reboots(tmp_path: Path) -> None:
    """Verify that identity, memory, and somatic state survive a restart."""
    config = BootstrapConfig(state_dir=tmp_path, session_id="reboot-a")

    # First boot
    ctx1 = await BootstrapPipeline(config).run()
    await ctx1.memory.save_episode(Episode(kind=EpisodeKind.TURN, content="first"))
    ctx1.identity.add_user_preference("theme", "dark")
    ctx1.somatic.set_fatigue(0.3)
    await ctx1.close()

    # Second boot
    ctx2 = await BootstrapPipeline(config).run()
    assert ctx2.identity.continuity.boot_count == 2
    assert ctx2.identity.user_model.explicit_preferences.get("theme") == "dark"
    assert ctx2.somatic.state.fatigue == 0.3

    episodes = await ctx2.memory.list_episodes()
    assert any(e.content == "first" for e in episodes)

    await ctx2.close()


@pytest.mark.asyncio
async def test_embedding_backfill_with_memory(tmp_path: Path) -> None:
    """Memory content can be backfilled with embeddings on demand."""
    config = BootstrapConfig(state_dir=tmp_path, session_id="embed-backfill")
    ctx = await BootstrapPipeline(config).run()

    text = "The quick brown fox"
    embed_record = await ctx.embeddings.embed(text)

    ep = Episode(
        kind=EpisodeKind.OBSERVATION,
        content=text,
        embedding_id=embed_record.source_hash,
    )
    await ctx.memory.save_episode(ep)

    fetched = await ctx.memory.get_episode(str(ep.episode_id))
    assert fetched is not None
    assert fetched.embedding_id == embed_record.source_hash

    health = await ctx.embeddings.health()
    assert health.total_records >= 1

    await ctx.close()


@pytest.mark.asyncio
async def test_memory_compaction_and_identity(tmp_path: Path) -> None:
    """Compact old episodes and verify identity continuity persists."""
    config = BootstrapConfig(state_dir=tmp_path, session_id="compact-test")
    ctx = await BootstrapPipeline(config).run()

    ep1 = Episode(kind=EpisodeKind.TURN, content="old turn 1")
    ep2 = Episode(kind=EpisodeKind.TURN, content="old turn 2")
    await ctx.memory.save_episode(ep1)
    await ctx.memory.save_episode(ep2)

    await ctx.memory.mark_compacted([str(ep1.episode_id), str(ep2.episode_id)])
    active = await ctx.memory.list_episodes(compacted=False)
    assert len(active) == 0

    compacted = await ctx.memory.list_episodes(compacted=True)
    assert len(compacted) == 2

    ctx.identity.record_shutdown(session_id="compact-test")
    assert ctx.identity.continuity.last_shutdown_time is not None

    await ctx.close()


@pytest.mark.asyncio
async def test_embedding_backfill_realigns_stale_episode_and_memory_models(tmp_path: Path) -> None:
    store_path = tmp_path / "memory.db"
    cache_path = tmp_path / "embeddings.db"

    config = BootstrapConfig(state_dir=tmp_path, session_id="realign-test")
    ctx = await BootstrapPipeline(config).run()

    old_cache = EmbeddingCache(cache_path)
    await old_cache.connect()
    old_embeddings = EmbeddingService(old_cache, model_id="local-fallback")
    old_episode_record = await old_embeddings.embed("old episode text", task_type="memory_episode")
    old_memory_record = await old_embeddings.embed("old distilled memory", task_type="memory_distilled")

    episode = Episode(
        kind=EpisodeKind.OBSERVATION,
        content="old episode text",
        embedding_id=old_episode_record.source_hash,
    )
    memory = Memory(
        content="old distilled memory",
        embedding_id=old_memory_record.source_hash,
    )
    await ctx.memory.save_episode(episode)
    await ctx.memory.save_memory(memory)
    await old_cache.close()

    new_cache = EmbeddingCache(cache_path)
    await new_cache.connect()
    new_embeddings = EmbeddingService(
        new_cache,
        model_id="google/gemini-embedding-2-preview",
        embed_fn=old_embeddings._fallback_embed,
    )
    backfill = EmbeddingBackfill(new_embeddings, ctx.memory)

    refreshed_episode_count = await backfill.align_episode_embeddings([episode])
    refreshed_memory_count = await backfill.align_memory_embeddings([memory])

    refreshed_episode = await ctx.memory.get_episode(str(episode.episode_id))
    refreshed_memory = await ctx.memory.get_memory(str(memory.memory_id))

    assert refreshed_episode_count == 1
    assert refreshed_memory_count == 1
    assert refreshed_episode is not None
    assert refreshed_memory is not None
    assert refreshed_episode.embedding_id != old_episode_record.source_hash
    assert refreshed_memory.embedding_id != old_memory_record.source_hash

    episode_embedding = await new_cache.get(refreshed_episode.embedding_id)
    memory_embedding = await new_cache.get(refreshed_memory.embedding_id)
    assert episode_embedding is not None
    assert memory_embedding is not None
    assert episode_embedding.model_id == "google/gemini-embedding-2-preview"
    assert memory_embedding.model_id == "google/gemini-embedding-2-preview"

    await new_cache.close()
    await ctx.close()
