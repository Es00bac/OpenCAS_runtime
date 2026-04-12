"""Tests for consolidation edge building and identity core promotion."""

import pytest
import pytest_asyncio
from datetime import datetime, timezone, timedelta
from pathlib import Path

from opencas.api import LLMClient
from opencas.consolidation import NightlyConsolidationEngine
from opencas.embeddings import EmbeddingCache, EmbeddingService
from opencas.identity import IdentityManager, IdentityStore
from opencas.memory import EdgeKind, Episode, EpisodeEdge, EpisodeKind, MemoryStore
from opencas.somatic.models import AffectState, PrimaryEmotion


@pytest_asyncio.fixture
async def engine(tmp_path: Path):
    store = MemoryStore(tmp_path / "memory.db")
    await store.connect()
    cache = EmbeddingCache(":memory:")
    await cache.connect()
    embeddings = EmbeddingService(cache=cache, model_id="local-fallback")
    identity_store = IdentityStore(tmp_path / "identity")
    identity = IdentityManager(identity_store)
    identity.load()
    llm = LLMClient(provider_manager=object())  # type: ignore
    consolidation = NightlyConsolidationEngine(
        memory=store,
        embeddings=embeddings,
        llm=llm,
        identity=identity,
    )
    yield consolidation
    await store.close()


@pytest_asyncio.fixture
async def embedded_eps(engine: NightlyConsolidationEngine):
    """Helper to create and embed episodes for fabric builder tests."""
    now = datetime.now(timezone.utc)
    ep1 = Episode(
        kind=EpisodeKind.TURN,
        content="learning rust",
        created_at=now - timedelta(hours=1),
        affect=AffectState(primary_emotion=PrimaryEmotion.CURIOUS, valence=0.3),
    )
    ep2 = Episode(
        kind=EpisodeKind.TURN,
        content="rust programming basics",
        created_at=now,
        affect=AffectState(primary_emotion=PrimaryEmotion.CURIOUS, valence=0.3),
    )
    for ep in [ep1, ep2]:
        rec = await engine.embeddings.embed(ep.content)
        ep.embedding_id = rec.source_hash
        await engine.memory.save_episode(ep)
    return ep1, ep2


@pytest.mark.asyncio
async def test_fabric_builder_creates_typed_links(engine: NightlyConsolidationEngine, embedded_eps) -> None:
    ep1, ep2 = embedded_eps

    count = await engine.fabric_builder.rebuild([ep1, ep2])
    assert count >= 1

    edges = await engine.memory.get_edges_for(str(ep1.episode_id))
    assert len(edges) >= 1
    assert edges[0].kind is not None
    assert isinstance(edges[0].kind, EdgeKind)


@pytest.mark.asyncio
async def test_fabric_builder_skips_distant_unrelated(engine: NightlyConsolidationEngine) -> None:
    now = datetime.now(timezone.utc)
    ep1 = Episode(
        kind=EpisodeKind.TURN,
        content="a",
        created_at=now - timedelta(days=10),
    )
    ep2 = Episode(
        kind=EpisodeKind.TURN,
        content="b",
        created_at=now,
    )
    for ep in [ep1, ep2]:
        rec = await engine.embeddings.embed(ep.content)
        ep.embedding_id = rec.source_hash
        await engine.memory.save_episode(ep)

    count = await engine.fabric_builder.rebuild([ep1, ep2], min_confidence=0.2)
    assert count == 0


@pytest.mark.asyncio
async def test_promote_identity_core(engine: NightlyConsolidationEngine) -> None:
    now = datetime.now(timezone.utc)
    # Create 6 episodes all linked to a central "hub" episode
    hub = Episode(
        kind=EpisodeKind.TURN,
        content="hub",
        created_at=now,
        affect=AffectState(primary_emotion=PrimaryEmotion.JOY, valence=0.5),
    )
    await engine.memory.save_episode(hub)

    for i in range(6):
        ep = Episode(
            kind=EpisodeKind.TURN,
            content=f"spoke {i}",
            created_at=now - timedelta(minutes=i),
            affect=AffectState(primary_emotion=PrimaryEmotion.JOY, valence=0.5),
        )
        await engine.memory.save_episode(ep)
        await engine.memory.save_edge(
            EpisodeEdge(
                source_id=str(hub.episode_id),
                target_id=str(ep.episode_id),
                kind=EdgeKind.SEMANTIC,
                confidence=0.5,
            )
        )

    promoted = await engine._promote_identity_core([hub])
    assert promoted >= 1

    fetched = await engine.memory.get_episode(str(hub.episode_id))
    assert fetched is not None
    assert fetched.identity_core is True


@pytest.mark.asyncio
async def test_consolidation_run_includes_edges(engine: NightlyConsolidationEngine) -> None:
    now = datetime.now(timezone.utc)
    ep1 = Episode(
        kind=EpisodeKind.TURN,
        content="rust basics",
        created_at=now - timedelta(hours=2),
        affect=AffectState(primary_emotion=PrimaryEmotion.CURIOUS, valence=0.3),
    )
    ep2 = Episode(
        kind=EpisodeKind.TURN,
        content="rust ownership",
        created_at=now - timedelta(hours=1),
        affect=AffectState(primary_emotion=PrimaryEmotion.CURIOUS, valence=0.3),
    )
    for ep in [ep1, ep2]:
        rec = await engine.embeddings.embed(ep.content)
        ep.embedding_id = rec.source_hash
        await engine.memory.save_episode(ep)

    result = await engine.run()
    assert result.edges_created >= 1
