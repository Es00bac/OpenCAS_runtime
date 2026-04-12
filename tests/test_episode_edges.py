"""Tests for episode edge storage and runtime creation."""

import pytest
import pytest_asyncio
from pathlib import Path

from opencas.memory import EdgeKind, Episode, EpisodeEdge, EpisodeKind, MemoryStore
from opencas.somatic.models import AffectState, PrimaryEmotion


@pytest_asyncio.fixture
async def store(tmp_path: Path):
    s = MemoryStore(tmp_path / "memory.db")
    await s.connect()
    yield s
    await s.close()


@pytest.mark.asyncio
async def test_save_and_get_edge(store: MemoryStore) -> None:
    edge = EpisodeEdge(
        source_id="ep-1",
        target_id="ep-2",
        kind=EdgeKind.CAUSAL,
        semantic_weight=0.5,
        emotional_weight=0.8,
        recency_weight=0.9,
        structural_weight=0.3,
        confidence=0.75,
    )
    await store.save_edge(edge)
    edges = await store.get_edges_for("ep-1")
    assert len(edges) == 1
    assert edges[0].confidence == 0.75
    assert edges[0].kind == EdgeKind.CAUSAL


@pytest.mark.asyncio
async def test_get_edges_for_kind_filter(store: MemoryStore) -> None:
    ep1 = Episode(kind=EpisodeKind.TURN, content="a")
    ep2 = Episode(kind=EpisodeKind.TURN, content="b")
    await store.save_episode(ep1)
    await store.save_episode(ep2)

    await store.save_edge(
        EpisodeEdge(
            source_id=str(ep1.episode_id),
            target_id=str(ep2.episode_id),
            kind=EdgeKind.TEMPORAL,
            confidence=0.5,
        )
    )
    all_edges = await store.get_edges_for(str(ep1.episode_id))
    assert len(all_edges) == 1
    temporal_edges = await store.get_edges_for(str(ep1.episode_id), kind=EdgeKind.TEMPORAL)
    assert len(temporal_edges) == 1
    semantic_edges = await store.get_edges_for(str(ep1.episode_id), kind=EdgeKind.SEMANTIC)
    assert len(semantic_edges) == 0


@pytest.mark.asyncio
async def test_edge_upsert(store: MemoryStore) -> None:
    edge1 = EpisodeEdge(
        source_id="ep-a",
        target_id="ep-b",
        confidence=0.2,
    )
    edge2 = EpisodeEdge(
        source_id="ep-a",
        target_id="ep-b",
        confidence=0.9,
    )
    await store.save_edge(edge1)
    await store.save_edge(edge2)
    edges = await store.get_edges_for("ep-a")
    assert len(edges) == 1
    assert edges[0].confidence == 0.9


@pytest.mark.asyncio
async def test_get_edges_for_both_directions(store: MemoryStore) -> None:
    edge = EpisodeEdge(source_id="x", target_id="y", confidence=0.5)
    await store.save_edge(edge)
    x_edges = await store.get_edges_for("x")
    y_edges = await store.get_edges_for("y")
    assert len(x_edges) == 1
    assert len(y_edges) == 1


@pytest.mark.asyncio
async def test_delete_edges_for(store: MemoryStore) -> None:
    edge = EpisodeEdge(source_id="del-a", target_id="del-b", confidence=0.5)
    await store.save_edge(edge)
    await store.delete_edges_for("del-a")
    edges = await store.get_edges_for("del-a")
    assert len(edges) == 0


@pytest.mark.asyncio
async def test_episode_with_affect_roundtrip(store: MemoryStore) -> None:
    affect = AffectState(
        primary_emotion=PrimaryEmotion.JOY,
        valence=0.8,
        arousal=0.7,
        certainty=0.6,
        intensity=0.5,
    )
    ep = Episode(
        kind=EpisodeKind.TURN,
        content="Great news!",
        affect=affect,
        identity_core=True,
    )
    await store.save_episode(ep)
    fetched = await store.get_episode(str(ep.episode_id))
    assert fetched is not None
    assert fetched.affect is not None
    assert fetched.affect.primary_emotion == PrimaryEmotion.JOY
    assert fetched.affect.valence == 0.8
    assert fetched.identity_core is True


@pytest.mark.asyncio
async def test_get_episodes_by_ids(store: MemoryStore) -> None:
    ep1 = Episode(kind=EpisodeKind.TURN, content="a")
    ep2 = Episode(kind=EpisodeKind.TURN, content="b")
    await store.save_episode(ep1)
    await store.save_episode(ep2)
    results = await store.get_episodes_by_ids([str(ep1.episode_id), str(ep2.episode_id)])
    assert len(results) == 2


@pytest.mark.asyncio
async def test_update_episode_affect(store: MemoryStore) -> None:
    ep = Episode(kind=EpisodeKind.TURN, content="Initial")
    await store.save_episode(ep)
    new_affect = AffectState(primary_emotion=PrimaryEmotion.ANGER, valence=-0.8)
    await store.update_episode_affect(str(ep.episode_id), new_affect)
    fetched = await store.get_episode(str(ep.episode_id))
    assert fetched is not None
    assert fetched.affect is not None
    assert fetched.affect.primary_emotion == PrimaryEmotion.ANGER
    assert fetched.affect.valence == -0.8
