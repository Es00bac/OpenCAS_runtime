"""Tests for EpisodeGraph."""

import pytest
import pytest_asyncio
from pathlib import Path

from opencas.memory import EdgeKind, Episode, EpisodeEdge, EpisodeKind, MemoryStore
from opencas.memory.fabric.graph import EpisodeGraph


@pytest_asyncio.fixture
async def store(tmp_path: Path):
    s = MemoryStore(tmp_path / "memory.db")
    await s.connect()
    yield s
    await s.close()


@pytest_asyncio.fixture
async def graph(store: MemoryStore):
    return EpisodeGraph(store=store)


@pytest.mark.asyncio
async def test_get_neighbors_with_kind_filter(store: MemoryStore, graph: EpisodeGraph) -> None:
    ep1 = Episode(kind=EpisodeKind.TURN, content="a")
    ep2 = Episode(kind=EpisodeKind.TURN, content="b")
    ep3 = Episode(kind=EpisodeKind.TURN, content="c")
    await store.save_episode(ep1)
    await store.save_episode(ep2)
    await store.save_episode(ep3)

    await store.save_edge(
        EpisodeEdge(
            source_id=str(ep1.episode_id),
            target_id=str(ep2.episode_id),
            kind=EdgeKind.TEMPORAL,
            confidence=0.8,
        )
    )
    await store.save_edge(
        EpisodeEdge(
            source_id=str(ep1.episode_id),
            target_id=str(ep3.episode_id),
            kind=EdgeKind.SEMANTIC,
            confidence=0.7,
        )
    )

    all_edges = await graph.get_neighbors(str(ep1.episode_id))
    assert len(all_edges) == 2

    temporal_edges = await graph.get_neighbors(str(ep1.episode_id), kind=EdgeKind.TEMPORAL)
    assert len(temporal_edges) == 1
    assert temporal_edges[0].kind == EdgeKind.TEMPORAL


@pytest.mark.asyncio
async def test_walk_returns_hop_distances(store: MemoryStore, graph: EpisodeGraph) -> None:
    ep1 = Episode(kind=EpisodeKind.TURN, content="1")
    ep2 = Episode(kind=EpisodeKind.TURN, content="2")
    ep3 = Episode(kind=EpisodeKind.TURN, content="3")
    await store.save_episode(ep1)
    await store.save_episode(ep2)
    await store.save_episode(ep3)

    await store.save_edge(
        EpisodeEdge(source_id=str(ep1.episode_id), target_id=str(ep2.episode_id), confidence=0.9)
    )
    await store.save_edge(
        EpisodeEdge(source_id=str(ep2.episode_id), target_id=str(ep3.episode_id), confidence=0.9)
    )

    visited = await graph.walk(str(ep1.episode_id), steps=2)
    assert visited[str(ep1.episode_id)] == 0
    assert visited[str(ep2.episode_id)] == 1
    assert visited[str(ep3.episode_id)] == 2


@pytest.mark.asyncio
async def test_walk_respects_kind_filter(store: MemoryStore, graph: EpisodeGraph) -> None:
    ep1 = Episode(kind=EpisodeKind.TURN, content="1")
    ep2 = Episode(kind=EpisodeKind.TURN, content="2")
    ep3 = Episode(kind=EpisodeKind.TURN, content="3")
    await store.save_episode(ep1)
    await store.save_episode(ep2)
    await store.save_episode(ep3)

    await store.save_edge(
        EpisodeEdge(
            source_id=str(ep1.episode_id),
            target_id=str(ep2.episode_id),
            kind=EdgeKind.TEMPORAL,
            confidence=0.9,
        )
    )
    await store.save_edge(
        EpisodeEdge(
            source_id=str(ep2.episode_id),
            target_id=str(ep3.episode_id),
            kind=EdgeKind.SEMANTIC,
            confidence=0.9,
        )
    )

    visited = await graph.walk(str(ep1.episode_id), steps=2, kind_filter=EdgeKind.TEMPORAL)
    assert str(ep1.episode_id) in visited
    assert str(ep2.episode_id) in visited
    assert str(ep3.episode_id) not in visited


@pytest.mark.asyncio
async def test_subgraph_returns_only_internal_edges(store: MemoryStore, graph: EpisodeGraph) -> None:
    ep1 = Episode(kind=EpisodeKind.TURN, content="1")
    ep2 = Episode(kind=EpisodeKind.TURN, content="2")
    ep3 = Episode(kind=EpisodeKind.TURN, content="3")
    await store.save_episode(ep1)
    await store.save_episode(ep2)
    await store.save_episode(ep3)

    await store.save_edge(
        EpisodeEdge(source_id=str(ep1.episode_id), target_id=str(ep2.episode_id), confidence=0.9)
    )
    await store.save_edge(
        EpisodeEdge(source_id=str(ep2.episode_id), target_id=str(ep3.episode_id), confidence=0.9)
    )

    edges = await graph.subgraph([str(ep1.episode_id), str(ep2.episode_id)])
    assert len(edges) == 1
    ids = {edges[0].source_id, edges[0].target_id}
    assert ids == {str(ep1.episode_id), str(ep2.episode_id)}
