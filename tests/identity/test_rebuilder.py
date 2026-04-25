"""Tests for identity rebuild from autobiographical memory."""

from pathlib import Path
import pytest
import pytest_asyncio

from opencas.identity import IdentityManager, IdentityRebuilder, IdentityRebuildResult, IdentityStore
from opencas.memory import Episode, EpisodeEdge, EpisodeKind, EdgeKind, MemoryStore


@pytest_asyncio.fixture
async def memory(tmp_path: Path):
    store = MemoryStore(tmp_path / "memory.db")
    await store.connect()
    yield store
    await store.close()


@pytest.fixture
def identity(tmp_path: Path):
    store = IdentityStore(tmp_path / "identity")
    mgr = IdentityManager(store)
    mgr.load()
    return mgr


@pytest.mark.asyncio
async def test_rebuild_heuristic_fallback(memory: MemoryStore, identity: IdentityManager) -> None:
    ep1 = Episode(
        kind=EpisodeKind.OBSERVATION,
        content="I am curious and want to learn python.",
        identity_core=True,
    )
    ep2 = Episode(
        kind=EpisodeKind.OBSERVATION,
        content="I should build something useful and help others.",
        identity_core=True,
    )
    await memory.save_episode(ep1)
    await memory.save_episode(ep2)

    rebuilder = IdentityRebuilder(memory=memory, llm=None)
    result = await rebuilder.rebuild()

    assert result.narrative is not None
    assert len(result.source_episode_ids) >= 2
    assert result.confidence > 0
    assert any("growth" in v for v in result.values) or any("agency" in v for v in result.values)

    await rebuilder.apply(result, identity)
    assert identity.self_model.narrative == result.narrative


@pytest.mark.asyncio
async def test_rebuild_with_llm(memory: MemoryStore, identity: IdentityManager) -> None:
    class MockLLM:
        async def chat_completion(self, messages, **kwargs):
            return {
                "choices": [
                    {
                        "message": {
                            "content": (
                                '{"narrative": "A learner builder.", '
                                '"values": ["growth", "honesty"], '
                                '"traits": ["curious"], '
                                '"goals": ["learn python"]}'
                            )
                        }
                    }
                ]
            }

    ep = Episode(kind=EpisodeKind.OBSERVATION, content="I love learning.", identity_core=True)
    await memory.save_episode(ep)

    rebuilder = IdentityRebuilder(memory=memory, llm=MockLLM())
    result = await rebuilder.rebuild()

    assert result.narrative == "A learner builder."
    assert "growth" in result.values
    assert "curious" in result.traits
    assert "learn python" in result.goals


@pytest.mark.asyncio
async def test_rebuild_uses_graph_walk(memory: MemoryStore, identity: IdentityManager) -> None:
    from opencas.memory.fabric.graph import EpisodeGraph

    ep1 = Episode(kind=EpisodeKind.OBSERVATION, content="seed one", identity_core=True)
    ep2 = Episode(kind=EpisodeKind.OBSERVATION, content="related two", identity_core=False)
    await memory.save_episode(ep1)
    await memory.save_episode(ep2)

    graph = EpisodeGraph(store=memory)
    edge = EpisodeEdge(
        source_id=str(ep1.episode_id),
        target_id=str(ep2.episode_id),
        kind=EdgeKind.TEMPORAL,
        confidence=0.8,
    )
    await memory.save_edge(edge)

    rebuilder = IdentityRebuilder(memory=memory, episode_graph=graph, llm=None)
    result = await rebuilder.rebuild(seed_episode_ids=[str(ep1.episode_id)])

    assert str(ep2.episode_id) in result.source_episode_ids


@pytest.mark.asyncio
async def test_rebuild_falls_back_to_recent_episodes(memory: MemoryStore) -> None:
    ep = Episode(kind=EpisodeKind.TURN, content="recent turn", identity_core=False)
    await memory.save_episode(ep)

    rebuilder = IdentityRebuilder(memory=memory, llm=None)
    result = await rebuilder.rebuild()

    assert len(result.source_episode_ids) >= 1
    assert result.confidence > 0
