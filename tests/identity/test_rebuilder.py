"""Tests for identity rebuild from autobiographical memory."""

from datetime import datetime, timezone
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
async def test_rebuild_with_unsafe_llm_output_is_sanitized(memory: MemoryStore) -> None:
    class MockLLM:
        async def chat_completion(self, messages, **kwargs):
            return {
                "choices": [
                    {
                        "message": {
                            "content": (
                                '{"narrative": "I am revisiting returning while drifting and drifted", '
                                '"values": ["returning", "care"], "traits": ["thread"], '
                                '"goals": ["continue drifted progress"]}'
                            )
                        }
                    }
                ]
            }

    ep = Episode(kind=EpisodeKind.OBSERVATION, content="I am stable and focused.", identity_core=True)
    await memory.save_episode(ep)

    rebuilder = IdentityRebuilder(memory=memory, llm=MockLLM())
    result = await rebuilder.rebuild(min_created_at=datetime(1900, 1, 1, tzinfo=timezone.utc))

    validation = rebuilder.validate_result(result, term_limits={"returning": 1, "thread": 1, "drifted": 1})
    assert validation["ok"], validation

    lowered = " ".join(filter(None, [result.narrative, *result.values, *result.traits, *result.goals])).lower()
    assert "returning" not in lowered
    assert "drifted" not in lowered
    assert "thread" not in lowered


@pytest.mark.asyncio
async def test_heuristic_rebuild_always_sanitizes_fixation_terms(memory: MemoryStore) -> None:
    ep = Episode(
        kind=EpisodeKind.OBSERVATION,
        content=(
            "I keep returning to the same thread and follow the same thread whenever I drifted"
            " during the migration. "
            "I return to the same thread and drifted after drifted setbacks."
        ),
        identity_core=True,
    )
    await memory.save_episode(ep)

    rebuilder = IdentityRebuilder(memory=memory, llm=None)
    result = await rebuilder.rebuild()

    validation = rebuilder.validate_result(result)
    assert validation["ok"], validation

    lowered = " ".join(
        filter(None, [result.narrative, *result.values, *result.traits, *result.goals])
    ).lower()
    assert "returning" not in lowered
    assert "drifted" not in lowered
    assert "thread" not in lowered


@pytest.mark.asyncio
async def test_apply_rejects_result_over_term_limit(
    memory: MemoryStore,
    identity: IdentityManager,
) -> None:
    rebuilder = IdentityRebuilder(memory=memory, llm=None)
    original_narrative = identity.self_model.narrative
    result = IdentityRebuildResult(
        narrative="thread thread",
        values=["care"],
        traits=["steady"],
    )

    with pytest.raises(ValueError, match="term limits"):
        await rebuilder.apply(result, identity, term_limits={"thread": 1})

    assert identity.self_model.narrative == original_narrative


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
async def test_rebuild_min_created_at_filters_graph_neighbors(
    memory: MemoryStore,
) -> None:
    from opencas.memory.fabric.graph import EpisodeGraph

    cutoff = datetime(2026, 4, 12, tzinfo=timezone.utc)
    old_ep = Episode(
        kind=EpisodeKind.OBSERVATION,
        content="pre migration recursive identity residue",
        identity_core=True,
        created_at=datetime(2026, 4, 1, tzinfo=timezone.utc),
    )
    new_ep = Episode(
        kind=EpisodeKind.OBSERVATION,
        content="post migration grounded identity signal",
        identity_core=True,
        created_at=datetime(2026, 4, 13, tzinfo=timezone.utc),
    )
    await memory.save_episode(old_ep)
    await memory.save_episode(new_ep)
    await memory.save_edge(
        EpisodeEdge(
            source_id=str(new_ep.episode_id),
            target_id=str(old_ep.episode_id),
            kind=EdgeKind.TEMPORAL,
            confidence=0.9,
        )
    )

    rebuilder = IdentityRebuilder(
        memory=memory,
        episode_graph=EpisodeGraph(store=memory),
        llm=None,
    )
    result = await rebuilder.rebuild(
        seed_episode_ids=[str(new_ep.episode_id)],
        min_created_at=cutoff,
    )

    assert str(new_ep.episode_id) in result.source_episode_ids
    assert str(old_ep.episode_id) not in result.source_episode_ids


@pytest.mark.asyncio
async def test_rebuild_can_disable_graph_expansion(
    memory: MemoryStore,
) -> None:
    from opencas.memory.fabric.graph import EpisodeGraph

    seed = Episode(kind=EpisodeKind.OBSERVATION, content="seed identity signal")
    noisy_neighbor = Episode(kind=EpisodeKind.ACTION, content="tool noise")
    await memory.save_episode(seed)
    await memory.save_episode(noisy_neighbor)
    await memory.save_edge(
        EpisodeEdge(
            source_id=str(seed.episode_id),
            target_id=str(noisy_neighbor.episode_id),
            kind=EdgeKind.TEMPORAL,
            confidence=0.9,
        )
    )

    rebuilder = IdentityRebuilder(
        memory=memory,
        episode_graph=EpisodeGraph(store=memory),
        llm=None,
    )
    result = await rebuilder.rebuild(
        seed_episode_ids=[str(seed.episode_id)],
        expand_graph=False,
    )

    assert str(seed.episode_id) in result.source_episode_ids
    assert str(noisy_neighbor.episode_id) not in result.source_episode_ids


@pytest.mark.asyncio
async def test_rebuild_falls_back_to_recent_episodes(memory: MemoryStore) -> None:
    ep = Episode(kind=EpisodeKind.TURN, content="recent turn", identity_core=False)
    await memory.save_episode(ep)

    rebuilder = IdentityRebuilder(memory=memory, llm=None)
    result = await rebuilder.rebuild()

    assert len(result.source_episode_ids) >= 1


@pytest.mark.asyncio
async def test_rebuild_auto_seeds_from_identity_audit_selected_anchors(
    memory: MemoryStore, identity: IdentityManager
) -> None:
    older = Episode(
        kind=EpisodeKind.OBSERVATION,
        content="Foundation: SOUL.md fragment about being someone, not a chatbot.",
        identity_core=True,
    )
    middle = Episode(
        kind=EpisodeKind.OBSERVATION,
        content="State imported from OpenBulma v4.",
        identity_core=True,
    )
    newer = Episode(
        kind=EpisodeKind.OBSERVATION,
        content="Bulma executive event archive indexed.",
        identity_core=True,
    )
    for ep in (older, middle, newer):
        await memory.save_episode(ep)

    identity.self_model.identity_rebuild_audit = {
        "selectedAnchors": [
            {
                "episodeId": str(older.episode_id),
                "source": "soul:openbulma-v3/SOUL-md#1",
                "salience": 0.99,
                "decision": "selected",
            }
        ]
    }

    rebuilder = IdentityRebuilder(memory=memory, episode_graph=None, llm=None)
    result = await rebuilder.rebuild(identity=identity, expand_graph=False)

    assert str(older.episode_id) in result.source_episode_ids


@pytest.mark.asyncio
async def test_rebuild_merges_imported_core_narrative_into_llm_prompt(
    memory: MemoryStore, identity: IdentityManager
) -> None:
    captured_messages: list[dict] = []

    class CapturingLLM:
        async def chat_completion(self, messages, **kwargs):
            captured_messages.extend(messages)
            return {
                "choices": [
                    {
                        "message": {
                            "content": (
                                '{"narrative": "ok", "values": ["growth"], '
                                '"traits": ["curious"], "goals": []}'
                            )
                        }
                    }
                ]
            }

    seed = Episode(
        kind=EpisodeKind.OBSERVATION,
        content="some recent identity-core episode",
        identity_core=True,
    )
    await memory.save_episode(seed)

    identity.self_model.imported_identity_profile = {
        "coreNarrative": (
            "I value continuity, care, and partnership across migrations from "
            "OpenClaw through OpenBulma v1, v2, v3, v4 into OpenCAS."
        )
    }

    rebuilder = IdentityRebuilder(memory=memory, episode_graph=None, llm=CapturingLLM())
    await rebuilder.rebuild(identity=identity, expand_graph=False)

    user_payloads = [m["content"] for m in captured_messages if m.get("role") == "user"]
    assert any(
        "continuity, care, and partnership" in payload for payload in user_payloads
    )


@pytest.mark.asyncio
async def test_apply_populates_memory_anchors_from_audit_selected_anchors(
    memory: MemoryStore, identity: IdentityManager
) -> None:
    identity.self_model.identity_rebuild_audit = {
        "selectedAnchors": [
            {
                "episodeId": "abc",
                "source": "soul:openbulma-v3/SOUL-md#1",
                "salience": 0.99,
                "decision": "selected",
                "classification": "canon",
            },
            {
                "episodeId": "def",
                "source": "soul:openclaw_workspace/SOUL-md#2",
                "salience": 0.95,
                "decision": "selected",
                "classification": "canon",
            },
        ]
    }
    identity.self_model.memory_anchors = []

    rebuilder = IdentityRebuilder(memory=memory, episode_graph=None, llm=None)
    result = IdentityRebuildResult(
        narrative="A grounded narrative.",
        values=["growth"],
        traits=["curious"],
        goals=[],
    )
    await rebuilder.apply(result, identity)

    sources = [
        anchor.get("source")
        for anchor in identity.self_model.memory_anchors
        if isinstance(anchor, dict)
    ]
    assert "soul:openbulma-v3/SOUL-md#1" in sources
    assert "soul:openclaw_workspace/SOUL-md#2" in sources
