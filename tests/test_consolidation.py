"""Tests for NightlyConsolidationEngine."""

import pytest
import pytest_asyncio
from unittest.mock import AsyncMock, MagicMock

from opencas.consolidation import NightlyConsolidationEngine
from opencas.embeddings import EmbeddingCache, EmbeddingService
from opencas.identity import IdentityManager, IdentityStore
from opencas.memory import Episode, EpisodeKind, Memory, MemoryStore


@pytest_asyncio.fixture
async def deps(tmp_path):
    mem_store = MemoryStore(tmp_path / "memory.db")
    await mem_store.connect()

    cache = EmbeddingCache(tmp_path / "embeddings.db")
    await cache.connect()
    embed_service = EmbeddingService(cache=cache, model_id="local-fallback")

    id_store = IdentityStore(tmp_path / "identity")
    identity = IdentityManager(id_store)
    identity.load()

    mgr = MagicMock()
    resolved = MagicMock()
    resolved.provider_id = "test-provider"
    resolved.model_id = "test-model"
    resolved.provider = MagicMock()
    resolved.provider.chat_completion = AsyncMock(
        return_value={"choices": [{"message": {"content": "Cluster summary"}}]}
    )
    mgr.resolve.return_value = resolved
    from opencas.api import LLMClient
    llm = LLMClient(mgr, default_model="test/model")

    engine = NightlyConsolidationEngine(
        memory=mem_store,
        embeddings=embed_service,
        llm=llm,
        identity=identity,
    )
    yield mem_store, embed_service, identity, engine
    await mem_store.close()
    await cache.close()


@pytest.mark.asyncio
async def test_consolidation_empty(deps):
    _store, _embeds, _identity, engine = deps
    result = await engine.run()
    assert result.candidate_episodes == 0
    assert result.memories_created == 0


@pytest.mark.asyncio
async def test_consolidation_clusters_and_creates_memories(deps):
    store, _embeds, _identity, engine = deps
    # Create episodes with similar content so they cluster
    for i in range(5):
        ep = Episode(kind=EpisodeKind.OBSERVATION, content=f"fact about planets {i}")
        await store.save_episode(ep)
    for i in range(3):
        ep = Episode(kind=EpisodeKind.OBSERVATION, content=f"unrelated math {i}")
        await store.save_episode(ep)

    result = await engine.run(similarity_threshold=0.5)
    assert result.candidate_episodes == 8
    assert result.clusters_formed >= 1
    assert result.memories_created >= 1

    memories = await store.list_memories(limit=10)
    assert len(memories) >= 1
    assert any("consolidation" in m.tags for m in memories)


@pytest.mark.asyncio
async def test_consolidation_reweights_salience(deps):
    store, _embeds, _identity, engine = deps
    # Save a memory with high access count
    mem = Memory(content="important fact", access_count=10, salience=1.0)
    await store.save_memory(mem)

    result = await engine.run()
    assert len(result.salience_updates) >= 1
    updated = [u for u in result.salience_updates if u.memory_id == str(mem.memory_id)]
    assert updated
    assert updated[0].new_salience > updated[0].old_salience


@pytest.mark.asyncio
async def test_consolidation_prunes_low_salience(deps):
    store, _embeds, _identity, engine = deps
    # Save a very low-salience episode
    ep = Episode(kind=EpisodeKind.OBSERVATION, content="noise", salience=0.1)
    await store.save_episode(ep)

    result = await engine.run(salience_threshold=0.5)
    assert result.episodes_pruned >= 1

    remaining = await store.list_episodes(limit=100)
    assert all(ep.content != "noise" or ep.salience >= 0.5 for ep in remaining)


@pytest.mark.asyncio
async def test_consolidation_updates_identity(deps):
    store, _embeds, identity, engine = deps
    for i in range(4):
        ep = Episode(kind=EpisodeKind.OBSERVATION, content=f"theme {i}")
        await store.save_episode(ep)

    result = await engine.run(similarity_threshold=0.1)
    assert result.identity_updates
    key = list(result.identity_updates.keys())[0]
    assert "consolidation_themes" in key
    assert identity.self_model.self_beliefs.get(key)


@pytest.mark.asyncio
async def test_consolidation_skips_previously_rejected(deps, tmp_path):
    store, _embeds, _identity, engine = deps
    from opencas.consolidation import ConsolidationCurationStore
    curation = ConsolidationCurationStore(tmp_path / "curation_test.db")
    await curation.connect()
    engine.curation_store = curation

    # Create similar episodes that will cluster
    for i in range(3):
        ep = Episode(kind=EpisodeKind.OBSERVATION, content=f"same theme repeated {i}")
        await store.save_episode(ep)

    # First run should create a memory
    result1 = await engine.run(similarity_threshold=0.1)
    assert result1.memories_created >= 1

    # Create another batch that clusters to the same episodes... actually the episodes are fixed.
    # Instead, mark the cluster as rejected and re-run on the same episodes.
    # The easiest way: create new episodes, get their cluster hash, reject it, then run again.
    from opencas.consolidation.engine import NightlyConsolidationEngine
    episodes = await store.list_non_compacted_episodes(limit=100)
    cluster_hash = engine._cluster_hash(episodes[:3])
    await curation.record_rejection(cluster_hash, [str(e.episode_id) for e in episodes[:3]], "test")

    # Running again with the same store should skip the rejected cluster
    result2 = await engine.run(similarity_threshold=0.1)
    assert result2.merges_rejected >= 1

    await curation.close()


@pytest.mark.asyncio
async def test_consolidation_promotes_strong_signals(deps):
    store, _embeds, identity, engine = deps
    # Seed identity goals to boost identity relevance
    identity.update_self_belief("current_goals", ["learn astronomy"])

    # Create one lone high-salience episode that likely won't cluster
    ep = Episode(
        kind=EpisodeKind.OBSERVATION,
        content="I learned that Jupiter has 95 moons and this fascinates me deeply",
        salience=8.5,
    )
    await store.save_episode(ep)

    result = await engine.run(signal_threshold=0.3)
    assert result.signals_promoted >= 1

    memories = await store.list_memories(limit=10)
    strong = [m for m in memories if "strong_signal" in m.tags]
    assert len(strong) >= 1
    assert str(ep.episode_id) in strong[0].source_episode_ids


@pytest.mark.asyncio
async def test_consolidation_no_duplicate_signal_when_clustered(deps):
    store, _embeds, _identity, engine = deps
    # Create a batch of similar episodes that will cluster
    for i in range(3):
        ep = Episode(kind=EpisodeKind.OBSERVATION, content=f"daily running habit progress update {i}", salience=7.0)
        await store.save_episode(ep)

    result = await engine.run(similarity_threshold=0.1, signal_threshold=0.1)
    # They should cluster and not also be promoted as individual signals
    assert result.memories_created >= 1
    memories = await store.list_memories(limit=10)
    strong_signals = [m for m in memories if "strong_signal" in m.tags]
    cluster_memories = [m for m in memories if m.tags == ["consolidation"]]
    # No strong_signal memories should come from clustered episodes
    for sm in strong_signals:
        for cm in cluster_memories:
            assert not set(sm.source_episode_ids).intersection(set(cm.source_episode_ids))


@pytest.mark.asyncio
async def test_consolidation_reweights_episode_salience(deps):
    store, _embeds, _identity, engine = deps
    # Create a high-salience episode that should be promoted
    ep = Episode(kind=EpisodeKind.OBSERVATION, content="critical insight about system design", salience=8.0)
    await store.save_episode(ep)

    old_salience = ep.salience
    result = await engine.run(signal_threshold=0.1)
    assert result.episode_salience_updates >= 1

    updated_ep = await store.get_episode(str(ep.episode_id))
    assert updated_ep is not None
    assert updated_ep.salience != old_salience


from datetime import datetime, timezone, timedelta

@pytest.mark.asyncio
async def test_consolidation_recovers_orphans(deps):
    store, embeds, _identity, engine = deps
    # Create episodes far apart in time with different kinds so bridge affinity is low.
    # Temporal path affinity = 1/(1+10) ≈ 0.09, cross_source_echo = 0.3 -> avg ≈ 0.195 < 0.3
    now = datetime.now(timezone.utc)
    ep1 = Episode(
        kind=EpisodeKind.OBSERVATION,
        content="quantum computing advances",
        created_at=now - timedelta(days=300),
    )
    ep2 = Episode(
        kind=EpisodeKind.ACTION,
        content="medieval history of pottery",
        created_at=now,
    )
    for ep in [ep1, ep2]:
        rec = await embeds.embed(ep.content)
        ep.embedding_id = rec.source_hash
        await store.save_episode(ep)

    # Remove any pre-existing edges
    await store.delete_edges_for(str(ep1.episode_id))
    await store.delete_edges_for(str(ep2.episode_id))

    result = await engine.run(similarity_threshold=0.99, signal_threshold=1.0)
    assert result.orphans_recovered >= 1

    # At least one orphan should now have an edge
    edges1 = await store.get_edges_for(str(ep1.episode_id), min_confidence=0.0, limit=1)
    edges2 = await store.get_edges_for(str(ep2.episode_id), min_confidence=0.0, limit=1)
    assert len(edges1) >= 1 or len(edges2) >= 1


@pytest.mark.asyncio
async def test_sweep_belief_consistency_no_tom_store(deps):
    _store, _embeds, _identity, engine = deps
    assert engine.tom_store is None
    decayed = await engine._sweep_belief_consistency()
    assert decayed == 0


@pytest.mark.asyncio
async def test_sweep_belief_consistency_decays_stale_beliefs(deps, tmp_path):
    store, _embeds, _identity, engine = deps
    from opencas.tom import Belief, BeliefSubject, TomStore

    tom_store = TomStore(tmp_path / "tom_sweep.db")
    await tom_store.connect()
    engine.tom_store = tom_store

    # Create a recent episode and an old episode
    now = datetime.now(timezone.utc)
    recent_ep = Episode(kind=EpisodeKind.OBSERVATION, content="recent event", created_at=now)
    old_ep = Episode(kind=EpisodeKind.OBSERVATION, content="old event", created_at=now - timedelta(days=30))
    await store.save_episode(recent_ep)
    await store.save_episode(old_ep)

    # Belief with high confidence but only old evidence -> should decay
    stale_belief = Belief(
        subject=BeliefSubject.USER,
        predicate="likes stale thing",
        confidence=0.9,
        evidence_ids=[str(old_ep.episode_id)],
        belief_revision_score=0.5,
    )
    await tom_store.save_belief(stale_belief)

    # Belief with high confidence and recent evidence -> should NOT decay
    fresh_belief = Belief(
        subject=BeliefSubject.USER,
        predicate="likes fresh thing",
        confidence=0.9,
        evidence_ids=[str(recent_ep.episode_id)],
        belief_revision_score=0.5,
    )
    await tom_store.save_belief(fresh_belief)

    # Belief below confidence threshold -> should NOT decay
    low_conf_belief = Belief(
        subject=BeliefSubject.USER,
        predicate="maybe likes something",
        confidence=0.5,
        evidence_ids=[str(old_ep.episode_id)],
        belief_revision_score=0.5,
    )
    await tom_store.save_belief(low_conf_belief)

    result = await engine.run()
    assert result.beliefs_decayed >= 1

    # Verify the stale belief was decayed
    beliefs = await tom_store.list_beliefs()
    stale = next(b for b in beliefs if b.predicate == "likes stale thing")
    assert stale.confidence < 0.9
    assert stale.belief_revision_score < 0.5

    # Verify the fresh belief was untouched
    fresh = next(b for b in beliefs if b.predicate == "likes fresh thing")
    assert fresh.confidence == 0.9
    assert fresh.belief_revision_score == 0.5

    await tom_store.close()


@pytest.mark.asyncio
async def test_sweep_belief_consistency_no_evidence_decays(deps, tmp_path):
    store, _embeds, _identity, engine = deps
    from opencas.tom import Belief, BeliefSubject, TomStore

    tom_store = TomStore(tmp_path / "tom_no_evidence.db")
    await tom_store.connect()
    engine.tom_store = tom_store

    # Belief with high confidence but no evidence at all -> should decay
    belief = Belief(
        subject=BeliefSubject.SELF,
        predicate="is always correct",
        confidence=0.85,
        evidence_ids=[],
        belief_revision_score=0.3,
    )
    await tom_store.save_belief(belief)

    decayed = await engine._sweep_belief_consistency(confidence_threshold=0.8, decay_factor=0.8)
    assert decayed == 1

    updated = await tom_store.list_beliefs()
    assert updated[0].confidence == round(0.85 * 0.8, 3)
    assert updated[0].belief_revision_score == round(0.3 - 0.1, 3)

    await tom_store.close()
