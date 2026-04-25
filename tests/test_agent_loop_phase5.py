"""Integration tests for Phase 5 Theory of Mind & Identity wiring in AgentRuntime."""

import pytest
import pytest_asyncio

from opencas.bootstrap import BootstrapConfig, BootstrapPipeline
from opencas.identity import IdentityRebuildResult
from opencas.memory import Episode, EpisodeKind
from opencas.runtime.agent_loop import AgentRuntime
from opencas.tom import BeliefSubject, IntentionStatus


@pytest_asyncio.fixture
async def runtime(tmp_path_factory):
    config = BootstrapConfig(
        state_dir=tmp_path_factory.mktemp("state"),
        session_id="phase5-test",
    )
    ctx = await BootstrapPipeline(config).run()
    rt = AgentRuntime(ctx)
    await rt.tom.load()
    yield rt
    await rt._close_stores()


@pytest.mark.asyncio
async def test_tom_load_restores_beliefs(runtime: AgentRuntime) -> None:
    # Pre-seed a belief directly through the engine (store persists it)
    belief = await runtime.tom.record_belief(
        BeliefSubject.USER, "likes coffee", confidence=0.9
    )

    # Create a fresh engine pointing to the same store and load
    from opencas.tom import ToMEngine

    fresh_tom = ToMEngine(identity=runtime.ctx.identity, store=runtime.ctx.tom_store)
    await fresh_tom.load()

    assert len(fresh_tom.list_beliefs()) >= 1
    preds = {b.predicate for b in fresh_tom.list_beliefs()}
    assert "likes coffee" in preds


@pytest.mark.asyncio
async def test_tom_load_restores_intentions(runtime: AgentRuntime) -> None:
    await runtime.tom.record_intention(BeliefSubject.SELF, "write tests")
    await runtime.tom.resolve_intention("write tests", IntentionStatus.COMPLETED)

    from opencas.tom import ToMEngine

    fresh_tom = ToMEngine(identity=runtime.ctx.identity, store=runtime.ctx.tom_store)
    await fresh_tom.load()

    completed = fresh_tom.list_intentions(status=IntentionStatus.COMPLETED)
    assert any(i.content == "write tests" for i in completed)


@pytest.mark.asyncio
async def test_rebuild_identity_updates_self_model(runtime: AgentRuntime) -> None:
    # Seed an identity-core episode
    ep = Episode(
        kind=EpisodeKind.OBSERVATION,
        content="I am curious and want to learn python to help others.",
        identity_core=True,
    )
    await runtime.memory.save_episode(ep)

    result = await runtime.rebuild_identity()
    assert isinstance(result, dict)
    assert result.get("narrative") is not None
    assert len(result.get("source_episode_ids", [])) >= 1

    # Identity should have been updated
    assert runtime.ctx.identity.self_model.narrative is not None


@pytest.mark.asyncio
async def test_self_knowledge_registry_round_trip(runtime: AgentRuntime) -> None:
    # Record a high-confidence self-belief via ToM
    await runtime.tom.record_belief(BeliefSubject.SELF, "focused", confidence=0.8)

    # The registry should contain the entry
    registry = runtime.ctx.identity.registry
    assert registry is not None
    entry = registry.get("tom", "belief_focused")
    assert entry is not None
    assert entry.value["predicate"] == "focused"

    # Identity self_beliefs should also reflect it after save (nested from registry merge)
    assert "tom" in runtime.ctx.identity.self_model.self_beliefs
    assert "belief_focused" in runtime.ctx.identity.self_model.self_beliefs["tom"]
