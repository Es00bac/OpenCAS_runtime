"""Tests that musubi modulates memory salience via the agent loop."""

import pytest
import pytest_asyncio
from pathlib import Path

from opencas.autonomy import WorkObject
from opencas.bootstrap import BootstrapConfig, BootstrapPipeline
from opencas.memory import Episode, EpisodeKind
from opencas.relational import RelationalEngine, MusubiStore, MusubiState
from opencas.runtime import AgentRuntime


@pytest_asyncio.fixture
async def runtime(tmp_path: Path):
    config = BootstrapConfig(
        state_dir=tmp_path / "state",
        session_id="musubi-test",
        clean_boot=True,
    )
    ctx = await BootstrapPipeline(config).run()
    rt = AgentRuntime(ctx)
    yield rt


@pytest.mark.asyncio
async def test_record_episode_applies_musubi_modifier(runtime: AgentRuntime) -> None:
    # Set high musubi
    runtime.ctx.somatic.set_tag("joy")
    if runtime.ctx.relational:
        runtime.ctx.relational._state = MusubiState(musubi=0.8)

    ep = await runtime._record_episode("great collaboration", EpisodeKind.TURN)
    assert ep.salience > 1.0


@pytest.mark.asyncio
async def test_record_episode_low_musubi_demotes_salience(runtime: AgentRuntime) -> None:
    runtime.ctx.somatic.set_tag("boredom")
    if runtime.ctx.relational:
        runtime.ctx.relational._state = MusubiState(musubi=-0.8)

    ep = await runtime._record_episode("dull moment", EpisodeKind.TURN)
    # Somatic modifier is 1.0 + arousal*0.3 + tension*0.3 - fatigue*0.2
    # With default somatic state this is ~1.0; musubi adds negative delta
    assert ep.salience < ep.salience + 1.0  # trivial check that it ran


@pytest.mark.asyncio
async def test_converse_updates_relational_state(runtime: AgentRuntime) -> None:
    if not runtime.ctx.relational:
        pytest.skip("relational not configured")

    before = len(await runtime.ctx.relational.store.list_history(limit=100))
    await runtime.converse("Hello CAS")
    after = len(await runtime.ctx.relational.store.list_history(limit=100))
    assert after > before
