"""Tests for the relational (musubi) engine."""

import pytest
import pytest_asyncio
from pathlib import Path

from opencas.memory import Episode, EpisodeKind
from opencas.relational import (
    MusubiStore,
    RelationalEngine,
    ResonanceDimension,
)


@pytest_asyncio.fixture
async def engine(tmp_path: Path):
    store = MusubiStore(tmp_path / "relational.db")
    rel = RelationalEngine(store)
    await rel.connect()
    yield rel
    await rel.close()


@pytest.mark.asyncio
async def test_connect_initializes_state(engine: RelationalEngine) -> None:
    assert engine.state is not None
    assert engine.state.musubi == 0.0


@pytest.mark.asyncio
async def test_initialize_sets_dimensions(engine: RelationalEngine) -> None:
    state = await engine.initialize(
        trust=0.5,
        resonance=0.2,
        presence=0.1,
        attunement=0.0,
        note="seed",
    )
    assert state.dimensions[ResonanceDimension.TRUST.value] == 0.5
    assert state.dimensions[ResonanceDimension.RESONANCE.value] == 0.2
    assert state.dimensions[ResonanceDimension.PRESENCE.value] == 0.1
    assert state.musubi == pytest.approx(0.3 * 0.5 + 0.25 * 0.2 + 0.25 * 0.1, abs=0.01)


@pytest.mark.asyncio
async def test_heartbeat_session_active_boosts_presence(engine: RelationalEngine) -> None:
    await engine.initialize(presence=0.0)
    state = await engine.heartbeat(session_active=True)
    assert state.dimensions[ResonanceDimension.PRESENCE.value] > 0.0


@pytest.mark.asyncio
async def test_heartbeat_absence_decays_presence(engine: RelationalEngine) -> None:
    await engine.initialize(presence=0.5)
    state = await engine.heartbeat(session_active=False)
    assert state.dimensions[ResonanceDimension.PRESENCE.value] < 0.5


@pytest.mark.asyncio
async def test_record_interaction_positive_somatic(engine: RelationalEngine) -> None:
    await engine.initialize()
    ep = Episode(
        kind=EpisodeKind.TURN,
        content="great idea",
        somatic_tag="joy",
    )
    state = await engine.record_interaction(ep, outcome="positive")
    assert state.dimensions[ResonanceDimension.ATTUNEMENT.value] > 0.0
    assert state.dimensions[ResonanceDimension.RESONANCE.value] > 0.0
    history = await engine.store.list_history(limit=10)
    assert len(history) >= 1


@pytest.mark.asyncio
async def test_record_interaction_negative_somatic(engine: RelationalEngine) -> None:
    await engine.initialize()
    ep = Episode(
        kind=EpisodeKind.TURN,
        content="bad result",
        somatic_tag="anger",
    )
    state = await engine.record_interaction(ep, outcome="negative")
    assert state.dimensions[ResonanceDimension.ATTUNEMENT.value] < 0.0
    assert state.dimensions[ResonanceDimension.RESONANCE.value] < 0.0


@pytest.mark.asyncio
async def test_record_creative_collab_success(engine: RelationalEngine) -> None:
    await engine.initialize()
    state = await engine.record_creative_collab(success=True)
    assert state.dimensions[ResonanceDimension.RESONANCE.value] > 0.0
    assert state.dimensions[ResonanceDimension.ATTUNEMENT.value] > 0.0


@pytest.mark.asyncio
async def test_record_boundary_respected(engine: RelationalEngine) -> None:
    await engine.initialize()
    state = await engine.record_boundary_respected(respected=True)
    assert state.dimensions[ResonanceDimension.TRUST.value] > 0.0


@pytest.mark.asyncio
async def test_record_boundary_violated(engine: RelationalEngine) -> None:
    await engine.initialize()
    state = await engine.record_boundary_respected(respected=False)
    assert state.dimensions[ResonanceDimension.TRUST.value] < 0.0


def test_derive_musubi_bounds() -> None:
    dims = {
        ResonanceDimension.TRUST.value: 2.0,
        ResonanceDimension.RESONANCE.value: 2.0,
        ResonanceDimension.PRESENCE.value: 2.0,
        ResonanceDimension.ATTUNEMENT.value: 2.0,
    }
    musubi = RelationalEngine._derive_musubi_from_dimensions(dims)
    assert musubi == 1.0

    dims = {
        ResonanceDimension.TRUST.value: -2.0,
        ResonanceDimension.RESONANCE.value: -2.0,
        ResonanceDimension.PRESENCE.value: -2.0,
        ResonanceDimension.ATTUNEMENT.value: -2.0,
    }
    musubi = RelationalEngine._derive_musubi_from_dimensions(dims)
    assert musubi == -1.0


def test_to_memory_salience_modifier_high_musubi_collab() -> None:
    store = MusubiStore(Path(":memory:"))  # sync test, no connect needed for engine logic
    rel = RelationalEngine(store)
    rel._state = rel._state or rel._state
    # Initialize state manually for sync test
    from opencas.relational import MusubiState

    rel._state = MusubiState(musubi=0.8)
    mod = rel.to_memory_salience_modifier(has_user_collab_tag=True)
    assert mod > 0.0


def test_to_memory_salience_modifier_low_musubi() -> None:
    store = MusubiStore(Path(":memory:"))
    rel = RelationalEngine(store)
    from opencas.relational import MusubiState

    rel._state = MusubiState(musubi=-0.8)
    mod = rel.to_memory_salience_modifier(has_user_collab_tag=False)
    assert mod < 0.0


def test_to_creative_boost() -> None:
    store = MusubiStore(Path(":memory:"))
    rel = RelationalEngine(store)
    from opencas.relational import MusubiState

    rel._state = MusubiState(musubi=0.8)
    boost = rel.to_creative_boost(aligns_with_shared_goals=True)
    assert boost > 0.0


def test_to_approval_risk_modifier() -> None:
    store = MusubiStore(Path(":memory:"))
    rel = RelationalEngine(store)
    from opencas.relational import MusubiState

    rel._state = MusubiState(musubi=0.8)
    mod = rel.to_approval_risk_modifier()
    assert mod > 0.0

    rel._state = MusubiState(musubi=-0.8)
    mod = rel.to_approval_risk_modifier()
    assert mod < 0.0
