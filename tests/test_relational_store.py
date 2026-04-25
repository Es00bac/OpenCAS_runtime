"""Tests for the relational (musubi) store."""

import pytest
import pytest_asyncio
from pathlib import Path

from opencas.relational import MusubiStore, MusubiState, MusubiRecord


@pytest_asyncio.fixture
async def store(tmp_path: Path):
    s = MusubiStore(tmp_path / "relational.db")
    await s.connect()
    yield s
    await s.close()


@pytest.mark.asyncio
async def test_save_and_load_state(store: MusubiStore) -> None:
    state = MusubiState(musubi=0.5)
    state.dimensions["trust"] = 0.5
    await store.save_state(state)
    loaded = await store.load_state()
    assert loaded is not None
    assert loaded.musubi == 0.5
    assert loaded.dimensions["trust"] == 0.5


@pytest.mark.asyncio
async def test_state_overwrite(store: MusubiStore) -> None:
    s1 = MusubiState(musubi=0.2)
    s2 = MusubiState(musubi=0.8)
    await store.save_state(s1)
    await store.save_state(s2)
    loaded = await store.load_state()
    assert loaded is not None
    assert loaded.musubi == 0.8


@pytest.mark.asyncio
async def test_append_record_and_list_history(store: MusubiStore) -> None:
    state = MusubiState()
    await store.save_state(state)
    record = MusubiRecord(
        musubi_before=0.0,
        musubi_after=0.1,
        delta=0.1,
        dimension_deltas={"trust": 0.1},
        trigger_event="test",
        note="test note",
    )
    await store.append_record(record)
    history = await store.list_history(limit=10)
    assert len(history) == 1
    assert history[0].trigger_event == "test"
    assert history[0].note == "test note"


@pytest.mark.asyncio
async def test_list_history_limit(store: MusubiStore) -> None:
    state = MusubiState()
    await store.save_state(state)
    for i in range(5):
        record = MusubiRecord(
            musubi_before=0.0,
            musubi_after=0.1,
            delta=0.1,
            dimension_deltas={},
            trigger_event=f"test_{i}",
        )
        await store.append_record(record)
    history = await store.list_history(limit=2)
    assert len(history) == 2
    # Most recent first
    assert history[0].trigger_event == "test_4"
    assert history[1].trigger_event == "test_3"


@pytest.mark.asyncio
async def test_list_history_offset(store: MusubiStore) -> None:
    state = MusubiState()
    await store.save_state(state)
    for i in range(3):
        record = MusubiRecord(
            musubi_before=0.0,
            musubi_after=0.1,
            delta=0.1,
            dimension_deltas={},
            trigger_event=f"test_{i}",
        )
        await store.append_record(record)
    history = await store.list_history(limit=1, offset=1)
    assert len(history) == 1
    assert history[0].trigger_event == "test_1"
