"""Tests for the relational (musubi) store."""

import sqlite3

import pytest
import pytest_asyncio
from pathlib import Path

from opencas.relational import (
    DirectionalAttribution,
    MusubiRecord,
    MusubiState,
    MusubiStore,
    MutualAcknowledgment,
)


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
async def test_directional_attribution_persists_for_state_and_history(
    store: MusubiStore,
) -> None:
    acknowledgment = MutualAcknowledgment(
        agent_leaning_in=True,
        user_leaning_in=True,
        boundary_note="both sides explicitly acknowledged",
    )
    state = MusubiState(
        musubi=0.99,
        directional_attribution=DirectionalAttribution.USER,
        mutual_acknowledgment=acknowledgment,
    )
    await store.save_state(state)

    loaded = await store.load_state()
    assert loaded is not None
    assert loaded.directional_attribution == DirectionalAttribution.USER
    assert loaded.mutual_acknowledgment.shared_boundary_held is True
    assert loaded.mutual_acknowledgment.boundary_note == "both sides explicitly acknowledged"

    record = MusubiRecord(
        musubi_before=0.5,
        musubi_after=0.99,
        delta=0.49,
        trigger_event="user_acknowledged_boundary",
        directional_attribution=DirectionalAttribution.USER,
        mutual_acknowledgment_snapshot=acknowledgment,
    )
    await store.append_record(record)

    history = await store.list_history(limit=1)
    assert history[0].directional_attribution == DirectionalAttribution.USER
    assert history[0].mutual_acknowledgment_snapshot is not None
    assert history[0].mutual_acknowledgment_snapshot.shared_boundary_held is True


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


@pytest.mark.asyncio
async def test_connect_migrates_older_musubi_schema(tmp_path: Path) -> None:
    db_path = tmp_path / "old-relational.db"
    conn = sqlite3.connect(db_path)
    conn.executescript(
        """
        CREATE TABLE musubi_state (
            state_id TEXT PRIMARY KEY,
            updated_at TEXT NOT NULL,
            musubi REAL NOT NULL DEFAULT 0.0,
            dimensions TEXT NOT NULL DEFAULT '{}',
            continuity_breadcrumb TEXT NOT NULL DEFAULT '',
            source_tag TEXT
        );
        CREATE TABLE musubi_history (
            record_id TEXT PRIMARY KEY,
            created_at TEXT NOT NULL,
            musubi_before REAL NOT NULL DEFAULT 0.0,
            musubi_after REAL NOT NULL DEFAULT 0.0,
            delta REAL NOT NULL DEFAULT 0.0,
            dimension_deltas TEXT NOT NULL DEFAULT '{}',
            trigger_event TEXT NOT NULL,
            episode_id TEXT,
            note TEXT,
            continuity_breadcrumb TEXT NOT NULL DEFAULT ''
        );
        INSERT INTO musubi_state (
            state_id, updated_at, musubi, dimensions, continuity_breadcrumb, source_tag
        ) VALUES (
            '11111111-1111-1111-1111-111111111111', '2026-04-23T00:00:00+00:00', 0.4, '{}', '', 'legacy'
        );
        """
    )
    conn.commit()
    conn.close()

    store = await MusubiStore(db_path).connect()
    try:
        loaded = await store.load_state()
        assert loaded is not None
        assert loaded.directional_attribution == DirectionalAttribution.UNKNOWN
        assert loaded.mutual_acknowledgment.shared_boundary_held is False

        await store.save_state(
            loaded.model_copy(update={"directional_attribution": DirectionalAttribution.AGENT})
        )
        reloaded = await store.load_state()
        assert reloaded is not None
        assert reloaded.directional_attribution == DirectionalAttribution.AGENT

        await store.append_record(
            MusubiRecord(
                trigger_event="migrated_write",
                directional_attribution=DirectionalAttribution.USER,
            )
        )
        history = await store.list_history(limit=1)
        assert history[0].directional_attribution == DirectionalAttribution.USER
    finally:
        await store.close()
