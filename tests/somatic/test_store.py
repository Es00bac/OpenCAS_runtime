"""Tests for SomaticStore."""

from datetime import datetime, timezone

import pytest

from opencas.somatic.store import SomaticStore
from opencas.somatic.models import SomaticSnapshot, PrimaryEmotion


@pytest.mark.asyncio
async def test_store_saves_and_retrieves_latest(tmp_path) -> None:
    store = SomaticStore(tmp_path / "somatic.db")
    await store.connect()

    snap = SomaticSnapshot(
        arousal=0.8,
        fatigue=0.2,
        tension=0.5,
        valence=0.3,
        primary_emotion=PrimaryEmotion.ANTICIPATION,
        source="test",
    )
    await store.save(snap)

    latest = await store.get_latest()
    assert latest is not None
    assert round(latest.arousal, 3) == 0.8
    assert latest.primary_emotion == PrimaryEmotion.ANTICIPATION
    assert latest.source == "test"

    await store.close()


@pytest.mark.asyncio
async def test_store_lists_recent(tmp_path) -> None:
    store = SomaticStore(tmp_path / "somatic.db")
    await store.connect()

    for i in range(3):
        await store.save(
            SomaticSnapshot(
                arousal=0.1 * i,
                fatigue=0.0,
                tension=0.0,
                valence=0.0,
                source=f"s{i}",
            )
        )

    recent = await store.list_recent(limit=2)
    assert len(recent) == 2
    # descending by recorded_at
    assert recent[0].source == "s2"
    assert recent[1].source == "s1"

    await store.close()


@pytest.mark.asyncio
async def test_store_trajectory_range(tmp_path) -> None:
    store = SomaticStore(tmp_path / "somatic.db")
    await store.connect()

    t0 = datetime(2026, 1, 1, 12, 0, 0, tzinfo=timezone.utc)
    t1 = datetime(2026, 1, 1, 13, 0, 0, tzinfo=timezone.utc)
    t2 = datetime(2026, 1, 1, 14, 0, 0, tzinfo=timezone.utc)

    await store.save(SomaticSnapshot(recorded_at=t0, source="a"))
    await store.save(SomaticSnapshot(recorded_at=t1, source="b"))
    await store.save(SomaticSnapshot(recorded_at=t2, source="c"))

    traj = await store.trajectory(start=t0, end=t1)
    assert len(traj) == 2
    assert traj[0].source == "a"
    assert traj[1].source == "b"

    await store.close()


@pytest.mark.asyncio
async def test_store_save_batch(tmp_path) -> None:
    store = SomaticStore(tmp_path / "somatic.db")
    await store.connect()

    snaps = [
        SomaticSnapshot(arousal=0.1, source="batch1"),
        SomaticSnapshot(arousal=0.2, source="batch2"),
    ]
    await store.save_batch(snaps)

    recent = await store.list_recent(limit=10)
    assert len(recent) == 2
    sources = {s.source for s in recent}
    assert sources == {"batch1", "batch2"}

    await store.close()
