"""Tests for ConflictStore and DaydreamStore."""

from pathlib import Path
import pytest
import pytest_asyncio

from opencas.daydream import ConflictRecord, ConflictStore, DaydreamReflection, DaydreamStore


@pytest_asyncio.fixture
async def conflict_store(tmp_path: Path):
    store = ConflictStore(tmp_path / "conflicts.db")
    await store.connect()
    yield store
    await store.close()


@pytest_asyncio.fixture
async def daydream_store(tmp_path: Path):
    store = DaydreamStore(tmp_path / "daydreams.db")
    await store.connect()
    yield store
    await store.close()


@pytest.mark.asyncio
async def test_record_conflict(conflict_store: ConflictStore) -> None:
    record = ConflictRecord(kind="test_conflict", description="A test tension")
    saved = await conflict_store.record_conflict(record)
    assert saved.occurrence_count >= 1


@pytest.mark.asyncio
async def test_conflict_upsert(conflict_store: ConflictStore) -> None:
    record = ConflictRecord(kind="repeat", description="Same tension")
    await conflict_store.record_conflict(record)
    await conflict_store.record_conflict(record)
    active = await conflict_store.list_active_conflicts()
    matches = [r for r in active if r.kind == "repeat"]
    assert len(matches) == 1
    assert matches[0].occurrence_count == 2


@pytest.mark.asyncio
async def test_resolve_conflict(conflict_store: ConflictStore) -> None:
    record = ConflictRecord(kind="resolved", description="Goes away")
    saved = await conflict_store.record_conflict(record)
    await conflict_store.resolve_conflict(str(saved.conflict_id), auto=True)
    active = await conflict_store.list_active_conflicts()
    assert all(r.kind != "resolved" for r in active)


@pytest.mark.asyncio
async def test_auto_resolve_chronic(conflict_store: ConflictStore) -> None:
    record = ConflictRecord(kind="chronic", description="Old tension")
    saved = await conflict_store.record_conflict(record)
    # Manually backdate created_at by updating raw DB
    from datetime import datetime, timezone, timedelta
    old = (datetime.now(timezone.utc) - timedelta(days=15)).isoformat()
    await conflict_store._db.execute(
        "UPDATE conflicts SET created_at = ?, occurrence_count = 30 WHERE conflict_id = ?",
        (old, str(saved.conflict_id)),
    )
    await conflict_store._db.commit()
    resolved = await conflict_store.auto_resolve_chronic(threshold=25, min_days=10)
    assert resolved == 1


@pytest.mark.asyncio
async def test_save_and_list_reflections(daydream_store: DaydreamStore) -> None:
    reflection = DaydreamReflection(
        spark_content="A bright idea",
        synthesis="forward motion",
    )
    await daydream_store.save_reflection(reflection)
    recent = await daydream_store.list_recent(limit=1)
    assert len(recent) == 1
    assert recent[0].spark_content == "A bright idea"
    assert recent[0].synthesis == "forward motion"


@pytest.mark.asyncio
async def test_list_recent_order(daydream_store: DaydreamStore) -> None:
    r1 = DaydreamReflection(spark_content="first")
    r2 = DaydreamReflection(spark_content="second")
    await daydream_store.save_reflection(r1)
    await daydream_store.save_reflection(r2)
    recent = await daydream_store.list_recent(limit=2)
    assert recent[0].spark_content == "second"
    assert recent[1].spark_content == "first"
