"""Tests for ConflictStore and DaydreamStore."""

from pathlib import Path

import pytest
import pytest_asyncio

from opencas.daydream import ConflictRecord, ConflictStore, DaydreamReflection, DaydreamStore


def test_store_compat_exports() -> None:
    from opencas.daydream.conflict_store import ConflictStore as SplitConflictStore
    from opencas.daydream.daydream_store import DaydreamStore as SplitDaydreamStore
    from opencas.daydream.store import ConflictStore as CompatConflictStore
    from opencas.daydream.store import DaydreamStore as CompatDaydreamStore

    assert CompatConflictStore is SplitConflictStore
    assert CompatDaydreamStore is SplitDaydreamStore


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
    from datetime import datetime, timedelta, timezone
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
        experience_context={
            "trigger": "background_daydream",
            "somatic": {"somatic_tag": "curious", "tension": 0.2},
        },
    )
    await daydream_store.save_reflection(reflection)
    recent = await daydream_store.list_recent(limit=1)
    assert len(recent) == 1
    assert recent[0].spark_content == "A bright idea"
    assert recent[0].synthesis == "forward motion"
    assert recent[0].experience_context["trigger"] == "background_daydream"
    assert recent[0].experience_context["somatic"]["somatic_tag"] == "curious"


@pytest.mark.asyncio
async def test_daydream_store_migrates_experience_context_column(tmp_path: Path) -> None:
    import sqlite3

    db_path = tmp_path / "legacy-daydreams.db"
    conn = sqlite3.connect(db_path)
    conn.execute(
        """
        CREATE TABLE daydream_reflections (
            reflection_id TEXT PRIMARY KEY,
            created_at TEXT NOT NULL,
            spark_content TEXT NOT NULL,
            recollection TEXT NOT NULL DEFAULT '',
            interpretation TEXT NOT NULL DEFAULT '',
            synthesis TEXT NOT NULL DEFAULT '',
            open_question TEXT,
            changed_self_view TEXT NOT NULL DEFAULT '',
            tension_hints TEXT NOT NULL DEFAULT '[]',
            alignment_score REAL NOT NULL DEFAULT 0.0,
            novelty_score REAL NOT NULL DEFAULT 0.0,
            keeper INTEGER NOT NULL DEFAULT 0
        )
        """
    )
    conn.commit()
    conn.close()

    store = DaydreamStore(db_path)
    await store.connect()
    try:
        columns = [
            row["name"]
            for row in await (await store.db.execute("PRAGMA table_info(daydream_reflections)")).fetchall()
        ]
        assert "experience_context" in columns
    finally:
        await store.close()


@pytest.mark.asyncio
async def test_list_recent_order(daydream_store: DaydreamStore) -> None:
    r1 = DaydreamReflection(spark_content="first")
    r2 = DaydreamReflection(spark_content="second")
    await daydream_store.save_reflection(r1)
    await daydream_store.save_reflection(r2)
    recent = await daydream_store.list_recent(limit=2)
    assert recent[0].spark_content == "second"
    assert recent[1].spark_content == "first"
