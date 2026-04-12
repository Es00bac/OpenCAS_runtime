"""Tests for ConsolidationCurationStore."""

import pytest
import pytest_asyncio
from datetime import datetime, timezone, timedelta
from opencas.consolidation import ConsolidationCurationStore


@pytest_asyncio.fixture
async def store(tmp_path):
    s = ConsolidationCurationStore(tmp_path / "curation.db")
    await s.connect()
    yield s
    await s.close()


@pytest.mark.asyncio
async def test_record_and_check_rejection(store):
    await store.record_rejection(
        "hash123",
        ["ep1", "ep2"],
        reason="empty_summary",
    )
    assert await store.is_rejected("hash123") is True
    assert await store.is_rejected("hash999") is False


@pytest.mark.asyncio
async def test_list_rejected(store):
    await store.record_rejection("h1", ["a"], "r1")
    await store.record_rejection("h2", ["b"], "r2")
    rejected = await store.list_rejected(limit=10)
    assert len(rejected) == 2
    hashes = {r.cluster_hash for r in rejected}
    assert hashes == {"h1", "h2"}


@pytest.mark.asyncio
async def test_list_rejected_since(store):
    old = datetime.now(timezone.utc) - timedelta(days=10)
    # Manually insert an old record
    await store._db.execute(
        "INSERT INTO rejected_merges VALUES (?, ?, ?, ?)",
        ("old_hash", "x", "old", old.isoformat()),
    )
    await store._db.commit()
    await store.record_rejection("new_hash", ["y"], "new")

    since = datetime.now(timezone.utc) - timedelta(days=1)
    rejected = await store.list_rejected(since=since)
    assert len(rejected) == 1
    assert rejected[0].cluster_hash == "new_hash"


@pytest.mark.asyncio
async def test_prune_old(store):
    old = datetime.now(timezone.utc) - timedelta(days=40)
    await store._db.execute(
        "INSERT INTO rejected_merges VALUES (?, ?, ?, ?)",
        ("old_hash", "x", "old", old.isoformat()),
    )
    await store._db.commit()
    await store.record_rejection("new_hash", ["y"], "new")

    pruned = await store.prune_old(max_age_days=30)
    assert pruned == 1
    assert await store.is_rejected("old_hash") is False
    assert await store.is_rejected("new_hash") is True
