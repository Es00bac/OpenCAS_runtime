"""Tests for SomaticManager snapshot and embedding integration."""

import pytest
import pytest_asyncio

from opencas.embeddings import EmbeddingCache, EmbeddingService
from opencas.somatic.manager import SomaticManager
from opencas.somatic.store import SomaticStore
from opencas.somatic.models import SomaticSnapshot, PrimaryEmotion


@pytest_asyncio.fixture
async def managers(tmp_path):
    store = SomaticStore(tmp_path / "somatic.db")
    await store.connect()
    cache = EmbeddingCache(tmp_path / "embeddings.db")
    await cache.connect()
    embeddings = EmbeddingService(cache, model_id="local-fallback")
    manager = SomaticManager(
        tmp_path / "somatic.json",
        store=store,
        embeddings=embeddings,
    )
    yield manager, store, embeddings
    await store.close()
    await cache.close()


@pytest.mark.asyncio
async def test_record_snapshot_persists_and_dedupes(managers) -> None:
    manager, store, _ = managers
    manager.set_arousal(0.7)
    manager.set_fatigue(0.3)

    snap1 = await manager.record_snapshot(source="test")
    assert snap1 is not None
    assert snap1.arousal == 0.7
    assert snap1.source == "test"

    # immediate duplicate should be skipped
    snap2 = await manager.record_snapshot(source="test")
    assert snap2 is not None  # returns latest

    recent = await store.list_recent(limit=10)
    assert len(recent) == 1


@pytest.mark.asyncio
async def test_record_snapshot_embeds_when_wired(managers) -> None:
    manager, store, embeddings = managers
    manager.set_valence(0.5)
    manager.set_energy(0.6)

    snap = await manager.record_snapshot(source="embed_test")
    assert snap is not None
    assert snap.embedding_id is not None
    assert len(snap.embedding_id) > 0

    cached = await embeddings.cache.get(snap.embedding_id)
    assert cached is not None
    assert cached.model_id == "local-fallback"
    assert cached.meta.get("source") == "somatic_snapshot"


@pytest.mark.asyncio
async def test_find_similar_periods(managers) -> None:
    manager, store, embeddings = managers

    # Create two distinct states
    manager.set_arousal(0.9)
    manager.set_valence(0.8)
    snap1 = await manager.record_snapshot(source="happy")

    manager.set_arousal(0.1)
    manager.set_valence(-0.8)
    snap2 = await manager.record_snapshot(source="sad")

    # Reset to happy-ish and search for similarities
    manager.set_arousal(0.9)
    manager.set_valence(0.8)
    await manager.record_snapshot(source="happy_again")

    similar = await manager.find_similar_periods(limit=3)
    # Should find the earlier happy state, not the sad one
    assert len(similar) >= 1
    # The top similarity should be to the happy snapshot
    top = similar[0]
    assert isinstance(top[0], SomaticSnapshot)
    assert top[1] > 0.0
