from __future__ import annotations

from datetime import datetime, timedelta, timezone

import pytest

from opencas.memory import MemoryStore
from opencas.memory.autobiography import SessionAnchor, SessionAnchorStore
from opencas.memory.models import Episode, EpisodeKind
from scripts.backfill_session_anchors import backfill_session_anchors


@pytest.mark.asyncio
async def test_idempotent_upsert_and_evidence_hash_invalidation(tmp_path) -> None:
    memory = await MemoryStore(tmp_path / "memory.db").connect()
    assert memory._db is not None
    store = SessionAnchorStore(memory._db)

    created = datetime(2026, 5, 5, 12, tzinfo=timezone.utc)
    await store.upsert(
        SessionAnchor(
            session_id="s1",
            created_at=created,
            ended_at=created,
            duration_s=0,
            agent_name="OpenCAS",
            episode_kind_tally={"action": 1},
            evidence_episode_ids=["e1"],
            evidence_hash="hash-1",
            evidence_strength="high",
        )
    )
    await store.update_gist("s1", "I wrote a file.", "high", "hash-1")

    await store.upsert(
        SessionAnchor(
            session_id="s1",
            created_at=created,
            ended_at=created + timedelta(minutes=1),
            duration_s=60,
            agent_name="OpenCAS",
            episode_kind_tally={"action": 2},
            evidence_episode_ids=["e1", "e2"],
            evidence_hash="hash-2",
            evidence_strength="high",
        )
    )

    anchor = await store.get("s1")
    assert anchor is not None
    assert anchor.duration_s == 60
    assert anchor.evidence_episode_ids == ["e1", "e2"]
    assert anchor.gist is None
    assert anchor.gist_version == 0
    assert anchor.evidence_hash == "hash-2"

    missing = await store.list_missing_for_sessions(["s1", "s2"])
    assert missing == ["s2"]

    await store.upsert(
        SessionAnchor(
            session_id="s1",
            created_at=created,
            ended_at=created + timedelta(seconds=30),
            duration_s=30,
            agent_name="OpenCAS",
            episode_kind_tally={"action": 1},
            evidence_episode_ids=["e1"],
            evidence_hash="hash-3",
            evidence_strength="high",
        )
    )
    anchor = await store.get("s1")
    assert anchor is not None
    assert anchor.duration_s == 60
    assert anchor.ended_at == created + timedelta(minutes=1)

    await memory.close()


@pytest.mark.asyncio
async def test_list_recent_and_skeletons(tmp_path) -> None:
    memory = await MemoryStore(tmp_path / "memory.db").connect()
    assert memory._db is not None
    store = SessionAnchorStore(memory._db)
    now = datetime(2026, 5, 5, 12, tzinfo=timezone.utc)

    await store.upsert(
        SessionAnchor(session_id="old", created_at=now - timedelta(days=2), agent_name="OpenCAS")
    )
    await store.upsert(
        SessionAnchor(session_id="new", created_at=now, agent_name="OpenCAS")
    )
    await store.update_gist("new", "I handled a session.", "medium", "hash-new")

    recent = await store.list_recent(now - timedelta(days=1), limit=10)
    assert [item.session_id for item in recent] == ["new"]

    skeletons = await store.list_skeletons(limit=10)
    assert [item.session_id for item in skeletons] == ["old"]

    await memory.close()


@pytest.mark.asyncio
async def test_backfill_script_idempotent(tmp_path) -> None:
    memory_path = tmp_path / "memory.db"
    memory = await MemoryStore(memory_path).connect()
    assert memory._db is not None
    await memory.save_episode(
        Episode(
            session_id="s1",
            kind=EpisodeKind.TURN,
            content="A remembered session.",
        )
    )
    await memory.save_episode(
        Episode(
            session_id="s2",
            kind=EpisodeKind.ACTION,
            content="tool fs_write_file path=/mnt/xtra/OpenCAS/workspace/a.md bytes=1 checksum=abc",
            payload={"tool_name": "fs_write_file"},
        )
    )
    await memory.close()

    first = await backfill_session_anchors(memory_path)
    second = await backfill_session_anchors(memory_path)

    assert first.total_sessions == 2
    assert first.anchors_created == 2
    assert first.anchors_skipped == 0
    assert second.total_sessions == 2
    assert second.anchors_created == 0
    assert second.anchors_skipped == 2
