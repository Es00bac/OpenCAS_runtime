from __future__ import annotations

from datetime import datetime, timedelta, timezone

import pytest

from opencas.memory import CompactionRecord, Episode, EpisodeKind, MemoryStore
from opencas.memory.autobiography import (
    SessionAnchorStore,
    SessionAutobiographyComposer,
)
from opencas.somatic.models import AffectState, PrimaryEmotion


@pytest.mark.asyncio
async def test_skeleton_anchor_empty_session(tmp_path) -> None:
    memory = await MemoryStore(tmp_path / "memory.db").connect()
    assert memory._db is not None
    store = SessionAnchorStore(memory._db)
    composer = SessionAutobiographyComposer(memory_store=memory, anchor_store=store)

    anchor = await composer.compose_skeleton("empty")

    assert anchor.session_id == "empty"
    assert anchor.episode_kind_tally == {}
    assert anchor.evidence_episode_ids == []
    assert anchor.gist is None
    assert anchor.evidence_strength == "low"

    await memory.close()


@pytest.mark.asyncio
async def test_skeleton_anchor_compacted_session(tmp_path) -> None:
    memory = await MemoryStore(tmp_path / "memory.db").connect()
    assert memory._db is not None
    store = SessionAnchorStore(memory._db)
    created = datetime(2026, 5, 5, 12, tzinfo=timezone.utc)
    episode = Episode(
        created_at=created,
        kind=EpisodeKind.TURN,
        session_id="s1",
        content="I compacted this earlier.",
        compacted=True,
        payload={"role": "assistant"},
    )
    await memory.save_episode(episode)
    record = CompactionRecord(
        episode_ids=[str(episode.episode_id)],
        summary="Assistant compressed prior context.",
        removed_count=1,
    )
    await memory.record_compaction(record)

    composer = SessionAutobiographyComposer(memory_store=memory, anchor_store=store)
    anchor = await composer.compose_skeleton("s1")

    assert str(episode.episode_id) in anchor.evidence_episode_ids
    assert str(record.compaction_id) in anchor.compaction_record_ids

    await memory.close()


@pytest.mark.asyncio
async def test_skeleton_anchor_identity_affect_artifacts_gaps_and_strength(tmp_path) -> None:
    memory = await MemoryStore(tmp_path / "memory.db").connect()
    assert memory._db is not None
    store = SessionAnchorStore(memory._db)
    t0 = datetime(2026, 5, 5, 12, tzinfo=timezone.utc)
    episodes = [
        Episode(
            created_at=t0,
            kind=EpisodeKind.TURN,
            session_id="s1",
            content="Please revise writing project 4246.",
            payload={"role": "user"},
        ),
        Episode(
            created_at=t0 + timedelta(minutes=1),
            kind=EpisodeKind.ACTION,
            session_id="s1",
            content=(
                "tool fs_write_file path=/mnt/xtra/OpenCAS/workspace/writing/4246/ch1.md "
                "bytes=12 checksum=abc"
            ),
            payload={"tool_name": "fs_write_file"},
            identity_mutagen=True,
            affect=AffectState(primary_emotion=PrimaryEmotion.JOY, intensity=0.9, valence=0.7),
        ),
        Episode(
            created_at=t0 + timedelta(minutes=50),
            kind=EpisodeKind.ACTION,
            session_id="s1",
            content='tool edit_file: {"file_path": "/mnt/xtra/OpenCAS/workspace/writing/4246/ch2.md"}',
            payload={"tool_name": "edit_file", "args": {"file_path": "/mnt/xtra/OpenCAS/workspace/writing/4246/ch2.md"}},
            affect=AffectState(primary_emotion=PrimaryEmotion.CURIOUS, intensity=0.6, valence=0.4),
        ),
    ]
    for episode in episodes:
        await memory.save_episode(episode)

    composer = SessionAutobiographyComposer(memory_store=memory, anchor_store=store)
    anchor = await composer.compose_skeleton("s1")

    assert anchor.episode_kind_tally == {"turn": 1, "action": 2}
    assert anchor.evidence_strength == "high"
    assert str(episodes[1].episode_id) in anchor.identity_mutagen_episode_ids
    assert anchor.affect_peaks[0]["primary_emotion"] == "joy"
    assert "workspace/writing/4246/ch1.md" in anchor.artifact_paths_touched
    assert "workspace/writing/4246/ch2.md" in anchor.artifact_paths_touched
    assert anchor.gaps_noted
    assert anchor.recorded_affect_summary
    assert anchor.evidence_hash

    await memory.close()


@pytest.mark.asyncio
async def test_skeleton_recollects_prior_failure_to_recollect(tmp_path) -> None:
    memory = await MemoryStore(tmp_path / "memory.db").connect()
    assert memory._db is not None
    store = SessionAnchorStore(memory._db)
    t0 = datetime(2026, 5, 5, 12, tzinfo=timezone.utc)
    failed = Episode(
        created_at=t0,
        kind=EpisodeKind.TURN,
        session_id="s1",
        content="I don't have continuous autobiographical memory of writing writing project 4246. I only retrieve records.",
        payload={"role": "assistant"},
    )
    recovered = Episode(
        created_at=t0 + timedelta(minutes=1),
        kind=EpisodeKind.ACTION,
        session_id="s1",
        content='tool search_memories: {"query": "writing project 4246", "limit": 20}',
        payload={"tool_name": "search_memories"},
    )
    await memory.save_episode(failed)
    await memory.save_episode(recovered)

    composer = SessionAutobiographyComposer(memory_store=memory, anchor_store=store)
    anchor = await composer.compose_skeleton("s1")

    assert str(failed.episode_id) in anchor.recall_failure_episode_ids
    assert str(recovered.episode_id) in anchor.recall_recovery_episode_ids

    await memory.close()


@pytest.mark.asyncio
async def test_gist_lazy_fill_uses_recorded_affect_and_deterministic_low_fallback(tmp_path) -> None:
    memory = await MemoryStore(tmp_path / "memory.db").connect()
    assert memory._db is not None
    store = SessionAnchorStore(memory._db)
    t0 = datetime(2026, 5, 5, 12, tzinfo=timezone.utc)
    episode = Episode(
        created_at=t0,
        kind=EpisodeKind.ACTION,
        session_id="s1",
        content="tool fs_write_file path=/mnt/xtra/OpenCAS/workspace/a.md bytes=3 checksum=abc",
        payload={"tool_name": "fs_write_file"},
        affect=AffectState(primary_emotion=PrimaryEmotion.CURIOUS, intensity=0.7, valence=0.2),
    )
    await memory.save_episode(episode)

    class FakeLLM:
        def __init__(self) -> None:
            self.calls = []

        async def chat_completion(self, messages, **kwargs):
            self.calls.append({"messages": messages, **kwargs})
            return {
                "choices": [
                    {
                        "message": {
                            "content": (
                                "I wrote a.md while curiosity was active "
                                f"[ep:{episode.episode_id}]."
                            )
                        }
                    }
                ]
            }

    llm = FakeLLM()
    composer = SessionAutobiographyComposer(memory_store=memory, anchor_store=store, llm=llm)
    await store.upsert(await composer.compose_skeleton("s1"))

    gist = await composer.compose_gist("s1")

    assert gist == f"I wrote a.md while curiosity was active [ep:{episode.episode_id}]."
    assert "Affect trajectory during session" in llm.calls[0]["messages"][1]["content"]
    assert "evidence_episodes" in llm.calls[0]["messages"][1]["content"]
    assert "current somatic" not in llm.calls[0]["messages"][1]["content"].lower()
    anchor = await store.get("s1")
    assert anchor is not None
    assert anchor.gist_version == 1
    assert anchor.confidence == "high"

    await memory.save_episode(
        Episode(
            created_at=t0,
            kind=EpisodeKind.TURN,
            session_id="low",
            content="Did you write this?",
            payload={"role": "user"},
        )
    )
    await store.upsert(await composer.compose_skeleton("low"))
    low_gist = await composer.compose_gist("low")
    assert "Primarily operator mentions" in low_gist
    assert len(llm.calls) == 1

    await memory.close()


@pytest.mark.asyncio
async def test_gist_validator_drops_uncited_sentences(tmp_path) -> None:
    memory = await MemoryStore(tmp_path / "memory.db").connect()
    assert memory._db is not None
    store = SessionAnchorStore(memory._db)
    t0 = datetime(2026, 5, 5, 12, tzinfo=timezone.utc)
    episode = Episode(
        created_at=t0,
        kind=EpisodeKind.ACTION,
        session_id="s1",
        content="tool fs_write_file path=/mnt/xtra/OpenCAS/workspace/a.md",
        payload={"tool_name": "fs_write_file"},
    )
    await memory.save_episode(episode)

    class FakeLLM:
        async def chat_completion(self, messages, **kwargs):
            return {
                "choices": [
                    {
                        "message": {
                            "content": (
                                f"I wrote a.md [ep:{episode.episode_id}]. "
                                "I also fixed the whole system."
                            )
                        }
                    }
                ]
            }

    composer = SessionAutobiographyComposer(memory_store=memory, anchor_store=store, llm=FakeLLM())
    await store.upsert(await composer.compose_skeleton("s1"))

    gist = await composer.compose_gist("s1")

    assert gist == f"I wrote a.md [ep:{episode.episode_id}]."
    assert "whole system" not in gist
    await memory.close()


@pytest.mark.asyncio
async def test_gist_validator_falls_back_to_deterministic_when_all_sentences_uncited(tmp_path) -> None:
    memory = await MemoryStore(tmp_path / "memory.db").connect()
    assert memory._db is not None
    store = SessionAnchorStore(memory._db)
    t0 = datetime(2026, 5, 5, 12, tzinfo=timezone.utc)
    await memory.save_episode(
        Episode(
            created_at=t0,
            kind=EpisodeKind.ACTION,
            session_id="s1",
            content="tool fs_write_file path=/mnt/xtra/OpenCAS/workspace/a.md",
            payload={"tool_name": "fs_write_file"},
        )
    )

    class FakeLLM:
        async def chat_completion(self, messages, **kwargs):
            return {"choices": [{"message": {"content": "I wrote a.md without citation."}}]}

    composer = SessionAutobiographyComposer(memory_store=memory, anchor_store=store, llm=FakeLLM())
    await store.upsert(await composer.compose_skeleton("s1"))

    gist = await composer.compose_gist("s1")

    assert gist.startswith("Session s1 on 2026-05-05")
    assert "artifacts touched" in gist
    await memory.close()


@pytest.mark.asyncio
async def test_boot_recovery_creates_missing_skeletons(tmp_path) -> None:
    from opencas.runtime.autobiography_hooks import recover_missing_anchors

    memory = await MemoryStore(tmp_path / "memory.db").connect()
    assert memory._db is not None
    store = SessionAnchorStore(memory._db)
    now = datetime(2026, 5, 5, 12, tzinfo=timezone.utc)
    for idx in range(3):
        await memory.save_episode(
            Episode(
                created_at=now + timedelta(minutes=idx),
                kind=EpisodeKind.TURN,
                session_id=f"s{idx}",
                content=f"message {idx}",
                payload={"role": "assistant"},
            )
        )

    composer = SessionAutobiographyComposer(memory_store=memory, anchor_store=store)
    created = await recover_missing_anchors(memory, store, composer, since=now - timedelta(minutes=1))

    assert created == 3
    assert await store.get("s0") is not None
    assert await store.get("s1") is not None
    assert await store.get("s2") is not None

    await memory.close()
