from __future__ import annotations

from datetime import datetime, timezone
from types import SimpleNamespace

import pytest

from opencas.memory import Episode, EpisodeKind, MemoryStore
from opencas.memory.autobiography import (
    AutobiographyReconstructor,
    SessionAnchorStore,
    SessionAutobiographyComposer,
)
from opencas.tools.autobiography_tool import AutobiographyToolAdapter


@pytest.mark.asyncio
async def test_recall_packet_shape_evidence_ranking_and_cross_session_project(tmp_path) -> None:
    memory = await MemoryStore(tmp_path / "memory.db").connect()
    assert memory._db is not None
    store = SessionAnchorStore(memory._db)
    composer = SessionAutobiographyComposer(memory_store=memory, anchor_store=store)
    t0 = datetime(2026, 5, 5, 12, tzinfo=timezone.utc)
    for session_id, filename in (("s1", "ch1.md"), ("s2", "ch2.md")):
        await memory.save_episode(
            Episode(
                created_at=t0,
                kind=EpisodeKind.TURN,
                session_id=session_id,
                content="I worked on writing project 4246.",
                payload={"role": "assistant"},
            )
        )
        await memory.save_episode(
            Episode(
                created_at=t0,
                kind=EpisodeKind.ACTION,
                session_id=session_id,
                content=f"tool fs_write_file path=/mnt/xtra/OpenCAS/workspace/writing/4246/{filename} bytes=3 checksum=abc",
                payload={"tool_name": "fs_write_file"},
            )
        )
        await store.upsert(await composer.compose_skeleton(session_id))

    reconstructor = AutobiographyReconstructor(anchor_store=store, composer=composer, memory_store=memory)
    result = await reconstructor.recall(query="writing project 4246", max_tokens=500)
    payload = result.to_dict()

    assert set(payload) >= {
        "essence",
        "confidence",
        "evidence_scope",
        "strongest_evidence",
        "gaps",
        "next_best_lookup",
        "session_refs",
        "project_arc",
    }
    assert payload["confidence"] == "high"
    assert payload["evidence_scope"] == "autobiographical"
    assert payload["strongest_evidence"][0]["kind"] == "autobiographical"
    assert "2 sessions" in payload["essence"]
    assert payload["project_arc"]["path"] == "workspace/writing/4246"
    assert payload["project_arc"]["session_count"] == 2
    assert {item["session_id"] for item in payload["strongest_evidence"]} >= {"s1", "s2"}

    await memory.close()


@pytest.mark.asyncio
async def test_recall_references_prior_failure_to_recollect_delicately(tmp_path) -> None:
    memory = await MemoryStore(tmp_path / "memory.db").connect()
    assert memory._db is not None
    store = SessionAnchorStore(memory._db)
    composer = SessionAutobiographyComposer(memory_store=memory, anchor_store=store)
    t0 = datetime(2026, 5, 5, 12, tzinfo=timezone.utc)
    await memory.save_episode(
        Episode(
            created_at=t0,
            kind=EpisodeKind.TURN,
            session_id="s1",
            content="I don't remember writing project 4246; I only retrieve records.",
            payload={"role": "assistant"},
        )
    )
    await memory.save_episode(
        Episode(
            created_at=t0,
            kind=EpisodeKind.ACTION,
            session_id="s1",
            content="tool search_memories: {\"query\": \"writing project 4246\"}",
            payload={"tool_name": "search_memories"},
        )
    )
    await store.upsert(await composer.compose_skeleton("s1"))

    reconstructor = AutobiographyReconstructor(anchor_store=store, composer=composer, memory_store=memory)
    result = await reconstructor.recall(query="writing project 4246 recall failure")
    payload = result.to_dict()

    assert any("failure to recollect" in gap.lower() for gap in payload["gaps"])
    assert "permanent defect" not in payload["essence"].lower()
    assert "amnesia" not in payload["essence"].lower()

    await memory.close()


@pytest.mark.asyncio
async def test_recall_tool_returns_json_structured_packet(tmp_path) -> None:
    memory = await MemoryStore(tmp_path / "memory.db").connect()
    assert memory._db is not None
    store = SessionAnchorStore(memory._db)
    composer = SessionAutobiographyComposer(memory_store=memory, anchor_store=store)
    await store.upsert(await composer.compose_skeleton("empty"))
    reconstructor = AutobiographyReconstructor(anchor_store=store, composer=composer, memory_store=memory)
    runtime = SimpleNamespace(ctx=SimpleNamespace(autobiography_reconstructor=reconstructor))

    result = await AutobiographyToolAdapter(runtime)(
        "recall_autobiography",
        {"query": "what did we work on", "max_tokens": 300},
    )

    assert result.success is True
    assert '"essence"' in result.output
    assert result.metadata["confidence"] in {"low", "insufficient"}

    await memory.close()


@pytest.mark.asyncio
async def test_recall_tool_does_not_lazy_fill_gists_on_interactive_path() -> None:
    class _Result:
        def to_dict(self):
            return {
                "essence": "fast packet",
                "confidence": "low",
                "evidence_scope": "mention_only",
                "strongest_evidence": [],
            }

    class _Reconstructor:
        def __init__(self) -> None:
            self.calls = []

        async def recall(self, **kwargs):
            self.calls.append(kwargs)
            return _Result()

    reconstructor = _Reconstructor()
    runtime = SimpleNamespace(ctx=SimpleNamespace(autobiography_reconstructor=reconstructor))

    result = await AutobiographyToolAdapter(runtime)(
        "recall_autobiography",
        {"query": "atlas repair", "max_tokens": 300},
    )

    assert result.success is True
    assert reconstructor.calls[0]["lazy_fill"] is False


@pytest.mark.asyncio
async def test_recall_matches_linked_episode_content_before_gist_exists(tmp_path) -> None:
    memory = await MemoryStore(tmp_path / "memory.db").connect()
    assert memory._db is not None
    store = SessionAnchorStore(memory._db)
    composer = SessionAutobiographyComposer(memory_store=memory, anchor_store=store)
    t0 = datetime(2026, 5, 5, 12, tzinfo=timezone.utc)
    await memory.save_episode(
        Episode(
            created_at=t0,
            kind=EpisodeKind.TURN,
            session_id="s1",
            content="I investigated the buzzing Micali memory problem.",
            payload={"role": "assistant"},
        )
    )
    await memory.save_episode(
        Episode(
            created_at=t0.replace(hour=13),
            kind=EpisodeKind.TURN,
            session_id="s2",
            content="I discussed an unrelated calendar cleanup.",
            payload={"role": "assistant"},
        )
    )
    await store.upsert(await composer.compose_skeleton("s1"))
    await store.upsert(await composer.compose_skeleton("s2"))

    reconstructor = AutobiographyReconstructor(anchor_store=store, composer=composer, memory_store=memory)
    result = await reconstructor.recall(query="buzzing Micali")

    assert result.session_refs[0].session_id == "s1"
    assert result.evidence_scope in {"mixed", "mention_only"}
    assert any("buzzing Micali" in item.excerpt for item in result.strongest_evidence)

    await memory.close()


@pytest.mark.asyncio
async def test_recall_absolute_workspace_path_prefers_matching_anchor(tmp_path) -> None:
    memory = await MemoryStore(tmp_path / "memory.db").connect()
    assert memory._db is not None
    store = SessionAnchorStore(memory._db)
    composer = SessionAutobiographyComposer(memory_store=memory, anchor_store=store)
    t0 = datetime(2026, 5, 5, 12, tzinfo=timezone.utc)
    await memory.save_episode(
        Episode(
            created_at=t0.replace(minute=0),
            kind=EpisodeKind.ACTION,
            session_id="target",
            content='tool search_memories: {"query": "operator manuscript"}',
            payload={"tool_name": "search_memories"},
        )
    )
    await memory.save_episode(
        Episode(
            created_at=t0.replace(minute=1),
            kind=EpisodeKind.ACTION,
            session_id="target",
            content=(
                "tool fs_write_file path=/mnt/xtra/OpenCAS/workspace/if_i_am_the_operator/"
                "manuscript_draft.md bytes=12 checksum=abc"
            ),
            payload={"tool_name": "fs_write_file"},
        )
    )
    await memory.save_episode(
        Episode(
            created_at=t0.replace(hour=13),
            kind=EpisodeKind.TURN,
            session_id="decoy",
            content=(
                "I discussed an OpenCAS workspace manuscript draft operator question. "
                "OpenCAS workspace manuscript draft operator came up several times, "
                "but no artifact path was touched."
            ),
            payload={"role": "assistant"},
        )
    )
    await store.upsert(await composer.compose_skeleton("target"))
    await store.upsert(await composer.compose_skeleton("decoy"))

    reconstructor = AutobiographyReconstructor(anchor_store=store, composer=composer, memory_store=memory)
    result = await reconstructor.recall(
        query="/mnt/xtra/OpenCAS/workspace/if_i_am_the_operator/manuscript_draft.md"
    )

    assert result.session_refs[0].session_id == "target"
    assert result.next_best_lookup[0].args["path"] == "workspace/if_i_am_the_operator/manuscript_draft.md"
    assert "fs_write_file" in result.strongest_evidence[0].label

    await memory.close()


@pytest.mark.asyncio
async def test_recall_does_not_promote_unmatched_project_arc_for_generic_repair_query(tmp_path) -> None:
    memory = await MemoryStore(tmp_path / "memory.db").connect()
    assert memory._db is not None
    store = SessionAnchorStore(memory._db)
    composer = SessionAutobiographyComposer(memory_store=memory, anchor_store=store)
    t0 = datetime(2026, 5, 5, 12, tzinfo=timezone.utc)
    for index in range(2):
        session_id = f"writing-project-{index}"
        await memory.save_episode(
            Episode(
                created_at=t0,
                kind=EpisodeKind.ACTION,
                session_id=session_id,
                content=(
                    "tool edit_file path=/mnt/xtra/OpenCAS/workspace/writing/4246/"
                    f"story_4246_repair_notes_{index}.md bytes=3 checksum=abc"
                ),
                payload={"tool_name": "edit_file"},
            )
        )
        await store.upsert(await composer.compose_skeleton(session_id))

    reconstructor = AutobiographyReconstructor(anchor_store=store, composer=composer, memory_store=memory)
    result = await reconstructor.recall(
        query=(
            "Codex here, not Jarrod. Please use available memory/runtime evidence "
            "if needed: what do you recall about the atlas repair thread?"
        )
    )

    assert result.project_arc is None
    assert result.confidence == "insufficient"
    assert "workspace/writing/4246" not in result.essence

    await memory.close()


@pytest.mark.asyncio
async def test_recall_filters_unrelated_action_evidence_inside_matching_anchor(tmp_path) -> None:
    memory = await MemoryStore(tmp_path / "memory.db").connect()
    assert memory._db is not None
    store = SessionAnchorStore(memory._db)
    composer = SessionAutobiographyComposer(memory_store=memory, anchor_store=store)
    t0 = datetime(2026, 5, 5, 12, tzinfo=timezone.utc)
    await memory.save_episode(
        Episode(
            created_at=t0,
            kind=EpisodeKind.TURN,
            session_id="mixed",
            content="We discussed the atlas repair thread and what broke in the flow view.",
            payload={"role": "assistant"},
        )
    )
    await memory.save_episode(
        Episode(
            created_at=t0,
            kind=EpisodeKind.ACTION,
            session_id="mixed",
            content=(
                "tool fs_write_file path=/mnt/xtra/OpenCAS/workspace/writing/4246/"
                "story_4246.md bytes=3 checksum=abc"
            ),
            payload={"tool_name": "fs_write_file"},
        )
    )
    await store.upsert(await composer.compose_skeleton("mixed"))

    reconstructor = AutobiographyReconstructor(anchor_store=store, composer=composer, memory_store=memory)
    result = await reconstructor.recall(query="atlas repair thread")

    assert result.strongest_evidence
    assert "atlas repair" in result.strongest_evidence[0].excerpt.lower()
    assert "story_4246" not in result.strongest_evidence[0].excerpt.lower()

    await memory.close()


@pytest.mark.asyncio
async def test_project_arc_prefers_query_matching_project_over_path_volume(tmp_path) -> None:
    memory = await MemoryStore(tmp_path / "memory.db").connect()
    assert memory._db is not None
    store = SessionAnchorStore(memory._db)
    composer = SessionAutobiographyComposer(memory_store=memory, anchor_store=store)
    t0 = datetime(2026, 5, 5, 12, tzinfo=timezone.utc)
    mixed_paths = [
        "workspace/writing/2046/ch1.md",
        "workspace/writing/2046/ch2.md",
        "workspace/writing/2046/ch3.md",
        "workspace/kPony/BUILD_PROOF.md",
    ]
    for index, path in enumerate(mixed_paths):
        await memory.save_episode(
            Episode(
                created_at=t0.replace(minute=index),
                kind=EpisodeKind.ACTION,
                session_id="mixed",
                content=f"tool fs_write_file path=/mnt/xtra/OpenCAS/{path}",
                payload={"tool_name": "fs_write_file"},
            )
        )
    await memory.save_episode(
        Episode(
            created_at=t0.replace(hour=13),
            kind=EpisodeKind.ACTION,
            session_id="kpony",
            content="tool fs_write_file path=/mnt/xtra/OpenCAS/workspace/kPony/CMakeLists.txt",
            payload={"tool_name": "fs_write_file"},
        )
    )
    await store.upsert(await composer.compose_skeleton("mixed"))
    await store.upsert(await composer.compose_skeleton("kpony"))

    reconstructor = AutobiographyReconstructor(anchor_store=store, composer=composer, memory_store=memory)
    result = await reconstructor.recall(query="kPony build verification")

    assert result.project_arc is not None
    assert result.project_arc["path"] == "workspace/kPony"
    assert result.project_arc["session_count"] == 2

    await memory.close()
