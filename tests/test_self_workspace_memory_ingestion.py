"""Self-workspace writes must land in searchable memory at write time.

Without this contract, reflective artifacts (self notes, prototypes, research,
daydream digests) live only as files on disk and never surface in retrieval —
the exact gap that prevented creative_writing work from drawing on prior reflections.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from opencas.context.retriever import MemoryRetriever
from opencas.daydream.self_workspace import SelfWorkspaceService
from opencas.daydream.signals import (
    PossibilitySignal,
    PossibilitySignalRoute,
    SelfWorkKind,
)
from opencas.embeddings import EmbeddingCache, EmbeddingService
from opencas.memory import ArtifactMemoryBridge, MemoryStore


async def _build_bridge(tmp_path: Path) -> tuple[ArtifactMemoryBridge, MemoryStore, EmbeddingCache]:
    state_dir = tmp_path / ".opencas"
    state_dir.mkdir(parents=True, exist_ok=True)
    memory = MemoryStore(state_dir / "memory.db")
    await memory.connect()
    cache = EmbeddingCache(state_dir / "embeddings.db")
    await cache.connect()
    embeddings = EmbeddingService(cache=cache, model_id="local-fallback")
    bridge = ArtifactMemoryBridge(state_dir=state_dir, memory=memory, embeddings=embeddings)
    return bridge, memory, cache


def _signal(summary: str, route: PossibilitySignalRoute, kind: SelfWorkKind) -> PossibilitySignal:
    return PossibilitySignal(
        source_reflection_id=f"reflection-{kind.value}",
        source_thought_index=0,
        summary=summary,
        imaginative_branch="(test branch)",
        practical_branch="(test practical branch)",
        bridge="(test bridge)",
        suggested_route=route,
        self_work_kind=kind,
    )


@pytest.mark.asyncio
async def test_self_note_is_searchable_immediately_after_write(tmp_path: Path) -> None:
    bridge, memory, cache = await _build_bridge(tmp_path)
    workspace_root = tmp_path / "workspace"
    service = SelfWorkspaceService(
        workspace_root=workspace_root,
        artifact_bridge=bridge,
    )

    signal = _signal(
        "Theo's voice anchors writing project 4246's central testimony arc.",
        PossibilitySignalRoute.SELF_NOTE,
        SelfWorkKind.NOTE,
    )
    receipt = await service.write_note(signal, reason="character-voice anchor for later chapters")

    note_paths = [Path(p) for p in receipt.artifact_paths if p.endswith(".md")]
    assert note_paths, "expected at least one markdown artifact"

    episodes = await memory.list_artifact_episodes(
        str(note_paths[0].resolve().relative_to(tmp_path))
    )
    assert episodes, "self note must produce a searchable artifact episode"
    assert episodes[0].kind.value == "artifact"
    assert "theo" in episodes[0].content.lower()

    retriever = MemoryRetriever(memory=memory, embeddings=EmbeddingService(cache=cache, model_id="local-fallback"))
    results = await retriever.retrieve("Who anchors testimony in writing project 4246?", limit=10)
    assert any("theo" in item.content.lower() for item in results), \
        "retriever must surface the freshly-written reflection"

    await memory.close()
    await cache.close()


@pytest.mark.asyncio
async def test_self_workspace_without_bridge_still_writes_files(tmp_path: Path) -> None:
    """Bridge is optional — without it the writer must still succeed."""
    workspace_root = tmp_path / "workspace"
    service = SelfWorkspaceService(workspace_root=workspace_root, artifact_bridge=None)

    signal = _signal(
        "A prototype outline.",
        PossibilitySignalRoute.SELF_NOTE,
        SelfWorkKind.NOTE,
    )
    receipt = await service.write_note(signal, reason="no bridge")
    note_paths = [Path(p) for p in receipt.artifact_paths if p.endswith(".md")]
    assert note_paths and note_paths[0].exists()


@pytest.mark.asyncio
async def test_self_research_ingests_both_note_and_manifest(tmp_path: Path) -> None:
    bridge, memory, cache = await _build_bridge(tmp_path)
    service = SelfWorkspaceService(
        workspace_root=tmp_path / "workspace",
        artifact_bridge=bridge,
    )

    signal = _signal(
        "Research thread on epistolary structure for writing project 4246 chapter 8.",
        PossibilitySignalRoute.RESEARCH,
        SelfWorkKind.RESEARCH,
    )
    receipt = await service.write_research(signal, reason="testing ingestion")

    md_paths = [Path(p) for p in receipt.artifact_paths if p.endswith(".md")]
    json_paths = [Path(p) for p in receipt.artifact_paths if p.endswith(".json") and "MANIFEST" in p]
    assert md_paths and json_paths

    md_episodes = await memory.list_artifact_episodes(
        str(md_paths[0].resolve().relative_to(tmp_path))
    )
    json_episodes = await memory.list_artifact_episodes(
        str(json_paths[0].resolve().relative_to(tmp_path))
    )
    assert md_episodes, "research note markdown must be ingested"
    assert json_episodes, "research manifest must be ingested too"

    await memory.close()
    await cache.close()
