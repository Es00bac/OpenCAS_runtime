from __future__ import annotations

from pathlib import Path

import pytest

from opencas.memory import Episode, EpisodeKind, MemoryStore
from scripts.backfill_compact_action_heads import backfill_compact_action_heads


@pytest.mark.asyncio
async def test_backfill_dry_run_reports_old_fat_action_without_mutating(tmp_path: Path) -> None:
    store = await MemoryStore(tmp_path / "memory.sqlite").connect()
    try:
        original_content = "tool archive_large_context: " + ("full prompt body " * 200)
        episode = Episode(
            kind=EpisodeKind.ACTION,
            content=original_content,
            payload={
                "tool_name": "archive_large_context",
                "args": {"prompt": "full prompt body " * 200, "output_path": "workspace/reports/trace.md"},
                "result_metadata": {"ok": True},
            },
        )
        await store.save_episode(episode)

        report = await backfill_compact_action_heads(
            store,
            dry_run=True,
            limit=10,
            content_threshold=256,
        )
        reloaded = await store.get_episode(str(episode.episode_id))

        assert report.scanned == 1
        assert report.eligible == 1
        assert report.updated == 0
        assert reloaded is not None
        assert reloaded.content == original_content
        assert "compact_action_backfill" not in reloaded.payload
    finally:
        await store.close()


@pytest.mark.asyncio
async def test_backfill_apply_compacts_and_preserves_legacy_content(tmp_path: Path) -> None:
    store = await MemoryStore(tmp_path / "memory.sqlite").connect()
    try:
        original_content = "tool archive_large_context: " + ("full prompt body " * 200)
        episode = Episode(
            kind=EpisodeKind.ACTION,
            content=original_content,
            payload={
                "tool_name": "archive_large_context",
                "args": {"prompt": "full prompt body " * 200, "artifact_path": "workspace/reports/trace.md"},
                "result_metadata": {"ok": True},
            },
        )
        await store.save_episode(episode)

        report = await backfill_compact_action_heads(
            store,
            dry_run=False,
            limit=10,
            path_filter="workspace/reports",
            content_threshold=256,
        )
        reloaded = await store.get_episode(str(episode.episode_id))

        assert report.scanned == 1
        assert report.eligible == 1
        assert report.updated == 1
        assert reloaded is not None
        assert reloaded.content.startswith("tool archive_large_context args_digest=")
        assert "full prompt body full prompt body" not in reloaded.content
        assert "artifact=workspace/reports/trace.md" in reloaded.content
        assert reloaded.payload["compact_action_backfill"]["version"] == 1
        assert reloaded.payload["compact_action_backfill"]["original_content"] == original_content
        assert reloaded.payload["compact_action_backfill"]["original_content_sha256"]
        assert reloaded.payload["compact_action_backfill"]["dry_run"] is False
        assert reloaded.payload["args"]["prompt"].startswith("full prompt body")
    finally:
        await store.close()


@pytest.mark.asyncio
async def test_backfill_parses_legacy_action_args_from_content_when_payload_missing(
    tmp_path: Path,
) -> None:
    store = await MemoryStore(tmp_path / "memory.sqlite").connect()
    try:
        original_content = (
            'tool fs_write_file: {"file_path": "workspace/kPony/BUILD_PROOF.md", '
            '"content": "'
            + ("build proof body " * 200)
            + '"}'
        )
        episode = Episode(
            kind=EpisodeKind.ACTION,
            content=original_content,
            payload={},
        )
        await store.save_episode(episode)

        report = await backfill_compact_action_heads(
            store,
            dry_run=False,
            limit=10,
            path_filter="workspace/kPony/BUILD_PROOF.md",
            content_threshold=256,
        )
        reloaded = await store.get_episode(str(episode.episode_id))

        assert report.scanned == 1
        assert report.eligible == 1
        assert report.updated == 1
        assert reloaded is not None
        assert reloaded.content.startswith("tool fs_write_file args_digest=")
        assert "artifact=workspace/kPony/BUILD_PROOF.md" in reloaded.content
        assert reloaded.payload["tool_name"] == "fs_write_file"
        assert reloaded.payload["args"]["file_path"] == "workspace/kPony/BUILD_PROOF.md"
        assert reloaded.payload["compact_action_backfill"]["original_content"] == original_content
    finally:
        await store.close()
