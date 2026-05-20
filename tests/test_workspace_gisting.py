from __future__ import annotations

import json
from pathlib import Path

import pytest

from opencas.embeddings import EmbeddingCache, EmbeddingService
from opencas.workspace.gisting import generate_validated_gist
from opencas.workspace.scanner import FileSnapshot, sha256_file_sync
from opencas.workspace.service import WorkspaceIndexService
from opencas.workspace.store import WorkspaceStore


class _JsonLLM:
    async def chat_completion(self, **kwargs):
        del kwargs
        return {
            "choices": [
                {
                    "message": {
                        "content": json.dumps(
                            {
                                "summary": "Binary file metadata only.",
                                "purpose": "Track the file without pretending to know its contents.",
                                "why_exists": "It was encountered during workspace indexing.",
                                "notes": "No text was available for semantic validation.",
                            }
                        )
                    }
                }
            ]
        }


@pytest.mark.asyncio
async def test_unreadable_workspace_file_does_not_store_synthetic_content_vector(
    tmp_path: Path,
) -> None:
    cache = await EmbeddingCache(tmp_path / "embeddings.db").connect()
    embeddings = EmbeddingService(cache=cache, model_id="local-fallback")
    store = await WorkspaceStore(tmp_path / "workspace.db").connect()

    try:
        record = await generate_validated_gist(
            path=tmp_path / "asset.bin",
            checksum="a" * 64,
            file_kind="binary",
            content_excerpt="",
            extracted_text="",
            llm_client=_JsonLLM(),
            embeddings_client=embeddings,
            store=store,
            llm_model="test-model",
            embedding_model="local-fallback",
        )

        cursor = await store._db.execute(
            """
            SELECT content_text_status, content_embedding_ref, content_embedding_model, content_embedding_dim
            FROM workspace_checksums
            WHERE checksum = ?
            """,
            ("a" * 64,),
        )
        row = await cursor.fetchone()

        assert row["content_text_status"] == "unreadable"
        assert row["content_embedding_ref"] is None
        assert row["content_embedding_model"] is None
        assert row["content_embedding_dim"] is None
        assert record.gist_embedding_ref is not None
        assert record.gist_embedding_dim == 256
        assert record.accepted_flag is True
    finally:
        await store.close()
        await cache.close()


@pytest.mark.asyncio
async def test_list_directory_with_status_reports_divergence_when_index_is_non_empty(
    tmp_path: Path,
) -> None:
    store = await WorkspaceStore(tmp_path / "workspace.db").connect()
    root = tmp_path / "workspace"
    root.mkdir()
    indexed_path = root / "indexed.md"
    missing_path = root / "missing.md"
    live_only_path = root / "live_only.md"
    indexed_path.write_text("indexed", encoding="utf-8")
    missing_path.write_text("missing", encoding="utf-8")
    live_only_path.write_text("live only", encoding="utf-8")

    async def index_file(path: Path) -> None:
        stat = path.stat()
        checksum = sha256_file_sync(path)
        await store.upsert_path_snapshot(
            FileSnapshot(
                abs_path=path.resolve(),
                rel_path=path.relative_to(root),
                parent_dir=path.parent.resolve(),
                file_name=path.name,
                extension=path.suffix.lower() or None,
                size_bytes=stat.st_size,
                mtime_ns=stat.st_mtime_ns,
                file_kind="text",
            ),
            checksum=checksum,
        )
        await store.upsert_checksum(
            checksum,
            stat.st_size,
            "text",
            "text/markdown",
            "full",
            path.read_text(encoding="utf-8"),
            None,
            None,
            None,
        )

    try:
        await index_file(indexed_path)
        await index_file(missing_path)
        missing_path.unlink()
        service = WorkspaceIndexService(
            store=store,
            embeddings_client=None,
            llm_client=None,
            workspace_roots=[root],
            llm_model="test-llm",
            embedding_model="test-embedding",
        )

        payload = await service.list_directory_with_status(root)

        assert sorted(item["name"] for item in payload["indexed_files"]) == [
            "indexed.md",
            "missing.md",
        ]
        assert sorted(item["name"] for item in payload["disk_listing"]) == [
            "indexed.md",
            "live_only.md",
        ]
        assert payload["index_status"]["indexed_count"] == 2
        assert payload["index_status"]["disk_count"] == 2
        assert payload["index_status"]["fallback_used"] is False
        assert payload["index_status"]["live_listing_complete"] is True
        assert payload["index_status"]["not_indexed"] == [
            {
                "name": "live_only.md",
                "path": str(live_only_path.resolve()),
                "kind": "text",
                "indexed": False,
                "gist_pending": True,
                "size_bytes": 9,
            }
        ]
        assert payload["index_status"]["missing_on_disk"] == [
            {
                "name": "missing.md",
                "path": str(missing_path.resolve()),
                "kind": "text",
            }
        ]
        assert payload["index_status"]["stale_index_count"] == 1
    finally:
        await store.close()
