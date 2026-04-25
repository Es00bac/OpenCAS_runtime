from __future__ import annotations

import json
from pathlib import Path

import pytest

from opencas.embeddings import EmbeddingCache, EmbeddingService
from opencas.workspace.gisting import generate_validated_gist
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
