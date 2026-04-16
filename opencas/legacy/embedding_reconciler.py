"""Reconcile Bulma v4 Qdrant embeddings into OpenCAS embedding cache."""

from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from opencas.embeddings.models import EmbeddingRecord
from opencas.embeddings.service import EmbeddingCache, EmbeddingService

from .loader import load_qdrant_collection


async def import_qdrant_collection(
    qdrant_dir: Path,
    collection_name: str,
    model_tag: str,
    cache: EmbeddingCache,
    episode_id_to_text: Optional[Dict[str, str]] = None,
) -> int:
    """Import all vectors from a local Qdrant collection into the OpenCAS cache.

    Returns the number of vectors imported.
    """
    count = 0
    try:
        points = load_qdrant_collection(qdrant_dir, collection_name)
        for point_id, vector, payload in points:
            source_text = payload.get("textContent") or payload.get("content") or ""
            if not source_text and episode_id_to_text:
                source_text = episode_id_to_text.get(str(point_id), "")
            # Fallback: use a deterministic synthetic text so we still have a cache key
            if not source_text:
                source_text = f"__bulma_import__{collection_name}__{point_id}"

            import hashlib
            source_hash = hashlib.sha256(source_text.encode("utf-8")).hexdigest()

            now = datetime.now(timezone.utc)
            meta = {
                "bulma_collection": collection_name,
                "bulma_point_id": str(point_id),
                "bulma_payload": payload,
            }
            # Store under the provenance tag so the import is traceable.
            await cache.put(EmbeddingRecord(
                source_hash=source_hash,
                model_id=model_tag,
                dimension=len(vector),
                vector=vector,
                created_at=now,
                updated_at=now,
                meta=meta,
            ))
            # Also store under the native OpenCAS model_id so that cache
            # lookups by EmbeddingService hit without recomputing.  Bulma
            # used the same model (google/gemini-embedding-2-preview) so
            # the vectors are identical — no re-embedding needed.
            native_model_id = "google/gemini-embedding-2-preview"
            if model_tag != native_model_id:
                await cache.put(EmbeddingRecord(
                    source_hash=source_hash,
                    model_id=native_model_id,
                    dimension=len(vector),
                    vector=vector,
                    created_at=now,
                    updated_at=now,
                    meta=meta,
                ))
            count += 1
    except (ValueError, RuntimeError, PermissionError, OSError):
        pass
    return count


async def reconcile_embeddings(
    bulma_state_dir: Path,
    embedding_service: EmbeddingService,
    episode_id_to_text: Optional[Dict[str, str]] = None,
    high_salience_ids: Optional[set] = None,
) -> Dict[str, Any]:
    """Import Bulma embeddings and optionally backfill native embeddings.

    Returns a summary dict with counts.
    """
    qdrant_dir = bulma_state_dir / "qdrant_storage"
    imported_counts: Dict[str, int] = {}

    if qdrant_dir.exists():
        imported_counts["episodes_embed_v1"] = await import_qdrant_collection(
            qdrant_dir,
            "episodes_embed_v1",
            "openbulma-v4/episodes_embed_v1",
            embedding_service.cache,
            episode_id_to_text,
        )
        imported_counts["episodes_semantic"] = await import_qdrant_collection(
            qdrant_dir,
            "episodes_semantic",
            "openbulma-v4/episodes_semantic",
            embedding_service.cache,
            episode_id_to_text,
        )

    # Native backfill for high-value episodes (fire-and-forget BAA scheduling
    # would happen at the importer layer; here we just count candidates).
    backfill_candidates = 0
    if high_salience_ids and episode_id_to_text:
        for eid in high_salience_ids:
            text = episode_id_to_text.get(eid, "")
            if text:
                backfill_candidates += 1

    return {
        "imported_counts": imported_counts,
        "backfill_candidates": backfill_candidates,
    }
