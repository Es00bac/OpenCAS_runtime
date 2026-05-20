"""Embedding service with compute-once, cache-many semantics."""

from __future__ import annotations

import asyncio
import hashlib
import json
import time
from collections import deque
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Callable, Coroutine, Dict, List, Optional, Sequence

import numpy as np
import logging

from open_llm_auth.auth.manager import ProviderManager

from opencas.embeddings.models import EmbeddingHealth, EmbeddingRecord

logger = logging.getLogger(__name__)


class EmbeddingCache:
    """SQLite-backed cache for embedding vectors."""

    def __init__(
        self,
        db_path: Path | str,
        vector_backend=None,
        hnsw_backend=None,
    ) -> None:
        self.db_path = Path(db_path)
        self._db = None
        self.vector_backend = vector_backend
        self.hnsw_backend = hnsw_backend
        self._search_history: deque = deque(maxlen=1000)

    async def connect(self) -> "EmbeddingCache":
        import aiosqlite

        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._db = await aiosqlite.connect(str(self.db_path), timeout=30)
        await self._db.execute("PRAGMA journal_mode=WAL")
        await self._db.execute("PRAGMA busy_timeout=30000")
        await self._db.execute("PRAGMA synchronous=NORMAL")
        await self._db.executescript(
            """
            CREATE TABLE IF NOT EXISTS embeddings (
                embedding_id TEXT PRIMARY KEY,
                source_hash TEXT NOT NULL UNIQUE,
                model_id TEXT NOT NULL,
                dimension INTEGER NOT NULL,
                vector TEXT NOT NULL,
                created_at TEXT NOT NULL,
                updated_at TEXT NOT NULL,
                meta TEXT NOT NULL DEFAULT '{}'
            );
            CREATE INDEX IF NOT EXISTS idx_source_hash ON embeddings(source_hash);
            CREATE INDEX IF NOT EXISTS idx_model_id ON embeddings(model_id);
            """
        )
        await self._db.commit()
        return self

    async def close(self) -> None:
        if self._db:
            await self._db.close()
            self._db = None
        if self.vector_backend is not None:
            close = getattr(self.vector_backend, "close", None)
            if callable(close):
                maybe_result = close()
                if hasattr(maybe_result, "__await__"):
                    await maybe_result
        if self.hnsw_backend is not None:
            close = getattr(self.hnsw_backend, "close", None)
            if callable(close):
                close()

    async def get(self, source_hash: str) -> Optional[EmbeddingRecord]:
        assert self._db is not None
        cursor = await self._db.execute(
            "SELECT * FROM embeddings WHERE source_hash = ?", (source_hash,)
        )
        row = await cursor.fetchone()
        if row is None:
            cursor = await self._db.execute(
                "SELECT * FROM embeddings WHERE embedding_id = ?", (source_hash,)
            )
            row = await cursor.fetchone()
        if row is None:
            return None
        return self._row_to_record(row)

    async def get_many(self, identifiers: Sequence[str]) -> Dict[str, EmbeddingRecord]:
        """Return embedding records keyed by both source_hash and embedding_id.

        Callers such as retrieval reranking often have a mixed list of source
        hashes and embedding ids. Fetching them in batches avoids hundreds of
        sequential SQLite round-trips on recall-heavy turns while preserving the
        existing single-record lookup contract.
        """

        assert self._db is not None
        unique_ids = [
            item
            for item in dict.fromkeys(str(identifier or "").strip() for identifier in identifiers)
            if item
        ]
        if not unique_ids:
            return {}

        records: Dict[str, EmbeddingRecord] = {}
        chunk_size = 400
        for offset in range(0, len(unique_ids), chunk_size):
            chunk = unique_ids[offset : offset + chunk_size]
            placeholders = ",".join("?" for _ in chunk)
            cursor = await self._db.execute(
                f"""
                SELECT *
                FROM embeddings
                WHERE source_hash IN ({placeholders})
                   OR embedding_id IN ({placeholders})
                """,
                tuple(chunk + chunk),
            )
            rows = await cursor.fetchall()
            for row in rows:
                record = self._row_to_record(row)
                records[str(record.source_hash)] = record
                records[str(record.embedding_id)] = record
        return records

    async def put(self, record: EmbeddingRecord) -> None:
        assert self._db is not None
        record.meta = dict(record.meta or {})
        await self._db.execute(
            """
            INSERT INTO embeddings (
                embedding_id, source_hash, model_id, dimension, vector,
                created_at, updated_at, meta
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            ON CONFLICT(source_hash) DO UPDATE SET
                embedding_id = excluded.embedding_id,
                model_id = excluded.model_id,
                dimension = excluded.dimension,
                vector = excluded.vector,
                updated_at = excluded.updated_at,
                meta = excluded.meta
            """,
            (
                str(record.embedding_id),
                record.source_hash,
                record.model_id,
                record.dimension,
                json.dumps(record.vector),
                record.created_at.isoformat(),
                record.updated_at.isoformat(),
                json.dumps(record.meta),
            ),
        )
        await self._db.commit()
        meta_dirty = False
        if self.vector_backend is not None:
            try:
                ok = await self.vector_backend.upsert(record)
                record.meta["vector_status"] = "ok" if ok else "stale"
                meta_dirty = True
                if not ok:
                    logger.warning(
                        "Embedding vector backend rejected upsert for %s",
                        record.source_hash,
                    )
            except Exception:
                record.meta["vector_status"] = "stale"
                meta_dirty = True
                logger.warning(
                    "Embedding vector backend upsert failed for %s",
                    record.source_hash,
                    exc_info=True,
                )
        else:
            record.meta["vector_status"] = "skipped"
            meta_dirty = True
        if self.hnsw_backend is not None:
            try:
                ok = await self.hnsw_backend.upsert(record)
                record.meta["hnsw_status"] = "ok" if ok else "stale"
                meta_dirty = True
                if not ok:
                    logger.warning(
                        "Embedding HNSW backend rejected upsert for %s",
                        record.source_hash,
                    )
            except Exception:
                record.meta["hnsw_status"] = "stale"
                meta_dirty = True
                logger.warning(
                    "Embedding HNSW backend upsert failed for %s",
                    record.source_hash,
                    exc_info=True,
                )
        else:
            record.meta["hnsw_status"] = "skipped"
            meta_dirty = True
        if meta_dirty:
            await self._update_record_meta(record.source_hash, record.meta)

    async def reindex_stale(self, limit: int = 1000) -> Dict[str, int]:
        """Retry vector backend writes for cache rows marked stale."""
        assert self._db is not None
        cursor = await self._db.execute(
            """
            SELECT * FROM embeddings
            WHERE meta LIKE '%vector_status%stale%'
               OR meta LIKE '%hnsw_status%stale%'
            ORDER BY updated_at ASC
            LIMIT ?
            """,
            (max(1, int(limit)),),
        )
        rows = await cursor.fetchall()
        stats = {
            "records_checked": len(rows),
            "vector_reindexed": 0,
            "hnsw_reindexed": 0,
            "failed": 0,
        }
        for row in rows:
            record = self._row_to_record(row)
            meta_dirty = False
            if record.meta.get("vector_status") == "stale":
                if self.vector_backend is None:
                    record.meta["vector_status"] = "skipped"
                    meta_dirty = True
                else:
                    try:
                        ok = await self.vector_backend.upsert(record)
                    except Exception:
                        ok = False
                        logger.warning(
                            "Embedding vector backend reindex failed for %s",
                            record.source_hash,
                            exc_info=True,
                        )
                    if ok:
                        record.meta["vector_status"] = "ok"
                        stats["vector_reindexed"] += 1
                        meta_dirty = True
                    else:
                        stats["failed"] += 1
            if record.meta.get("hnsw_status") == "stale":
                if self.hnsw_backend is None:
                    record.meta["hnsw_status"] = "skipped"
                    meta_dirty = True
                else:
                    try:
                        ok = await self.hnsw_backend.upsert(record)
                    except Exception:
                        ok = False
                        logger.warning(
                            "Embedding HNSW backend reindex failed for %s",
                            record.source_hash,
                            exc_info=True,
                        )
                    if ok:
                        record.meta["hnsw_status"] = "ok"
                        stats["hnsw_reindexed"] += 1
                        meta_dirty = True
                    else:
                        stats["failed"] += 1
            if meta_dirty:
                await self._update_record_meta(record.source_hash, record.meta)
        return stats

    async def _update_record_meta(self, source_hash: str, meta: Dict[str, Any]) -> None:
        assert self._db is not None
        await self._db.execute(
            "UPDATE embeddings SET meta = ? WHERE source_hash = ?",
            (json.dumps(meta), source_hash),
        )
        await self._db.commit()

    async def health(self) -> EmbeddingHealth:
        assert self._db is not None
        cursor = await self._db.execute("SELECT COUNT(*) FROM embeddings")
        total = (await cursor.fetchone())[0]
        cursor = await self._db.execute(
            "SELECT COUNT(DISTINCT model_id) FROM embeddings"
        )
        models = (await cursor.fetchone())[0]
        cursor = await self._db.execute(
            "SELECT AVG(dimension) FROM embeddings"
        )
        avg_dim = (await cursor.fetchone())[0]
        return EmbeddingHealth(
            total_records=total,
            total_models=models,
            average_vector_dimension=int(avg_dim) if avg_dim else None,
        )

    async def recent_records(self, limit: int = 10) -> List[EmbeddingRecord]:
        assert self._db is not None
        cursor = await self._db.execute(
            """
            SELECT * FROM embeddings
            ORDER BY updated_at DESC
            LIMIT ?
            """,
            (max(1, int(limit)),),
        )
        rows = await cursor.fetchall()
        return [self._row_to_record(row) for row in rows]

    async def recent_records_since(
        self,
        since: datetime,
        limit: int = 10000,
    ) -> List[EmbeddingRecord]:
        """Return cache rows updated after *since*, newest first."""
        assert self._db is not None
        cursor = await self._db.execute(
            """
            SELECT * FROM embeddings
            WHERE datetime(updated_at) >= datetime(?)
            ORDER BY updated_at DESC
            LIMIT ?
            """,
            (since.isoformat(), max(1, int(limit))),
        )
        rows = await cursor.fetchall()
        return [self._row_to_record(row) for row in rows]

    async def search_similar(
        self,
        vector: Sequence[float],
        limit: int = 10,
        model_id: Optional[str] = None,
        project_id: Optional[str] = None,
        task_type: Optional[str] = None,
        query_text: Optional[str] = None,
        max_scan_rows: int = 2000,
    ) -> List[tuple[EmbeddingRecord, float]]:
        """Return the top *limit* cached embeddings by cosine similarity to *vector*."""
        assert self._db is not None
        start = time.perf_counter()

        # Tier 1: Qdrant
        if self.vector_backend is not None:
            try:
                hits = await self.vector_backend.search(
                    vector,
                    limit=limit,
                    model_id=model_id,
                    project_id=project_id,
                    task_type=task_type,
                    with_scores=True,
                )
                if hits:
                    scored: List[tuple[EmbeddingRecord, float]] = []
                    for source_hash, sim in hits:
                        record = await self.get(source_hash)
                        if record is not None:
                            if task_type and record.meta.get("task_type") != task_type:
                                continue
                            scored.append((record, sim))
                    if scored:
                        latency_ms = (time.perf_counter() - start) * 1000
                        self._record_search_path("qdrant", latency_ms)
                        return scored
            except Exception:
                pass

        # Tier 2: HNSW local ANN
        if self.hnsw_backend is not None:
            try:
                hits = await self.hnsw_backend.search(
                    vector,
                    limit=limit,
                    model_id=model_id,
                    project_id=project_id,
                    task_type=task_type,
                    with_scores=True,
                )
                if hits:
                    scored = []
                    for source_hash, sim in hits:
                        record = await self.get(source_hash)
                        if record is not None:
                            if task_type and record.meta.get("task_type") != task_type:
                                continue
                            scored.append((record, sim))
                    if scored:
                        latency_ms = (time.perf_counter() - start) * 1000
                        self._record_search_path("hnsw", latency_ms)
                        return scored
            except Exception:
                pass

        # Tier 3: Bounded SQLite scan ordered by recency
        scan_limit = max(1, int(max_scan_rows))
        if model_id is not None:
            cursor = await self._db.execute(
                "SELECT * FROM embeddings WHERE model_id = ? ORDER BY updated_at DESC LIMIT ?",
                (model_id, scan_limit),
            )
        else:
            cursor = await self._db.execute(
                "SELECT * FROM embeddings ORDER BY updated_at DESC LIMIT ?",
                (scan_limit,),
            )
        rows = await cursor.fetchall()
        query = np.array(vector, dtype=np.float32)
        q_norm = np.linalg.norm(query)
        if q_norm == 0:
            q_norm = 1.0
        scored = []
        for row in rows:
            record = self._row_to_record(row)
            if model_id and record.model_id != model_id:
                continue
            if project_id:
                meta_project = record.meta.get("project_id") if record.meta else None
                if meta_project != project_id:
                    continue
            if task_type:
                meta_task_type = record.meta.get("task_type") if record.meta else None
                if meta_task_type != task_type:
                    continue
            cand = np.array(record.vector, dtype=np.float32)
            if cand.shape != query.shape:
                continue
            c_norm = np.linalg.norm(cand)
            if c_norm == 0:
                c_norm = 1.0
            sim = float(np.dot(query, cand) / (q_norm * c_norm))
            scored.append((record, sim))
        scored.sort(key=lambda x: x[1], reverse=True)

        # Tier 4: Lexical fallback
        max_sim = scored[0][1] if scored else 0.0
        if max_sim < 0.1 and query_text:
            query_lower = query_text.lower()
            scored = []
            for row in rows:
                record = self._row_to_record(row)
                if model_id and record.model_id != model_id:
                    continue
                if project_id:
                    meta_project = record.meta.get("project_id") if record.meta else None
                    if meta_project != project_id:
                        continue
                if task_type:
                    meta_task_type = record.meta.get("task_type") if record.meta else None
                    if meta_task_type != task_type:
                        continue
                text = (record.meta.get("text") or "").lower()
                if not text:
                    continue
                overlap = sum(1 for word in query_lower.split() if word in text)
                sim = min(1.0, overlap / max(1, len(query_lower.split())))
                if sim > 0:
                    scored.append((record, sim))
            scored.sort(key=lambda x: x[1], reverse=True)
            if scored:
                latency_ms = (time.perf_counter() - start) * 1000
                self._record_search_path("lexical_fallback", latency_ms)
                return scored[:limit]

        latency_ms = (time.perf_counter() - start) * 1000
        self._record_search_path("sqlite_brute_force", latency_ms)
        return scored[:limit]

    def _record_search_path(self, path: str, latency_ms: float) -> None:
        """Append a search history entry."""
        self._search_history.append(
            {
                "timestamp": datetime.now(timezone.utc),
                "path": path,
                "latency_ms": latency_ms,
            }
        )

    @staticmethod
    def _row_to_record(row) -> EmbeddingRecord:
        from datetime import datetime

        return EmbeddingRecord(
            embedding_id=row[0],
            source_hash=row[1],
            model_id=row[2],
            dimension=row[3],
            vector=json.loads(row[4]),
            created_at=datetime.fromisoformat(row[5]),
            updated_at=datetime.fromisoformat(row[6]),
            meta=json.loads(row[7]),
        )


class EmbeddingService:
    """Compute embeddings once and reuse them via cache."""

    def __init__(
        self,
        cache: EmbeddingCache,
        embed_fn: Callable[[str], Coroutine[Any, Any, Sequence[float]]] | None = None,
        embed_batch_fn: Callable[[List[str]], Coroutine[Any, Any, List[Sequence[float]]]] | None = None,
        model_id: Optional[str] = None,
        expected_dimension: Optional[int] = None,
        store=None,
    ) -> None:
        self.cache = cache
        self.local_embedding_model_id = ProviderManager.default_embedding_model_ref()
        self.offline_embedding_model_id = ProviderManager.offline_embedding_model_ref()
        self.model_id = model_id or self.offline_embedding_model_id
        self.expected_dimension = expected_dimension
        self._embed_fn = embed_fn or self._fallback_embed_wrapper
        self._has_custom_embed_fn = embed_fn is not None
        self._embed_batch_fn = embed_batch_fn
        self._request_count = 0
        self._hit_count = 0
        self.store = store
        self._embed_history: deque = deque(maxlen=1000)

    async def _fallback_embed_wrapper(self, text: str) -> List[float]:
        return list(await self._fallback_embed(text, dim=self._fallback_hash_dimension()))

    async def _fallback_embed_batch_wrapper(self, texts: List[str]) -> List[List[float]]:
        results = []
        for t in texts:
            results.append(list(await self._fallback_embed(t, dim=self._fallback_hash_dimension())))
        return results

    async def embed(
        self,
        text: str,
        meta: Optional[Dict[str, Any]] = None,
        task_type: str = "general",
    ) -> EmbeddingRecord:
        """Return a cached or freshly computed embedding for *text*."""
        results = await self.embed_batch([text], meta=meta, task_type=task_type)
        return results[0]

    async def embed_batch(
        self,
        texts: List[str],
        meta: Optional[Dict[str, Any]] = None,
        task_type: str = "general",
    ) -> List[EmbeddingRecord]:
        """Return cached or freshly computed embeddings for a list of texts."""
        if not texts:
            return []

        project_id = (meta or {}).get("project_id")
        project_scope = str(project_id) if project_id else None
        source_hashes = [
            self._build_source_hash(
                t,
                task_type=task_type,
                project_id=project_scope,
            )
            for t in texts
        ]
        self._request_count += len(texts)

        # 1. Try cache
        results: List[Optional[EmbeddingRecord]] = [None] * len(texts)
        missing_indices: List[int] = []
        for i, source_hash in enumerate(source_hashes):
            cached = await self.cache.get(source_hash)
            if cached is not None and self._cached_record_is_usable(
                cached,
                project_id=project_scope,
            ):
                self._hit_count += 1
                results[i] = cached
            else:
                missing_indices.append(i)

        if not missing_indices:
            return results  # type: ignore

        # 2. Compute missing
        missing_texts = [texts[i] for i in missing_indices]
        started = time.perf_counter()

        actual_model_id = self.model_id
        degraded_reason: Optional[str] = None

        try:
            if self._embed_batch_fn is not None:
                vectors = await self._embed_batch_fn(missing_texts)
            elif self._has_custom_embed_fn:
                vectors = []
                for text in missing_texts:
                    vectors.append(await self._embed_fn(text))
            else:
                # Fallback to deterministic hashing. Production OpenCAS paths
                # should provide embed_fn/embed_batch_fn so model execution is
                # resolved by OpenLLMAuth, including local embedding runtimes.
                vectors = await self._fallback_embed_batch_wrapper(missing_texts)
                if self.model_id != self.offline_embedding_model_id:
                    actual_model_id = self.offline_embedding_model_id
                    degraded_reason = (
                        f"No live embedding provider available for {self.model_id}; "
                        "using deterministic hash fallback"
                    )
        except Exception as exc:
            degraded_reason = f"{type(exc).__name__}: {exc}"
            vectors = await self._fallback_embed_batch_wrapper(missing_texts)
            actual_model_id = self.offline_embedding_model_id

        coerced_vectors: List[List[float]] = []
        dimension_meta: List[Dict[str, Any]] = []
        for vector in vectors:
            coerced, meta_update = self._coerce_vector_dimension(vector)
            coerced_vectors.append(coerced)
            dimension_meta.append(meta_update)
        vectors = coerced_vectors

        latency_ms = (time.perf_counter() - started) * 1000
        # Record one history entry for the batch
        self._embed_history.append(
            {
                "timestamp": datetime.now(timezone.utc),
                "latency_ms": latency_ms,
                "batch_size": len(missing_texts),
                "task_type": task_type,
            }
        )

        # 3. Create records and update cache
        for i, idx in enumerate(missing_indices):
            text = texts[idx]
            source_hash = source_hashes[idx]
            vector = list(vectors[i])

            merged_meta = meta or {}
            merged_meta = {
                **merged_meta,
                "text": text,
                "task_type": task_type,
                "content_hash": hashlib.sha256(text.encode("utf-8")).hexdigest(),
            }
            if degraded_reason is not None:
                merged_meta["embedding_degraded"] = True
                merged_meta["embedding_degraded_reason"] = degraded_reason
            if actual_model_id != self.model_id:
                merged_meta["requested_model_id"] = self.model_id
            if dimension_meta[i]:
                merged_meta.update(dimension_meta[i])

            record = EmbeddingRecord(
                source_hash=source_hash,
                model_id=actual_model_id,
                dimension=len(vector),
                vector=vector,
                meta=merged_meta,
            )
            await self.cache.put(record)
            results[idx] = record

        return results  # type: ignore

    async def health(self) -> EmbeddingHealth:
        health = await self.cache.health()
        if self._request_count > 0:
            health.cache_hit_rate_1h = round(self._hit_count / self._request_count, 3)

        # Compute search-path metrics from ring buffer
        cutoff = datetime.now(timezone.utc) - timedelta(hours=1)
        recent = [h for h in self.cache._search_history if h["timestamp"] >= cutoff]
        health.semantic_success_count_1h = sum(
            1 for h in recent if h["path"] in ("qdrant", "hnsw")
        )
        health.lexical_fallback_count_1h = sum(
            1 for h in recent if h["path"] == "lexical_fallback"
        )
        if recent:
            health.avg_latency_ms_1h = round(
                sum(h["latency_ms"] for h in recent) / len(recent), 3
            )
        recent_embeds = [h for h in self._embed_history if h["timestamp"] >= cutoff]
        if recent_embeds:
            total_items = sum(max(1, int(h.get("batch_size", 1))) for h in recent_embeds)
            health.avg_embed_latency_ms_1h = round(
                sum(h["latency_ms"] for h in recent_embeds) / total_items, 3
            )

        # Compute ready_ratio from a sample of episodes
        if self.store is not None:
            try:
                sample = await self.store.list_episodes(compacted=False, limit=1000)
                if sample:
                    ready = sum(1 for ep in sample if ep.embedding_id)
                    health.ready_ratio = round(ready / len(sample), 3)
                else:
                    health.ready_ratio = 1.0
            except Exception:
                health.ready_ratio = 1.0
        else:
            health.ready_ratio = 1.0

        return health

    async def recent_records(self, limit: int = 10) -> List[EmbeddingRecord]:
        return await self.cache.recent_records(limit=limit)

    def _fallback_hash_dimension(self) -> int:
        if self.model_id == self.local_embedding_model_id:
            definition = ProviderManager.local_embedding_model_definition(self.local_embedding_model_id) or {}
            try:
                return int(definition.get("dimensions") or self.expected_dimension or 256)
            except (TypeError, ValueError):
                return int(self.expected_dimension or 256)
        return 256

    @staticmethod
    async def _fallback_embed(text: str, dim: int = 256) -> Sequence[float]:
        """Deterministic lightweight fallback embedder.

        Produces a vector based on character n-gram hashing.
        Good enough for testing and basic similarity when no model is available.
        """
        vec = np.zeros(dim, dtype=np.float32)
        text = text.lower()
        for i in range(len(text) - 2):
            tri = text[i : i + 3]
            idx = int(hashlib.md5(tri.encode()).hexdigest(), 16) % dim
            vec[idx] += 1.0
        norm = np.linalg.norm(vec)
        if norm > 0:
            vec = vec / norm
        return vec.tolist()

    def _build_source_hash(
        self,
        text: str,
        task_type: str,
        project_id: Optional[str] = None,
    ) -> str:
        payload = "\0".join([self.model_id, task_type, project_id or "", text])
        return hashlib.sha256(payload.encode("utf-8")).hexdigest()

    def _cached_record_is_usable(
        self,
        record: EmbeddingRecord,
        project_id: Optional[str] = None,
    ) -> bool:
        if record.model_id != self.model_id:
            return False
        if project_id is not None and record.meta.get("project_id") != project_id:
            return False
        if self.expected_dimension is not None and record.dimension != self.expected_dimension:
            return False
        return True

    def _coerce_vector_dimension(
        self,
        vector: Sequence[float],
    ) -> tuple[List[float], Dict[str, Any]]:
        vector_list = list(vector)
        actual_dimension = len(vector_list)
        if self.expected_dimension is None or actual_dimension == self.expected_dimension:
            return vector_list, {}
        raise ValueError(
            f"Embedding dimension mismatch for {self.model_id}: "
            f"expected {self.expected_dimension}, got {actual_dimension}"
        )
