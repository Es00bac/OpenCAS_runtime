"""Embedding service with compute-once, cache-many semantics."""

from __future__ import annotations

import hashlib
import json
import time
from collections import deque
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Callable, Coroutine, Dict, List, Optional, Sequence

import numpy as np

from opencas.embeddings.models import EmbeddingHealth, EmbeddingRecord


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
        self._db = await aiosqlite.connect(str(self.db_path))
        await self._db.execute("PRAGMA journal_mode=WAL")
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

    async def put(self, record: EmbeddingRecord) -> None:
        assert self._db is not None
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
        if self.vector_backend is not None:
            try:
                await self.vector_backend.upsert(record)
            except Exception:
                pass
        if self.hnsw_backend is not None:
            try:
                await self.hnsw_backend.upsert(record)
            except Exception:
                pass

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

    async def search_similar(
        self,
        vector: Sequence[float],
        limit: int = 10,
        model_id: Optional[str] = None,
        project_id: Optional[str] = None,
        query_text: Optional[str] = None,
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
                    with_scores=True,
                )
                if hits:
                    scored: List[tuple[EmbeddingRecord, float]] = []
                    for source_hash, sim in hits:
                        record = await self.get(source_hash)
                        if record is not None:
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
                    with_scores=True,
                )
                if hits:
                    scored = []
                    for source_hash, sim in hits:
                        record = await self.get(source_hash)
                        if record is not None:
                            scored.append((record, sim))
                    if scored:
                        latency_ms = (time.perf_counter() - start) * 1000
                        self._record_search_path("hnsw", latency_ms)
                        return scored
            except Exception:
                pass

        # Tier 3: Filtered SQLite scan
        if model_id is not None:
            cursor = await self._db.execute(
                "SELECT * FROM embeddings WHERE model_id = ?", (model_id,)
            )
        else:
            cursor = await self._db.execute("SELECT * FROM embeddings")
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
        model_id: str = "local-fallback",
        store=None,
    ) -> None:
        self.cache = cache
        self.model_id = model_id
        self._embed_fn = embed_fn or self._fallback_embed
        self._request_count = 0
        self._hit_count = 0
        self.store = store
        self._embed_history: deque = deque(maxlen=1000)

    async def embed(
        self,
        text: str,
        meta: Optional[Dict[str, Any]] = None,
        task_type: str = "general",
    ) -> EmbeddingRecord:
        """Return a cached or freshly computed embedding for *text*."""
        source_hash = self._build_source_hash(text, task_type=task_type)
        self._request_count += 1
        cached = await self.cache.get(source_hash)
        if cached is not None:
            self._hit_count += 1
            return cached

        started = time.perf_counter()
        degraded_reason: Optional[str] = None
        try:
            vector = list(await self._embed_fn(text))
        except Exception as exc:
            degraded_reason = f"{type(exc).__name__}: {exc}"
            vector = list(await self._fallback_embed(text))
        latency_ms = (time.perf_counter() - started) * 1000
        self._embed_history.append(
            {
                "timestamp": datetime.now(timezone.utc),
                "latency_ms": latency_ms,
                "task_type": task_type,
            }
        )
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
        record = EmbeddingRecord(
            source_hash=source_hash,
            model_id=self.model_id,
            dimension=len(vector),
            vector=vector,
            meta=merged_meta,
        )
        await self.cache.put(record)
        return record

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
            health.avg_embed_latency_ms_1h = round(
                sum(h["latency_ms"] for h in recent_embeds) / len(recent_embeds), 3
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

    @staticmethod
    async def _fallback_embed(text: str) -> Sequence[float]:
        """Deterministic lightweight fallback embedder.

        Produces a 256-dim vector based on character n-gram hashing.
        Good enough for testing and basic similarity when no model is available.
        """
        dim = 256
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

    def _build_source_hash(self, text: str, task_type: str) -> str:
        payload = "\0".join([self.model_id, task_type, text])
        return hashlib.sha256(payload.encode("utf-8")).hexdigest()
