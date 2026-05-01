"""Async SQLite memory store."""

from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path
from typing import Any, List, Optional

import aiosqlite

from .models import CompactionRecord, EdgeKind, Episode, EpisodeEdge, EpisodeKind, Memory
from .store_edges import (
    boost_edge_confidence as boost_edge_confidence_impl,
    decay_all_edges as decay_all_edges_impl,
    delete_edges_for as delete_edges_for_impl,
    edge_exists as edge_exists_impl,
    get_edges_for as get_edges_for_impl,
    get_edges_for_batch as get_edges_for_batch_impl,
    prune_weak_edges as prune_weak_edges_impl,
    save_edge as save_edge_impl,
    save_edges_batch as save_edges_batch_impl,
)
from .store_episodes import (
    delete_episodes as delete_episodes_impl,
    get_episode as get_episode_impl,
    get_episodes_by_ids as get_episodes_by_ids_impl,
    list_artifact_episodes as list_artifact_episodes_impl,
    list_episodes as list_episodes_impl,
    list_episodes_by_embedding_ids as list_episodes_by_embedding_ids_impl,
    list_identity_core_episodes as list_identity_core_episodes_impl,
    list_non_compacted_episodes as list_non_compacted_episodes_impl,
    list_recent_episodes as list_recent_episodes_impl,
    mark_compacted as mark_compacted_impl,
    mark_episode_failed as mark_episode_failed_impl,
    mark_episode_successful as mark_episode_successful_impl,
    prune_episodes_by_salience as prune_episodes_by_salience_impl,
    save_episode as save_episode_impl,
    save_episodes_batch as save_episodes_batch_impl,
    search_episodes_by_content as search_episodes_by_content_impl,
    touch_episode as touch_episode_impl,
    update_episode_affect as update_episode_affect_impl,
)
from .store_schema import MEMORY_STORE_MIGRATIONS, MEMORY_STORE_SCHEMA
from .store_serialization import (
    compaction_db_params,
    memory_db_params,
    row_to_memory as deserialize_memory,
)


class MemoryStore:
    """Async SQLite store for episodes, semantic memories, and compactions."""

    def __init__(self, path: Path | str) -> None:
        self.path = Path(path)
        self._db: aiosqlite.Connection | None = None

    async def connect(self) -> "MemoryStore":
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._db = await aiosqlite.connect(str(self.path))
        self._db.row_factory = aiosqlite.Row
        await self._executescript(MEMORY_STORE_SCHEMA)
        await self._migrate()
        return self

    async def _migrate(self) -> None:
        """Lightweight migrations for existing stores."""
        assert self._db is not None
        for sql in MEMORY_STORE_MIGRATIONS:
            try:
                await self._db.execute(sql)
            except Exception:
                pass  # column likely already exists
        await self._db.commit()

    async def close(self) -> None:
        if self._db:
            await self._db.close()
            self._db = None

    async def _executescript(self, sql: str) -> None:
        assert self._db is not None
        await self._db.executescript(sql)
        await self._db.commit()

    async def _execute(self, sql: str, parameters: tuple = ()) -> aiosqlite.Cursor:
        assert self._db is not None
        return await self._db.execute(sql, parameters)

    async def _executemany(self, sql: str, parameters: List[tuple]) -> None:
        assert self._db is not None
        await self._db.executemany(sql, parameters)
        await self._db.commit()

    # Episodes

    async def update_episode_embedding(self, episode_id: str, embedding_id: str) -> None:
        from .store_episodes import update_episode_embedding as update_episode_embedding_impl
        await update_episode_embedding_impl(self, episode_id, embedding_id)

    async def update_memory_embedding(self, memory_id: str, embedding_id: str) -> None:
        assert self._db is not None
        await self._db.execute(
            "UPDATE memories SET embedding_id = ? WHERE memory_id = ?",
            (embedding_id, memory_id),
        )
        await self._db.commit()

    async def save_episodes_batch(self, episodes: List[Episode]) -> None:
        await save_episodes_batch_impl(self, episodes)

    async def save_episode(self, episode: Episode) -> None:
        await save_episode_impl(self, episode)

    async def get_episode(self, episode_id: str) -> Optional[Episode]:
        return await get_episode_impl(self, episode_id)

    async def touch_episode(self, episode_id: str) -> None:
        await touch_episode_impl(self, episode_id)

    async def list_episodes(
        self,
        session_id: Optional[str] = None,
        compacted: Optional[bool] = None,
        limit: int = 100,
        offset: int = 0,
    ) -> List[Episode]:
        return await list_episodes_impl(
            self,
            session_id=session_id,
            compacted=compacted,
            limit=limit,
            offset=offset,
        )

    async def list_artifact_episodes(self, artifact_path: str) -> List[Episode]:
        return await list_artifact_episodes_impl(self, artifact_path)

    async def delete_episodes(self, episode_ids: List[str]) -> int:
        return await delete_episodes_impl(self, episode_ids)

    async def mark_compacted(self, episode_ids: List[str]) -> None:
        await mark_compacted_impl(self, episode_ids)

    async def search_episodes_by_content(
        self,
        query: str,
        limit: int = 20,
    ) -> List[Episode]:
        return await search_episodes_by_content_impl(self, query, limit=limit)

    async def list_recent_episodes(
        self,
        session_id: Optional[str] = None,
        limit: int = 20,
    ) -> List[Episode]:
        return await list_recent_episodes_impl(self, session_id=session_id, limit=limit)

    async def get_episodes_by_ids(self, episode_ids: List[str]) -> List[Episode]:
        return await get_episodes_by_ids_impl(self, episode_ids)

    async def update_episode_affect(
        self,
        episode_id: str,
        affect: Any,
    ) -> None:
        await update_episode_affect_impl(self, episode_id, affect)

    async def mark_episode_successful(self, episode_id: str) -> None:
        await mark_episode_successful_impl(self, episode_id)

    async def mark_episode_failed(self, episode_id: str) -> None:
        await mark_episode_failed_impl(self, episode_id)

    async def edge_exists(self, source_id: str, target_id: str) -> bool:
        return await edge_exists_impl(self, source_id, target_id)

    async def save_edge(self, edge: EpisodeEdge) -> None:
        await save_edge_impl(self, edge)

    async def save_edges_batch(self, edges: List[EpisodeEdge]) -> None:
        await save_edges_batch_impl(self, edges)

    async def get_edges_for(
        self,
        episode_id: str,
        min_confidence: float = 0.0,
        limit: int = 24,
        kind: Optional["EdgeKind"] = None,
    ) -> List[EpisodeEdge]:
        return await get_edges_for_impl(
            self,
            episode_id,
            min_confidence=min_confidence,
            limit=limit,
            kind=kind,
        )

    async def get_edges_for_batch(
        self,
        episode_ids: List[str],
        min_confidence: float = 0.0,
        kind: Optional["EdgeKind"] = None,
    ) -> List[EpisodeEdge]:
        return await get_edges_for_batch_impl(
            self,
            episode_ids,
            min_confidence=min_confidence,
            kind=kind,
        )

    async def delete_edges_for(self, episode_id: str) -> None:
        await delete_edges_for_impl(self, episode_id)

    async def boost_edge_confidence(
        self,
        episode_id: str,
        boost: float = 0.05,
        max_confidence: float = 1.0,
    ) -> int:
        return await boost_edge_confidence_impl(
            self,
            episode_id,
            boost=boost,
            max_confidence=max_confidence,
        )

    async def decay_all_edges(self, decay: float = 0.95) -> int:
        return await decay_all_edges_impl(self, decay=decay)

    async def prune_weak_edges(self, min_confidence: float = 0.05) -> int:
        return await prune_weak_edges_impl(self, min_confidence=min_confidence)

    # Memories

    async def save_memory(self, memory: Memory) -> None:
        assert self._db is not None
        await self._db.execute(
            """
            INSERT INTO memories (
                memory_id, created_at, updated_at, content, embedding_id,
                source_episode_ids, tags, salience, access_count, last_accessed,
                identity_mutagen, confidence_score
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            ON CONFLICT(memory_id) DO UPDATE SET
                updated_at = excluded.updated_at,
                content = excluded.content,
                embedding_id = excluded.embedding_id,
                source_episode_ids = excluded.source_episode_ids,
                tags = excluded.tags,
                salience = excluded.salience,
                access_count = excluded.access_count,
                last_accessed = excluded.last_accessed,
                identity_mutagen = excluded.identity_mutagen,
                confidence_score = excluded.confidence_score
            """,
            memory_db_params(memory),
        )
        await self._db.commit()

    async def get_memory(self, memory_id: str) -> Optional[Memory]:
        cursor = await self._execute(
            "SELECT * FROM memories WHERE memory_id = ?", (memory_id,)
        )
        row = await cursor.fetchone()
        if row is None:
            return None
        return self._row_to_memory(row)

    async def list_memories(self, limit: int = 100, offset: int = 0) -> List[Memory]:
        cursor = await self._execute(
            "SELECT * FROM memories ORDER BY salience DESC, updated_at DESC LIMIT ? OFFSET ?",
            (limit, offset),
        )
        rows = await cursor.fetchall()
        return [self._row_to_memory(r) for r in rows]

    async def touch_memory(self, memory_id: str) -> None:
        now = datetime.now(timezone.utc).isoformat()
        await self._execute(
            "UPDATE memories SET access_count = access_count + 1, last_accessed = ? WHERE memory_id = ?",
            (now, memory_id),
        )
        assert self._db is not None
        await self._db.commit()

    async def list_episodes_by_embedding_ids(
        self,
        embedding_ids: List[str],
    ) -> List[Episode]:
        return await list_episodes_by_embedding_ids_impl(self, embedding_ids)

    async def list_memories_by_embedding_ids(
        self,
        embedding_ids: List[str],
    ) -> List[Memory]:
        """Fetch memories whose embedding_id is in the given list."""
        if not embedding_ids:
            return []
        placeholders = ", ".join("?" for _ in embedding_ids)
        cursor = await self._execute(
            f"SELECT * FROM memories WHERE embedding_id IN ({placeholders})",
            tuple(embedding_ids),
        )
        rows = await cursor.fetchall()
        return [self._row_to_memory(r) for r in rows]

    async def list_memories_by_tag(
        self,
        tag: str,
        limit: int = 100,
        offset: int = 0,
    ) -> List[Memory]:
        """Return memories tagged with *tag*."""
        cursor = await self._execute(
            """
            SELECT * FROM memories
            WHERE EXISTS (
                SELECT 1 FROM json_each(tags)
                WHERE json_each.value = ?
            )
            ORDER BY salience DESC, updated_at DESC
            LIMIT ? OFFSET ?
            """,
            (tag, limit, offset),
        )
        rows = await cursor.fetchall()
        return [self._row_to_memory(r) for r in rows]

    async def search_memories_by_content(
        self,
        query: str,
        limit: int = 20,
    ) -> List[Memory]:
        """Return memory summaries whose content or tags mention *query*."""
        needle = query.strip().lower()
        if not needle:
            return []
        cursor = await self._execute(
            "SELECT * FROM memories WHERE LOWER(content) LIKE ?",
            (f"%{needle}%",),
        )
        rows = await cursor.fetchall()

        tokens = [
            token
            for token in needle.replace("_", " ").split()
            if len(token) >= 2
        ]
        scored: list[tuple[float, Memory]] = []
        for row in rows:
            memory = self._row_to_memory(row)
            content = memory.content.lower()
            tags = " ".join(memory.tags).lower()
            exact_hits = int(needle in content) + int(needle in tags)
            token_hits = sum(1 for token in tokens if token in content or token in tags)
            if exact_hits == 0 and token_hits == 0:
                continue
            score = (exact_hits * 3.0) + (token_hits / max(1, len(tokens)))
            score += min(memory.salience / 10.0, 0.2)
            scored.append((score, memory))

        scored.sort(
            key=lambda item: (item[0], item[1].updated_at, item[1].created_at),
            reverse=True,
        )
        return [memory for _, memory in scored[:limit]]

    async def list_non_compacted_episodes(
        self,
        session_id: Optional[str] = None,
        limit: int = 100,
    ) -> List[Episode]:
        return await list_non_compacted_episodes_impl(self, session_id=session_id, limit=limit)

    async def list_identity_core_episodes(
        self,
        limit: int = 100,
    ) -> List[Episode]:
        return await list_identity_core_episodes_impl(self, limit=limit)

    async def update_memory_salience(
        self,
        memory_id: str,
        salience: float,
    ) -> None:
        """Update the salience of a memory."""
        now = datetime.now(timezone.utc).isoformat()
        await self._execute(
            "UPDATE memories SET salience = ?, updated_at = ? WHERE memory_id = ?",
            (salience, now, memory_id),
        )
        assert self._db is not None
        await self._db.commit()

    async def prune_episodes_by_salience(
        self,
        threshold: float,
    ) -> int:
        return await prune_episodes_by_salience_impl(self, threshold)

    async def get_stats(self) -> Dict[str, Any]:
        """Return aggregate statistics for episodes, memories, and edges."""
        assert self._db is not None
        db = self._db

        async def _count(table: str) -> int:
            cursor = await db.execute(f"SELECT COUNT(*) FROM {table}")
            row = await cursor.fetchone()
            return row[0] if row else 0

        episode_count = await _count("episodes")
        memory_count = await _count("memories")
        edge_count = await _count("episode_edges")

        cursor = await db.execute("SELECT COUNT(*) FROM episodes WHERE compacted = 1")
        row = await cursor.fetchone()
        compacted_count = row[0] if row else 0

        cursor = await db.execute("SELECT COUNT(*) FROM episodes WHERE identity_core = 1")
        row = await cursor.fetchone()
        identity_core_count = row[0] if row else 0

        cursor = await db.execute("SELECT AVG(salience) FROM episodes")
        row = await cursor.fetchone()
        avg_salience = row[0] if row and row[0] is not None else 0.0

        cursor = await db.execute(
            "SELECT affect_primary, COUNT(*) as c FROM episodes WHERE affect_primary IS NOT NULL GROUP BY affect_primary"
        )
        rows = await cursor.fetchall()
        affect_distribution = {row["affect_primary"]: row["c"] for row in rows}

        return {
            "episode_count": episode_count,
            "memory_count": memory_count,
            "edge_count": edge_count,
            "compacted_count": compacted_count,
            "identity_core_count": identity_core_count,
            "avg_salience": round(avg_salience, 3),
            "affect_distribution": affect_distribution,
        }

    # Compactions

    async def record_compaction(self, record: CompactionRecord) -> None:
        assert self._db is not None
        await self._db.execute(
            """
            INSERT INTO compactions (
                compaction_id, created_at, episode_ids, summary, removed_count
            ) VALUES (?, ?, ?, ?, ?)
            """,
            compaction_db_params(record),
        )
        await self._db.commit()

    # Helpers

    @staticmethod
    def _row_to_memory(row: aiosqlite.Row) -> Memory:
        return deserialize_memory(row)
