"""Async SQLite memory store."""

from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path
from typing import Any, List, Optional

import aiosqlite

from .models import CompactionRecord, EdgeKind, Episode, EpisodeEdge, EpisodeKind, Memory


_SCHEMA = """
CREATE TABLE IF NOT EXISTS episodes (
    episode_id TEXT PRIMARY KEY,
    created_at TEXT NOT NULL,
    kind TEXT NOT NULL,
    session_id TEXT,
    content TEXT NOT NULL,
    embedding_id TEXT,
    somatic_tag TEXT,
    affect_primary TEXT,
    affect_valence REAL,
    affect_arousal REAL,
    affect_certainty REAL,
    affect_intensity REAL,
    affect_social_target TEXT,
    affect_tags TEXT,
    salience REAL NOT NULL DEFAULT 1.0,
    compacted INTEGER NOT NULL DEFAULT 0,
    identity_core INTEGER NOT NULL DEFAULT 0,
    confidence_score REAL NOT NULL DEFAULT 0.8,
    access_count INTEGER NOT NULL DEFAULT 0,
    last_accessed TEXT,
    used_successfully INTEGER NOT NULL DEFAULT 0,
    used_unsuccessfully INTEGER NOT NULL DEFAULT 0,
    payload TEXT NOT NULL DEFAULT '{}'
);

CREATE INDEX IF NOT EXISTS idx_episodes_created_at ON episodes(created_at);
CREATE INDEX IF NOT EXISTS idx_episodes_session_id ON episodes(session_id);
CREATE INDEX IF NOT EXISTS idx_episodes_compacted ON episodes(compacted);

CREATE TABLE IF NOT EXISTS memories (
    memory_id TEXT PRIMARY KEY,
    created_at TEXT NOT NULL,
    updated_at TEXT NOT NULL,
    content TEXT NOT NULL,
    embedding_id TEXT,
    source_episode_ids TEXT NOT NULL DEFAULT '[]',
    tags TEXT NOT NULL DEFAULT '[]',
    salience REAL NOT NULL DEFAULT 1.0,
    access_count INTEGER NOT NULL DEFAULT 0,
    last_accessed TEXT
);

CREATE INDEX IF NOT EXISTS idx_memories_salience ON memories(salience);

CREATE TABLE IF NOT EXISTS compactions (
    compaction_id TEXT PRIMARY KEY,
    created_at TEXT NOT NULL,
    episode_ids TEXT NOT NULL,
    summary TEXT NOT NULL,
    removed_count INTEGER NOT NULL
);

CREATE TABLE IF NOT EXISTS episode_edges (
    edge_id TEXT PRIMARY KEY,
    source_id TEXT NOT NULL,
    target_id TEXT NOT NULL,
    kind TEXT NOT NULL DEFAULT 'semantic',
    semantic_weight REAL NOT NULL DEFAULT 0.0,
    emotional_weight REAL NOT NULL DEFAULT 0.0,
    recency_weight REAL NOT NULL DEFAULT 0.0,
    structural_weight REAL NOT NULL DEFAULT 0.0,
    salience_weight REAL NOT NULL DEFAULT 0.0,
    causal_weight REAL NOT NULL DEFAULT 0.0,
    verification_weight REAL NOT NULL DEFAULT 0.0,
    actor_affinity_weight REAL NOT NULL DEFAULT 0.0,
    confidence REAL NOT NULL DEFAULT 0.0,
    created_at TEXT NOT NULL,
    UNIQUE(source_id, target_id)
);
CREATE INDEX IF NOT EXISTS idx_edges_source ON episode_edges(source_id);
CREATE INDEX IF NOT EXISTS idx_edges_target ON episode_edges(target_id);
CREATE INDEX IF NOT EXISTS idx_edges_confidence ON episode_edges(confidence);
CREATE INDEX IF NOT EXISTS idx_edges_kind ON episode_edges(kind);

CREATE VIRTUAL TABLE IF NOT EXISTS episodes_fts USING fts5(
    content,
    content='episodes',
    content_rowid='rowid'
);

CREATE TRIGGER IF NOT EXISTS episodes_fts_insert AFTER INSERT ON episodes BEGIN
    INSERT INTO episodes_fts(rowid, content) VALUES (new.rowid, new.content);
END;

CREATE TRIGGER IF NOT EXISTS episodes_fts_delete AFTER DELETE ON episodes BEGIN
    INSERT INTO episodes_fts(episodes_fts, rowid, content) VALUES ('delete', old.rowid, old.content);
END;

CREATE TRIGGER IF NOT EXISTS episodes_fts_update AFTER UPDATE ON episodes BEGIN
    INSERT INTO episodes_fts(episodes_fts, rowid, content) VALUES ('delete', old.rowid, old.content);
    INSERT INTO episodes_fts(rowid, content) VALUES (new.rowid, new.content);
END;
"""


class MemoryStore:
    """Async SQLite store for episodes, semantic memories, and compactions."""

    def __init__(self, path: Path | str) -> None:
        self.path = Path(path)
        self._db: aiosqlite.Connection | None = None

    async def connect(self) -> "MemoryStore":
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._db = await aiosqlite.connect(str(self.path))
        self._db.row_factory = aiosqlite.Row
        await self._executescript(_SCHEMA)
        await self._migrate()
        return self

    async def _migrate(self) -> None:
        """Lightweight migrations for existing stores."""
        assert self._db is not None
        migrations = [
            "ALTER TABLE episodes ADD COLUMN affect_primary TEXT",
            "ALTER TABLE episodes ADD COLUMN affect_valence REAL",
            "ALTER TABLE episodes ADD COLUMN affect_arousal REAL",
            "ALTER TABLE episodes ADD COLUMN affect_certainty REAL",
            "ALTER TABLE episodes ADD COLUMN affect_intensity REAL",
            "ALTER TABLE episodes ADD COLUMN affect_social_target TEXT",
            "ALTER TABLE episodes ADD COLUMN affect_tags TEXT",
            "ALTER TABLE episodes ADD COLUMN identity_core INTEGER NOT NULL DEFAULT 0",
            "ALTER TABLE episode_edges ADD COLUMN kind TEXT NOT NULL DEFAULT 'semantic'",
            "ALTER TABLE episodes ADD COLUMN confidence_score REAL NOT NULL DEFAULT 0.8",
            "ALTER TABLE episodes ADD COLUMN access_count INTEGER NOT NULL DEFAULT 0",
            "ALTER TABLE episodes ADD COLUMN last_accessed TEXT",
            "ALTER TABLE episodes ADD COLUMN used_successfully INTEGER NOT NULL DEFAULT 0",
            "ALTER TABLE episodes ADD COLUMN used_unsuccessfully INTEGER NOT NULL DEFAULT 0",
            "ALTER TABLE episode_edges ADD COLUMN salience_weight REAL NOT NULL DEFAULT 0.0",
            "ALTER TABLE episode_edges ADD COLUMN causal_weight REAL NOT NULL DEFAULT 0.0",
            "ALTER TABLE episode_edges ADD COLUMN verification_weight REAL NOT NULL DEFAULT 0.0",
            "ALTER TABLE episode_edges ADD COLUMN actor_affinity_weight REAL NOT NULL DEFAULT 0.0",
        ]
        for sql in migrations:
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

    async def save_episodes_batch(self, episodes: List[Episode]) -> None:
        """Batch insert or update episodes in a single transaction."""
        assert self._db is not None
        params: List[tuple] = []
        for episode in episodes:
            affect = episode.affect
            params.append(
                (
                    str(episode.episode_id),
                    episode.created_at.isoformat(),
                    episode.kind.value,
                    episode.session_id,
                    episode.content,
                    episode.embedding_id,
                    episode.somatic_tag,
                    affect.primary_emotion.value if affect else None,
                    affect.valence if affect else None,
                    affect.arousal if affect else None,
                    affect.certainty if affect else None,
                    affect.intensity if affect else None,
                    affect.social_target.value if affect else None,
                    (
                        __import__("json").dumps(affect.emotion_tags)
                        if affect and affect.emotion_tags
                        else None
                    ),
                    episode.salience,
                    int(episode.compacted),
                    int(episode.identity_core),
                    episode.confidence_score,
                    episode.access_count,
                    episode.last_accessed.isoformat() if episode.last_accessed else None,
                    episode.used_successfully,
                    episode.used_unsuccessfully,
                    episode.model_dump_json(include={"payload"}),
                )
            )
        await self._db.executemany(
            """
            INSERT INTO episodes (
                episode_id, created_at, kind, session_id, content,
                embedding_id, somatic_tag, affect_primary, affect_valence,
                affect_arousal, affect_certainty, affect_intensity,
                affect_social_target, affect_tags, salience, compacted,
                identity_core, confidence_score, access_count, last_accessed,
                used_successfully, used_unsuccessfully, payload
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            ON CONFLICT(episode_id) DO UPDATE SET
                created_at = excluded.created_at,
                kind = excluded.kind,
                session_id = excluded.session_id,
                content = excluded.content,
                embedding_id = excluded.embedding_id,
                somatic_tag = excluded.somatic_tag,
                affect_primary = excluded.affect_primary,
                affect_valence = excluded.affect_valence,
                affect_arousal = excluded.affect_arousal,
                affect_certainty = excluded.affect_certainty,
                affect_intensity = excluded.affect_intensity,
                affect_social_target = excluded.affect_social_target,
                affect_tags = excluded.affect_tags,
                salience = excluded.salience,
                compacted = excluded.compacted,
                identity_core = excluded.identity_core,
                confidence_score = excluded.confidence_score,
                access_count = excluded.access_count,
                last_accessed = excluded.last_accessed,
                used_successfully = excluded.used_successfully,
                used_unsuccessfully = excluded.used_unsuccessfully,
                payload = excluded.payload
            """,
            params,
        )
        await self._db.commit()

    async def save_episode(self, episode: Episode) -> None:
        assert self._db is not None
        affect = episode.affect
        await self._db.execute(
            """
            INSERT INTO episodes (
                episode_id, created_at, kind, session_id, content,
                embedding_id, somatic_tag, affect_primary, affect_valence,
                affect_arousal, affect_certainty, affect_intensity,
                affect_social_target, affect_tags, salience, compacted,
                identity_core, confidence_score, access_count, last_accessed,
                used_successfully, used_unsuccessfully, payload
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            ON CONFLICT(episode_id) DO UPDATE SET
                created_at = excluded.created_at,
                kind = excluded.kind,
                session_id = excluded.session_id,
                content = excluded.content,
                embedding_id = excluded.embedding_id,
                somatic_tag = excluded.somatic_tag,
                affect_primary = excluded.affect_primary,
                affect_valence = excluded.affect_valence,
                affect_arousal = excluded.affect_arousal,
                affect_certainty = excluded.affect_certainty,
                affect_intensity = excluded.affect_intensity,
                affect_social_target = excluded.affect_social_target,
                affect_tags = excluded.affect_tags,
                salience = excluded.salience,
                compacted = excluded.compacted,
                identity_core = excluded.identity_core,
                confidence_score = excluded.confidence_score,
                access_count = excluded.access_count,
                last_accessed = excluded.last_accessed,
                used_successfully = excluded.used_successfully,
                used_unsuccessfully = excluded.used_unsuccessfully,
                payload = excluded.payload
            """,
            (
                str(episode.episode_id),
                episode.created_at.isoformat(),
                episode.kind.value,
                episode.session_id,
                episode.content,
                episode.embedding_id,
                episode.somatic_tag,
                affect.primary_emotion.value if affect else None,
                affect.valence if affect else None,
                affect.arousal if affect else None,
                affect.certainty if affect else None,
                affect.intensity if affect else None,
                affect.social_target.value if affect else None,
                (
                    __import__("json").dumps(affect.emotion_tags)
                    if affect and affect.emotion_tags
                    else None
                ),
                episode.salience,
                int(episode.compacted),
                int(episode.identity_core),
                episode.confidence_score,
                episode.access_count,
                episode.last_accessed.isoformat() if episode.last_accessed else None,
                episode.used_successfully,
                episode.used_unsuccessfully,
                episode.model_dump_json(include={"payload"}),
            ),
        )
        await self._db.commit()

    async def get_episode(self, episode_id: str) -> Optional[Episode]:
        cursor = await self._execute(
            "SELECT * FROM episodes WHERE episode_id = ?", (episode_id,)
        )
        row = await cursor.fetchone()
        if row is None:
            return None
        return self._row_to_episode(row)

    async def touch_episode(self, episode_id: str) -> None:
        """Increment episodic access counters when retrieval surfaces an episode."""
        now = datetime.now(timezone.utc).isoformat()
        await self._execute(
            "UPDATE episodes SET access_count = access_count + 1, last_accessed = ? WHERE episode_id = ?",
            (now, episode_id),
        )
        assert self._db is not None
        await self._db.commit()

    async def list_episodes(
        self,
        session_id: Optional[str] = None,
        compacted: Optional[bool] = None,
        limit: int = 100,
        offset: int = 0,
    ) -> List[Episode]:
        conditions: List[str] = []
        params: List[Any] = []
        if session_id is not None:
            conditions.append("session_id = ?")
            params.append(session_id)
        if compacted is not None:
            conditions.append("compacted = ?")
            params.append(int(compacted))
        where = "WHERE " + " AND ".join(conditions) if conditions else ""
        cursor = await self._execute(
            f"""
            SELECT * FROM episodes
            {where}
            ORDER BY created_at DESC
            LIMIT ? OFFSET ?
            """,
            (*params, limit, offset),
        )
        rows = await cursor.fetchall()
        return [self._row_to_episode(r) for r in rows]

    async def list_artifact_episodes(self, artifact_path: str) -> List[Episode]:
        """Return artifact-backed episodes for a specific relative artifact path."""
        assert self._db is not None
        cursor = await self._db.execute(
            """
            SELECT * FROM episodes
            WHERE json_extract(payload, '$.payload.artifact.path') = ?
            ORDER BY CAST(COALESCE(json_extract(payload, '$.payload.artifact.chunk_index'), 0) AS INTEGER) ASC
            """,
            (artifact_path,),
        )
        rows = await cursor.fetchall()
        return [self._row_to_episode(r) for r in rows]

    async def delete_episodes(self, episode_ids: List[str]) -> int:
        """Delete a set of episodes by id."""
        if not episode_ids:
            return 0
        assert self._db is not None
        placeholders = ", ".join("?" for _ in episode_ids)
        cursor = await self._db.execute(
            f"DELETE FROM episodes WHERE episode_id IN ({placeholders})",
            tuple(episode_ids),
        )
        await self._db.commit()
        return int(cursor.rowcount or 0)

    async def mark_compacted(self, episode_ids: List[str]) -> None:
        if not episode_ids:
            return
        placeholders = ", ".join("?" for _ in episode_ids)
        await self._execute(
            f"UPDATE episodes SET compacted = 1 WHERE episode_id IN ({placeholders})",
            tuple(episode_ids),
        )
        assert self._db is not None
        await self._db.commit()

    async def search_episodes_by_content(
        self,
        query: str,
        limit: int = 20,
    ) -> List[Episode]:
        """Search episodes using FTS5 over content."""
        assert self._db is not None
        
        # Escape quotes to prevent FTS syntax errors
        escaped_query = query.replace('"', '""')
        # Wrap query in quotes for safe exact matching
        safe_query = f'"{escaped_query}"'
        
        cursor = await self._db.execute(
            """
            SELECT e.* FROM episodes e
            JOIN episodes_fts fts ON e.rowid = fts.rowid
            WHERE episodes_fts MATCH ?
            ORDER BY fts.rank
            LIMIT ?
            """,
            (safe_query, limit),
        )
        rows = await cursor.fetchall()
        return [self._row_to_episode(r) for r in rows]

    async def list_recent_episodes(
        self,
        session_id: Optional[str] = None,
        limit: int = 20,
    ) -> List[Episode]:
        """Return the most recent episodes, optionally filtered by session."""
        return await self.list_episodes(session_id=session_id, limit=limit)

    async def get_episodes_by_ids(self, episode_ids: List[str]) -> List[Episode]:
        """Fetch multiple episodes by their IDs."""
        if not episode_ids:
            return []
        placeholders = ", ".join("?" for _ in episode_ids)
        cursor = await self._execute(
            f"SELECT * FROM episodes WHERE episode_id IN ({placeholders})",
            tuple(episode_ids),
        )
        rows = await cursor.fetchall()
        return [self._row_to_episode(r) for r in rows]

    async def update_episode_affect(
        self,
        episode_id: str,
        affect: Any,
    ) -> None:
        """Update the affect fields of an episode."""
        import json

        assert self._db is not None
        await self._db.execute(
            """
            UPDATE episodes SET
                affect_primary = ?,
                affect_valence = ?,
                affect_arousal = ?,
                affect_certainty = ?,
                affect_intensity = ?,
                affect_social_target = ?,
                affect_tags = ?
            WHERE episode_id = ?
            """,
            (
                affect.primary_emotion.value if affect else None,
                affect.valence if affect else None,
                affect.arousal if affect else None,
                affect.certainty if affect else None,
                affect.intensity if affect else None,
                affect.social_target.value if affect else None,
                json.dumps(affect.emotion_tags) if affect and getattr(affect, "emotion_tags", None) else None,
                episode_id,
            ),
        )
        await self._db.commit()

    async def mark_episode_successful(self, episode_id: str) -> None:
        """Increment the successful-use counter for an episode."""
        assert self._db is not None
        await self._db.execute(
            "UPDATE episodes SET used_successfully = used_successfully + 1 WHERE episode_id = ?",
            (episode_id,),
        )
        await self._db.commit()

    async def mark_episode_failed(self, episode_id: str) -> None:
        """Increment the unsuccessful-use counter for an episode."""
        assert self._db is not None
        await self._db.execute(
            "UPDATE episodes SET used_unsuccessfully = used_unsuccessfully + 1 WHERE episode_id = ?",
            (episode_id,),
        )
        await self._db.commit()

    async def edge_exists(self, source_id: str, target_id: str) -> bool:
        """Check whether an edge already exists between two episodes."""
        assert self._db is not None
        cursor = await self._db.execute(
            "SELECT 1 FROM episode_edges WHERE source_id = ? AND target_id = ? LIMIT 1",
            (source_id, target_id),
        )
        row = await cursor.fetchone()
        return row is not None

    async def save_edge(self, edge: EpisodeEdge) -> None:
        """Upsert an episode edge."""
        assert self._db is not None
        await self._db.execute(
            """
            INSERT INTO episode_edges (
                edge_id, source_id, target_id, kind, semantic_weight,
                emotional_weight, recency_weight, structural_weight,
                salience_weight, causal_weight, verification_weight, actor_affinity_weight,
                confidence, created_at
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            ON CONFLICT(source_id, target_id) DO UPDATE SET
                kind = excluded.kind,
                semantic_weight = excluded.semantic_weight,
                emotional_weight = excluded.emotional_weight,
                recency_weight = excluded.recency_weight,
                structural_weight = excluded.structural_weight,
                salience_weight = excluded.salience_weight,
                causal_weight = excluded.causal_weight,
                verification_weight = excluded.verification_weight,
                actor_affinity_weight = excluded.actor_affinity_weight,
                confidence = excluded.confidence,
                created_at = excluded.created_at
            """,
            (
                str(edge.edge_id),
                edge.source_id,
                edge.target_id,
                edge.kind.value,
                edge.semantic_weight,
                edge.emotional_weight,
                edge.recency_weight,
                edge.structural_weight,
                edge.salience_weight,
                edge.causal_weight,
                edge.verification_weight,
                edge.actor_affinity_weight,
                edge.confidence,
                edge.created_at.isoformat(),
            ),
        )
        await self._db.commit()

    async def save_edges_batch(self, edges: List[EpisodeEdge]) -> None:
        """Batch upsert episode edges."""
        assert self._db is not None
        params: List[tuple] = []
        for edge in edges:
            params.append(
                (
                    str(edge.edge_id),
                    edge.source_id,
                    edge.target_id,
                    edge.kind.value,
                    edge.semantic_weight,
                    edge.emotional_weight,
                    edge.recency_weight,
                    edge.structural_weight,
                    edge.salience_weight,
                    edge.causal_weight,
                    edge.verification_weight,
                    edge.actor_affinity_weight,
                    edge.confidence,
                    edge.created_at.isoformat(),
                )
            )
        await self._db.executemany(
            """
            INSERT INTO episode_edges (
                edge_id, source_id, target_id, kind, semantic_weight,
                emotional_weight, recency_weight, structural_weight,
                salience_weight, causal_weight, verification_weight, actor_affinity_weight,
                confidence, created_at
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            ON CONFLICT(source_id, target_id) DO UPDATE SET
                kind = excluded.kind,
                semantic_weight = excluded.semantic_weight,
                emotional_weight = excluded.emotional_weight,
                recency_weight = excluded.recency_weight,
                structural_weight = excluded.structural_weight,
                salience_weight = excluded.salience_weight,
                causal_weight = excluded.causal_weight,
                verification_weight = excluded.verification_weight,
                actor_affinity_weight = excluded.actor_affinity_weight,
                confidence = excluded.confidence,
                created_at = excluded.created_at
            """,
            params,
        )
        await self._db.commit()

    async def get_edges_for(
        self,
        episode_id: str,
        min_confidence: float = 0.0,
        limit: int = 24,
        kind: Optional["EdgeKind"] = None,
    ) -> List[EpisodeEdge]:
        """Return edges connected to an episode, ordered by confidence desc."""
        assert self._db is not None
        if kind is not None:
            cursor = await self._db.execute(
                """
                SELECT * FROM episode_edges
                WHERE (source_id = ? OR target_id = ?)
                  AND confidence >= ?
                  AND kind = ?
                ORDER BY confidence DESC, created_at DESC
                LIMIT ?
                """,
                (episode_id, episode_id, min_confidence, kind.value, limit),
            )
        else:
            cursor = await self._db.execute(
                """
                SELECT * FROM episode_edges
                WHERE (source_id = ? OR target_id = ?)
                  AND confidence >= ?
                ORDER BY confidence DESC, created_at DESC
                LIMIT ?
                """,
                (episode_id, episode_id, min_confidence, limit),
            )
        rows = await cursor.fetchall()
        return [self._row_to_edge(r) for r in rows]

    async def get_edges_for_batch(
        self,
        episode_ids: List[str],
        min_confidence: float = 0.0,
        kind: Optional["EdgeKind"] = None,
    ) -> List[EpisodeEdge]:
        """Return edges connected to a set of episodes, ordered by confidence desc."""
        if not episode_ids:
            return []
        assert self._db is not None
        placeholders = ", ".join("?" for _ in episode_ids)
        if kind is not None:
            cursor = await self._db.execute(
                f"""
                SELECT * FROM episode_edges
                WHERE (source_id IN ({placeholders}) OR target_id IN ({placeholders}))
                  AND confidence >= ?
                  AND kind = ?
                ORDER BY confidence DESC, created_at DESC
                """,
                (*episode_ids, *episode_ids, min_confidence, kind.value),
            )
        else:
            cursor = await self._db.execute(
                f"""
                SELECT * FROM episode_edges
                WHERE (source_id IN ({placeholders}) OR target_id IN ({placeholders}))
                  AND confidence >= ?
                ORDER BY confidence DESC, created_at DESC
                """,
                (*episode_ids, *episode_ids, min_confidence),
            )
        rows = await cursor.fetchall()
        return [self._row_to_edge(r) for r in rows]

    async def delete_edges_for(self, episode_id: str) -> None:
        """Delete all edges referencing an episode."""
        assert self._db is not None
        await self._db.execute(
            "DELETE FROM episode_edges WHERE source_id = ? OR target_id = ?",
            (episode_id, episode_id),
        )
        await self._db.commit()

    async def boost_edge_confidence(
        self,
        episode_id: str,
        boost: float = 0.05,
        max_confidence: float = 1.0,
    ) -> int:
        """Boost confidence of all edges connected to *episode_id*."""
        assert self._db is not None
        cursor = await self._db.execute(
            """
            UPDATE episode_edges
            SET confidence = min(?, confidence + ?)
            WHERE source_id = ? OR target_id = ?
            """,
            (max_confidence, boost, episode_id, episode_id),
        )
        await self._db.commit()
        return cursor.rowcount

    async def decay_all_edges(self, decay: float = 0.95) -> int:
        """Decay confidence of all edges by *decay* factor."""
        assert self._db is not None
        cursor = await self._db.execute(
            "UPDATE episode_edges SET confidence = confidence * ?",
            (decay,),
        )
        await self._db.commit()
        return cursor.rowcount

    async def prune_weak_edges(self, min_confidence: float = 0.05) -> int:
        """Remove edges whose confidence fell below *min_confidence*."""
        assert self._db is not None
        cursor = await self._db.execute(
            "DELETE FROM episode_edges WHERE confidence < ?",
            (min_confidence,),
        )
        await self._db.commit()
        return cursor.rowcount

    @staticmethod
    def _row_to_edge(row: aiosqlite.Row) -> EpisodeEdge:
        return EpisodeEdge(
            edge_id=row["edge_id"],
            source_id=row["source_id"],
            target_id=row["target_id"],
            kind=EdgeKind(row["kind"]) if row["kind"] else EdgeKind.SEMANTIC,
            semantic_weight=row["semantic_weight"],
            emotional_weight=row["emotional_weight"],
            recency_weight=row["recency_weight"],
            structural_weight=row["structural_weight"],
            salience_weight=row["salience_weight"],
            causal_weight=row["causal_weight"],
            verification_weight=row["verification_weight"],
            actor_affinity_weight=row["actor_affinity_weight"],
            confidence=row["confidence"],
            created_at=datetime.fromisoformat(row["created_at"]),
        )

    # Memories

    async def save_memory(self, memory: Memory) -> None:
        assert self._db is not None
        await self._db.execute(
            """
            INSERT INTO memories (
                memory_id, created_at, updated_at, content, embedding_id,
                source_episode_ids, tags, salience, access_count, last_accessed
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            ON CONFLICT(memory_id) DO UPDATE SET
                updated_at = excluded.updated_at,
                content = excluded.content,
                embedding_id = excluded.embedding_id,
                source_episode_ids = excluded.source_episode_ids,
                tags = excluded.tags,
                salience = excluded.salience,
                access_count = excluded.access_count,
                last_accessed = excluded.last_accessed
            """,
            (
                str(memory.memory_id),
                memory.created_at.isoformat(),
                memory.updated_at.isoformat(),
                memory.content,
                memory.embedding_id,
                memory.model_dump_json(include={"source_episode_ids"}),
                memory.model_dump_json(include={"tags"}),
                memory.salience,
                memory.access_count,
                memory.last_accessed.isoformat() if memory.last_accessed else None,
            ),
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
        """Fetch episodes whose embedding_id is in the given list."""
        if not embedding_ids:
            return []
        placeholders = ", ".join("?" for _ in embedding_ids)
        cursor = await self._execute(
            f"SELECT * FROM episodes WHERE embedding_id IN ({placeholders})",
            tuple(embedding_ids),
        )
        rows = await cursor.fetchall()
        return [self._row_to_episode(r) for r in rows]

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
            WHERE tags LIKE ?
            ORDER BY salience DESC, updated_at DESC
            LIMIT ? OFFSET ?
            """,
            (f'%"{tag}"%', limit, offset),
        )
        rows = await cursor.fetchall()
        return [self._row_to_memory(r) for r in rows]

    async def list_non_compacted_episodes(
        self,
        session_id: Optional[str] = None,
        limit: int = 100,
    ) -> List[Episode]:
        """Return episodes that have not been compacted, oldest first."""
        eps = await self.list_episodes(
            session_id=session_id, compacted=False, limit=limit
        )
        eps.reverse()
        return eps

    async def list_identity_core_episodes(
        self,
        limit: int = 100,
    ) -> List[Episode]:
        """Return episodes marked as identity core, most recent first."""
        assert self._db is not None
        cursor = await self._db.execute(
            "SELECT * FROM episodes WHERE identity_core = 1 ORDER BY created_at DESC LIMIT ?",
            (limit,),
        )
        rows = await cursor.fetchall()
        return [self._row_to_episode(r) for r in rows]

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
        """Remove episodes with salience below threshold. Returns deleted count."""
        assert self._db is not None
        cursor = await self._db.execute(
            "DELETE FROM episodes WHERE salience < ?",
            (threshold,),
        )
        await self._db.commit()
        return cursor.rowcount

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
            (
                str(record.compaction_id),
                record.created_at.isoformat(),
                record.model_dump_json(include={"episode_ids"}),
                record.summary,
                record.removed_count,
            ),
        )
        await self._db.commit()

    # Helpers

    @staticmethod
    def _row_to_episode(row: aiosqlite.Row) -> Episode:
        import json

        raw_payload = row["payload"]
        payload = json.loads(raw_payload) if raw_payload else {}
        # The payload column stores {"payload": {...}} because of model_dump_json include
        if "payload" in payload:
            payload = payload["payload"]

        affect = None
        if row["affect_primary"] is not None:
            from opencas.somatic.models import AffectState, PrimaryEmotion, SocialTarget

            raw_tags = row["affect_tags"]
            tags = json.loads(raw_tags) if raw_tags else []
            affect = AffectState(
                primary_emotion=PrimaryEmotion(row["affect_primary"]),
                valence=row["affect_valence"] or 0.0,
                arousal=row["affect_arousal"] or 0.0,
                certainty=row["affect_certainty"] or 0.0,
                intensity=row["affect_intensity"] or 0.0,
                social_target=SocialTarget(row["affect_social_target"] or "user"),
                emotion_tags=tags,
            )

        return Episode(
            episode_id=row["episode_id"],
            created_at=datetime.fromisoformat(row["created_at"]),
            kind=EpisodeKind(row["kind"]),
            session_id=row["session_id"],
            content=row["content"],
            embedding_id=row["embedding_id"],
            somatic_tag=row["somatic_tag"],
            affect=affect,
            salience=row["salience"],
            compacted=bool(row["compacted"]),
            identity_core=bool(row["identity_core"]),
            confidence_score=row["confidence_score"],
            access_count=row["access_count"] if row["access_count"] is not None else 0,
            last_accessed=datetime.fromisoformat(row["last_accessed"]) if row["last_accessed"] else None,
            used_successfully=row["used_successfully"],
            used_unsuccessfully=row["used_unsuccessfully"],
            payload=payload,
        )

    @staticmethod
    def _row_to_memory(row: aiosqlite.Row) -> Memory:
        import json

        raw_source_ids = row["source_episode_ids"]
        source_ids = json.loads(raw_source_ids) if raw_source_ids else []
        if isinstance(source_ids, dict) and "source_episode_ids" in source_ids:
            source_ids = source_ids["source_episode_ids"]
        raw_tags = row["tags"]
        tags = json.loads(raw_tags) if raw_tags else []
        if isinstance(tags, dict) and "tags" in tags:
            tags = tags["tags"]
        return Memory(
            memory_id=row["memory_id"],
            created_at=datetime.fromisoformat(row["created_at"]),
            updated_at=datetime.fromisoformat(row["updated_at"]),
            content=row["content"],
            embedding_id=row["embedding_id"],
            source_episode_ids=source_ids,
            tags=tags,
            salience=row["salience"],
            access_count=row["access_count"],
            last_accessed=datetime.fromisoformat(row["last_accessed"]) if row["last_accessed"] else None,
        )
