"""Portfolio clustering for daydream sparks and work items."""

from __future__ import annotations

import re
import sqlite3
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional
from uuid import UUID, uuid4

import aiosqlite
from pydantic import BaseModel, Field

_STOP_WORDS = {
    "the", "a", "an", "is", "are", "was", "were", "be", "been", "being",
    "have", "has", "had", "do", "does", "did", "will", "would", "could",
    "should", "may", "might", "must", "shall", "can", "need", "dare",
    "ought", "used", "to", "of", "in", "for", "on", "with", "at", "by",
    "from", "as", "into", "through", "during", "before", "after",
    "above", "below", "between", "out", "off", "over", "under", "again",
    "further", "then", "once", "here", "there", "when", "where", "why",
    "how", "all", "each", "few", "more", "most", "other", "some", "such",
    "no", "nor", "not", "only", "own", "same", "so", "than", "too", "very",
    "just", "and", "but", "if", "or", "because", "until", "while", "this",
    "that", "these", "those", "i", "me", "my", "myself", "we", "our",
    "you", "your", "he", "him", "his", "she", "her", "it", "its", "they",
    "them", "their", "what", "which", "who", "whom", "about", "up", "down",
}


class PortfolioCluster(BaseModel):
    """A thematic cluster of sparks and initiatives."""

    cluster_id: UUID = Field(default_factory=uuid4)
    created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    updated_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    fascination_key: str
    spark_count: int = 0
    initiative_count: int = 0
    artifact_count: int = 0
    last_touched_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    tags: List[str] = Field(default_factory=list)
    embedding_id: Optional[str] = None
    meta: Dict[str, Any] = Field(default_factory=dict)


_SCHEMA = """
CREATE TABLE IF NOT EXISTS portfolio_clusters (
    cluster_id TEXT PRIMARY KEY,
    created_at TEXT NOT NULL,
    updated_at TEXT NOT NULL,
    fascination_key TEXT NOT NULL UNIQUE,
    spark_count INTEGER NOT NULL DEFAULT 0,
    initiative_count INTEGER NOT NULL DEFAULT 0,
    artifact_count INTEGER NOT NULL DEFAULT 0,
    last_touched_at TEXT NOT NULL,
    tags TEXT NOT NULL DEFAULT '[]',
    embedding_id TEXT,
    meta TEXT NOT NULL DEFAULT '{}'
);

CREATE INDEX IF NOT EXISTS idx_portfolio_key ON portfolio_clusters(fascination_key);
"""


class PortfolioStore:
    """Async SQLite store for portfolio clusters."""

    def __init__(self, path: Path | str) -> None:
        self.path = Path(path)
        self._db: aiosqlite.Connection | None = None

    async def connect(self) -> "PortfolioStore":
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._db = await aiosqlite.connect(str(self.path))
        self._db.row_factory = aiosqlite.Row
        await self._db.executescript(_SCHEMA)
        try:
            await self._db.execute("ALTER TABLE portfolio_clusters ADD COLUMN meta TEXT NOT NULL DEFAULT '{}'")
        except Exception:
            pass
        await self._db.commit()
        return self

    async def close(self) -> None:
        if self._db:
            await self._db.close()
            self._db = None

    async def save(self, cluster: PortfolioCluster) -> None:
        assert self._db is not None
        await self._db.execute(
            """
            INSERT INTO portfolio_clusters (
                cluster_id, created_at, updated_at, fascination_key,
                spark_count, initiative_count, artifact_count, last_touched_at,
                tags, embedding_id, meta
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            ON CONFLICT(fascination_key) DO UPDATE SET
                updated_at = excluded.updated_at,
                spark_count = excluded.spark_count,
                initiative_count = excluded.initiative_count,
                artifact_count = excluded.artifact_count,
                last_touched_at = excluded.last_touched_at,
                tags = excluded.tags,
                embedding_id = excluded.embedding_id,
                meta = excluded.meta
            """,
            (
                str(cluster.cluster_id),
                cluster.created_at.isoformat(),
                cluster.updated_at.isoformat(),
                cluster.fascination_key,
                cluster.spark_count,
                cluster.initiative_count,
                cluster.artifact_count,
                cluster.last_touched_at.isoformat(),
                __import__("json").dumps(cluster.tags),
                cluster.embedding_id,
                __import__("json").dumps(cluster.meta),
            ),
        )
        await self._db.commit()

    async def get_by_key(self, key: str) -> Optional[PortfolioCluster]:
        assert self._db is not None
        cursor = await self._db.execute(
            "SELECT * FROM portfolio_clusters WHERE fascination_key = ?", (key,)
        )
        row = await cursor.fetchone()
        if row is None:
            return None
        return self._row_to_cluster(row)

    async def list_all(self, limit: int = 100) -> List[PortfolioCluster]:
        assert self._db is not None
        cursor = await self._db.execute(
            "SELECT * FROM portfolio_clusters ORDER BY updated_at DESC LIMIT ?",
            (limit,),
        )
        rows = await cursor.fetchall()
        return [self._row_to_cluster(r) for r in rows]

    async def increment_counts(
        self,
        fascination_key: str,
        sparks: int = 0,
        initiatives: int = 0,
        artifacts: int = 0,
    ) -> bool:
        assert self._db is not None
        now = datetime.now(timezone.utc).isoformat()
        cursor = await self._db.execute(
            """
            UPDATE portfolio_clusters SET
                spark_count = spark_count + ?,
                initiative_count = initiative_count + ?,
                artifact_count = artifact_count + ?,
                last_touched_at = ?,
                updated_at = ?
            WHERE fascination_key = ?
            """,
            (sparks, initiatives, artifacts, now, now, fascination_key),
        )
        await self._db.commit()
        return cursor.rowcount > 0

    @staticmethod
    def _row_to_cluster(row: aiosqlite.Row) -> PortfolioCluster:
        tags = __import__("json").loads(row["tags"]) if row["tags"] else []
        return PortfolioCluster(
            cluster_id=row["cluster_id"],
            created_at=datetime.fromisoformat(row["created_at"]),
            updated_at=datetime.fromisoformat(row["updated_at"]),
            fascination_key=row["fascination_key"],
            spark_count=row["spark_count"],
            initiative_count=row["initiative_count"],
            artifact_count=row["artifact_count"],
            last_touched_at=datetime.fromisoformat(row["last_touched_at"]),
            tags=tags,
            embedding_id=row["embedding_id"],
            meta=__import__("json").loads(row["meta"]) if "meta" in row.keys() and row["meta"] else {},
        )


def build_fascination_key(content: str, tags: Optional[List[str]] = None) -> str:
    """Build a canonical fascination key from content and tags."""
    if tags:
        tokens = sorted({t.lower().strip() for t in tags if t.strip()})
        return "+".join(tokens)
    text = content.lower()
    # strip punctuation, split into words
    words = re.findall(r"[a-z]+", text)
    tokens = sorted({w for w in words if w not in _STOP_WORDS and len(w) > 2})
    # limit to first 8 tokens to keep keys compact
    return "+".join(tokens[:8])
