"""Curation store for rejected consolidation merges."""

from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path
from typing import List, Optional

import aiosqlite

from .models import RejectedMerge

_SCHEMA = """
CREATE TABLE IF NOT EXISTS rejected_merges (
    cluster_hash TEXT PRIMARY KEY,
    episode_ids TEXT NOT NULL,
    reason TEXT NOT NULL DEFAULT '',
    rejected_at TEXT NOT NULL
);

CREATE INDEX IF NOT EXISTS idx_rejected_merges_at ON rejected_merges(rejected_at);
"""


class ConsolidationCurationStore:
    """SQLite store that remembers rejected merges so they are skipped later."""

    def __init__(self, path: Path | str) -> None:
        self.path = Path(path)
        self._db: aiosqlite.Connection | None = None

    async def connect(self) -> "ConsolidationCurationStore":
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._db = await aiosqlite.connect(str(self.path))
        self._db.row_factory = aiosqlite.Row
        await self._db.executescript(_SCHEMA)
        await self._db.commit()
        return self

    async def close(self) -> None:
        if self._db:
            await self._db.close()
            self._db = None

    async def is_rejected(self, cluster_hash: str) -> bool:
        """Check whether a cluster hash was previously rejected."""
        assert self._db is not None
        cursor = await self._db.execute(
            "SELECT 1 FROM rejected_merges WHERE cluster_hash = ?",
            (cluster_hash,),
        )
        row = await cursor.fetchone()
        return row is not None

    async def record_rejection(
        self,
        cluster_hash: str,
        episode_ids: List[str],
        reason: str = "",
    ) -> None:
        """Persist a rejected merge."""
        assert self._db is not None
        await self._db.execute(
            """
            INSERT OR REPLACE INTO rejected_merges
            (cluster_hash, episode_ids, reason, rejected_at)
            VALUES (?, ?, ?, ?)
            """,
            (
                cluster_hash,
                ",".join(episode_ids),
                reason,
                datetime.now(timezone.utc).isoformat(),
            ),
        )
        await self._db.commit()

    async def list_rejected(
        self,
        limit: int = 1000,
        since: Optional[datetime] = None,
    ) -> List[RejectedMerge]:
        """List recently rejected merges."""
        assert self._db is not None
        if since is not None:
            cursor = await self._db.execute(
                """
                SELECT * FROM rejected_merges
                WHERE rejected_at > ?
                ORDER BY rejected_at DESC
                LIMIT ?
                """,
                (since.isoformat(), limit),
            )
        else:
            cursor = await self._db.execute(
                """
                SELECT * FROM rejected_merges
                ORDER BY rejected_at DESC
                LIMIT ?
                """,
                (limit,),
            )
        rows = await cursor.fetchall()
        return [self._row_to_model(r) for r in rows]

    async def prune_old(self, max_age_days: int = 30) -> int:
        """Remove rejection records older than *max_age_days*."""
        assert self._db is not None
        cutoff = datetime.now(timezone.utc).isoformat()
        cursor = await self._db.execute(
            """
            DELETE FROM rejected_merges
            WHERE rejected_at < datetime(?, '-{} days')
            """.format(max_age_days),
            (cutoff,),
        )
        await self._db.commit()
        return cursor.rowcount

    @staticmethod
    def _row_to_model(row: aiosqlite.Row) -> RejectedMerge:
        return RejectedMerge(
            cluster_hash=row["cluster_hash"],
            episode_ids=row["episode_ids"].split(",") if row["episode_ids"] else [],
            reason=row["reason"],
            rejected_at=datetime.fromisoformat(row["rejected_at"]),
        )
