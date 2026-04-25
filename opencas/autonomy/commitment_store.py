"""Persistent SQLite store for Commitments."""

from __future__ import annotations

import json
import sqlite3
from datetime import datetime
from pathlib import Path
from typing import List, Optional

import aiosqlite

from .commitment import Commitment, CommitmentStatus

_SCHEMA = """
CREATE TABLE IF NOT EXISTS commitments (
    commitment_id TEXT PRIMARY KEY,
    created_at TEXT NOT NULL,
    updated_at TEXT NOT NULL,
    content TEXT NOT NULL,
    status TEXT NOT NULL,
    priority REAL NOT NULL DEFAULT 5.0,
    deadline TEXT,
    linked_work_ids TEXT NOT NULL DEFAULT '[]',
    linked_task_ids TEXT NOT NULL DEFAULT '[]',
    tags TEXT NOT NULL DEFAULT '[]',
    meta TEXT NOT NULL DEFAULT '{}'
);

CREATE INDEX IF NOT EXISTS idx_commitments_status ON commitments(status);
CREATE INDEX IF NOT EXISTS idx_commitments_priority ON commitments(priority);
CREATE INDEX IF NOT EXISTS idx_commitments_updated_at ON commitments(updated_at);
"""

_MIGRATIONS: List[str] = []


class CommitmentStore:
    """Async SQLite store for Commitments."""

    def __init__(self, path: Path | str) -> None:
        self.path = Path(path)
        self._db: aiosqlite.Connection | None = None

    async def connect(self) -> "CommitmentStore":
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._db = await aiosqlite.connect(str(self.path))
        self._db.row_factory = aiosqlite.Row
        await self._db.executescript(_SCHEMA)
        await self._db.commit()
        await self._migrate()
        return self

    async def close(self) -> None:
        if self._db:
            await self._db.close()
            self._db = None

    async def _migrate(self) -> None:
        for sql in _MIGRATIONS:
            try:
                assert self._db is not None
                await self._db.execute(sql)
                await self._db.commit()
            except sqlite3.OperationalError:
                pass

    async def save(self, commitment: Commitment) -> None:
        """Insert or replace a Commitment."""
        assert self._db is not None
        await self._db.execute(
            """
            INSERT INTO commitments (
                commitment_id, created_at, updated_at, content, status,
                priority, deadline, linked_work_ids, linked_task_ids, tags, meta
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            ON CONFLICT(commitment_id) DO UPDATE SET
                updated_at = excluded.updated_at,
                content = excluded.content,
                status = excluded.status,
                priority = excluded.priority,
                deadline = excluded.deadline,
                linked_work_ids = excluded.linked_work_ids,
                linked_task_ids = excluded.linked_task_ids,
                tags = excluded.tags,
                meta = excluded.meta
            """,
            (
                str(commitment.commitment_id),
                commitment.created_at.isoformat(),
                commitment.updated_at.isoformat(),
                commitment.content,
                commitment.status.value,
                commitment.priority,
                commitment.deadline.isoformat() if commitment.deadline else None,
                json.dumps(commitment.linked_work_ids),
                json.dumps(commitment.linked_task_ids),
                json.dumps(commitment.tags),
                json.dumps(commitment.meta),
            ),
        )
        await self._db.commit()

    async def get(self, commitment_id: str) -> Optional[Commitment]:
        """Fetch a single Commitment by ID."""
        assert self._db is not None
        cursor = await self._db.execute(
            "SELECT * FROM commitments WHERE commitment_id = ?", (commitment_id,)
        )
        row = await cursor.fetchone()
        if row is None:
            return None
        return self._row_to_commitment(row)

    async def list_by_status(
        self,
        status: CommitmentStatus,
        limit: int = 100,
        offset: int = 0,
    ) -> List[Commitment]:
        """Return commitments filtered by status."""
        assert self._db is not None
        cursor = await self._db.execute(
            """
            SELECT * FROM commitments
            WHERE status = ?
            ORDER BY priority DESC, updated_at DESC
            LIMIT ? OFFSET ?
            """,
            (status.value, limit, offset),
        )
        rows = await cursor.fetchall()
        return [self._row_to_commitment(r) for r in rows]

    async def list_active(
        self,
        limit: int = 100,
        offset: int = 0,
    ) -> List[Commitment]:
        """Return active commitments ordered by priority."""
        return await self.list_by_status(CommitmentStatus.ACTIVE, limit, offset)

    async def count_by_status(self, status: CommitmentStatus) -> int:
        """Return the number of commitments in a given status."""
        assert self._db is not None
        cursor = await self._db.execute(
            "SELECT COUNT(*) FROM commitments WHERE status = ?",
            (status.value,),
        )
        row = await cursor.fetchone()
        return int(row[0]) if row else 0

    async def update_status(
        self,
        commitment_id: str,
        status: CommitmentStatus,
    ) -> bool:
        """Update the status of a commitment."""
        assert self._db is not None
        from datetime import timezone

        now = datetime.now(timezone.utc).isoformat()
        cursor = await self._db.execute(
            """
            UPDATE commitments
            SET status = ?, updated_at = ?
            WHERE commitment_id = ?
            """,
            (status.value, now, commitment_id),
        )
        await self._db.commit()
        return cursor.rowcount > 0

    async def link_work(self, commitment_id: str, work_id: str) -> bool:
        """Append a work_id to the commitment's linked_work_ids."""
        assert self._db is not None
        cursor = await self._db.execute(
            "SELECT linked_work_ids FROM commitments WHERE commitment_id = ?",
            (commitment_id,),
        )
        row = await cursor.fetchone()
        if row is None:
            return False
        linked = json.loads(row["linked_work_ids"]) if row["linked_work_ids"] else []
        if work_id not in linked:
            linked.append(work_id)
            await self._db.execute(
                "UPDATE commitments SET linked_work_ids = ? WHERE commitment_id = ?",
                (json.dumps(linked), commitment_id),
            )
            await self._db.commit()
        return True

    async def link_task(self, commitment_id: str, task_id: str) -> bool:
        """Append a task_id to the commitment's linked_task_ids."""
        assert self._db is not None
        cursor = await self._db.execute(
            "SELECT linked_task_ids FROM commitments WHERE commitment_id = ?",
            (commitment_id,),
        )
        row = await cursor.fetchone()
        if row is None:
            return False
        linked = json.loads(row["linked_task_ids"]) if row["linked_task_ids"] else []
        if task_id not in linked:
            linked.append(task_id)
            await self._db.execute(
                "UPDATE commitments SET linked_task_ids = ? WHERE commitment_id = ?",
                (json.dumps(linked), commitment_id),
            )
            await self._db.commit()
        return True

    async def delete(self, commitment_id: str) -> bool:
        """Delete a Commitment by ID."""
        assert self._db is not None
        cursor = await self._db.execute(
            "DELETE FROM commitments WHERE commitment_id = ?", (commitment_id,)
        )
        await self._db.commit()
        return cursor.rowcount > 0

    @staticmethod
    def _row_to_commitment(row: aiosqlite.Row) -> Commitment:
        linked_work_ids = json.loads(row["linked_work_ids"]) if row["linked_work_ids"] else []
        linked_task_ids = json.loads(row["linked_task_ids"]) if row["linked_task_ids"] else []
        tags = json.loads(row["tags"]) if row["tags"] else []
        meta = json.loads(row["meta"]) if row["meta"] else {}
        return Commitment(
            commitment_id=row["commitment_id"],
            created_at=datetime.fromisoformat(row["created_at"]),
            updated_at=datetime.fromisoformat(row["updated_at"]),
            content=row["content"],
            status=CommitmentStatus(row["status"]),
            priority=row["priority"],
            deadline=datetime.fromisoformat(row["deadline"]) if row["deadline"] else None,
            linked_work_ids=linked_work_ids,
            linked_task_ids=linked_task_ids,
            tags=tags,
            meta=meta,
        )
