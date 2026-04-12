"""Persistent SQLite store for WorkObjects."""

from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path
from typing import List, Optional

import aiosqlite
import sqlite3

from .models import WorkObject, WorkStage

_SCHEMA = """
CREATE TABLE IF NOT EXISTS work_objects (
    work_id TEXT PRIMARY KEY,
    created_at TEXT NOT NULL,
    updated_at TEXT NOT NULL,
    stage TEXT NOT NULL,
    content TEXT NOT NULL,
    embedding_id TEXT,
    source_memory_ids TEXT NOT NULL DEFAULT '[]',
    promotion_score REAL NOT NULL DEFAULT 0.0,
    access_count INTEGER NOT NULL DEFAULT 0,
    last_accessed TEXT,
    meta TEXT NOT NULL DEFAULT '{}',
    dependency_ids TEXT NOT NULL DEFAULT '[]',
    blocked_by TEXT NOT NULL DEFAULT '[]',
    project_id TEXT,
    commitment_id TEXT,
    portfolio_id TEXT
);

CREATE INDEX IF NOT EXISTS idx_work_stage ON work_objects(stage);
CREATE INDEX IF NOT EXISTS idx_work_project ON work_objects(project_id);
CREATE INDEX IF NOT EXISTS idx_work_updated_at ON work_objects(updated_at);
CREATE INDEX IF NOT EXISTS idx_work_commitment ON work_objects(commitment_id);
CREATE INDEX IF NOT EXISTS idx_work_portfolio ON work_objects(portfolio_id);
"""

_MIGRATIONS: List[str] = [
    "ALTER TABLE work_objects ADD COLUMN commitment_id TEXT",
    "ALTER TABLE work_objects ADD COLUMN portfolio_id TEXT",
]


class WorkStore:
    """Async SQLite store for WorkObjects."""

    def __init__(self, path: Path | str) -> None:
        self.path = Path(path)
        self._db: aiosqlite.Connection | None = None

    async def connect(self) -> "WorkStore":
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._db = await aiosqlite.connect(str(self.path))
        self._db.row_factory = aiosqlite.Row
        await self._db.executescript(_SCHEMA)
        await self._db.commit()
        await self._migrate()
        return self

    async def _migrate(self) -> None:
        for sql in _MIGRATIONS:
            try:
                assert self._db is not None
                await self._db.execute(sql)
                await self._db.commit()
            except sqlite3.OperationalError:
                pass

    async def close(self) -> None:
        if self._db:
            await self._db.close()
            self._db = None

    async def save(self, work: WorkObject) -> None:
        """Insert or replace a WorkObject."""
        assert self._db is not None
        await self._db.execute(
            """
            INSERT INTO work_objects (
                work_id, created_at, updated_at, stage, content,
                embedding_id, source_memory_ids, promotion_score, access_count,
                last_accessed, meta, dependency_ids, blocked_by, project_id,
                commitment_id, portfolio_id
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            ON CONFLICT(work_id) DO UPDATE SET
                updated_at = excluded.updated_at,
                stage = excluded.stage,
                content = excluded.content,
                embedding_id = excluded.embedding_id,
                source_memory_ids = excluded.source_memory_ids,
                promotion_score = excluded.promotion_score,
                access_count = excluded.access_count,
                last_accessed = excluded.last_accessed,
                meta = excluded.meta,
                dependency_ids = excluded.dependency_ids,
                blocked_by = excluded.blocked_by,
                project_id = excluded.project_id,
                commitment_id = excluded.commitment_id,
                portfolio_id = excluded.portfolio_id
            """,
            (
                str(work.work_id),
                work.created_at.isoformat(),
                work.updated_at.isoformat(),
                work.stage.value,
                work.content,
                work.embedding_id,
                json.dumps(work.source_memory_ids),
                work.promotion_score,
                work.access_count,
                work.last_accessed.isoformat() if work.last_accessed else None,
                json.dumps(work.meta),
                json.dumps(work.dependency_ids),
                json.dumps(work.blocked_by),
                work.project_id,
                work.commitment_id,
                work.portfolio_id,
            ),
        )
        await self._db.commit()

    async def get(self, work_id: str) -> Optional[WorkObject]:
        """Fetch a single WorkObject by ID."""
        assert self._db is not None
        cursor = await self._db.execute(
            "SELECT * FROM work_objects WHERE work_id = ?", (work_id,)
        )
        row = await cursor.fetchone()
        if row is None:
            return None
        return self._row_to_work(row)

    async def list_all(
        self,
        limit: int = 100,
        offset: int = 0,
    ) -> List[WorkObject]:
        """Return all work objects ordered by updated_at DESC."""
        assert self._db is not None
        cursor = await self._db.execute(
            "SELECT * FROM work_objects ORDER BY updated_at DESC LIMIT ? OFFSET ?",
            (limit, offset),
        )
        rows = await cursor.fetchall()
        return [self._row_to_work(r) for r in rows]

    async def list_by_stage(
        self,
        stage: WorkStage,
        limit: int = 100,
        offset: int = 0,
    ) -> List[WorkObject]:
        """Return work objects filtered by stage."""
        assert self._db is not None
        cursor = await self._db.execute(
            """
            SELECT * FROM work_objects
            WHERE stage = ?
            ORDER BY updated_at DESC
            LIMIT ? OFFSET ?
            """,
            (stage.value, limit, offset),
        )
        rows = await cursor.fetchall()
        return [self._row_to_work(r) for r in rows]

    async def list_by_project(
        self,
        project_id: str,
        limit: int = 100,
        offset: int = 0,
    ) -> List[WorkObject]:
        """Return work objects for a given project."""
        assert self._db is not None
        cursor = await self._db.execute(
            """
            SELECT * FROM work_objects
            WHERE project_id = ?
            ORDER BY updated_at DESC
            LIMIT ? OFFSET ?
            """,
            (project_id, limit, offset),
        )
        rows = await cursor.fetchall()
        return [self._row_to_work(r) for r in rows]

    async def list_by_origin(
        self,
        origin: str,
        limit: int = 100,
        offset: int = 0,
    ) -> List[WorkObject]:
        """Return work objects whose meta.origin matches *origin*."""
        assert self._db is not None
        cursor = await self._db.execute(
            """
            SELECT * FROM work_objects
            WHERE json_extract(meta, '$.origin') = ?
            ORDER BY updated_at DESC
            LIMIT ? OFFSET ?
            """,
            (origin, limit, offset),
        )
        rows = await cursor.fetchall()
        return [self._row_to_work(r) for r in rows]

    async def list_blocked(
        self,
        limit: int = 100,
    ) -> List[WorkObject]:
        """Return work objects that have unresolved blocked_by entries."""
        assert self._db is not None
        cursor = await self._db.execute(
            """
            SELECT * FROM work_objects
            WHERE json_array_length(blocked_by) > 0
            ORDER BY updated_at DESC
            LIMIT ?
            """,
            (limit,),
        )
        rows = await cursor.fetchall()
        return [self._row_to_work(r) for r in rows]

    async def list_ready(
        self,
        limit: int = 100,
    ) -> List[WorkObject]:
        """Return work objects with no unresolved blocked_by entries."""
        assert self._db is not None
        cursor = await self._db.execute(
            """
            SELECT * FROM work_objects
            WHERE json_array_length(blocked_by) = 0
            ORDER BY updated_at DESC
            LIMIT ?
            """,
            (limit,),
        )
        rows = await cursor.fetchall()
        return [self._row_to_work(r) for r in rows]

    async def summary_counts(self) -> dict[str, int]:
        """Return aggregate counts for all, ready, and blocked work."""
        assert self._db is not None
        total_cursor = await self._db.execute("SELECT COUNT(*) FROM work_objects")
        total_row = await total_cursor.fetchone()
        ready_cursor = await self._db.execute(
            "SELECT COUNT(*) FROM work_objects WHERE json_array_length(blocked_by) = 0"
        )
        ready_row = await ready_cursor.fetchone()
        blocked_cursor = await self._db.execute(
            "SELECT COUNT(*) FROM work_objects WHERE json_array_length(blocked_by) > 0"
        )
        blocked_row = await blocked_cursor.fetchone()
        return {
            "total": int(total_row[0]) if total_row else 0,
            "ready": int(ready_row[0]) if ready_row else 0,
            "blocked": int(blocked_row[0]) if blocked_row else 0,
        }

    async def find_by_repair_task_id(self, repair_task_id: str) -> Optional[WorkObject]:
        """Find a work object by repair_task_id stored in meta."""
        assert self._db is not None
        cursor = await self._db.execute(
            "SELECT * FROM work_objects WHERE meta LIKE ? ORDER BY updated_at DESC LIMIT 25",
            (f'%"repair_task_id": "{repair_task_id}"%',),
        )
        rows = await cursor.fetchall()
        for row in rows:
            work = self._row_to_work(row)
            if work.meta.get("repair_task_id") == repair_task_id:
                return work
        return None

    async def unblock_dependencies(self, work_id: str) -> int:
        """Remove *work_id* from the blocked_by list of all work objects.

        Returns the number of rows modified.
        """
        assert self._db is not None
        # aiosqlite doesn't support JSON_REMOVE directly in parameterized queries
        # without SQLite JSON1; use Python fetch-update pattern for compatibility.
        cursor = await self._db.execute(
            "SELECT work_id, blocked_by FROM work_objects WHERE blocked_by LIKE ?",
            (f'%"{work_id}"%',),
        )
        rows = await cursor.fetchall()
        modified = 0
        for row in rows:
            blocked = json.loads(row["blocked_by"]) if row["blocked_by"] else []
            if work_id in blocked:
                blocked.remove(work_id)
                await self._db.execute(
                    "UPDATE work_objects SET blocked_by = ? WHERE work_id = ?",
                    (json.dumps(blocked), row["work_id"]),
                )
                modified += 1
        if modified:
            await self._db.commit()
        return modified

    async def delete(self, work_id: str) -> bool:
        """Delete a WorkObject by ID."""
        assert self._db is not None
        cursor = await self._db.execute(
            "DELETE FROM work_objects WHERE work_id = ?", (work_id,)
        )
        await self._db.commit()
        return cursor.rowcount > 0

    async def touch(self, work_id: str) -> bool:
        """Increment access_count and update last_accessed."""
        from datetime import timezone
        assert self._db is not None
        now = datetime.now(timezone.utc).isoformat()
        cursor = await self._db.execute(
            """
            UPDATE work_objects
            SET access_count = access_count + 1, last_accessed = ?
            WHERE work_id = ?
            """,
            (now, work_id),
        )
        await self._db.commit()
        return cursor.rowcount > 0

    @staticmethod
    def _row_to_work(row: aiosqlite.Row) -> WorkObject:
        source_memory_ids = json.loads(row["source_memory_ids"]) if row["source_memory_ids"] else []
        meta = json.loads(row["meta"]) if row["meta"] else {}
        dependency_ids = json.loads(row["dependency_ids"]) if row["dependency_ids"] else []
        blocked_by = json.loads(row["blocked_by"]) if row["blocked_by"] else []
        return WorkObject(
            work_id=row["work_id"],
            created_at=datetime.fromisoformat(row["created_at"]),
            updated_at=datetime.fromisoformat(row["updated_at"]),
            stage=WorkStage(row["stage"]),
            content=row["content"],
            embedding_id=row["embedding_id"],
            source_memory_ids=source_memory_ids,
            promotion_score=row["promotion_score"],
            access_count=row["access_count"],
            last_accessed=datetime.fromisoformat(row["last_accessed"]) if row["last_accessed"] else None,
            meta=meta,
            dependency_ids=dependency_ids,
            blocked_by=blocked_by,
            project_id=row["project_id"],
            commitment_id=row["commitment_id"],
            portfolio_id=row["portfolio_id"],
        )
