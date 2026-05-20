"""SQLite persistence for peripheral thread and bead records."""

from __future__ import annotations

from pathlib import Path

import aiosqlite

from .models import BeadEntry, BeadSourceKind, BeadStatus, ThreadAnchor, ThreadStatus

_SCHEMA = """
CREATE TABLE IF NOT EXISTS thread_anchors (
    anchor_id TEXT PRIMARY KEY,
    title TEXT NOT NULL,
    kind TEXT NOT NULL,
    status TEXT NOT NULL,
    created_at TEXT NOT NULL,
    updated_at TEXT NOT NULL,
    payload TEXT NOT NULL
);

CREATE INDEX IF NOT EXISTS idx_thread_anchors_status
ON thread_anchors(status);

CREATE INDEX IF NOT EXISTS idx_thread_anchors_kind
ON thread_anchors(kind);

CREATE TABLE IF NOT EXISTS bead_entries (
    bead_id TEXT PRIMARY KEY,
    thread_anchor_id TEXT NOT NULL,
    title TEXT NOT NULL,
    source_kind TEXT NOT NULL,
    source_ref TEXT NOT NULL,
    content_hash_full TEXT NOT NULL,
    content_hash_short TEXT NOT NULL,
    status TEXT NOT NULL,
    user_commissioned INTEGER NOT NULL,
    created_at TEXT NOT NULL,
    committed_at TEXT,
    payload TEXT NOT NULL
);

CREATE INDEX IF NOT EXISTS idx_bead_entries_thread_anchor_id
ON bead_entries(thread_anchor_id);

CREATE INDEX IF NOT EXISTS idx_bead_entries_status
ON bead_entries(status);

CREATE INDEX IF NOT EXISTS idx_bead_entries_source_kind
ON bead_entries(source_kind);

CREATE INDEX IF NOT EXISTS idx_bead_entries_source_ref
ON bead_entries(source_ref);

CREATE INDEX IF NOT EXISTS idx_bead_entries_content_hash_short
ON bead_entries(content_hash_short);

CREATE UNIQUE INDEX IF NOT EXISTS idx_bead_entries_hash_source
ON bead_entries(content_hash_full, source_ref);
"""


class ThreadRegistryStore:
    """Async SQLite store for thread anchors and bead entries."""

    def __init__(self, path: Path | str) -> None:
        self.path = Path(path)
        self._db: aiosqlite.Connection | None = None

    async def connect(self) -> "ThreadRegistryStore":
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._db = await aiosqlite.connect(str(self.path))
        self._db.row_factory = aiosqlite.Row
        await self._db.executescript(_SCHEMA)
        await self._db.commit()
        return self

    async def close(self) -> None:
        if self._db is not None:
            await self._db.close()
            self._db = None

    async def save_thread_anchor(self, anchor: ThreadAnchor) -> None:
        assert self._db is not None
        await self._db.execute(
            """
            INSERT INTO thread_anchors
                (anchor_id, title, kind, status, created_at, updated_at, payload)
            VALUES (?, ?, ?, ?, ?, ?, ?)
            ON CONFLICT(anchor_id) DO UPDATE SET
                title = excluded.title,
                kind = excluded.kind,
                status = excluded.status,
                updated_at = excluded.updated_at,
                payload = excluded.payload
            """,
            (
                anchor.anchor_id,
                anchor.title,
                anchor.kind,
                anchor.status.value,
                anchor.created_at.isoformat(),
                anchor.updated_at.isoformat(),
                anchor.model_dump_json(),
            ),
        )
        await self._db.commit()

    async def get_thread_anchor(self, anchor_id: str) -> ThreadAnchor | None:
        assert self._db is not None
        cursor = await self._db.execute(
            "SELECT payload FROM thread_anchors WHERE anchor_id = ?",
            (str(anchor_id),),
        )
        row = await cursor.fetchone()
        return ThreadAnchor.model_validate_json(row["payload"]) if row else None

    async def list_thread_anchors(
        self,
        *,
        status: ThreadStatus | str | None = None,
        limit: int = 50,
    ) -> list[ThreadAnchor]:
        assert self._db is not None
        capped_limit = max(1, min(200, int(limit)))
        if status:
            status_value = status.value if isinstance(status, ThreadStatus) else str(status)
            cursor = await self._db.execute(
                """
                SELECT payload FROM thread_anchors
                WHERE status = ?
                ORDER BY updated_at DESC
                LIMIT ?
                """,
                (status_value, capped_limit),
            )
        else:
            cursor = await self._db.execute(
                """
                SELECT payload FROM thread_anchors
                ORDER BY updated_at DESC
                LIMIT ?
                """,
                (capped_limit,),
            )
        rows = await cursor.fetchall()
        return [ThreadAnchor.model_validate_json(row["payload"]) for row in rows]

    async def save_bead(self, bead: BeadEntry) -> None:
        assert self._db is not None
        await self._db.execute(
            """
            INSERT INTO bead_entries
                (
                    bead_id, thread_anchor_id, title, source_kind, source_ref,
                    content_hash_full, content_hash_short, status, user_commissioned,
                    created_at, committed_at, payload
                )
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            ON CONFLICT(bead_id) DO UPDATE SET
                thread_anchor_id = excluded.thread_anchor_id,
                title = excluded.title,
                source_kind = excluded.source_kind,
                source_ref = excluded.source_ref,
                content_hash_full = excluded.content_hash_full,
                content_hash_short = excluded.content_hash_short,
                status = excluded.status,
                user_commissioned = excluded.user_commissioned,
                committed_at = excluded.committed_at,
                payload = excluded.payload
            """,
            (
                bead.bead_id,
                bead.thread_anchor_id,
                bead.title,
                bead.source_kind.value,
                bead.source_ref,
                bead.content_hash_full,
                bead.content_hash_short,
                bead.status.value,
                1 if bead.user_commissioned else 0,
                bead.created_at.isoformat(),
                bead.committed_at.isoformat() if bead.committed_at else None,
                bead.model_dump_json(),
            ),
        )
        await self._db.commit()

    async def get_bead(self, bead_id: str) -> BeadEntry | None:
        assert self._db is not None
        cursor = await self._db.execute(
            "SELECT payload FROM bead_entries WHERE bead_id = ?",
            (str(bead_id),),
        )
        row = await cursor.fetchone()
        return BeadEntry.model_validate_json(row["payload"]) if row else None

    async def find_bead_by_source_hash(
        self,
        *,
        source_ref: str,
        content_hash_full: str,
    ) -> BeadEntry | None:
        assert self._db is not None
        cursor = await self._db.execute(
            """
            SELECT payload FROM bead_entries
            WHERE source_ref = ? AND content_hash_full = ?
            LIMIT 1
            """,
            (source_ref, content_hash_full),
        )
        row = await cursor.fetchone()
        return BeadEntry.model_validate_json(row["payload"]) if row else None

    async def list_beads(
        self,
        *,
        thread_anchor_id: str | None = None,
        status: BeadStatus | str | None = None,
        source_kind: BeadSourceKind | str | None = None,
        limit: int = 50,
    ) -> list[BeadEntry]:
        assert self._db is not None
        clauses: list[str] = []
        params: list[str | int] = []
        if thread_anchor_id:
            clauses.append("thread_anchor_id = ?")
            params.append(str(thread_anchor_id))
        if status:
            clauses.append("status = ?")
            params.append(status.value if isinstance(status, BeadStatus) else str(status))
        if source_kind:
            clauses.append("source_kind = ?")
            params.append(source_kind.value if isinstance(source_kind, BeadSourceKind) else str(source_kind))
        where = f"WHERE {' AND '.join(clauses)}" if clauses else ""
        params.append(max(1, min(200, int(limit))))
        cursor = await self._db.execute(
            f"""
            SELECT payload FROM bead_entries
            {where}
            ORDER BY created_at DESC
            LIMIT ?
            """,
            tuple(params),
        )
        rows = await cursor.fetchall()
        return [BeadEntry.model_validate_json(row["payload"]) for row in rows]
