"""Session-scoped message store for conversation context."""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional
from uuid import UUID

import aiosqlite

from .models import MessageEntry, MessageRole

_SCHEMA = """
CREATE TABLE IF NOT EXISTS sessions (
    session_id TEXT PRIMARY KEY,
    name TEXT,
    status TEXT NOT NULL DEFAULT 'active',
    created_at TEXT NOT NULL,
    updated_at TEXT NOT NULL,
    last_message_at TEXT
);

CREATE INDEX IF NOT EXISTS idx_sessions_status_updated_at ON sessions(status, updated_at DESC);

CREATE TABLE IF NOT EXISTS messages (
    message_id TEXT PRIMARY KEY,
    session_id TEXT NOT NULL,
    role TEXT NOT NULL,
    content TEXT NOT NULL,
    created_at TEXT NOT NULL,
    meta TEXT NOT NULL DEFAULT '{}'
);

CREATE INDEX IF NOT EXISTS idx_messages_session_id ON messages(session_id);
CREATE INDEX IF NOT EXISTS idx_messages_created_at ON messages(created_at);

CREATE VIRTUAL TABLE IF NOT EXISTS messages_fts USING fts5(
    content,
    content='messages',
    content_rowid='rowid'
);

CREATE TRIGGER IF NOT EXISTS messages_fts_insert AFTER INSERT ON messages BEGIN
    INSERT INTO messages_fts(rowid, content) VALUES (new.rowid, new.content);
END;

CREATE TRIGGER IF NOT EXISTS messages_fts_delete AFTER DELETE ON messages BEGIN
    INSERT INTO messages_fts(messages_fts, rowid, content) VALUES ('delete', old.rowid, old.content);
END;

CREATE TRIGGER IF NOT EXISTS messages_fts_update AFTER UPDATE ON messages BEGIN
    INSERT INTO messages_fts(messages_fts, rowid, content) VALUES ('delete', old.rowid, old.content);
    INSERT INTO messages_fts(rowid, content) VALUES (new.rowid, new.content);
END;
"""


class SessionContextStore:
    """SQLite-backed store for per-session conversation messages."""

    def __init__(self, path: Path | str) -> None:
        self.path = Path(path)
        self._db: aiosqlite.Connection | None = None

    async def connect(self) -> "SessionContextStore":
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._db = await aiosqlite.connect(str(self.path))
        self._db.row_factory = aiosqlite.Row
        await self._db.executescript(_SCHEMA)
        await self._backfill_sessions()
        await self._db.commit()
        return self

    async def close(self) -> None:
        if self._db:
            await self._db.close()
            self._db = None

    async def append(
        self,
        session_id: str,
        role: MessageRole,
        content: str,
        meta: Optional[Dict[str, Any]] = None,
    ) -> MessageEntry:
        """Append a message to the session and return it."""
        assert self._db is not None
        entry = MessageEntry(role=role, content=content, meta=meta or {})
        await self._touch_session(session_id, entry.created_at.isoformat())
        await self._db.execute(
            """
            INSERT INTO messages (message_id, session_id, role, content, created_at, meta)
            VALUES (?, ?, ?, ?, ?, ?)
            """,
            (
                str(entry.message_id),
                session_id,
                role.value,
                content,
                entry.created_at.isoformat(),
                json.dumps(entry.meta),
            ),
        )
        await self._db.commit()
        return entry

    async def merge_message_meta(
        self,
        session_id: str,
        message_id: UUID | str,
        meta_patch: Dict[str, Any],
    ) -> Optional[MessageEntry]:
        """Shallow-merge *meta_patch* into an existing message meta payload."""
        assert self._db is not None
        cursor = await self._db.execute(
            """
            SELECT * FROM messages
            WHERE session_id = ? AND message_id = ?
            """,
            (session_id, str(message_id)),
        )
        row = await cursor.fetchone()
        if row is None:
            return None
        current_meta = json.loads(row["meta"]) if row["meta"] else {}
        if isinstance(current_meta, dict) and "meta" in current_meta:
            current_meta = current_meta["meta"]
        merged_meta = {**(current_meta or {}), **(meta_patch or {})}
        await self._db.execute(
            """
            UPDATE messages
            SET meta = ?
            WHERE session_id = ? AND message_id = ?
            """,
            (json.dumps(merged_meta), session_id, str(message_id)),
        )
        await self._db.commit()
        return self._row_to_entry({**dict(row), "meta": json.dumps(merged_meta)})

    async def import_entry(self, session_id: str, entry: MessageEntry) -> MessageEntry:
        """Insert an externally sourced message without rewriting its timestamp."""
        assert self._db is not None
        await self._touch_session(session_id, entry.created_at.isoformat())
        await self._db.execute(
            """
            INSERT INTO messages (message_id, session_id, role, content, created_at, meta)
            VALUES (?, ?, ?, ?, ?, ?)
            ON CONFLICT(message_id) DO UPDATE SET
                session_id = excluded.session_id,
                role = excluded.role,
                content = excluded.content,
                created_at = excluded.created_at,
                meta = excluded.meta
            """,
            (
                str(entry.message_id),
                session_id,
                entry.role.value,
                entry.content,
                entry.created_at.isoformat(),
                json.dumps(entry.meta),
            ),
        )
        await self._db.commit()
        return entry

    async def list_recent(
        self,
        session_id: str,
        limit: int = 50,
        *,
        include_hidden: bool = False,
    ) -> List[MessageEntry]:
        """Return the most recent messages for a session in chronological order."""
        assert self._db is not None
        params: tuple[Any, ...] = (session_id, limit)
        if include_hidden:
            query = """
                SELECT * FROM messages
                WHERE session_id = ?
                ORDER BY created_at DESC
                LIMIT ?
            """
        else:
            query = """
                SELECT * FROM messages
                WHERE session_id = ?
                  AND instr(meta, '"hidden": true') = 0
                ORDER BY created_at DESC
                LIMIT ?
            """
        cursor = await self._db.execute(query, params)
        rows = await cursor.fetchall()
        entries = [self._row_to_entry(r) for r in rows]
        entries.reverse()
        return entries

    async def search(
        self,
        query: str,
        session_id: Optional[str] = None,
        limit: int = 20,
    ) -> List[MessageEntry]:
        """Search messages using FTS5."""
        assert self._db is not None
        if session_id is not None:
            cursor = await self._db.execute(
                """
                SELECT m.* FROM messages m
                JOIN messages_fts fts ON m.rowid = fts.rowid
                WHERE m.session_id = ? AND messages_fts MATCH ?
                ORDER BY fts.rank
                LIMIT ?
                """,
                (session_id, query, limit),
            )
        else:
            cursor = await self._db.execute(
                """
                SELECT m.* FROM messages m
                JOIN messages_fts fts ON m.rowid = fts.rowid
                WHERE messages_fts MATCH ?
                ORDER BY fts.rank
                LIMIT ?
                """,
                (query, limit),
            )
        rows = await cursor.fetchall()
        return [self._row_to_entry(r) for r in rows]

    async def count(self, session_id: str) -> int:
        """Return the number of messages in a session."""
        assert self._db is not None
        cursor = await self._db.execute(
            """
            SELECT COUNT(*) FROM messages
            WHERE session_id = ?
              AND instr(meta, '"hidden": true') = 0
            """,
            (session_id,),
        )
        row = await cursor.fetchone()
        return row[0] if row else 0

    async def list_session_ids(
        self,
        limit: int = 50,
        status: str = "active",
    ) -> List[Dict[str, Any]]:
        """Return session summaries with last activity and visible message count."""
        assert self._db is not None
        status_filter = None if status == "all" else status
        cursor = await self._db.execute(
            """
            SELECT
                s.session_id,
                s.name,
                s.status,
                s.created_at,
                COALESCE(
                    MAX(CASE WHEN instr(m.meta, '"hidden": true') = 0 THEN m.created_at END),
                    s.last_message_at,
                    s.updated_at,
                    s.created_at
                ) AS last_at,
                COALESCE(
                    SUM(
                        CASE
                            WHEN m.message_id IS NULL THEN 0
                            WHEN instr(COALESCE(m.meta, '{}'), '"hidden": true') = 0 THEN 1
                            ELSE 0
                        END
                    ),
                    0
                ) AS message_count
            FROM sessions s
            LEFT JOIN messages m ON m.session_id = s.session_id
            WHERE (? IS NULL OR s.status = ?)
            GROUP BY s.session_id, s.name, s.status, s.created_at, s.updated_at, s.last_message_at
            ORDER BY last_at DESC
            LIMIT ?
            """,
            (status_filter, status_filter, limit),
        )
        rows = await cursor.fetchall()
        return [dict(row) for row in rows]

    async def search_sessions(
        self,
        query: str,
        status: str = "active",
        limit: int = 20,
    ) -> List[Dict[str, Any]]:
        """Search sessions by id or optional display name."""
        assert self._db is not None
        search = f"%{query.strip().lower()}%"
        if not search.strip("%"):
            return await self.list_session_ids(limit=limit, status=status)
        status_filter = None if status == "all" else status
        cursor = await self._db.execute(
            """
            SELECT
                s.session_id,
                s.name,
                s.status,
                s.created_at,
                COALESCE(
                    MAX(CASE WHEN instr(m.meta, '"hidden": true') = 0 THEN m.created_at END),
                    s.last_message_at,
                    s.updated_at,
                    s.created_at
                ) AS last_at,
                COALESCE(
                    SUM(
                        CASE
                            WHEN m.message_id IS NULL THEN 0
                            WHEN instr(COALESCE(m.meta, '{}'), '"hidden": true') = 0 THEN 1
                            ELSE 0
                        END
                    ),
                    0
                ) AS message_count
            FROM sessions s
            LEFT JOIN messages m ON m.session_id = s.session_id
            WHERE (? IS NULL OR s.status = ?)
              AND (
                    LOWER(s.session_id) LIKE ?
                 OR LOWER(COALESCE(s.name, '')) LIKE ?
              )
            GROUP BY s.session_id, s.name, s.status, s.created_at, s.updated_at, s.last_message_at
            ORDER BY last_at DESC
            LIMIT ?
            """,
            (status_filter, status_filter, search, search, limit),
        )
        rows = await cursor.fetchall()
        return [dict(row) for row in rows]

    async def get_session_meta(self, session_id: str) -> Optional[Dict[str, Any]]:
        """Return one session summary row."""
        assert self._db is not None
        cursor = await self._db.execute(
            """
            SELECT
                s.session_id,
                s.name,
                s.status,
                s.created_at,
                COALESCE(
                    MAX(CASE WHEN instr(m.meta, '"hidden": true') = 0 THEN m.created_at END),
                    s.last_message_at,
                    s.updated_at,
                    s.created_at
                ) AS last_at,
                COALESCE(
                    SUM(
                        CASE
                            WHEN m.message_id IS NULL THEN 0
                            WHEN instr(COALESCE(m.meta, '{}'), '"hidden": true') = 0 THEN 1
                            ELSE 0
                        END
                    ),
                    0
                ) AS message_count
            FROM sessions s
            LEFT JOIN messages m ON m.session_id = s.session_id
            WHERE s.session_id = ?
            GROUP BY s.session_id, s.name, s.status, s.created_at, s.updated_at, s.last_message_at
            """,
            (session_id,),
        )
        row = await cursor.fetchone()
        return dict(row) if row else None

    async def ensure_session(self, session_id: str) -> None:
        """Ensure a session row exists so it appears in listings before any user message."""
        await self._touch_session(session_id)

    async def update_session_name(self, session_id: str, name: Optional[str]) -> None:
        """Update a session display name."""
        assert self._db is not None
        await self.ensure_session(session_id)
        await self._db.execute(
            """
            UPDATE sessions
            SET name = ?, updated_at = ?
            WHERE session_id = ?
            """,
            ((name or "").strip() or None, datetime.now(timezone.utc).isoformat(), session_id),
        )
        await self._db.commit()

    async def set_session_status(self, session_id: str, status: str) -> None:
        """Update a session status."""
        normalized = str(status or "").strip().lower()
        if normalized not in {"active", "archived"}:
            raise ValueError(f"Unsupported session status: {status}")
        assert self._db is not None
        await self.ensure_session(session_id)
        await self._db.execute(
            """
            UPDATE sessions
            SET status = ?, updated_at = ?
            WHERE session_id = ?
            """,
            (normalized, datetime.now(timezone.utc).isoformat(), session_id),
        )
        await self._db.commit()

    async def _backfill_sessions(self) -> None:
        """Create session rows for legacy databases that only contain messages."""
        assert self._db is not None
        await self._db.execute(
            """
            INSERT OR IGNORE INTO sessions (session_id, created_at, updated_at, last_message_at)
            SELECT
                session_id,
                MIN(created_at) AS created_at,
                MAX(created_at) AS updated_at,
                MAX(created_at) AS last_message_at
            FROM messages
            GROUP BY session_id
            """
        )

    async def _touch_session(self, session_id: str, timestamp: Optional[str] = None) -> None:
        """Ensure a session exists and refresh its activity timestamps."""
        assert self._db is not None
        ts = timestamp or datetime.now(timezone.utc).isoformat()
        await self._db.execute(
            """
            INSERT INTO sessions (session_id, created_at, updated_at, last_message_at)
            VALUES (?, ?, ?, ?)
            ON CONFLICT(session_id) DO UPDATE SET
                created_at = MIN(sessions.created_at, excluded.created_at),
                updated_at = MAX(sessions.updated_at, excluded.updated_at),
                last_message_at = CASE
                    WHEN sessions.last_message_at IS NULL THEN excluded.last_message_at
                    ELSE MAX(sessions.last_message_at, excluded.last_message_at)
                END
            """,
            (session_id, ts, ts, ts),
        )
        await self._db.commit()

    @staticmethod
    def _row_to_entry(row: aiosqlite.Row) -> MessageEntry:
        meta = json.loads(row["meta"]) if row["meta"] else {}
        if isinstance(meta, dict) and "meta" in meta:
            meta = meta["meta"]
        return MessageEntry(
            message_id=UUID(row["message_id"]),
            role=MessageRole(row["role"]),
            content=row["content"],
            created_at=datetime.fromisoformat(row["created_at"]),
            meta=meta,
        )
