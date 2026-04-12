"""Session-scoped message store for conversation context."""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional
import uuid
from uuid import UUID

import aiosqlite

from .models import MessageEntry, MessageRole

_SCHEMA = """
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

    async def import_entry(self, session_id: str, entry: MessageEntry) -> MessageEntry:
        """Insert an externally sourced message without rewriting its timestamp."""
        assert self._db is not None
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
    ) -> List[MessageEntry]:
        """Return the most recent messages for a session in chronological order."""
        assert self._db is not None
        cursor = await self._db.execute(
            """
            SELECT * FROM messages
            WHERE session_id = ?
            ORDER BY created_at DESC
            LIMIT ?
            """,
            (session_id, limit),
        )
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
            "SELECT COUNT(*) FROM messages WHERE session_id = ?",
            (session_id,),
        )
        row = await cursor.fetchone()
        return row[0] if row else 0

    async def list_session_ids(self, limit: int = 50) -> List[Dict[str, Any]]:
        """Return distinct session IDs with last activity and message count."""
        assert self._db is not None
        cursor = await self._db.execute(
            """
            SELECT session_id, MAX(created_at) as last_at, COUNT(*) as msg_count
            FROM messages
            GROUP BY session_id
            ORDER BY last_at DESC
            LIMIT ?
            """,
            (limit,),
        )
        rows = await cursor.fetchall()
        return [
            {
                "session_id": row["session_id"],
                "last_at": row["last_at"],
                "message_count": row["msg_count"],
            }
            for row in rows
        ]

    async def ensure_session(self, session_id: str) -> None:
        """Ensure a session row exists so it appears in listings before any user message."""
        assert self._db is not None
        await self._db.execute(
            """
            INSERT OR IGNORE INTO messages (session_id, message_id, role, content, meta, created_at)
            VALUES (?, ?, ?, ?, ?, ?)
            """,
            (
                session_id,
                str(uuid.uuid4()),
                MessageRole.SYSTEM.value,
                "",
                json.dumps({"hidden": True}),
                datetime.now(timezone.utc).isoformat(),
            ),
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
