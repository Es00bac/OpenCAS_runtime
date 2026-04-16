"""SQLite-backed persistence for reflective conflict records."""

from __future__ import annotations

import sqlite3
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import List, Optional

import aiosqlite

from .models import ConflictRecord
from .sqlite_base import SqliteBackedStore


_CONFLICT_SCHEMA = """
CREATE TABLE IF NOT EXISTS conflicts (
    conflict_id TEXT PRIMARY KEY,
    created_at TEXT NOT NULL,
    updated_at TEXT NOT NULL,
    kind TEXT NOT NULL,
    description TEXT NOT NULL,
    source_daydream_id TEXT,
    occurrence_count INTEGER NOT NULL DEFAULT 1,
    resolved INTEGER NOT NULL DEFAULT 0,
    auto_resolved INTEGER NOT NULL DEFAULT 0,
    UNIQUE(kind, description)
);

CREATE INDEX IF NOT EXISTS idx_conflicts_kind ON conflicts(kind);
CREATE INDEX IF NOT EXISTS idx_conflicts_resolved ON conflicts(resolved);
"""


class ConflictStore(SqliteBackedStore):
    """Persistent registry of detected tensions."""

    SCHEMA = _CONFLICT_SCHEMA

    async def _migrate(self) -> None:
        migrations = [
            "ALTER TABLE conflicts ADD COLUMN somatic_context TEXT;",
            "ALTER TABLE conflicts ADD COLUMN resolved_at TEXT;",
            "ALTER TABLE conflicts ADD COLUMN resolution_notes TEXT NOT NULL DEFAULT '';",
        ]
        for sql in migrations:
            try:
                await self.db.execute(sql)
            except sqlite3.OperationalError:
                pass

    async def record_conflict(self, record: ConflictRecord) -> ConflictRecord:
        now = datetime.now(timezone.utc).isoformat()
        somatic_context_json = (
            record.somatic_context.model_dump_json() if record.somatic_context else None
        )
        await self.db.execute(
            """
            INSERT INTO conflicts (
                conflict_id, created_at, updated_at, kind, description,
                source_daydream_id, occurrence_count, resolved, auto_resolved,
                somatic_context, resolved_at, resolution_notes
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            ON CONFLICT(kind, description) DO UPDATE SET
                updated_at = excluded.updated_at,
                occurrence_count = occurrence_count + 1,
                source_daydream_id = excluded.source_daydream_id,
                resolved = 0,
                auto_resolved = 0,
                somatic_context = excluded.somatic_context
            """,
            (
                str(record.conflict_id),
                record.created_at.isoformat(),
                now,
                record.kind,
                record.description,
                record.source_daydream_id,
                record.occurrence_count,
                int(record.resolved),
                int(record.auto_resolved),
                somatic_context_json,
                record.resolved_at.isoformat() if record.resolved_at else None,
                record.resolution_notes,
            ),
        )
        await self.db.commit()
        cursor = await self.db.execute(
            "SELECT * FROM conflicts WHERE kind = ? AND description = ?",
            (record.kind, record.description),
        )
        row = await cursor.fetchone()
        assert row is not None
        return self._row_to_conflict(row)

    async def list_active_conflicts(self, limit: int = 20) -> List[ConflictRecord]:
        cursor = await self.db.execute(
            """
            SELECT * FROM conflicts
            WHERE resolved = 0 AND auto_resolved = 0
            ORDER BY occurrence_count DESC, updated_at DESC
            LIMIT ?
            """,
            (limit,),
        )
        rows = await cursor.fetchall()
        return [self._row_to_conflict(r) for r in rows]

    async def list_conflicts(
        self,
        limit: int = 20,
        resolved: Optional[bool] = None,
    ) -> List[ConflictRecord]:
        sql = "SELECT * FROM conflicts"
        params: List[object] = []
        if resolved is not None:
            sql += " WHERE resolved = ?"
            params.append(int(resolved))
        sql += " ORDER BY updated_at DESC, occurrence_count DESC LIMIT ?"
        params.append(limit)
        cursor = await self.db.execute(sql, tuple(params))
        rows = await cursor.fetchall()
        return [self._row_to_conflict(r) for r in rows]

    async def resolve_conflict(
        self,
        conflict_id: str,
        auto: bool = False,
        resolution_notes: str = "",
    ) -> None:
        now = datetime.now(timezone.utc).isoformat()
        await self.db.execute(
            """
            UPDATE conflicts
            SET resolved = 1, auto_resolved = ?, resolved_at = ?, resolution_notes = ?
            WHERE conflict_id = ?
            """,
            (int(auto), now, resolution_notes, conflict_id),
        )
        await self.db.commit()

    async def auto_resolve_chronic(
        self,
        threshold: int = 25,
        min_days: int = 10,
    ) -> int:
        """Auto-resolve conflicts that have occurred many times over many days."""
        cutoff = (datetime.now(timezone.utc) - timedelta(days=min_days)).isoformat()
        cursor = await self.db.execute(
            """
            SELECT conflict_id FROM conflicts
            WHERE occurrence_count >= ?
              AND created_at <= ?
              AND resolved = 0
            """,
            (threshold, cutoff),
        )
        rows = await cursor.fetchall()
        resolved = 0
        for row in rows:
            await self.resolve_conflict(
                row["conflict_id"], auto=True, resolution_notes="auto-resolved chronic conflict"
            )
            resolved += 1
        return resolved

    @staticmethod
    def _row_to_conflict(row: aiosqlite.Row) -> ConflictRecord:
        from opencas.somatic.models import SomaticSnapshot

        resolved_at = None
        if row["resolved_at"]:
            resolved_at = datetime.fromisoformat(row["resolved_at"])
        somatic_context = None
        if row["somatic_context"]:
            try:
                somatic_context = SomaticSnapshot.model_validate_json(row["somatic_context"])
            except (ValueError, TypeError):
                somatic_context = None
        return ConflictRecord(
            conflict_id=row["conflict_id"],
            created_at=datetime.fromisoformat(row["created_at"]),
            kind=row["kind"],
            description=row["description"],
            source_daydream_id=row["source_daydream_id"],
            occurrence_count=row["occurrence_count"],
            resolved=bool(row["resolved"]),
            auto_resolved=bool(row["auto_resolved"]),
            somatic_context=somatic_context,
            resolved_at=resolved_at,
            resolution_notes=row["resolution_notes"] or "",
        )
