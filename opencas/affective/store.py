"""SQLite store for evidence-linked affective examinations."""

from __future__ import annotations

import json
import sqlite3
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, List, Optional

import aiosqlite

from .models import (
    AffectiveActionPressure,
    AffectiveConsumedBy,
    AffectiveExamination,
    AffectiveSourceType,
    AffectiveTarget,
)

_SCHEMA = """
CREATE TABLE IF NOT EXISTS affective_examinations (
    examination_id TEXT PRIMARY KEY,
    created_at TEXT NOT NULL,
    session_id TEXT,
    source_type TEXT NOT NULL,
    source_id TEXT NOT NULL,
    source_excerpt TEXT NOT NULL DEFAULT '',
    source_hash TEXT NOT NULL DEFAULT '',
    target TEXT NOT NULL,
    affect TEXT NOT NULL,
    intensity REAL NOT NULL,
    confidence REAL NOT NULL,
    action_pressure TEXT NOT NULL,
    bounded_reason TEXT NOT NULL DEFAULT '',
    consumed_by TEXT NOT NULL DEFAULT 'none',
    expires_at TEXT,
    appraisal_version TEXT NOT NULL DEFAULT 'v1',
    meta TEXT NOT NULL DEFAULT '{}',
    UNIQUE(source_type, source_id, appraisal_version)
);

CREATE INDEX IF NOT EXISTS idx_affective_recent
    ON affective_examinations(created_at DESC);
CREATE INDEX IF NOT EXISTS idx_affective_session
    ON affective_examinations(session_id, created_at DESC);
CREATE INDEX IF NOT EXISTS idx_affective_source
    ON affective_examinations(source_type, source_id);
CREATE INDEX IF NOT EXISTS idx_affective_pressure
    ON affective_examinations(action_pressure, consumed_by, created_at DESC);
"""


class AffectiveExaminationStore:
    """Async SQLite persistence for affective examination records."""

    def __init__(self, path: Path | str) -> None:
        self.path = Path(path)
        self._db: aiosqlite.Connection | None = None

    async def connect(self) -> "AffectiveExaminationStore":
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

    async def upsert(self, record: AffectiveExamination) -> AffectiveExamination:
        """Insert a record idempotently, returning the existing row on conflict."""
        existing = await self.get_by_source(
            record.source_type,
            record.source_id,
            appraisal_version=record.appraisal_version,
        )
        if existing is not None:
            return existing
        assert self._db is not None
        try:
            await self._db.execute(
                """
                INSERT INTO affective_examinations (
                    examination_id, created_at, session_id, source_type, source_id,
                    source_excerpt, source_hash, target, affect, intensity,
                    confidence, action_pressure, bounded_reason, consumed_by,
                    expires_at, appraisal_version, meta
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                self._record_params(record),
            )
            await self._db.commit()
        except sqlite3.IntegrityError:
            existing = await self.get_by_source(
                record.source_type,
                record.source_id,
                appraisal_version=record.appraisal_version,
            )
            if existing is not None:
                return existing
            raise
        return record

    async def get_by_source(
        self,
        source_type: AffectiveSourceType | str,
        source_id: str,
        *,
        appraisal_version: str = "v1",
    ) -> Optional[AffectiveExamination]:
        assert self._db is not None
        cursor = await self._db.execute(
            """
            SELECT * FROM affective_examinations
            WHERE source_type = ? AND source_id = ? AND appraisal_version = ?
            """,
            (
                source_type.value if isinstance(source_type, AffectiveSourceType) else str(source_type),
                source_id,
                appraisal_version,
            ),
        )
        row = await cursor.fetchone()
        return self._row_to_record(row) if row else None

    async def list_recent(
        self,
        *,
        limit: int = 50,
        session_id: Optional[str] = None,
        source_type: Optional[AffectiveSourceType | str] = None,
        action_pressure: Optional[AffectiveActionPressure | str] = None,
        primary_emotion: Optional[str] = None,
        consumed_by: Optional[AffectiveConsumedBy | str] = None,
        decay_state: Optional[str] = None,
    ) -> List[AffectiveExamination]:
        assert self._db is not None
        clauses: list[str] = []
        params: list[Any] = []
        now = datetime.now(timezone.utc).isoformat()
        if session_id is not None:
            clauses.append("session_id = ?")
            params.append(session_id)
        if source_type is not None:
            clauses.append("source_type = ?")
            params.append(
                source_type.value if isinstance(source_type, AffectiveSourceType) else str(source_type)
            )
        if action_pressure is not None:
            clauses.append("action_pressure = ?")
            params.append(
                action_pressure.value
                if isinstance(action_pressure, AffectiveActionPressure)
                else str(action_pressure)
            )
        if consumed_by is not None:
            clauses.append("consumed_by = ?")
            params.append(
                consumed_by.value
                if isinstance(consumed_by, AffectiveConsumedBy)
                else str(consumed_by)
            )
        if primary_emotion is not None:
            clauses.append("json_extract(affect, '$.primary_emotion') = ?")
            params.append(str(primary_emotion))
        if decay_state == "active":
            clauses.append("(expires_at IS NULL OR expires_at > ?)")
            params.append(now)
        elif decay_state == "expired":
            clauses.append("expires_at IS NOT NULL AND expires_at <= ?")
            params.append(now)
        where = f"WHERE {' AND '.join(clauses)}" if clauses else ""
        cursor = await self._db.execute(
            f"""
            SELECT * FROM affective_examinations
            {where}
            ORDER BY created_at DESC, examination_id DESC
            LIMIT ?
            """,
            (*params, limit),
        )
        rows = await cursor.fetchall()
        return [self._row_to_record(row) for row in rows]

    async def list_unresolved_pressures(
        self,
        *,
        session_id: Optional[str] = None,
        limit: int = 50,
    ) -> List[AffectiveExamination]:
        assert self._db is not None
        now = datetime.now(timezone.utc).isoformat()
        clauses = ["consumed_by = ?", "(expires_at IS NULL OR expires_at > ?)"]
        params: list[Any] = [AffectiveConsumedBy.NONE.value, now]
        if session_id is not None:
            clauses.append("session_id = ?")
            params.append(session_id)
        cursor = await self._db.execute(
            f"""
            SELECT * FROM affective_examinations
            WHERE {' AND '.join(clauses)}
            ORDER BY created_at DESC, examination_id DESC
            LIMIT ?
            """,
            (*params, limit),
        )
        rows = await cursor.fetchall()
        return [self._row_to_record(row) for row in rows]

    async def mark_consumed(
        self,
        examination_id: str,
        consumed_by: AffectiveConsumedBy,
    ) -> None:
        assert self._db is not None
        await self._db.execute(
            """
            UPDATE affective_examinations
            SET consumed_by = ?
            WHERE examination_id = ?
            """,
            (consumed_by.value, examination_id),
        )
        await self._db.commit()

    @staticmethod
    def _record_params(record: AffectiveExamination) -> tuple[Any, ...]:
        return (
            str(record.examination_id),
            record.created_at.isoformat(),
            record.session_id,
            record.source_type.value,
            record.source_id,
            record.source_excerpt,
            record.source_hash,
            record.target.value,
            record.affect.model_dump_json(),
            record.intensity,
            record.confidence,
            record.action_pressure.value,
            record.bounded_reason,
            record.consumed_by.value,
            record.expires_at.isoformat() if record.expires_at else None,
            record.appraisal_version,
            json.dumps(record.meta),
        )

    @staticmethod
    def _row_to_record(row: aiosqlite.Row) -> AffectiveExamination:
        return AffectiveExamination(
            examination_id=row["examination_id"],
            created_at=datetime.fromisoformat(row["created_at"]),
            session_id=row["session_id"],
            source_type=AffectiveSourceType(row["source_type"]),
            source_id=row["source_id"],
            source_excerpt=row["source_excerpt"],
            source_hash=row["source_hash"],
            target=AffectiveTarget(row["target"]),
            affect=json.loads(row["affect"]),
            intensity=float(row["intensity"]),
            confidence=float(row["confidence"]),
            action_pressure=AffectiveActionPressure(row["action_pressure"]),
            bounded_reason=row["bounded_reason"],
            consumed_by=AffectiveConsumedBy(row["consumed_by"]),
            expires_at=datetime.fromisoformat(row["expires_at"]) if row["expires_at"] else None,
            appraisal_version=row["appraisal_version"],
            meta=json.loads(row["meta"]) if row["meta"] else {},
        )
