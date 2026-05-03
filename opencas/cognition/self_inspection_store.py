"""SQLite persistence for self-inspection records."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import aiosqlite

from .self_inspection import CommitmentGapStatus, SelfInspectionPhase, SelfInspectionRecord

_SCHEMA = """
CREATE TABLE IF NOT EXISTS self_inspection_records (
    record_id TEXT PRIMARY KEY,
    created_at TEXT NOT NULL,
    session_id TEXT NOT NULL,
    phase TEXT NOT NULL,
    has_open_commitment_gap INTEGER NOT NULL DEFAULT 0,
    raw TEXT NOT NULL
);

CREATE INDEX IF NOT EXISTS idx_self_inspection_session_created_at
    ON self_inspection_records(session_id, created_at);
CREATE INDEX IF NOT EXISTS idx_self_inspection_phase_created_at
    ON self_inspection_records(phase, created_at);
CREATE INDEX IF NOT EXISTS idx_self_inspection_open_gap_created_at
    ON self_inspection_records(has_open_commitment_gap, created_at);
"""


class SelfInspectionStore:
    """Persist pre/post turn self-inspection packets."""

    def __init__(self, path: Path | str) -> None:
        self.path = Path(path)
        self._db: aiosqlite.Connection | None = None

    async def connect(self) -> "SelfInspectionStore":
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

    async def save(self, record: SelfInspectionRecord) -> None:
        """Insert or replace a self-inspection record."""
        assert self._db is not None
        has_open_gap = any(
            gap.status in {CommitmentGapStatus.OPEN, CommitmentGapStatus.WATCH}
            for gap in record.commitment_gaps
        )
        await self._db.execute(
            """
            INSERT INTO self_inspection_records (
                record_id, created_at, session_id, phase, has_open_commitment_gap, raw
            ) VALUES (?, ?, ?, ?, ?, ?)
            ON CONFLICT(record_id) DO UPDATE SET
                created_at = excluded.created_at,
                session_id = excluded.session_id,
                phase = excluded.phase,
                has_open_commitment_gap = excluded.has_open_commitment_gap,
                raw = excluded.raw
            """,
            (
                str(record.record_id),
                record.created_at.isoformat(),
                record.session_id,
                record.phase.value,
                int(has_open_gap),
                json.dumps(record.model_dump(mode="json")),
            ),
        )
        await self._db.commit()

    async def list_recent(
        self,
        *,
        session_id: str | None = None,
        phase: SelfInspectionPhase | str | None = None,
        limit: int = 20,
    ) -> list[SelfInspectionRecord]:
        """Return recent self-inspection records in chronological order."""
        assert self._db is not None
        clauses: list[str] = []
        params: list[Any] = []
        if session_id is not None:
            clauses.append("session_id = ?")
            params.append(session_id)
        if phase is not None:
            phase_value = phase.value if isinstance(phase, SelfInspectionPhase) else str(phase)
            clauses.append("phase = ?")
            params.append(phase_value)
        where = f"WHERE {' AND '.join(clauses)}" if clauses else ""
        params.append(max(1, int(limit)))
        cursor = await self._db.execute(
            f"""
            SELECT raw
            FROM self_inspection_records
            {where}
            ORDER BY created_at DESC
            LIMIT ?
            """,
            tuple(params),
        )
        rows = await cursor.fetchall()
        records = [self._row_to_record(row) for row in rows]
        records.reverse()
        return records

    async def list_unresolved_commitment_gaps(
        self,
        *,
        limit: int = 20,
    ) -> list[SelfInspectionRecord]:
        """Return recent records containing open or watch commitment gaps."""
        assert self._db is not None
        cursor = await self._db.execute(
            """
            SELECT raw
            FROM self_inspection_records
            WHERE has_open_commitment_gap = 1
            ORDER BY created_at DESC
            LIMIT ?
            """,
            (max(1, int(limit)),),
        )
        rows = await cursor.fetchall()
        return [self._row_to_record(row) for row in rows]

    async def search(
        self,
        query: str,
        *,
        session_id: str | None = None,
        limit: int = 20,
    ) -> list[SelfInspectionRecord]:
        """Search structured inspection records for a compact term or phrase."""
        assert self._db is not None
        term = " ".join(str(query or "").split())
        if not term:
            return []
        clauses = ["raw LIKE ? ESCAPE '\\' COLLATE NOCASE"]
        params: list[Any] = [f"%{_escape_like(term)}%"]
        if session_id is not None:
            clauses.append("session_id = ?")
            params.append(session_id)
        params.append(max(1, int(limit)))
        cursor = await self._db.execute(
            f"""
            SELECT raw
            FROM self_inspection_records
            WHERE {' AND '.join(clauses)}
            ORDER BY created_at DESC
            LIMIT ?
            """,
            tuple(params),
        )
        rows = await cursor.fetchall()
        return [self._row_to_record(row) for row in rows]

    @staticmethod
    def _row_to_record(row: aiosqlite.Row) -> SelfInspectionRecord:
        payload = json.loads(row["raw"])
        return SelfInspectionRecord.model_validate(payload)


def _escape_like(value: str) -> str:
    return value.replace("\\", "\\\\").replace("%", "\\%").replace("_", "\\_")
