"""SQLite store for nightly dream records."""

from __future__ import annotations

from pathlib import Path

import aiosqlite

from .models import DreamMode, DreamRecord

_SCHEMA = """
CREATE TABLE IF NOT EXISTS dream_records (
    dream_id TEXT PRIMARY KEY,
    created_at TEXT NOT NULL,
    mode TEXT NOT NULL,
    source TEXT NOT NULL,
    consolidation_result_id TEXT NOT NULL,
    payload TEXT NOT NULL
);

CREATE INDEX IF NOT EXISTS idx_dream_records_created_at
ON dream_records(created_at);

CREATE INDEX IF NOT EXISTS idx_dream_records_mode_created_at
ON dream_records(mode, created_at);
"""


class DreamStore:
    """Async SQLite persistence for nightly dreams."""

    def __init__(self, path: Path | str) -> None:
        self.path = Path(path)
        self._db: aiosqlite.Connection | None = None

    async def connect(self) -> "DreamStore":
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

    async def save(self, record: DreamRecord) -> None:
        assert self._db is not None
        await self._db.execute(
            """
            INSERT INTO dream_records (
                dream_id, created_at, mode, source, consolidation_result_id, payload
            ) VALUES (?, ?, ?, ?, ?, ?)
            ON CONFLICT(dream_id) DO UPDATE SET
                created_at = excluded.created_at,
                mode = excluded.mode,
                source = excluded.source,
                consolidation_result_id = excluded.consolidation_result_id,
                payload = excluded.payload
            """,
            (
                str(record.dream_id),
                record.created_at.isoformat(),
                record.mode.value,
                record.source,
                record.consolidation_result_id,
                record.model_dump_json(),
            ),
        )
        await self._db.commit()

    async def list_recent(
        self,
        *,
        mode: DreamMode | str | None = None,
        limit: int = 20,
    ) -> list[DreamRecord]:
        assert self._db is not None
        bounded_limit = max(1, min(200, int(limit)))
        if mode is not None:
            mode_value = mode.value if isinstance(mode, DreamMode) else str(mode)
            cursor = await self._db.execute(
                """
                SELECT payload FROM dream_records
                WHERE mode = ?
                ORDER BY created_at DESC
                LIMIT ?
                """,
                (mode_value, bounded_limit),
            )
        else:
            cursor = await self._db.execute(
                """
                SELECT payload FROM dream_records
                ORDER BY created_at DESC
                LIMIT ?
                """,
                (bounded_limit,),
            )
        rows = await cursor.fetchall()
        return [DreamRecord.model_validate_json(row["payload"]) for row in rows]

