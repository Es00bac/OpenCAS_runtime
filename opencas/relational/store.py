"""SQLite store for musubi state and history."""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import List, Optional

import aiosqlite

from .models import MusubiRecord, MusubiState, ResonanceDimension

_SCHEMA = """
CREATE TABLE IF NOT EXISTS musubi_state (
    state_id TEXT PRIMARY KEY,
    updated_at TEXT NOT NULL,
    musubi REAL NOT NULL DEFAULT 0.0,
    dimensions TEXT NOT NULL DEFAULT '{}',
    source_tag TEXT
);

CREATE TABLE IF NOT EXISTS musubi_history (
    record_id TEXT PRIMARY KEY,
    created_at TEXT NOT NULL,
    musubi_before REAL NOT NULL DEFAULT 0.0,
    musubi_after REAL NOT NULL DEFAULT 0.0,
    delta REAL NOT NULL DEFAULT 0.0,
    dimension_deltas TEXT NOT NULL DEFAULT '{}',
    trigger_event TEXT NOT NULL,
    episode_id TEXT,
    note TEXT
);

CREATE INDEX IF NOT EXISTS idx_musubi_history_created_at ON musubi_history(created_at);
CREATE INDEX IF NOT EXISTS idx_musubi_history_trigger ON musubi_history(trigger_event);
"""


class MusubiStore:
    """Async SQLite store for relational resonance state and time-series."""

    def __init__(self, path: Path | str) -> None:
        self.path = Path(path)
        self._db: aiosqlite.Connection | None = None

    async def connect(self) -> "MusubiStore":
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

    async def save_state(self, state: MusubiState) -> None:
        """Upsert the current musubi state (single row)."""
        assert self._db is not None
        await self._db.execute(
            """
            INSERT INTO musubi_state (state_id, updated_at, musubi, dimensions, source_tag)
            VALUES (?, ?, ?, ?, ?)
            ON CONFLICT(state_id) DO UPDATE SET
                updated_at = excluded.updated_at,
                musubi = excluded.musubi,
                dimensions = excluded.dimensions,
                source_tag = excluded.source_tag
            """,
            (
                str(state.state_id),
                state.updated_at.isoformat(),
                state.musubi,
                json.dumps(state.dimensions),
                state.source_tag,
            ),
        )
        await self._db.commit()

    async def load_state(self) -> Optional[MusubiState]:
        """Fetch the most recent musubi state."""
        assert self._db is not None
        cursor = await self._db.execute(
            "SELECT * FROM musubi_state ORDER BY updated_at DESC LIMIT 1"
        )
        row = await cursor.fetchone()
        if row is None:
            return None
        return self._row_to_state(row)

    async def append_record(self, record: MusubiRecord) -> None:
        """Append a musubi record to the history."""
        assert self._db is not None
        await self._db.execute(
            """
            INSERT INTO musubi_history (
                record_id, created_at, musubi_before, musubi_after, delta,
                dimension_deltas, trigger_event, episode_id, note
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                str(record.record_id),
                record.created_at.isoformat(),
                record.musubi_before,
                record.musubi_after,
                record.delta,
                json.dumps(record.dimension_deltas),
                record.trigger_event,
                record.episode_id,
                record.note,
            ),
        )
        await self._db.commit()

    async def list_history(
        self,
        limit: int = 100,
        offset: int = 0,
    ) -> List[MusubiRecord]:
        """Return musubi history ordered by most recent first."""
        assert self._db is not None
        cursor = await self._db.execute(
            "SELECT * FROM musubi_history ORDER BY created_at DESC LIMIT ? OFFSET ?",
            (limit, offset),
        )
        rows = await cursor.fetchall()
        return [self._row_to_record(r) for r in rows]

    async def get_dimension_trend(
        self,
        dimension: ResonanceDimension,
        limit: int = 10,
    ) -> List[float]:
        """Return the most recent values for a dimension from history records."""
        assert self._db is not None
        cursor = await self._db.execute(
            """
            SELECT dimension_deltas FROM musubi_history
            ORDER BY created_at DESC LIMIT ?
            """,
            (limit,),
        )
        rows = await cursor.fetchall()
        values: List[float] = []
        # We approximate trend by summing deltas from oldest to newest for the dimension
        for row in reversed(rows):
            deltas = json.loads(row["dimension_deltas"]) if row["dimension_deltas"] else {}
            values.append(deltas.get(dimension.value, 0.0))
        return values

    @staticmethod
    def _row_to_state(row: aiosqlite.Row) -> MusubiState:
        dims = json.loads(row["dimensions"]) if row["dimensions"] else {}
        return MusubiState(
            state_id=row["state_id"],
            updated_at=datetime.fromisoformat(row["updated_at"]),
            musubi=row["musubi"],
            dimensions=dims,
            source_tag=row["source_tag"],
        )

    @staticmethod
    def _row_to_record(row: aiosqlite.Row) -> MusubiRecord:
        deltas = json.loads(row["dimension_deltas"]) if row["dimension_deltas"] else {}
        return MusubiRecord(
            record_id=row["record_id"],
            created_at=datetime.fromisoformat(row["created_at"]),
            musubi_before=row["musubi_before"],
            musubi_after=row["musubi_after"],
            delta=row["delta"],
            dimension_deltas=deltas,
            trigger_event=row["trigger_event"],
            episode_id=row["episode_id"],
            note=row["note"],
        )
