"""SQLite-backed store for durable somatic snapshots."""

from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path
from typing import List, Optional

from opencas.somatic.models import SomaticSnapshot


class SomaticStore:
    """Async SQLite store for somatic snapshot history."""

    def __init__(self, db_path: Path | str) -> None:
        self.db_path = Path(db_path)
        self._db = None

    async def connect(self) -> "SomaticStore":
        import aiosqlite

        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._db = await aiosqlite.connect(str(self.db_path))
        await self._db.executescript(
            """
            CREATE TABLE IF NOT EXISTS somatic_snapshots (
                snapshot_id TEXT PRIMARY KEY,
                recorded_at TEXT NOT NULL,
                arousal REAL NOT NULL,
                fatigue REAL NOT NULL,
                tension REAL NOT NULL,
                valence REAL NOT NULL,
                focus REAL DEFAULT 0.5,
                energy REAL DEFAULT 0.5,
                musubi REAL,
                primary_emotion TEXT NOT NULL,
                somatic_tag TEXT,
                certainty REAL DEFAULT 0.5,
                source TEXT NOT NULL,
                trigger_event_id TEXT,
                embedding_id TEXT
            );
            CREATE INDEX IF NOT EXISTS idx_somatic_recorded_at
                ON somatic_snapshots(recorded_at);
            CREATE INDEX IF NOT EXISTS idx_somatic_source
                ON somatic_snapshots(source);
            CREATE INDEX IF NOT EXISTS idx_somatic_trigger
                ON somatic_snapshots(trigger_event_id);
            CREATE INDEX IF NOT EXISTS idx_somatic_embedding
                ON somatic_snapshots(embedding_id);
            """
        )
        await self._db.commit()
        return self

    async def close(self) -> None:
        if self._db:
            await self._db.close()
            self._db = None

    async def save(self, snapshot: SomaticSnapshot) -> None:
        assert self._db is not None
        await self._db.execute(
            """
            INSERT INTO somatic_snapshots (
                snapshot_id, recorded_at, arousal, fatigue, tension, valence,
                focus, energy, musubi, primary_emotion, somatic_tag, certainty,
                source, trigger_event_id, embedding_id
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                str(snapshot.snapshot_id),
                snapshot.recorded_at.isoformat(),
                snapshot.arousal,
                snapshot.fatigue,
                snapshot.tension,
                snapshot.valence,
                snapshot.focus,
                snapshot.energy,
                snapshot.musubi,
                snapshot.primary_emotion.value,
                snapshot.somatic_tag,
                snapshot.certainty,
                snapshot.source,
                snapshot.trigger_event_id,
                snapshot.embedding_id,
            ),
        )
        await self._db.commit()

    async def save_batch(self, snapshots: List[SomaticSnapshot]) -> None:
        assert self._db is not None
        rows = [
            (
                str(s.snapshot_id),
                s.recorded_at.isoformat(),
                s.arousal,
                s.fatigue,
                s.tension,
                s.valence,
                s.focus,
                s.energy,
                s.musubi,
                s.primary_emotion.value,
                s.somatic_tag,
                s.certainty,
                s.source,
                s.trigger_event_id,
                s.embedding_id,
            )
            for s in snapshots
        ]
        await self._db.executemany(
            """
            INSERT INTO somatic_snapshots (
                snapshot_id, recorded_at, arousal, fatigue, tension, valence,
                focus, energy, musubi, primary_emotion, somatic_tag, certainty,
                source, trigger_event_id, embedding_id
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            rows,
        )
        await self._db.commit()

    async def get_latest(self) -> Optional[SomaticSnapshot]:
        assert self._db is not None
        cursor = await self._db.execute(
            "SELECT * FROM somatic_snapshots ORDER BY recorded_at DESC LIMIT 1"
        )
        row = await cursor.fetchone()
        if row is None:
            return None
        return self._row_to_snapshot(row)

    async def list_recent(
        self,
        limit: int = 100,
        before: Optional[datetime] = None,
    ) -> List[SomaticSnapshot]:
        assert self._db is not None
        if before:
            cursor = await self._db.execute(
                "SELECT * FROM somatic_snapshots WHERE recorded_at <= ? "
                "ORDER BY recorded_at DESC LIMIT ?",
                (before.isoformat(), limit),
            )
        else:
            cursor = await self._db.execute(
                "SELECT * FROM somatic_snapshots ORDER BY recorded_at DESC LIMIT ?",
                (limit,),
            )
        rows = await cursor.fetchall()
        return [self._row_to_snapshot(r) for r in rows]

    async def trajectory(
        self,
        start: Optional[datetime] = None,
        end: Optional[datetime] = None,
    ) -> List[SomaticSnapshot]:
        assert self._db is not None
        query = "SELECT * FROM somatic_snapshots WHERE 1=1"
        params: List[str] = []
        if start:
            query += " AND recorded_at >= ?"
            params.append(start.isoformat())
        if end:
            query += " AND recorded_at <= ?"
            params.append(end.isoformat())
        query += " ORDER BY recorded_at ASC"
        cursor = await self._db.execute(query, params)
        rows = await cursor.fetchall()
        return [self._row_to_snapshot(r) for r in rows]

    @staticmethod
    def _row_to_snapshot(row) -> SomaticSnapshot:
        from opencas.somatic.models import PrimaryEmotion

        return SomaticSnapshot(
            snapshot_id=row[0],
            recorded_at=datetime.fromisoformat(row[1]),
            arousal=row[2],
            fatigue=row[3],
            tension=row[4],
            valence=row[5],
            focus=row[6],
            energy=row[7],
            musubi=row[8],
            primary_emotion=PrimaryEmotion(row[9]),
            somatic_tag=row[10],
            certainty=row[11],
            source=row[12],
            trigger_event_id=row[13],
            embedding_id=row[14],
        )
