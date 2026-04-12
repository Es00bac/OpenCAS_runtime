"""SQLite store for scheduled tasks, events, and run history."""

from __future__ import annotations

import json
import sqlite3
from datetime import datetime, timezone
from pathlib import Path
from typing import List, Optional

import aiosqlite

from .models import (
    ScheduleAction,
    ScheduleItem,
    ScheduleKind,
    ScheduleRecurrence,
    ScheduleRun,
    ScheduleRunStatus,
    ScheduleStatus,
)

_SCHEMA = """
CREATE TABLE IF NOT EXISTS schedules (
    schedule_id TEXT PRIMARY KEY,
    created_at TEXT NOT NULL,
    updated_at TEXT NOT NULL,
    kind TEXT NOT NULL,
    action TEXT NOT NULL,
    status TEXT NOT NULL,
    title TEXT NOT NULL,
    description TEXT NOT NULL DEFAULT '',
    objective TEXT,
    start_at TEXT NOT NULL,
    end_at TEXT,
    timezone TEXT NOT NULL DEFAULT 'America/Denver',
    next_run_at TEXT,
    last_run_at TEXT,
    recurrence TEXT NOT NULL DEFAULT 'none',
    interval_hours REAL,
    weekdays TEXT NOT NULL DEFAULT '[]',
    max_occurrences INTEGER,
    occurrence_count INTEGER NOT NULL DEFAULT 0,
    priority REAL NOT NULL DEFAULT 5.0,
    tags TEXT NOT NULL DEFAULT '[]',
    commitment_id TEXT,
    plan_id TEXT,
    meta TEXT NOT NULL DEFAULT '{}'
);
CREATE INDEX IF NOT EXISTS idx_schedules_status_next_run ON schedules(status, next_run_at);
CREATE INDEX IF NOT EXISTS idx_schedules_kind ON schedules(kind);

CREATE TABLE IF NOT EXISTS schedule_runs (
    run_id TEXT PRIMARY KEY,
    schedule_id TEXT NOT NULL,
    scheduled_for TEXT NOT NULL,
    started_at TEXT NOT NULL,
    finished_at TEXT,
    status TEXT NOT NULL,
    task_id TEXT,
    error TEXT,
    meta TEXT NOT NULL DEFAULT '{}',
    FOREIGN KEY (schedule_id) REFERENCES schedules(schedule_id)
);
CREATE INDEX IF NOT EXISTS idx_schedule_runs_schedule ON schedule_runs(schedule_id);
CREATE INDEX IF NOT EXISTS idx_schedule_runs_started ON schedule_runs(started_at);
"""

_MIGRATIONS: List[str] = []


class ScheduleStore:
    """Async SQLite store for schedules and schedule runs."""

    def __init__(self, path: Path | str) -> None:
        self.path = Path(path)
        self._db: aiosqlite.Connection | None = None

    async def connect(self) -> "ScheduleStore":
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._db = await aiosqlite.connect(str(self.path))
        self._db.row_factory = aiosqlite.Row
        await self._db.executescript(_SCHEMA)
        await self._db.commit()
        await self._migrate()
        return self

    async def close(self) -> None:
        if self._db:
            await self._db.close()
            self._db = None

    async def _migrate(self) -> None:
        for sql in _MIGRATIONS:
            try:
                assert self._db is not None
                await self._db.execute(sql)
                await self._db.commit()
            except sqlite3.OperationalError:
                pass

    async def save(self, item: ScheduleItem) -> None:
        assert self._db is not None
        item.updated_at = datetime.now(timezone.utc)
        await self._db.execute(
            """
            INSERT INTO schedules (
                schedule_id, created_at, updated_at, kind, action, status, title,
                description, objective, start_at, end_at, timezone, next_run_at,
                last_run_at, recurrence, interval_hours, weekdays, max_occurrences,
                occurrence_count, priority, tags, commitment_id, plan_id, meta
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            ON CONFLICT(schedule_id) DO UPDATE SET
                updated_at = excluded.updated_at,
                kind = excluded.kind,
                action = excluded.action,
                status = excluded.status,
                title = excluded.title,
                description = excluded.description,
                objective = excluded.objective,
                start_at = excluded.start_at,
                end_at = excluded.end_at,
                timezone = excluded.timezone,
                next_run_at = excluded.next_run_at,
                last_run_at = excluded.last_run_at,
                recurrence = excluded.recurrence,
                interval_hours = excluded.interval_hours,
                weekdays = excluded.weekdays,
                max_occurrences = excluded.max_occurrences,
                occurrence_count = excluded.occurrence_count,
                priority = excluded.priority,
                tags = excluded.tags,
                commitment_id = excluded.commitment_id,
                plan_id = excluded.plan_id,
                meta = excluded.meta
            """,
            self._item_values(item),
        )
        await self._db.commit()

    async def get(self, schedule_id: str) -> Optional[ScheduleItem]:
        assert self._db is not None
        cursor = await self._db.execute(
            "SELECT * FROM schedules WHERE schedule_id = ?", (schedule_id,)
        )
        row = await cursor.fetchone()
        return self._row_to_item(row) if row else None

    async def list_items(
        self,
        status: Optional[ScheduleStatus] = None,
        kind: Optional[ScheduleKind] = None,
        limit: int = 100,
        offset: int = 0,
    ) -> List[ScheduleItem]:
        assert self._db is not None
        clauses: List[str] = []
        params: List[object] = []
        if status is not None:
            clauses.append("status = ?")
            params.append(status.value)
        if kind is not None:
            clauses.append("kind = ?")
            params.append(kind.value)
        where = f"WHERE {' AND '.join(clauses)}" if clauses else ""
        cursor = await self._db.execute(
            f"""
            SELECT * FROM schedules
            {where}
            ORDER BY COALESCE(next_run_at, start_at) ASC, priority DESC
            LIMIT ? OFFSET ?
            """,
            tuple(params + [limit, offset]),
        )
        rows = await cursor.fetchall()
        return [self._row_to_item(row) for row in rows]

    async def list_due(
        self,
        now: datetime,
        limit: int = 50,
    ) -> List[ScheduleItem]:
        assert self._db is not None
        now_iso = now.astimezone(timezone.utc).isoformat()
        cursor = await self._db.execute(
            """
            SELECT * FROM schedules
            WHERE status = ? AND next_run_at IS NOT NULL AND next_run_at <= ?
            ORDER BY next_run_at ASC, priority DESC
            LIMIT ?
            """,
            (ScheduleStatus.ACTIVE.value, now_iso, limit),
        )
        rows = await cursor.fetchall()
        return [self._row_to_item(row) for row in rows]

    async def record_run(self, run: ScheduleRun) -> None:
        assert self._db is not None
        await self._db.execute(
            """
            INSERT INTO schedule_runs (
                run_id, schedule_id, scheduled_for, started_at, finished_at, status, task_id, error, meta
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                str(run.run_id),
                str(run.schedule_id),
                run.scheduled_for.isoformat(),
                run.started_at.isoformat(),
                run.finished_at.isoformat() if run.finished_at else None,
                run.status.value,
                run.task_id,
                run.error,
                json.dumps(run.meta),
            ),
        )
        await self._db.commit()

    async def list_runs(
        self,
        schedule_id: Optional[str] = None,
        limit: int = 100,
        offset: int = 0,
    ) -> List[ScheduleRun]:
        assert self._db is not None
        if schedule_id:
            cursor = await self._db.execute(
                """
                SELECT * FROM schedule_runs
                WHERE schedule_id = ?
                ORDER BY started_at DESC
                LIMIT ? OFFSET ?
                """,
                (schedule_id, limit, offset),
            )
        else:
            cursor = await self._db.execute(
                """
                SELECT * FROM schedule_runs
                ORDER BY started_at DESC
                LIMIT ? OFFSET ?
                """,
                (limit, offset),
            )
        rows = await cursor.fetchall()
        return [self._row_to_run(row) for row in rows]

    async def cancel(self, schedule_id: str) -> bool:
        item = await self.get(schedule_id)
        if item is None:
            return False
        item.status = ScheduleStatus.CANCELLED
        item.next_run_at = None
        await self.save(item)
        return True

    @staticmethod
    def _item_values(item: ScheduleItem) -> tuple[object, ...]:
        return (
            str(item.schedule_id),
            item.created_at.isoformat(),
            item.updated_at.isoformat(),
            item.kind.value,
            item.action.value,
            item.status.value,
            item.title,
            item.description,
            item.objective,
            item.start_at.isoformat(),
            item.end_at.isoformat() if item.end_at else None,
            item.timezone,
            item.next_run_at.isoformat() if item.next_run_at else None,
            item.last_run_at.isoformat() if item.last_run_at else None,
            item.recurrence.value,
            item.interval_hours,
            json.dumps(item.weekdays),
            item.max_occurrences,
            item.occurrence_count,
            item.priority,
            json.dumps(item.tags),
            item.commitment_id,
            item.plan_id,
            json.dumps(item.meta),
        )

    @staticmethod
    def _row_to_item(row: aiosqlite.Row) -> ScheduleItem:
        return ScheduleItem(
            schedule_id=row["schedule_id"],
            created_at=datetime.fromisoformat(row["created_at"]),
            updated_at=datetime.fromisoformat(row["updated_at"]),
            kind=ScheduleKind(row["kind"]),
            action=ScheduleAction(row["action"]),
            status=ScheduleStatus(row["status"]),
            title=row["title"],
            description=row["description"] or "",
            objective=row["objective"],
            start_at=datetime.fromisoformat(row["start_at"]),
            end_at=datetime.fromisoformat(row["end_at"]) if row["end_at"] else None,
            timezone=row["timezone"] or "America/Denver",
            next_run_at=datetime.fromisoformat(row["next_run_at"]) if row["next_run_at"] else None,
            last_run_at=datetime.fromisoformat(row["last_run_at"]) if row["last_run_at"] else None,
            recurrence=ScheduleRecurrence(row["recurrence"]),
            interval_hours=row["interval_hours"],
            weekdays=json.loads(row["weekdays"]) if row["weekdays"] else [],
            max_occurrences=row["max_occurrences"],
            occurrence_count=row["occurrence_count"],
            priority=row["priority"],
            tags=json.loads(row["tags"]) if row["tags"] else [],
            commitment_id=row["commitment_id"],
            plan_id=row["plan_id"],
            meta=json.loads(row["meta"]) if row["meta"] else {},
        )

    @staticmethod
    def _row_to_run(row: aiosqlite.Row) -> ScheduleRun:
        return ScheduleRun(
            run_id=row["run_id"],
            schedule_id=row["schedule_id"],
            scheduled_for=datetime.fromisoformat(row["scheduled_for"]),
            started_at=datetime.fromisoformat(row["started_at"]),
            finished_at=datetime.fromisoformat(row["finished_at"]) if row["finished_at"] else None,
            status=ScheduleRunStatus(row["status"]),
            task_id=row["task_id"],
            error=row["error"],
            meta=json.loads(row["meta"]) if row["meta"] else {},
        )
