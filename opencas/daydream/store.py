"""SQLite stores for daydream reflections and conflict registry."""

from __future__ import annotations

from datetime import datetime, timezone, timedelta
from pathlib import Path
from typing import List, Optional

import aiosqlite
import sqlite3

from .models import (
    ConflictRecord,
    DaydreamInitiative,
    DaydreamNotification,
    DaydreamOutcome,
    DaydreamReflection,
    DaydreamSpark,
)


_SCHEMA = """
CREATE TABLE IF NOT EXISTS daydream_reflections (
    reflection_id TEXT PRIMARY KEY,
    created_at TEXT NOT NULL,
    spark_content TEXT NOT NULL,
    recollection TEXT NOT NULL DEFAULT '',
    interpretation TEXT NOT NULL DEFAULT '',
    synthesis TEXT NOT NULL DEFAULT '',
    open_question TEXT,
    changed_self_view TEXT NOT NULL DEFAULT '',
    tension_hints TEXT NOT NULL DEFAULT '[]',
    alignment_score REAL NOT NULL DEFAULT 0.0,
    novelty_score REAL NOT NULL DEFAULT 0.0,
    keeper INTEGER NOT NULL DEFAULT 0
);

CREATE INDEX IF NOT EXISTS idx_reflections_created_at ON daydream_reflections(created_at);
CREATE INDEX IF NOT EXISTS idx_reflections_keeper ON daydream_reflections(keeper);

CREATE TABLE IF NOT EXISTS daydream_sparks (
    spark_id TEXT PRIMARY KEY,
    created_at TEXT NOT NULL,
    mode TEXT NOT NULL DEFAULT '',
    trigger TEXT NOT NULL DEFAULT '',
    interest TEXT NOT NULL DEFAULT '',
    summary TEXT NOT NULL DEFAULT '',
    label TEXT NOT NULL DEFAULT '',
    kind TEXT NOT NULL DEFAULT '',
    intensity REAL NOT NULL DEFAULT 0.0,
    objective TEXT NOT NULL DEFAULT '',
    tags TEXT NOT NULL DEFAULT '[]',
    task_id TEXT,
    raw TEXT NOT NULL DEFAULT '{}'
);

CREATE INDEX IF NOT EXISTS idx_sparks_created_at ON daydream_sparks(created_at);
CREATE INDEX IF NOT EXISTS idx_sparks_kind ON daydream_sparks(kind);
CREATE INDEX IF NOT EXISTS idx_sparks_task_id ON daydream_sparks(task_id);

CREATE TABLE IF NOT EXISTS daydream_initiatives (
    initiative_id TEXT PRIMARY KEY,
    spark_id TEXT,
    created_at TEXT NOT NULL,
    mode TEXT NOT NULL DEFAULT '',
    trigger TEXT NOT NULL DEFAULT '',
    interest TEXT NOT NULL DEFAULT '',
    summary TEXT NOT NULL DEFAULT '',
    label TEXT NOT NULL DEFAULT '',
    kind TEXT NOT NULL DEFAULT '',
    intensity REAL NOT NULL DEFAULT 0.0,
    rung TEXT NOT NULL DEFAULT '',
    desired_rung TEXT NOT NULL DEFAULT '',
    objective TEXT NOT NULL DEFAULT '',
    focus TEXT NOT NULL DEFAULT '',
    source_kind TEXT NOT NULL DEFAULT '',
    source_label TEXT NOT NULL DEFAULT '',
    artifact_paths TEXT NOT NULL DEFAULT '[]',
    task_id TEXT,
    route_debug TEXT NOT NULL DEFAULT '{}',
    tags TEXT NOT NULL DEFAULT '[]',
    raw TEXT NOT NULL DEFAULT '{}'
);

CREATE INDEX IF NOT EXISTS idx_initiatives_created_at ON daydream_initiatives(created_at);
CREATE INDEX IF NOT EXISTS idx_initiatives_spark_id ON daydream_initiatives(spark_id);
CREATE INDEX IF NOT EXISTS idx_initiatives_task_id ON daydream_initiatives(task_id);
CREATE INDEX IF NOT EXISTS idx_initiatives_rung ON daydream_initiatives(rung);

CREATE TABLE IF NOT EXISTS daydream_outcomes (
    task_id TEXT PRIMARY KEY,
    recorded_at TEXT NOT NULL,
    outcome TEXT NOT NULL DEFAULT '',
    value_delivered INTEGER NOT NULL DEFAULT 0,
    raw TEXT NOT NULL DEFAULT '{}'
);

CREATE INDEX IF NOT EXISTS idx_outcomes_recorded_at ON daydream_outcomes(recorded_at);

CREATE TABLE IF NOT EXISTS daydream_notifications (
    notification_id TEXT PRIMARY KEY,
    spark_id TEXT,
    chat_id TEXT,
    sent_at TEXT NOT NULL,
    label TEXT NOT NULL DEFAULT '',
    intensity REAL NOT NULL DEFAULT 0.0,
    kind TEXT NOT NULL DEFAULT '',
    raw TEXT NOT NULL DEFAULT '{}'
);

CREATE INDEX IF NOT EXISTS idx_notifications_sent_at ON daydream_notifications(sent_at);
CREATE INDEX IF NOT EXISTS idx_notifications_spark_id ON daydream_notifications(spark_id);

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


class DaydreamStore:
    """Persist daydream reflections and history."""

    def __init__(self, path: Path | str) -> None:
        self.path = Path(path)
        self._db: aiosqlite.Connection | None = None

    async def connect(self) -> "DaydreamStore":
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._db = await aiosqlite.connect(str(self.path))
        self._db.row_factory = aiosqlite.Row
        await self._db.executescript(_SCHEMA)
        await self._migrate()
        await self._db.commit()
        return self

    async def _migrate(self) -> None:
        assert self._db is not None
        # No daydream-specific migrations yet; schema is handled in _SCHEMA
        pass

    async def close(self) -> None:
        if self._db:
            await self._db.close()
            self._db = None

    async def save_reflection(self, reflection: DaydreamReflection) -> None:
        assert self._db is not None
        import json

        await self._db.execute(
            """
            INSERT INTO daydream_reflections (
                reflection_id, created_at, spark_content, recollection,
                interpretation, synthesis, open_question, changed_self_view,
                tension_hints, alignment_score, novelty_score, keeper
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            ON CONFLICT(reflection_id) DO UPDATE SET
                created_at = excluded.created_at,
                spark_content = excluded.spark_content,
                recollection = excluded.recollection,
                interpretation = excluded.interpretation,
                synthesis = excluded.synthesis,
                open_question = excluded.open_question,
                changed_self_view = excluded.changed_self_view,
                tension_hints = excluded.tension_hints,
                alignment_score = excluded.alignment_score,
                novelty_score = excluded.novelty_score,
                keeper = excluded.keeper
            """,
            (
                str(reflection.reflection_id),
                reflection.created_at.isoformat(),
                reflection.spark_content,
                reflection.recollection,
                reflection.interpretation,
                reflection.synthesis,
                reflection.open_question,
                reflection.changed_self_view,
                json.dumps(reflection.tension_hints),
                reflection.alignment_score,
                reflection.novelty_score,
                int(reflection.keeper),
            ),
        )
        await self._db.commit()

    async def save_reflections_batch(self, reflections: List[DaydreamReflection]) -> None:
        """Batch insert reflections in a single transaction."""
        assert self._db is not None
        import json

        params: List[tuple] = []
        for reflection in reflections:
            params.append(
                (
                    str(reflection.reflection_id),
                    reflection.created_at.isoformat(),
                    reflection.spark_content,
                    reflection.recollection,
                    reflection.interpretation,
                    reflection.synthesis,
                    reflection.open_question,
                    reflection.changed_self_view,
                    json.dumps(reflection.tension_hints),
                    reflection.alignment_score,
                    reflection.novelty_score,
                    int(reflection.keeper),
                )
            )
        await self._db.executemany(
            """
            INSERT INTO daydream_reflections (
                reflection_id, created_at, spark_content, recollection,
                interpretation, synthesis, open_question, changed_self_view,
                tension_hints, alignment_score, novelty_score, keeper
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            ON CONFLICT(reflection_id) DO UPDATE SET
                created_at = excluded.created_at,
                spark_content = excluded.spark_content,
                recollection = excluded.recollection,
                interpretation = excluded.interpretation,
                synthesis = excluded.synthesis,
                open_question = excluded.open_question,
                changed_self_view = excluded.changed_self_view,
                tension_hints = excluded.tension_hints,
                alignment_score = excluded.alignment_score,
                novelty_score = excluded.novelty_score,
                keeper = excluded.keeper
            """,
            params,
        )
        await self._db.commit()

    async def list_recent(
        self,
        limit: int = 10,
        keeper_only: Optional[bool] = None,
    ) -> List[DaydreamReflection]:
        assert self._db is not None
        sql = """
            SELECT * FROM daydream_reflections
        """
        params: List[object] = []
        if keeper_only is not None:
            sql += " WHERE keeper = ?"
            params.append(int(keeper_only))
        sql += " ORDER BY created_at DESC LIMIT ?"
        params.append(limit)
        cursor = await self._db.execute(sql, tuple(params))
        rows = await cursor.fetchall()
        return [self._row_to_reflection(r) for r in rows]

    async def get_summary(self, window_days: int = 7) -> dict:
        assert self._db is not None
        cutoff = (datetime.now(timezone.utc) - timedelta(days=max(1, window_days))).isoformat()
        totals_cursor = await self._db.execute(
            """
            SELECT
                COUNT(*) AS total_reflections,
                SUM(CASE WHEN keeper = 1 THEN 1 ELSE 0 END) AS total_keepers
            FROM daydream_reflections
            """
        )
        window_cursor = await self._db.execute(
            """
            SELECT
                COUNT(*) AS window_reflections,
                SUM(CASE WHEN keeper = 1 THEN 1 ELSE 0 END) AS window_keepers
            FROM daydream_reflections
            WHERE created_at >= ?
            """,
            (cutoff,),
        )
        latest_cursor = await self._db.execute(
            """
            SELECT created_at
            FROM daydream_reflections
            ORDER BY created_at DESC
            LIMIT 1
            """
        )
        totals = await totals_cursor.fetchone()
        window = await window_cursor.fetchone()
        latest = await latest_cursor.fetchone()
        return {
            "total_reflections": int(totals["total_reflections"] or 0),
            "total_keepers": int(totals["total_keepers"] or 0),
            "window_days": max(1, window_days),
            "window_reflections": int(window["window_reflections"] or 0),
            "window_keepers": int(window["window_keepers"] or 0),
            "latest_reflection_at": latest["created_at"] if latest else None,
        }

    @staticmethod
    def _row_to_reflection(row: aiosqlite.Row) -> DaydreamReflection:
        import json

        return DaydreamReflection(
            reflection_id=row["reflection_id"],
            created_at=datetime.fromisoformat(row["created_at"]),
            spark_content=row["spark_content"],
            recollection=row["recollection"],
            interpretation=row["interpretation"],
            synthesis=row["synthesis"],
            open_question=row["open_question"],
            changed_self_view=row["changed_self_view"],
            tension_hints=json.loads(row["tension_hints"]) if row["tension_hints"] else [],
            alignment_score=row["alignment_score"],
            novelty_score=row["novelty_score"],
            keeper=bool(row["keeper"]),
        )

    async def save_spark(self, spark: DaydreamSpark) -> None:
        assert self._db is not None
        import json

        await self._db.execute(
            """
            INSERT INTO daydream_sparks (
                spark_id, created_at, mode, trigger, interest, summary, label,
                kind, intensity, objective, tags, task_id, raw
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            ON CONFLICT(spark_id) DO UPDATE SET
                created_at = excluded.created_at,
                mode = excluded.mode,
                trigger = excluded.trigger,
                interest = excluded.interest,
                summary = excluded.summary,
                label = excluded.label,
                kind = excluded.kind,
                intensity = excluded.intensity,
                objective = excluded.objective,
                tags = excluded.tags,
                task_id = excluded.task_id,
                raw = excluded.raw
            """,
            (
                spark.spark_id,
                spark.created_at.isoformat(),
                spark.mode,
                spark.trigger,
                spark.interest,
                spark.summary,
                spark.label,
                spark.kind,
                spark.intensity,
                spark.objective,
                json.dumps(spark.tags),
                spark.task_id,
                json.dumps(spark.raw),
            ),
        )
        await self._db.commit()

    async def save_initiative(self, initiative: DaydreamInitiative) -> None:
        assert self._db is not None
        import json

        await self._db.execute(
            """
            INSERT INTO daydream_initiatives (
                initiative_id, spark_id, created_at, mode, trigger, interest, summary,
                label, kind, intensity, rung, desired_rung, objective, focus,
                source_kind, source_label, artifact_paths, task_id, route_debug, tags, raw
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            ON CONFLICT(initiative_id) DO UPDATE SET
                spark_id = excluded.spark_id,
                created_at = excluded.created_at,
                mode = excluded.mode,
                trigger = excluded.trigger,
                interest = excluded.interest,
                summary = excluded.summary,
                label = excluded.label,
                kind = excluded.kind,
                intensity = excluded.intensity,
                rung = excluded.rung,
                desired_rung = excluded.desired_rung,
                objective = excluded.objective,
                focus = excluded.focus,
                source_kind = excluded.source_kind,
                source_label = excluded.source_label,
                artifact_paths = excluded.artifact_paths,
                task_id = excluded.task_id,
                route_debug = excluded.route_debug,
                tags = excluded.tags,
                raw = excluded.raw
            """,
            (
                initiative.initiative_id,
                initiative.spark_id,
                initiative.created_at.isoformat(),
                initiative.mode,
                initiative.trigger,
                initiative.interest,
                initiative.summary,
                initiative.label,
                initiative.kind,
                initiative.intensity,
                initiative.rung,
                initiative.desired_rung,
                initiative.objective,
                initiative.focus,
                initiative.source_kind,
                initiative.source_label,
                json.dumps(initiative.artifact_paths),
                initiative.task_id,
                json.dumps(initiative.route_debug),
                json.dumps(initiative.tags),
                json.dumps(initiative.raw),
            ),
        )
        await self._db.commit()

    async def save_outcome(self, outcome: DaydreamOutcome) -> None:
        assert self._db is not None
        import json

        await self._db.execute(
            """
            INSERT INTO daydream_outcomes (
                task_id, recorded_at, outcome, value_delivered, raw
            ) VALUES (?, ?, ?, ?, ?)
            ON CONFLICT(task_id) DO UPDATE SET
                recorded_at = excluded.recorded_at,
                outcome = excluded.outcome,
                value_delivered = excluded.value_delivered,
                raw = excluded.raw
            """,
            (
                outcome.task_id,
                outcome.recorded_at.isoformat(),
                outcome.outcome,
                int(outcome.value_delivered),
                json.dumps(outcome.raw),
            ),
        )
        await self._db.commit()

    async def save_notification(self, notification: DaydreamNotification) -> None:
        assert self._db is not None
        import json

        await self._db.execute(
            """
            INSERT INTO daydream_notifications (
                notification_id, spark_id, chat_id, sent_at, label, intensity, kind, raw
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            ON CONFLICT(notification_id) DO UPDATE SET
                spark_id = excluded.spark_id,
                chat_id = excluded.chat_id,
                sent_at = excluded.sent_at,
                label = excluded.label,
                intensity = excluded.intensity,
                kind = excluded.kind,
                raw = excluded.raw
            """,
            (
                notification.notification_id,
                notification.spark_id,
                notification.chat_id,
                notification.sent_at.isoformat(),
                notification.label,
                notification.intensity,
                notification.kind,
                json.dumps(notification.raw),
            ),
        )
        await self._db.commit()

    async def list_sparks(self, limit: int = 50) -> List[DaydreamSpark]:
        assert self._db is not None
        cursor = await self._db.execute(
            "SELECT * FROM daydream_sparks ORDER BY created_at DESC LIMIT ?",
            (limit,),
        )
        return [self._row_to_spark(row) for row in await cursor.fetchall()]

    async def list_initiatives(self, limit: int = 50) -> List[DaydreamInitiative]:
        assert self._db is not None
        cursor = await self._db.execute(
            "SELECT * FROM daydream_initiatives ORDER BY created_at DESC LIMIT ?",
            (limit,),
        )
        return [self._row_to_initiative(row) for row in await cursor.fetchall()]

    async def list_outcomes(self, limit: int = 50) -> List[DaydreamOutcome]:
        assert self._db is not None
        cursor = await self._db.execute(
            "SELECT * FROM daydream_outcomes ORDER BY recorded_at DESC LIMIT ?",
            (limit,),
        )
        return [self._row_to_outcome(row) for row in await cursor.fetchall()]

    async def list_notifications(self, limit: int = 50) -> List[DaydreamNotification]:
        assert self._db is not None
        cursor = await self._db.execute(
            "SELECT * FROM daydream_notifications ORDER BY sent_at DESC LIMIT ?",
            (limit,),
        )
        return [self._row_to_notification(row) for row in await cursor.fetchall()]

    async def get_lifecycle_for_spark(self, spark_id: str) -> dict:
        assert self._db is not None
        spark_cursor = await self._db.execute(
            "SELECT * FROM daydream_sparks WHERE spark_id = ?",
            (spark_id,),
        )
        spark_row = await spark_cursor.fetchone()
        initiatives_cursor = await self._db.execute(
            "SELECT * FROM daydream_initiatives WHERE spark_id = ? ORDER BY created_at DESC",
            (spark_id,),
        )
        notification_cursor = await self._db.execute(
            "SELECT * FROM daydream_notifications WHERE spark_id = ? ORDER BY sent_at DESC",
            (spark_id,),
        )
        initiatives = [self._row_to_initiative(row) for row in await initiatives_cursor.fetchall()]
        task_ids = [item.task_id for item in initiatives if item.task_id]
        outcomes: List[DaydreamOutcome] = []
        if task_ids:
            placeholders = ",".join("?" for _ in task_ids)
            outcome_cursor = await self._db.execute(
                f"SELECT * FROM daydream_outcomes WHERE task_id IN ({placeholders}) ORDER BY recorded_at DESC",
                tuple(task_ids),
            )
            outcomes = [self._row_to_outcome(row) for row in await outcome_cursor.fetchall()]
        return {
            "spark": self._row_to_spark(spark_row).model_dump(mode="json") if spark_row else None,
            "initiatives": [item.model_dump(mode="json") for item in initiatives],
            "outcomes": [item.model_dump(mode="json") for item in outcomes],
            "notifications": [
                self._row_to_notification(row).model_dump(mode="json")
                for row in await notification_cursor.fetchall()
            ],
        }

    @staticmethod
    def _row_to_spark(row: aiosqlite.Row) -> DaydreamSpark:
        import json

        return DaydreamSpark(
            spark_id=row["spark_id"],
            created_at=datetime.fromisoformat(row["created_at"]),
            mode=row["mode"],
            trigger=row["trigger"],
            interest=row["interest"],
            summary=row["summary"],
            label=row["label"],
            kind=row["kind"],
            intensity=row["intensity"],
            objective=row["objective"],
            tags=json.loads(row["tags"]) if row["tags"] else [],
            task_id=row["task_id"],
            raw=json.loads(row["raw"]) if row["raw"] else {},
        )

    @staticmethod
    def _row_to_initiative(row: aiosqlite.Row) -> DaydreamInitiative:
        import json

        return DaydreamInitiative(
            initiative_id=row["initiative_id"],
            spark_id=row["spark_id"],
            created_at=datetime.fromisoformat(row["created_at"]),
            mode=row["mode"],
            trigger=row["trigger"],
            interest=row["interest"],
            summary=row["summary"],
            label=row["label"],
            kind=row["kind"],
            intensity=row["intensity"],
            rung=row["rung"],
            desired_rung=row["desired_rung"],
            objective=row["objective"],
            focus=row["focus"],
            source_kind=row["source_kind"],
            source_label=row["source_label"],
            artifact_paths=json.loads(row["artifact_paths"]) if row["artifact_paths"] else [],
            task_id=row["task_id"],
            route_debug=json.loads(row["route_debug"]) if row["route_debug"] else {},
            tags=json.loads(row["tags"]) if row["tags"] else [],
            raw=json.loads(row["raw"]) if row["raw"] else {},
        )

    @staticmethod
    def _row_to_outcome(row: aiosqlite.Row) -> DaydreamOutcome:
        import json

        return DaydreamOutcome(
            task_id=row["task_id"],
            recorded_at=datetime.fromisoformat(row["recorded_at"]),
            outcome=row["outcome"],
            value_delivered=bool(row["value_delivered"]),
            raw=json.loads(row["raw"]) if row["raw"] else {},
        )

    @staticmethod
    def _row_to_notification(row: aiosqlite.Row) -> DaydreamNotification:
        import json

        return DaydreamNotification(
            notification_id=row["notification_id"],
            spark_id=row["spark_id"],
            chat_id=row["chat_id"],
            sent_at=datetime.fromisoformat(row["sent_at"]),
            label=row["label"],
            intensity=row["intensity"],
            kind=row["kind"],
            raw=json.loads(row["raw"]) if row["raw"] else {},
        )


class ConflictStore:
    """Persistent registry of detected tensions."""

    def __init__(self, path: Path | str) -> None:
        self.path = Path(path)
        self._db: aiosqlite.Connection | None = None

    async def connect(self) -> "ConflictStore":
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._db = await aiosqlite.connect(str(self.path))
        self._db.row_factory = aiosqlite.Row
        await self._db.executescript(_SCHEMA)
        await self._migrate()
        await self._db.commit()
        return self

    async def _migrate(self) -> None:
        assert self._db is not None
        migrations = [
            "ALTER TABLE conflicts ADD COLUMN somatic_context TEXT;",
            "ALTER TABLE conflicts ADD COLUMN resolved_at TEXT;",
            "ALTER TABLE conflicts ADD COLUMN resolution_notes TEXT NOT NULL DEFAULT '';",
        ]
        for sql in migrations:
            try:
                await self._db.execute(sql)
            except sqlite3.OperationalError:
                pass

    async def close(self) -> None:
        if self._db:
            await self._db.close()
            self._db = None

    async def record_conflict(
        self,
        record: ConflictRecord,
    ) -> ConflictRecord:
        assert self._db is not None
        import json
        now = datetime.now(timezone.utc).isoformat()
        somatic_context_json = (
            record.somatic_context.model_dump_json() if record.somatic_context else None
        )
        await self._db.execute(
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
        await self._db.commit()
        # Fetch updated record
        cursor = await self._db.execute(
            "SELECT * FROM conflicts WHERE kind = ? AND description = ?",
            (record.kind, record.description),
        )
        row = await cursor.fetchone()
        assert row is not None
        return self._row_to_conflict(row)

    async def list_active_conflicts(self, limit: int = 20) -> List[ConflictRecord]:
        assert self._db is not None
        cursor = await self._db.execute(
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
        assert self._db is not None
        sql = "SELECT * FROM conflicts"
        params: List[object] = []
        if resolved is not None:
            sql += " WHERE resolved = ?"
            params.append(int(resolved))
        sql += " ORDER BY updated_at DESC, occurrence_count DESC LIMIT ?"
        params.append(limit)
        cursor = await self._db.execute(sql, tuple(params))
        rows = await cursor.fetchall()
        return [self._row_to_conflict(r) for r in rows]

    async def resolve_conflict(
        self,
        conflict_id: str,
        auto: bool = False,
        resolution_notes: str = "",
    ) -> None:
        assert self._db is not None
        now = datetime.now(timezone.utc).isoformat()
        await self._db.execute(
            """
            UPDATE conflicts
            SET resolved = 1, auto_resolved = ?, resolved_at = ?, resolution_notes = ?
            WHERE conflict_id = ?
            """,
            (int(auto), now, resolution_notes, conflict_id),
        )
        await self._db.commit()

    async def auto_resolve_chronic(
        self,
        threshold: int = 25,
        min_days: int = 10,
    ) -> int:
        """Auto-resolve conflicts that have occurred many times over many days."""
        assert self._db is not None
        cutoff = (datetime.now(timezone.utc) - timedelta(days=min_days)).isoformat()
        cursor = await self._db.execute(
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
        import json

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
