"""SQLite-backed persistence for daydream reflections and lifecycle state."""

from __future__ import annotations

import json
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import List, Optional

import aiosqlite

from .models import (
    DaydreamInitiative,
    DaydreamNotification,
    DaydreamOutcome,
    DaydreamReflection,
    DaydreamSpark,
)
from .sqlite_base import SqliteBackedStore


_DAYDREAM_SCHEMA = """
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
"""


class DaydreamStore(SqliteBackedStore):
    """Persist daydream reflections and lifecycle history."""

    SCHEMA = _DAYDREAM_SCHEMA

    async def save_reflection(self, reflection: DaydreamReflection) -> None:
        await self.db.execute(
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
        await self.db.commit()

    async def save_reflections_batch(self, reflections: List[DaydreamReflection]) -> None:
        """Batch insert reflections in a single transaction."""
        params = [
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
            for reflection in reflections
        ]
        await self.db.executemany(
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
        await self.db.commit()

    async def list_recent(
        self,
        limit: int = 10,
        keeper_only: Optional[bool] = None,
    ) -> List[DaydreamReflection]:
        sql = """
            SELECT * FROM daydream_reflections
        """
        params: List[object] = []
        if keeper_only is not None:
            sql += " WHERE keeper = ?"
            params.append(int(keeper_only))
        sql += " ORDER BY created_at DESC LIMIT ?"
        params.append(limit)
        cursor = await self.db.execute(sql, tuple(params))
        rows = await cursor.fetchall()
        return [self._row_to_reflection(r) for r in rows]

    async def get_summary(self, window_days: int = 7) -> dict:
        cutoff = (datetime.now(timezone.utc) - timedelta(days=max(1, window_days))).isoformat()
        totals_cursor = await self.db.execute(
            """
            SELECT
                COUNT(*) AS total_reflections,
                SUM(CASE WHEN keeper = 1 THEN 1 ELSE 0 END) AS total_keepers
            FROM daydream_reflections
            """
        )
        window_cursor = await self.db.execute(
            """
            SELECT
                COUNT(*) AS window_reflections,
                SUM(CASE WHEN keeper = 1 THEN 1 ELSE 0 END) AS window_keepers
            FROM daydream_reflections
            WHERE created_at >= ?
            """,
            (cutoff,),
        )
        latest_cursor = await self.db.execute(
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

    async def save_spark(self, spark: DaydreamSpark) -> None:
        await self.db.execute(
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
        await self.db.commit()

    async def save_initiative(self, initiative: DaydreamInitiative) -> None:
        await self.db.execute(
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
        await self.db.commit()

    async def save_outcome(self, outcome: DaydreamOutcome) -> None:
        await self.db.execute(
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
        await self.db.commit()

    async def save_notification(self, notification: DaydreamNotification) -> None:
        await self.db.execute(
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
        await self.db.commit()

    async def list_sparks(self, limit: int = 50) -> List[DaydreamSpark]:
        cursor = await self.db.execute(
            "SELECT * FROM daydream_sparks ORDER BY created_at DESC LIMIT ?",
            (limit,),
        )
        return [self._row_to_spark(row) for row in await cursor.fetchall()]

    async def list_initiatives(self, limit: int = 50) -> List[DaydreamInitiative]:
        cursor = await self.db.execute(
            "SELECT * FROM daydream_initiatives ORDER BY created_at DESC LIMIT ?",
            (limit,),
        )
        return [self._row_to_initiative(row) for row in await cursor.fetchall()]

    async def list_outcomes(self, limit: int = 50) -> List[DaydreamOutcome]:
        cursor = await self.db.execute(
            "SELECT * FROM daydream_outcomes ORDER BY recorded_at DESC LIMIT ?",
            (limit,),
        )
        return [self._row_to_outcome(row) for row in await cursor.fetchall()]

    async def list_notifications(self, limit: int = 50) -> List[DaydreamNotification]:
        cursor = await self.db.execute(
            "SELECT * FROM daydream_notifications ORDER BY sent_at DESC LIMIT ?",
            (limit,),
        )
        return [self._row_to_notification(row) for row in await cursor.fetchall()]

    async def get_lifecycle_for_spark(self, spark_id: str) -> dict:
        spark_cursor = await self.db.execute(
            "SELECT * FROM daydream_sparks WHERE spark_id = ?",
            (spark_id,),
        )
        spark_row = await spark_cursor.fetchone()
        initiatives_cursor = await self.db.execute(
            "SELECT * FROM daydream_initiatives WHERE spark_id = ? ORDER BY created_at DESC",
            (spark_id,),
        )
        notification_cursor = await self.db.execute(
            "SELECT * FROM daydream_notifications WHERE spark_id = ? ORDER BY sent_at DESC",
            (spark_id,),
        )
        initiatives = [self._row_to_initiative(row) for row in await initiatives_cursor.fetchall()]
        task_ids = [item.task_id for item in initiatives if item.task_id]
        outcomes: List[DaydreamOutcome] = []
        if task_ids:
            placeholders = ",".join("?" for _ in task_ids)
            outcome_cursor = await self.db.execute(
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
    def _row_to_reflection(row: aiosqlite.Row) -> DaydreamReflection:
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

    @staticmethod
    def _row_to_spark(row: aiosqlite.Row) -> DaydreamSpark:
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
        return DaydreamOutcome(
            task_id=row["task_id"],
            recorded_at=datetime.fromisoformat(row["recorded_at"]),
            outcome=row["outcome"],
            value_delivered=bool(row["value_delivered"]),
            raw=json.loads(row["raw"]) if row["raw"] else {},
        )

    @staticmethod
    def _row_to_notification(row: aiosqlite.Row) -> DaydreamNotification:
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
