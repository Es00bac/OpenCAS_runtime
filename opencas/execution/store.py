"""Async SQLite task store for background repair execution."""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, List, Optional

import aiosqlite
import sqlite3

from .models import ExecutionStage, RepairResult, RepairTask, TaskTransitionRecord

_SCHEMA = """
CREATE TABLE IF NOT EXISTS tasks (
    task_id TEXT PRIMARY KEY,
    created_at TEXT NOT NULL,
    updated_at TEXT NOT NULL,
    objective TEXT NOT NULL,
    stage TEXT NOT NULL,
    status TEXT NOT NULL,
    artifacts TEXT NOT NULL DEFAULT '[]',
    attempt INTEGER NOT NULL DEFAULT 0,
    max_attempts INTEGER NOT NULL DEFAULT 3,
    verification_command TEXT,
    meta TEXT NOT NULL DEFAULT '{}',
    phases TEXT NOT NULL DEFAULT '[]',
    scratch_dir TEXT,
    checkpoint_commit TEXT,
    convergence_hashes TEXT NOT NULL DEFAULT '[]',
    retry_backoff_seconds REAL NOT NULL DEFAULT 1.0,
    depends_on TEXT NOT NULL DEFAULT '[]',
    project_id TEXT,
    commitment_id TEXT,
    success INTEGER,
    result_stage TEXT,
    result_output TEXT,
    result_timestamp TEXT
);

CREATE INDEX IF NOT EXISTS idx_tasks_stage ON tasks(stage);
CREATE INDEX IF NOT EXISTS idx_tasks_status ON tasks(status);
CREATE INDEX IF NOT EXISTS idx_tasks_updated_at ON tasks(updated_at);
CREATE INDEX IF NOT EXISTS idx_tasks_commitment ON tasks(commitment_id);

CREATE TABLE IF NOT EXISTS task_transitions (
    transition_id TEXT PRIMARY KEY,
    task_id TEXT NOT NULL,
    from_stage TEXT NOT NULL,
    to_stage TEXT NOT NULL,
    reason TEXT,
    timestamp TEXT NOT NULL
);

CREATE INDEX IF NOT EXISTS idx_task_transitions_task_id ON task_transitions(task_id);

CREATE TABLE IF NOT EXISTS task_lifecycle_transitions (
    transition_id TEXT PRIMARY KEY,
    task_id TEXT NOT NULL,
    from_stage TEXT NOT NULL,
    to_stage TEXT NOT NULL,
    reason TEXT,
    context TEXT NOT NULL DEFAULT '{}',
    timestamp TEXT NOT NULL
);

CREATE INDEX IF NOT EXISTS idx_lifecycle_task_id ON task_lifecycle_transitions(task_id);
"""

_MIGRATIONS: List[str] = [
    "ALTER TABLE tasks ADD COLUMN commitment_id TEXT",
]


class TaskStore:
    """Async SQLite store for repair tasks and results."""

    def __init__(self, path: Path | str) -> None:
        self.path = Path(path)
        self._db: aiosqlite.Connection | None = None

    async def connect(self) -> "TaskStore":
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._db = await aiosqlite.connect(str(self.path))
        self._db.row_factory = aiosqlite.Row
        await self._db.executescript(_SCHEMA)
        await self._db.commit()
        await self._migrate()
        return self

    async def _migrate(self) -> None:
        for sql in _MIGRATIONS:
            try:
                assert self._db is not None
                await self._db.execute(sql)
                await self._db.commit()
            except sqlite3.OperationalError:
                pass

    async def close(self) -> None:
        if self._db:
            await self._db.close()
            self._db = None

    async def save(self, task: RepairTask) -> None:
        """Insert or replace a task in the store."""
        assert self._db is not None
        await self._db.execute(
            """
            INSERT INTO tasks (
                task_id, created_at, updated_at, objective, stage, status,
                artifacts, attempt, max_attempts, verification_command, meta,
                phases, scratch_dir, checkpoint_commit, convergence_hashes, retry_backoff_seconds,
                depends_on, project_id, commitment_id
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            ON CONFLICT(task_id) DO UPDATE SET
                updated_at = excluded.updated_at,
                stage = excluded.stage,
                status = excluded.status,
                artifacts = excluded.artifacts,
                attempt = excluded.attempt,
                max_attempts = excluded.max_attempts,
                verification_command = excluded.verification_command,
                meta = excluded.meta,
                phases = excluded.phases,
                scratch_dir = excluded.scratch_dir,
                checkpoint_commit = excluded.checkpoint_commit,
                convergence_hashes = excluded.convergence_hashes,
                retry_backoff_seconds = excluded.retry_backoff_seconds,
                depends_on = excluded.depends_on,
                project_id = excluded.project_id,
                commitment_id = excluded.commitment_id,
                success = NULL,
                result_stage = NULL,
                result_output = NULL,
                result_timestamp = NULL
            """,
            (
                str(task.task_id),
                task.created_at.isoformat(),
                task.updated_at.isoformat(),
                task.objective,
                task.stage.value,
                task.status,
                json.dumps(task.artifacts),
                task.attempt,
                task.max_attempts,
                task.verification_command,
                json.dumps(task.meta),
                json.dumps([p.model_dump(mode="json") for p in task.phases]),
                task.scratch_dir,
                task.checkpoint_commit,
                json.dumps(task.convergence_hashes),
                task.retry_backoff_seconds,
                json.dumps(task.depends_on),
                task.project_id,
                task.commitment_id,
            ),
        )
        await self._db.commit()

    async def save_result(self, result: RepairResult) -> None:
        """Update the task row with its terminal result."""
        assert self._db is not None
        await self._db.execute(
            """
            UPDATE tasks SET
                success = ?,
                result_stage = ?,
                result_output = ?,
                result_timestamp = ?,
                stage = ?,
                status = ?
            WHERE task_id = ?
            """,
            (
                int(result.success),
                result.stage.value,
                result.output,
                result.timestamp.isoformat(),
                result.stage.value,
                "completed" if result.success else "failed",
                str(result.task_id),
            ),
        )
        await self._db.commit()

    async def get_result(self, task_id: str) -> Optional[RepairResult]:
        """Fetch the terminal result for a task if one exists."""
        assert self._db is not None
        cursor = await self._db.execute(
            """
            SELECT success, result_stage, result_output, result_timestamp
            FROM tasks WHERE task_id = ?
            """,
            (task_id,),
        )
        row = await cursor.fetchone()
        if row is None or row["success"] is None:
            return None
        return RepairResult(
            task_id=task_id,
            success=bool(row["success"]),
            stage=ExecutionStage(row["result_stage"]),
            output=row["result_output"] or "",
            timestamp=datetime.fromisoformat(row["result_timestamp"]),
        )

    async def get(self, task_id: str) -> Optional[RepairTask]:
        assert self._db is not None
        cursor = await self._db.execute(
            "SELECT * FROM tasks WHERE task_id = ?", (task_id,)
        )
        row = await cursor.fetchone()
        if row is None:
            return None
        return self._row_to_task(row)

    async def list_pending(
        self,
        limit: int = 100,
        offset: int = 0,
    ) -> List[RepairTask]:
        """Return tasks that are not in a terminal stage."""
        assert self._db is not None
        cursor = await self._db.execute(
            """
            SELECT * FROM tasks
            WHERE stage NOT IN ('done', 'failed')
            ORDER BY updated_at DESC
            LIMIT ? OFFSET ?
            """,
            (limit, offset),
        )
        rows = await cursor.fetchall()
        return [self._row_to_task(r) for r in rows]

    async def list_all(
        self,
        limit: int = 100,
        offset: int = 0,
    ) -> List[RepairTask]:
        assert self._db is not None
        cursor = await self._db.execute(
            "SELECT * FROM tasks ORDER BY updated_at DESC LIMIT ? OFFSET ?",
            (limit, offset),
        )
        rows = await cursor.fetchall()
        return [self._row_to_task(r) for r in rows]

    async def delete(self, task_id: str) -> bool:
        assert self._db is not None
        cursor = await self._db.execute(
            "DELETE FROM tasks WHERE task_id = ?", (task_id,)
        )
        await self._db.commit()
        return cursor.rowcount > 0

    async def record_transition(
        self,
        task_id: str,
        from_stage: ExecutionStage,
        to_stage: ExecutionStage,
        reason: Optional[str] = None,
    ) -> None:
        """Persist a stage transition for a task."""
        assert self._db is not None
        record = TaskTransitionRecord(
            task_id=task_id,
            from_stage=from_stage,
            to_stage=to_stage,
            reason=reason,
        )
        await self._db.execute(
            """
            INSERT INTO task_transitions (
                transition_id, task_id, from_stage, to_stage, reason, timestamp
            ) VALUES (?, ?, ?, ?, ?, ?)
            """,
            (
                str(record.transition_id),
                str(record.task_id),
                record.from_stage.value,
                record.to_stage.value,
                record.reason,
                record.timestamp.isoformat(),
            ),
        )
        await self._db.commit()

    async def list_transitions(
        self,
        task_id: str,
        limit: int = 100,
        offset: int = 0,
    ) -> List[TaskTransitionRecord]:
        """Return stage transitions for a task, newest first."""
        assert self._db is not None
        cursor = await self._db.execute(
            """
            SELECT * FROM task_transitions
            WHERE task_id = ?
            ORDER BY timestamp DESC
            LIMIT ? OFFSET ?
            """,
            (str(task_id), limit, offset),
        )
        rows = await cursor.fetchall()
        return [self._row_to_transition(r) for r in rows]

    async def record_lifecycle_transition(
        self,
        transition_id: str,
        task_id: str,
        from_stage: str,
        to_stage: str,
        reason: Optional[str] = None,
        context: Optional[Dict[str, Any]] = None,
    ) -> None:
        """Persist a unified lifecycle stage transition."""
        assert self._db is not None
        now = datetime.now(timezone.utc).isoformat()
        await self._db.execute(
            """
            INSERT INTO task_lifecycle_transitions (
                transition_id, task_id, from_stage, to_stage, reason, context, timestamp
            ) VALUES (?, ?, ?, ?, ?, ?, ?)
            """,
            (
                transition_id,
                task_id,
                from_stage,
                to_stage,
                reason,
                json.dumps(context or {}),
                now,
            ),
        )
        await self._db.commit()

    async def list_lifecycle_transitions(
        self,
        task_id: str,
        limit: int = 100,
        offset: int = 0,
    ) -> List[Dict[str, Any]]:
        """Return unified lifecycle transitions for a task, newest first."""
        assert self._db is not None
        cursor = await self._db.execute(
            """
            SELECT * FROM task_lifecycle_transitions
            WHERE task_id = ?
            ORDER BY timestamp DESC
            LIMIT ? OFFSET ?
            """,
            (str(task_id), limit, offset),
        )
        rows = await cursor.fetchall()
        return [
            {
                "transition_id": r["transition_id"],
                "task_id": r["task_id"],
                "from_stage": r["from_stage"],
                "to_stage": r["to_stage"],
                "reason": r["reason"],
                "context": json.loads(r["context"]) if r["context"] else {},
                "timestamp": datetime.fromisoformat(r["timestamp"]),
            }
            for r in rows
        ]

    @staticmethod
    def _row_to_transition(row: aiosqlite.Row) -> TaskTransitionRecord:
        return TaskTransitionRecord(
            transition_id=row["transition_id"],
            task_id=row["task_id"],
            from_stage=ExecutionStage(row["from_stage"]),
            to_stage=ExecutionStage(row["to_stage"]),
            reason=row["reason"],
            timestamp=datetime.fromisoformat(row["timestamp"]),
        )

    @staticmethod
    def _row_to_task(row: aiosqlite.Row) -> RepairTask:
        artifacts = json.loads(row["artifacts"]) if row["artifacts"] else []
        meta = json.loads(row["meta"]) if row["meta"] else {}
        phases_raw = json.loads(row["phases"]) if row["phases"] else []
        from opencas.execution.models import PhaseRecord
        phases = [PhaseRecord(**p) for p in phases_raw] if phases_raw else []
        convergence_hashes = json.loads(row["convergence_hashes"]) if row["convergence_hashes"] else []
        depends_on = json.loads(row["depends_on"]) if row["depends_on"] else []
        return RepairTask(
            task_id=row["task_id"],
            created_at=datetime.fromisoformat(row["created_at"]),
            updated_at=datetime.fromisoformat(row["updated_at"]),
            objective=row["objective"],
            stage=ExecutionStage(row["stage"]),
            status=row["status"],
            artifacts=artifacts,
            attempt=row["attempt"],
            max_attempts=row["max_attempts"],
            verification_command=row["verification_command"],
            meta=meta,
            phases=phases,
            scratch_dir=row["scratch_dir"],
            checkpoint_commit=row["checkpoint_commit"],
            convergence_hashes=convergence_hashes,
            retry_backoff_seconds=row["retry_backoff_seconds"],
            depends_on=depends_on,
            project_id=row["project_id"],
            commitment_id=row["commitment_id"],
        )
