"""Persistent SQLite store for plans and plan actions."""

from __future__ import annotations

import json
import sqlite3
from datetime import datetime, timezone
from pathlib import Path
from typing import List, Optional

import aiosqlite

from .models import PlanAction, PlanEntry

_SCHEMA = """
CREATE TABLE IF NOT EXISTS plans (
    plan_id TEXT PRIMARY KEY,
    status TEXT NOT NULL,
    content TEXT NOT NULL DEFAULT '',
    created_at TEXT NOT NULL,
    updated_at TEXT NOT NULL,
    project_id TEXT,
    task_id TEXT
);
CREATE INDEX IF NOT EXISTS idx_plans_status ON plans(status);
CREATE INDEX IF NOT EXISTS idx_plans_project ON plans(project_id);
CREATE INDEX IF NOT EXISTS idx_plans_task ON plans(task_id);

CREATE TABLE IF NOT EXISTS plan_actions (
    action_id TEXT PRIMARY KEY,
    plan_id TEXT NOT NULL,
    tool_name TEXT NOT NULL,
    args TEXT NOT NULL DEFAULT '{}',
    result_summary TEXT NOT NULL DEFAULT '',
    success INTEGER NOT NULL DEFAULT 0,
    timestamp TEXT NOT NULL,
    FOREIGN KEY (plan_id) REFERENCES plans(plan_id)
);
CREATE INDEX IF NOT EXISTS idx_plan_actions_plan_id ON plan_actions(plan_id);
CREATE INDEX IF NOT EXISTS idx_plan_actions_timestamp ON plan_actions(timestamp);
"""

_MIGRATIONS: List[str] = []


class PlanStore:
    """Async SQLite store for plans and their action ledgers."""

    def __init__(self, path: Path | str) -> None:
        self.path = Path(path)
        self._db: aiosqlite.Connection | None = None

    async def connect(self) -> "PlanStore":
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

    async def create_plan(
        self,
        plan_id: str,
        content: str = "",
        project_id: Optional[str] = None,
        task_id: Optional[str] = None,
    ) -> PlanEntry:
        assert self._db is not None
        now = datetime.now(timezone.utc)
        await self._db.execute(
            """
            INSERT INTO plans (plan_id, status, content, created_at, updated_at, project_id, task_id)
            VALUES (?, ?, ?, ?, ?, ?, ?)
            """,
            (plan_id, "draft", content, now.isoformat(), now.isoformat(), project_id, task_id),
        )
        await self._db.commit()
        return PlanEntry(
            plan_id=plan_id,
            status="draft",
            content=content,
            created_at=now,
            updated_at=now,
            project_id=project_id,
            task_id=task_id,
        )

    async def get_plan(self, plan_id: str) -> Optional[PlanEntry]:
        assert self._db is not None
        cursor = await self._db.execute("SELECT * FROM plans WHERE plan_id = ?", (plan_id,))
        row = await cursor.fetchone()
        if row is None:
            return None
        return self._row_to_plan(row)

    async def update_content(self, plan_id: str, content: str) -> bool:
        assert self._db is not None
        now = datetime.now(timezone.utc).isoformat()
        cursor = await self._db.execute(
            "UPDATE plans SET content = ?, updated_at = ? WHERE plan_id = ?",
            (content, now, plan_id),
        )
        await self._db.commit()
        return cursor.rowcount > 0

    async def set_status(self, plan_id: str, status: str) -> bool:
        assert self._db is not None
        now = datetime.now(timezone.utc).isoformat()
        cursor = await self._db.execute(
            "UPDATE plans SET status = ?, updated_at = ? WHERE plan_id = ?",
            (status, now, plan_id),
        )
        await self._db.commit()
        return cursor.rowcount > 0

    async def list_active(
        self,
        project_id: Optional[str] = None,
        task_id: Optional[str] = None,
    ) -> List[PlanEntry]:
        assert self._db is not None
        conditions = ["status = ?"]
        params: List[Optional[str]] = ["active"]
        if project_id is not None:
            conditions.append("project_id = ?")
            params.append(project_id)
        if task_id is not None:
            conditions.append("task_id = ?")
            params.append(task_id)
        where = " AND ".join(conditions)
        cursor = await self._db.execute(
            f"SELECT * FROM plans WHERE {where} ORDER BY updated_at DESC",
            tuple(params),
        )
        rows = await cursor.fetchall()
        return [self._row_to_plan(r) for r in rows]

    async def count_active(
        self,
        project_id: Optional[str] = None,
        task_id: Optional[str] = None,
    ) -> int:
        """Return the number of active plans with optional filters."""
        assert self._db is not None
        conditions = ["status = ?"]
        params: List[Optional[str]] = ["active"]
        if project_id is not None:
            conditions.append("project_id = ?")
            params.append(project_id)
        if task_id is not None:
            conditions.append("task_id = ?")
            params.append(task_id)
        where = " AND ".join(conditions)
        cursor = await self._db.execute(
            f"SELECT COUNT(*) FROM plans WHERE {where}",
            tuple(params),
        )
        row = await cursor.fetchone()
        return int(row[0]) if row else 0

    async def record_action(
        self,
        plan_id: str,
        tool_name: str,
        args: dict,
        result_summary: str,
        success: bool,
    ) -> None:
        assert self._db is not None
        now = datetime.now(timezone.utc)
        action_id = f"action-{now.timestamp()}-{tool_name}"
        await self._db.execute(
            """
            INSERT INTO plan_actions (action_id, plan_id, tool_name, args, result_summary, success, timestamp)
            VALUES (?, ?, ?, ?, ?, ?, ?)
            """,
            (
                action_id,
                plan_id,
                tool_name,
                json.dumps(args),
                result_summary,
                1 if success else 0,
                now.isoformat(),
            ),
        )
        await self._db.commit()

    async def get_actions(self, plan_id: str, limit: int = 100) -> List[PlanAction]:
        assert self._db is not None
        cursor = await self._db.execute(
            """
            SELECT * FROM plan_actions
            WHERE plan_id = ?
            ORDER BY timestamp DESC
            LIMIT ?
            """,
            (plan_id, limit),
        )
        rows = await cursor.fetchall()
        return [self._row_to_action(r) for r in rows]

    async def delete_plan(self, plan_id: str) -> bool:
        assert self._db is not None
        await self._db.execute("DELETE FROM plan_actions WHERE plan_id = ?", (plan_id,))
        cursor = await self._db.execute("DELETE FROM plans WHERE plan_id = ?", (plan_id,))
        await self._db.commit()
        return cursor.rowcount > 0

    @staticmethod
    def _row_to_plan(row: aiosqlite.Row) -> PlanEntry:
        return PlanEntry(
            plan_id=row["plan_id"],
            status=row["status"],
            content=row["content"] or "",
            created_at=datetime.fromisoformat(row["created_at"]),
            updated_at=datetime.fromisoformat(row["updated_at"]),
            project_id=row["project_id"],
            task_id=row["task_id"],
        )

    @staticmethod
    def _row_to_action(row: aiosqlite.Row) -> PlanAction:
        return PlanAction(
            action_id=row["action_id"],
            plan_id=row["plan_id"],
            tool_name=row["tool_name"],
            args=json.loads(row["args"]) if row["args"] else {},
            result_summary=row["result_summary"] or "",
            success=bool(row["success"]),
            timestamp=datetime.fromisoformat(row["timestamp"]),
        )
