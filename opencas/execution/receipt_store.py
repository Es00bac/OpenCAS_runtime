"""SQLite-backed store for execution receipts."""

from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path
from typing import List, Optional

import aiosqlite

from .models import ExecutionReceipt, PhaseRecord, RepairResult, RepairTask

_SCHEMA = """
CREATE TABLE IF NOT EXISTS receipts (
    receipt_id TEXT PRIMARY KEY,
    task_id TEXT NOT NULL,
    objective TEXT NOT NULL,
    plan TEXT,
    phases TEXT NOT NULL DEFAULT '[]',
    verification_result INTEGER,
    checkpoint_commit TEXT,
    created_at TEXT NOT NULL,
    completed_at TEXT,
    success INTEGER NOT NULL,
    output TEXT NOT NULL DEFAULT ''
);

CREATE INDEX IF NOT EXISTS idx_receipts_task_id ON receipts(task_id);
CREATE INDEX IF NOT EXISTS idx_receipts_created_at ON receipts(created_at);
"""


class ExecutionReceiptStore:
    """Async SQLite store for execution receipts."""

    def __init__(self, path: Path | str) -> None:
        self.path = Path(path)
        self._db: aiosqlite.Connection | None = None

    async def connect(self) -> "ExecutionReceiptStore":
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

    async def save(self, task: RepairTask, result: RepairResult) -> ExecutionReceipt:
        """Persist a receipt from a completed task and result."""
        assert self._db is not None
        plan = next(
            (a.split(":", 1)[1] for a in task.artifacts if a.startswith("plan:")),
            None,
        )
        verify_phase = next(
            (p for p in task.phases if p.phase.value == "verify"),
            None,
        )
        receipt = ExecutionReceipt(
            task_id=task.task_id,
            objective=task.objective,
            plan=plan,
            phases=task.phases,
            verification_result=verify_phase.success if verify_phase else None,
            checkpoint_commit=task.checkpoint_commit,
            completed_at=result.timestamp,
            success=result.success,
            output=result.output,
        )
        await self._db.execute(
            """
            INSERT INTO receipts (
                receipt_id, task_id, objective, plan, phases,
                verification_result, checkpoint_commit, created_at,
                completed_at, success, output
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                str(receipt.receipt_id),
                str(receipt.task_id),
                receipt.objective,
                receipt.plan,
                json.dumps([p.model_dump(mode="json") for p in receipt.phases]),
                int(receipt.verification_result) if receipt.verification_result is not None else None,
                receipt.checkpoint_commit,
                receipt.created_at.isoformat(),
                receipt.completed_at.isoformat() if receipt.completed_at else None,
                int(receipt.success),
                receipt.output,
            ),
        )
        await self._db.commit()
        return receipt

    async def save_direct(self, receipt: ExecutionReceipt) -> ExecutionReceipt:
        """Persist a pre-built execution receipt without requiring a RepairTask."""
        assert self._db is not None
        await self._db.execute(
            """
            INSERT INTO receipts (
                receipt_id, task_id, objective, plan, phases,
                verification_result, checkpoint_commit, created_at,
                completed_at, success, output
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            ON CONFLICT(receipt_id) DO UPDATE SET
                task_id = excluded.task_id,
                objective = excluded.objective,
                plan = excluded.plan,
                phases = excluded.phases,
                verification_result = excluded.verification_result,
                checkpoint_commit = excluded.checkpoint_commit,
                created_at = excluded.created_at,
                completed_at = excluded.completed_at,
                success = excluded.success,
                output = excluded.output
            """,
            (
                str(receipt.receipt_id),
                str(receipt.task_id),
                receipt.objective,
                receipt.plan,
                json.dumps([p.model_dump(mode="json") for p in receipt.phases]),
                int(receipt.verification_result) if receipt.verification_result is not None else None,
                receipt.checkpoint_commit,
                receipt.created_at.isoformat(),
                receipt.completed_at.isoformat() if receipt.completed_at else None,
                int(receipt.success),
                receipt.output,
            ),
        )
        await self._db.commit()
        return receipt

    async def get(self, receipt_id: str) -> Optional[ExecutionReceipt]:
        """Fetch a receipt by ID."""
        assert self._db is not None
        cursor = await self._db.execute(
            "SELECT * FROM receipts WHERE receipt_id = ?", (receipt_id,)
        )
        row = await cursor.fetchone()
        if row is None:
            return None
        return self._row_to_receipt(row)

    async def list_by_task(
        self,
        task_id: str,
        limit: int = 100,
        offset: int = 0,
    ) -> List[ExecutionReceipt]:
        """Return receipts for a given task, newest first."""
        assert self._db is not None
        cursor = await self._db.execute(
            """
            SELECT * FROM receipts
            WHERE task_id = ?
            ORDER BY created_at DESC
            LIMIT ? OFFSET ?
            """,
            (str(task_id), limit, offset),
        )
        rows = await cursor.fetchall()
        return [self._row_to_receipt(r) for r in rows]

    async def list_recent(
        self,
        limit: int = 20,
        offset: int = 0,
    ) -> List[ExecutionReceipt]:
        """Return the most recent execution receipts."""
        assert self._db is not None
        cursor = await self._db.execute(
            """
            SELECT * FROM receipts
            ORDER BY created_at DESC
            LIMIT ? OFFSET ?
            """,
            (limit, offset),
        )
        rows = await cursor.fetchall()
        return [self._row_to_receipt(r) for r in rows]

    @staticmethod
    def _row_to_receipt(row: aiosqlite.Row) -> ExecutionReceipt:
        phases_raw = json.loads(row["phases"]) if row["phases"] else []
        phases = [PhaseRecord(**p) for p in phases_raw] if phases_raw else []
        verify_value = row["verification_result"]
        return ExecutionReceipt(
            receipt_id=row["receipt_id"],
            task_id=row["task_id"],
            objective=row["objective"],
            plan=row["plan"],
            phases=phases,
            verification_result=bool(verify_value) if verify_value is not None else None,
            checkpoint_commit=row["checkpoint_commit"],
            created_at=datetime.fromisoformat(row["created_at"]),
            completed_at=datetime.fromisoformat(row["completed_at"]) if row["completed_at"] else None,
            success=bool(row["success"]),
            output=row["output"] or "",
        )
