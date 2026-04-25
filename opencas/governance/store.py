"""SQLite-backed store for approval ledger entries."""

from __future__ import annotations

from datetime import datetime
from pathlib import Path
from typing import List, Optional

import aiosqlite

from .models import ApprovalLedgerEntry

_SCHEMA = """
CREATE TABLE IF NOT EXISTS approval_ledger (
    entry_id TEXT PRIMARY KEY,
    decision_id TEXT NOT NULL,
    action_id TEXT NOT NULL,
    level TEXT NOT NULL,
    score REAL NOT NULL,
    reasoning TEXT NOT NULL,
    tool_name TEXT,
    tier TEXT NOT NULL,
    somatic_state TEXT,
    created_at TEXT NOT NULL
);

CREATE INDEX IF NOT EXISTS idx_ledger_action_id ON approval_ledger(action_id);
CREATE INDEX IF NOT EXISTS idx_ledger_decision_id ON approval_ledger(decision_id);
CREATE INDEX IF NOT EXISTS idx_ledger_created_at ON approval_ledger(created_at);
"""


class ApprovalLedgerStore:
    """Async SQLite store for approval ledger entries."""

    def __init__(self, path: Path | str) -> None:
        self.path = Path(path)
        self._db: aiosqlite.Connection | None = None

    async def connect(self) -> "ApprovalLedgerStore":
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

    async def save(self, entry: ApprovalLedgerEntry) -> None:
        """Persist an approval ledger entry."""
        assert self._db is not None
        await self._db.execute(
            """
            INSERT INTO approval_ledger (
                entry_id, decision_id, action_id, level, score, reasoning,
                tool_name, tier, somatic_state, created_at
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                str(entry.entry_id),
                str(entry.decision_id),
                str(entry.action_id),
                entry.level,
                entry.score,
                entry.reasoning,
                entry.tool_name,
                entry.tier.value,
                entry.somatic_state,
                entry.created_at.isoformat(),
            ),
        )
        await self._db.commit()

    async def get(self, entry_id: str) -> Optional[ApprovalLedgerEntry]:
        """Fetch a single ledger entry by ID."""
        assert self._db is not None
        cursor = await self._db.execute(
            "SELECT * FROM approval_ledger WHERE entry_id = ?", (entry_id,)
        )
        row = await cursor.fetchone()
        if row is None:
            return None
        return self._row_to_entry(row)

    async def list_recent(
        self,
        limit: int = 100,
        offset: int = 0,
    ) -> List[ApprovalLedgerEntry]:
        """Return recent ledger entries, newest first."""
        assert self._db is not None
        cursor = await self._db.execute(
            "SELECT * FROM approval_ledger ORDER BY created_at DESC LIMIT ? OFFSET ?",
            (limit, offset),
        )
        rows = await cursor.fetchall()
        return [self._row_to_entry(r) for r in rows]

    async def list_by_action(
        self,
        action_id: str,
        limit: int = 100,
        offset: int = 0,
    ) -> List[ApprovalLedgerEntry]:
        """Return ledger entries for a specific action."""
        assert self._db is not None
        cursor = await self._db.execute(
            """
            SELECT * FROM approval_ledger
            WHERE action_id = ?
            ORDER BY created_at DESC
            LIMIT ? OFFSET ?
            """,
            (str(action_id), limit, offset),
        )
        rows = await cursor.fetchall()
        return [self._row_to_entry(r) for r in rows]

    async def query_stats(self, window_days: int = 7) -> dict:
        """Return aggregate stats over the last *window_days*."""
        assert self._db is not None
        cursor = await self._db.execute(
            """
            SELECT
                COUNT(*) as total,
                AVG(score) as avg_score,
                tier,
                level
            FROM approval_ledger
            WHERE created_at >= datetime('now', ?)
            GROUP BY tier, level
            """,
            (f"-{window_days} days",),
        )
        rows = await cursor.fetchall()
        return {
            "window_days": window_days,
            "breakdown": [
                {
                    "tier": row["tier"],
                    "level": row["level"],
                    "count": row["total"],
                    "avg_score": row["avg_score"],
                }
                for row in rows
            ],
        }

    @staticmethod
    def _row_to_entry(row: aiosqlite.Row) -> ApprovalLedgerEntry:
        from opencas.autonomy.models import ActionRiskTier
        return ApprovalLedgerEntry(
            entry_id=row["entry_id"],
            decision_id=row["decision_id"],
            action_id=row["action_id"],
            level=row["level"],
            score=row["score"],
            reasoning=row["reasoning"],
            tool_name=row["tool_name"],
            tier=ActionRiskTier(row["tier"]),
            somatic_state=row["somatic_state"],
            created_at=datetime.fromisoformat(row["created_at"]),
        )
