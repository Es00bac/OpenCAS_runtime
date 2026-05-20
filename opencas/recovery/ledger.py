from __future__ import annotations

import json
import sqlite3
from datetime import datetime, timezone
from pathlib import Path

from opencas.recovery.models import RecoveryDecision


class RecoveryLedger:
    def __init__(self, db_path: str | Path) -> None:
        self.db_path = Path(db_path)
        self._conn: sqlite3.Connection | None = None

    async def connect(self) -> None:
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._conn = sqlite3.connect(self.db_path)
        self._conn.row_factory = sqlite3.Row
        self._conn.execute(
            """
            CREATE TABLE IF NOT EXISTS recovery_decisions (
                decision_id TEXT PRIMARY KEY,
                candidate_id TEXT NOT NULL,
                classification TEXT NOT NULL,
                strategy TEXT NOT NULL,
                result TEXT NOT NULL,
                created_task_id TEXT,
                created_loop_id TEXT,
                continuity_packet_id TEXT,
                evidence_refs TEXT NOT NULL,
                next_reconsideration_at TEXT,
                created_at TEXT NOT NULL
            )
            """
        )
        self._conn.execute(
            "CREATE INDEX IF NOT EXISTS idx_recovery_decisions_candidate ON recovery_decisions(candidate_id, created_at)"
        )
        self._conn.commit()

    async def close(self) -> None:
        if self._conn is not None:
            self._conn.close()
            self._conn = None

    async def record_decision(self, decision: RecoveryDecision) -> None:
        conn = self._require_conn()
        conn.execute(
            """
            INSERT OR REPLACE INTO recovery_decisions (
                decision_id, candidate_id, classification, strategy, result,
                created_task_id, created_loop_id, continuity_packet_id,
                evidence_refs, next_reconsideration_at, created_at
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                decision.decision_id,
                decision.candidate_id,
                decision.classification,
                decision.strategy,
                decision.result,
                decision.created_task_id,
                decision.created_loop_id,
                decision.continuity_packet_id,
                json.dumps(decision.evidence_refs),
                _format_dt(decision.next_reconsideration_at),
                _format_dt(datetime.now(timezone.utc)),
            ),
        )
        conn.commit()

    async def latest_decision(self, candidate_id: str) -> RecoveryDecision | None:
        conn = self._require_conn()
        row = conn.execute(
            """
            SELECT * FROM recovery_decisions
            WHERE candidate_id = ?
            ORDER BY created_at DESC
            LIMIT 1
            """,
            (candidate_id,),
        ).fetchone()
        if row is None:
            return None
        return _decision_from_row(row)

    async def should_skip_candidate(self, candidate_id: str) -> bool:
        latest = await self.latest_decision(candidate_id)
        if latest is None or latest.next_reconsideration_at is None:
            return False
        return latest.next_reconsideration_at > datetime.now(timezone.utc)

    def _require_conn(self) -> sqlite3.Connection:
        if self._conn is None:
            raise RuntimeError("RecoveryLedger.connect() must be called before use")
        return self._conn


def _format_dt(value: datetime | None) -> str | None:
    if value is None:
        return None
    return value.astimezone(timezone.utc).isoformat()


def _parse_dt(value: str | None) -> datetime | None:
    if not value:
        return None
    return datetime.fromisoformat(value)


def _decision_from_row(row: sqlite3.Row) -> RecoveryDecision:
    return RecoveryDecision(
        decision_id=row["decision_id"],
        candidate_id=row["candidate_id"],
        classification=row["classification"],
        strategy=row["strategy"],
        result=row["result"],
        created_task_id=row["created_task_id"],
        created_loop_id=row["created_loop_id"],
        continuity_packet_id=row["continuity_packet_id"],
        evidence_refs=json.loads(row["evidence_refs"]),
        next_reconsideration_at=_parse_dt(row["next_reconsideration_at"]),
    )
