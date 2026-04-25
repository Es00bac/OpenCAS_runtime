"""SQLite-backed store for durable ToM beliefs and intentions."""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

from opencas.tom.models import Belief, BeliefSubject, Intention, IntentionStatus


_SCHEMA = """
CREATE TABLE IF NOT EXISTS beliefs (
    belief_id TEXT PRIMARY KEY,
    timestamp TEXT NOT NULL,
    subject TEXT NOT NULL,
    predicate TEXT NOT NULL,
    confidence REAL NOT NULL,
    evidence_ids TEXT NOT NULL DEFAULT '[]',
    belief_revision_score REAL NOT NULL DEFAULT 0.0,
    reinforcement_count INTEGER NOT NULL DEFAULT 0,
    last_reinforced TEXT,
    meta TEXT NOT NULL DEFAULT '{}'
);

CREATE INDEX IF NOT EXISTS idx_beliefs_subject ON beliefs(subject);
CREATE INDEX IF NOT EXISTS idx_beliefs_subject_predicate ON beliefs(subject, predicate);
CREATE INDEX IF NOT EXISTS idx_beliefs_predicate ON beliefs(predicate);
CREATE INDEX IF NOT EXISTS idx_beliefs_timestamp ON beliefs(timestamp);

CREATE TABLE IF NOT EXISTS intentions (
    intention_id TEXT PRIMARY KEY,
    timestamp TEXT NOT NULL,
    actor TEXT NOT NULL,
    content TEXT NOT NULL,
    status TEXT NOT NULL,
    resolved_at TEXT,
    meta TEXT NOT NULL DEFAULT '{}'
);

CREATE INDEX IF NOT EXISTS idx_intentions_actor ON intentions(actor);
CREATE INDEX IF NOT EXISTS idx_intentions_status ON intentions(status);
CREATE INDEX IF NOT EXISTS idx_intentions_timestamp ON intentions(timestamp);
"""


class TomStore:
    """Async SQLite store for Theory of Mind beliefs and intentions."""

    def __init__(self, db_path: Path | str) -> None:
        self.db_path = Path(db_path)
        self._db = None

    async def connect(self) -> "TomStore":
        import aiosqlite

        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._db = await aiosqlite.connect(str(self.db_path))
        await self._db.executescript(_SCHEMA)
        await self._migrate()
        await self._db.commit()
        return self

    async def _migrate(self) -> None:
        """Lightweight migrations for existing stores."""
        assert self._db is not None
        migrations = [
            "ALTER TABLE beliefs ADD COLUMN belief_revision_score REAL NOT NULL DEFAULT 0.0",
            "ALTER TABLE beliefs ADD COLUMN reinforcement_count INTEGER NOT NULL DEFAULT 0",
            "ALTER TABLE beliefs ADD COLUMN last_reinforced TEXT",
        ]
        for sql in migrations:
            try:
                await self._db.execute(sql)
            except Exception:
                pass  # column likely already exists
        # Create composite index if not present (safe with IF NOT EXISTS)
        try:
            await self._db.execute(
                "CREATE INDEX IF NOT EXISTS idx_beliefs_subject_predicate ON beliefs(subject, predicate)"
            )
        except Exception:
            pass
        await self._db.commit()

    async def close(self) -> None:
        if self._db:
            await self._db.close()
            self._db = None

    async def save_belief(self, belief: Belief) -> Belief:
        assert self._db is not None
        await self._db.execute(
            """
            INSERT INTO beliefs (
                belief_id, timestamp, subject, predicate, confidence,
                evidence_ids, belief_revision_score, reinforcement_count,
                last_reinforced, meta
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            ON CONFLICT(belief_id) DO UPDATE SET
                timestamp = excluded.timestamp,
                subject = excluded.subject,
                predicate = excluded.predicate,
                confidence = excluded.confidence,
                evidence_ids = excluded.evidence_ids,
                belief_revision_score = excluded.belief_revision_score,
                reinforcement_count = excluded.reinforcement_count,
                last_reinforced = excluded.last_reinforced,
                meta = excluded.meta
            """,
            (
                str(belief.belief_id),
                belief.timestamp.isoformat(),
                belief.subject.value,
                belief.predicate,
                belief.confidence,
                json.dumps(belief.evidence_ids),
                belief.belief_revision_score,
                belief.reinforcement_count,
                belief.last_reinforced.isoformat() if belief.last_reinforced else None,
                json.dumps(belief.meta),
            ),
        )
        await self._db.commit()
        return belief

    async def list_beliefs(
        self,
        subject: Optional[BeliefSubject] = None,
        predicate: Optional[str] = None,
        limit: int = 100,
    ) -> List[Belief]:
        assert self._db is not None
        conditions: List[str] = []
        params: List[Any] = []
        if subject is not None:
            conditions.append("subject = ?")
            params.append(subject.value)
        if predicate is not None:
            conditions.append("predicate = ?")
            params.append(predicate.strip().lower())
        where = "WHERE " + " AND ".join(conditions) if conditions else ""
        cursor = await self._db.execute(
            f"""
            SELECT
                belief_id, timestamp, subject, predicate, confidence,
                evidence_ids, belief_revision_score, reinforcement_count,
                last_reinforced, meta
            FROM beliefs
            {where}
            ORDER BY timestamp DESC
            LIMIT ?
            """,
            (*params, limit),
        )
        rows = await cursor.fetchall()
        return [self._row_to_belief(r) for r in rows]

    async def save_intention(self, intention: Intention) -> Intention:
        assert self._db is not None
        await self._db.execute(
            """
            INSERT INTO intentions (
                intention_id, timestamp, actor, content, status,
                resolved_at, meta
            ) VALUES (?, ?, ?, ?, ?, ?, ?)
            ON CONFLICT(intention_id) DO UPDATE SET
                timestamp = excluded.timestamp,
                actor = excluded.actor,
                content = excluded.content,
                status = excluded.status,
                resolved_at = excluded.resolved_at,
                meta = excluded.meta
            """,
            (
                str(intention.intention_id),
                intention.timestamp.isoformat(),
                intention.actor.value,
                intention.content,
                intention.status.value,
                intention.resolved_at.isoformat() if intention.resolved_at else None,
                json.dumps(intention.meta),
            ),
        )
        await self._db.commit()
        return intention

    async def list_intentions(
        self,
        actor: Optional[BeliefSubject] = None,
        status: Optional[IntentionStatus] = None,
        limit: int = 100,
    ) -> List[Intention]:
        assert self._db is not None
        conditions: List[str] = []
        params: List[Any] = []
        if actor is not None:
            conditions.append("actor = ?")
            params.append(actor.value)
        if status is not None:
            conditions.append("status = ?")
            params.append(status.value)
        where = "WHERE " + " AND ".join(conditions) if conditions else ""
        cursor = await self._db.execute(
            f"""
            SELECT * FROM intentions
            {where}
            ORDER BY timestamp DESC
            LIMIT ?
            """,
            (*params, limit),
        )
        rows = await cursor.fetchall()
        return [self._row_to_intention(r) for r in rows]

    async def update_belief_confidence(
        self,
        belief_id: str,
        confidence: float,
        belief_revision_score: float,
    ) -> None:
        """Update a belief's confidence and revision score after a consistency sweep."""
        assert self._db is not None
        await self._db.execute(
            """
            UPDATE beliefs
            SET confidence = ?, belief_revision_score = ?
            WHERE belief_id = ?
            """,
            (confidence, belief_revision_score, belief_id),
        )
        await self._db.commit()

    async def get_belief_by_predicate(
        self,
        subject: BeliefSubject,
        predicate: str,
    ) -> Optional[Belief]:
        """Find the most recent belief matching subject+predicate exactly."""
        assert self._db is not None
        cursor = await self._db.execute(
            """
            SELECT
                belief_id, timestamp, subject, predicate, confidence,
                evidence_ids, belief_revision_score, reinforcement_count,
                last_reinforced, meta
            FROM beliefs
            WHERE subject = ? AND predicate = ?
            ORDER BY timestamp DESC
            LIMIT 1
            """,
            (subject.value, predicate.strip().lower()),
        )
        row = await cursor.fetchone()
        if row is None:
            return None
        return self._row_to_belief(row)

    async def increment_belief_reinforcement(
        self,
        belief_id: str,
        confidence: float,
        reinforcement_count: int,
        last_reinforced: datetime,
    ) -> None:
        """Update reinforcement metadata for an existing belief."""
        assert self._db is not None
        await self._db.execute(
            """
            UPDATE beliefs
            SET confidence = ?, reinforcement_count = ?, last_reinforced = ?
            WHERE belief_id = ?
            """,
            (confidence, reinforcement_count, last_reinforced.isoformat(), belief_id),
        )
        await self._db.commit()

    async def resolve_intention(
        self,
        intention_id: str,
        status: IntentionStatus,
        resolved_at: Optional[datetime] = None,
    ) -> None:
        assert self._db is not None
        resolved = resolved_at or datetime.now(timezone.utc)
        await self._db.execute(
            """
            UPDATE intentions
            SET status = ?, resolved_at = ?
            WHERE intention_id = ?
            """,
            (status.value, resolved.isoformat(), intention_id),
        )
        await self._db.commit()

    @staticmethod
    def _row_to_belief(row) -> Belief:
        return Belief(
            belief_id=row[0],
            timestamp=datetime.fromisoformat(row[1]),
            subject=BeliefSubject(row[2]),
            predicate=row[3],
            confidence=row[4],
            evidence_ids=json.loads(row[5]) if row[5] else [],
            belief_revision_score=row[6] if len(row) > 6 and row[6] is not None else 0.0,
            reinforcement_count=row[7] if len(row) > 7 and row[7] is not None else 0,
            last_reinforced=datetime.fromisoformat(row[8]) if len(row) > 8 and row[8] else None,
            meta=json.loads(row[9]) if len(row) > 9 and row[9] else {},
        )

    @staticmethod
    def _row_to_intention(row) -> Intention:
        return Intention(
            intention_id=row[0],
            timestamp=datetime.fromisoformat(row[1]),
            actor=BeliefSubject(row[2]),
            content=row[3],
            status=IntentionStatus(row[4]),
            resolved_at=datetime.fromisoformat(row[5]) if row[5] else None,
            meta=json.loads(row[6]) if row[6] else {},
        )
