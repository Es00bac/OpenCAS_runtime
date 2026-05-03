"""SQLite store for operational wellbeing records."""

from __future__ import annotations

import sqlite3
from pathlib import Path

import aiosqlite

from .models import (
    MaintenanceOutcome,
    SelfModificationProposal,
    WellbeingEvent,
    WellbeingRecommendation,
    WellbeingState,
)

_SCHEMA = """
CREATE TABLE IF NOT EXISTS wellbeing_states (
    state_id TEXT PRIMARY KEY,
    created_at TEXT NOT NULL,
    payload TEXT NOT NULL
);

CREATE INDEX IF NOT EXISTS idx_wellbeing_states_created_at
ON wellbeing_states(created_at);

CREATE TABLE IF NOT EXISTS wellbeing_events (
    event_id TEXT PRIMARY KEY,
    created_at TEXT NOT NULL,
    event_type TEXT NOT NULL,
    summary TEXT NOT NULL,
    payload TEXT NOT NULL
);

CREATE INDEX IF NOT EXISTS idx_wellbeing_events_created_at
ON wellbeing_events(created_at);

CREATE INDEX IF NOT EXISTS idx_wellbeing_events_type
ON wellbeing_events(event_type);

CREATE TABLE IF NOT EXISTS wellbeing_recommendations (
    recommendation_id TEXT PRIMARY KEY,
    created_at TEXT NOT NULL,
    status TEXT NOT NULL,
    payload TEXT NOT NULL
);

CREATE INDEX IF NOT EXISTS idx_wellbeing_recommendations_created_at
ON wellbeing_recommendations(created_at);

CREATE INDEX IF NOT EXISTS idx_wellbeing_recommendations_status
ON wellbeing_recommendations(status);

CREATE TABLE IF NOT EXISTS self_modification_proposals (
    proposal_id TEXT PRIMARY KEY,
    created_at TEXT NOT NULL,
    status TEXT NOT NULL,
    target_kind TEXT NOT NULL,
    payload TEXT NOT NULL
);

CREATE INDEX IF NOT EXISTS idx_self_modification_proposals_created_at
ON self_modification_proposals(created_at);

CREATE INDEX IF NOT EXISTS idx_self_modification_proposals_status
ON self_modification_proposals(status);

CREATE TABLE IF NOT EXISTS maintenance_outcomes (
    outcome_id TEXT PRIMARY KEY,
    created_at TEXT NOT NULL,
    action_type TEXT NOT NULL,
    outcome TEXT NOT NULL,
    payload TEXT NOT NULL
);

CREATE INDEX IF NOT EXISTS idx_maintenance_outcomes_created_at
ON maintenance_outcomes(created_at);

CREATE INDEX IF NOT EXISTS idx_maintenance_outcomes_action_type
ON maintenance_outcomes(action_type);
"""


class WellbeingStore:
    """Async SQLite persistence for wellbeing state and maintenance records."""

    def __init__(self, path: Path | str) -> None:
        self.path = Path(path)
        self._db: aiosqlite.Connection | None = None

    async def connect(self) -> "WellbeingStore":
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._db = await aiosqlite.connect(str(self.path))
        self._db.row_factory = aiosqlite.Row
        await self._db.executescript(_SCHEMA)
        await self._db.commit()
        return self

    async def close(self) -> None:
        if self._db is not None:
            await self._db.close()
            self._db = None

    async def save_state(self, state: WellbeingState) -> None:
        assert self._db is not None
        await self._db.execute(
            """
            INSERT INTO wellbeing_states (state_id, created_at, payload)
            VALUES (?, ?, ?)
            ON CONFLICT(state_id) DO UPDATE SET
                created_at = excluded.created_at,
                payload = excluded.payload
            """,
            (
                str(state.state_id),
                state.created_at.isoformat(),
                state.model_dump_json(),
            ),
        )
        await self._db.commit()

    async def latest_state(self) -> WellbeingState | None:
        assert self._db is not None
        cursor = await self._db.execute(
            """
            SELECT payload FROM wellbeing_states
            ORDER BY created_at DESC
            LIMIT 1
            """
        )
        row = await cursor.fetchone()
        return WellbeingState.model_validate_json(row["payload"]) if row else None

    async def list_states(self, limit: int = 20) -> list[WellbeingState]:
        assert self._db is not None
        cursor = await self._db.execute(
            """
            SELECT payload FROM wellbeing_states
            ORDER BY created_at DESC
            LIMIT ?
            """,
            (max(1, min(200, int(limit))),),
        )
        rows = await cursor.fetchall()
        return [WellbeingState.model_validate_json(row["payload"]) for row in rows]

    async def append_event(self, event: WellbeingEvent) -> None:
        assert self._db is not None
        await self._db.execute(
            """
            INSERT OR REPLACE INTO wellbeing_events
                (event_id, created_at, event_type, summary, payload)
            VALUES (?, ?, ?, ?, ?)
            """,
            (
                str(event.event_id),
                event.created_at.isoformat(),
                event.event_type,
                event.summary,
                event.model_dump_json(),
            ),
        )
        await self._db.commit()

    async def list_events(self, limit: int = 20) -> list[WellbeingEvent]:
        assert self._db is not None
        cursor = await self._db.execute(
            """
            SELECT payload FROM wellbeing_events
            ORDER BY created_at DESC
            LIMIT ?
            """,
            (max(1, min(200, int(limit))),),
        )
        rows = await cursor.fetchall()
        return [WellbeingEvent.model_validate_json(row["payload"]) for row in rows]

    async def save_recommendation(self, recommendation: WellbeingRecommendation) -> None:
        assert self._db is not None
        await self._db.execute(
            """
            INSERT INTO wellbeing_recommendations
                (recommendation_id, created_at, status, payload)
            VALUES (?, ?, ?, ?)
            ON CONFLICT(recommendation_id) DO UPDATE SET
                created_at = excluded.created_at,
                status = excluded.status,
                payload = excluded.payload
            """,
            (
                str(recommendation.recommendation_id),
                recommendation.created_at.isoformat(),
                recommendation.status,
                recommendation.model_dump_json(),
            ),
        )
        await self._db.commit()

    async def list_recommendations(
        self,
        *,
        status: str | None = None,
        limit: int = 20,
    ) -> list[WellbeingRecommendation]:
        assert self._db is not None
        if status:
            cursor = await self._db.execute(
                """
                SELECT payload FROM wellbeing_recommendations
                WHERE status = ?
                ORDER BY created_at DESC
                LIMIT ?
                """,
                (status, max(1, min(200, int(limit)))),
            )
        else:
            cursor = await self._db.execute(
                """
                SELECT payload FROM wellbeing_recommendations
                ORDER BY created_at DESC
                LIMIT ?
                """,
                (max(1, min(200, int(limit))),),
            )
        rows = await cursor.fetchall()
        return [WellbeingRecommendation.model_validate_json(row["payload"]) for row in rows]

    async def save_self_modification_proposal(self, proposal: SelfModificationProposal) -> None:
        assert self._db is not None
        await self._db.execute(
            """
            INSERT INTO self_modification_proposals
                (proposal_id, created_at, status, target_kind, payload)
            VALUES (?, ?, ?, ?, ?)
            ON CONFLICT(proposal_id) DO UPDATE SET
                created_at = excluded.created_at,
                status = excluded.status,
                target_kind = excluded.target_kind,
                payload = excluded.payload
            """,
            (
                str(proposal.proposal_id),
                proposal.created_at.isoformat(),
                proposal.status,
                proposal.target_kind,
                proposal.model_dump_json(),
            ),
        )
        await self._db.commit()

    async def list_self_modification_proposals(
        self,
        *,
        status: str | None = None,
        limit: int = 20,
    ) -> list[SelfModificationProposal]:
        assert self._db is not None
        if status:
            cursor = await self._db.execute(
                """
                SELECT payload FROM self_modification_proposals
                WHERE status = ?
                ORDER BY created_at DESC
                LIMIT ?
                """,
                (status, max(1, min(200, int(limit)))),
            )
        else:
            cursor = await self._db.execute(
                """
                SELECT payload FROM self_modification_proposals
                ORDER BY created_at DESC
                LIMIT ?
                """,
                (max(1, min(200, int(limit))),),
            )
        rows = await cursor.fetchall()
        return [SelfModificationProposal.model_validate_json(row["payload"]) for row in rows]

    async def save_maintenance_outcome(self, outcome: MaintenanceOutcome) -> None:
        assert self._db is not None
        await self._db.execute(
            """
            INSERT INTO maintenance_outcomes
                (outcome_id, created_at, action_type, outcome, payload)
            VALUES (?, ?, ?, ?, ?)
            ON CONFLICT(outcome_id) DO UPDATE SET
                created_at = excluded.created_at,
                action_type = excluded.action_type,
                outcome = excluded.outcome,
                payload = excluded.payload
            """,
            (
                str(outcome.outcome_id),
                outcome.created_at.isoformat(),
                outcome.action_type,
                outcome.outcome,
                outcome.model_dump_json(),
            ),
        )
        await self._db.commit()

    async def list_maintenance_outcomes(
        self,
        *,
        action_type: str | None = None,
        limit: int = 20,
    ) -> list[MaintenanceOutcome]:
        assert self._db is not None
        if action_type:
            cursor = await self._db.execute(
                """
                SELECT payload FROM maintenance_outcomes
                WHERE action_type = ?
                ORDER BY created_at DESC
                LIMIT ?
                """,
                (action_type, max(1, min(200, int(limit)))),
            )
        else:
            cursor = await self._db.execute(
                """
                SELECT payload FROM maintenance_outcomes
                ORDER BY created_at DESC
                LIMIT ?
                """,
                (max(1, min(200, int(limit))),),
            )
        rows = await cursor.fetchall()
        return [MaintenanceOutcome.model_validate_json(row["payload"]) for row in rows]

    async def vacuum(self) -> None:
        assert self._db is not None
        try:
            await self._db.execute("VACUUM")
        except sqlite3.OperationalError:
            pass
