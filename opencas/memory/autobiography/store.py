"""SQLite persistence for session autobiography anchors."""

from __future__ import annotations

import json
from datetime import datetime
from typing import Any, Optional

import aiosqlite

from .models import SessionAnchor

_JSON_FIELDS = {
    "episode_kind_tally",
    "decision_episode_ids",
    "affect_peaks",
    "artifact_paths_touched",
    "commitments_opened",
    "commitments_closed",
    "identity_mutagen_episode_ids",
    "compaction_record_ids",
    "narrative_bridge_message_ids",
    "continuity_breadcrumb_ids",
    "evidence_episode_ids",
    "recall_failure_episode_ids",
    "recall_recovery_episode_ids",
    "gaps_noted",
}

_COLUMNS = (
    "session_id",
    "created_at",
    "ended_at",
    "duration_s",
    "agent_name",
    "source_system",
    "episode_kind_tally",
    "decision_episode_ids",
    "affect_peaks",
    "recorded_affect_summary",
    "artifact_paths_touched",
    "commitments_opened",
    "commitments_closed",
    "identity_mutagen_episode_ids",
    "compaction_record_ids",
    "narrative_bridge_message_ids",
    "continuity_breadcrumb_ids",
    "evidence_episode_ids",
    "recall_failure_episode_ids",
    "recall_recovery_episode_ids",
    "evidence_hash",
    "evidence_strength",
    "gist",
    "gist_version",
    "confidence",
    "gaps_noted",
    "consolidation_run_id_at_time",
    "version",
)


class SessionAnchorStore:
    """CRUD wrapper for session_anchors sharing MemoryStore's connection."""

    def __init__(self, db: aiosqlite.Connection) -> None:
        self.db = db

    async def upsert(self, anchor: SessionAnchor) -> None:
        """Insert or structurally refresh an anchor.

        Skeleton refresh invalidates an existing gist only when the evidence hash
        changes. Otherwise non-null gists survive repeated lifecycle upserts.
        """

        params = self._values(anchor)
        await self.db.execute(
            """
            INSERT INTO session_anchors (
                session_id, created_at, ended_at, duration_s, agent_name,
                source_system, episode_kind_tally, decision_episode_ids,
                affect_peaks, recorded_affect_summary, artifact_paths_touched,
                commitments_opened, commitments_closed, identity_mutagen_episode_ids,
                compaction_record_ids, narrative_bridge_message_ids,
                continuity_breadcrumb_ids, evidence_episode_ids,
                recall_failure_episode_ids, recall_recovery_episode_ids,
                evidence_hash, evidence_strength, gist, gist_version, confidence,
                gaps_noted, consolidation_run_id_at_time, version
            ) VALUES (
                ?, ?, ?, ?, ?,
                ?, ?, ?,
                ?, ?, ?,
                ?, ?, ?,
                ?, ?,
                ?, ?,
                ?, ?,
                ?, ?, ?, ?, ?,
                ?, ?, ?
            )
            ON CONFLICT(session_id) DO UPDATE SET
                ended_at = CASE
                    WHEN excluded.ended_at IS NULL THEN session_anchors.ended_at
                    WHEN session_anchors.ended_at IS NULL THEN excluded.ended_at
                    WHEN excluded.ended_at > session_anchors.ended_at THEN excluded.ended_at
                    ELSE session_anchors.ended_at
                END,
                duration_s = CASE
                    WHEN excluded.duration_s IS NULL THEN session_anchors.duration_s
                    WHEN session_anchors.duration_s IS NULL THEN excluded.duration_s
                    WHEN excluded.duration_s > session_anchors.duration_s THEN excluded.duration_s
                    ELSE session_anchors.duration_s
                END,
                agent_name = excluded.agent_name,
                source_system = excluded.source_system,
                episode_kind_tally = excluded.episode_kind_tally,
                decision_episode_ids = excluded.decision_episode_ids,
                affect_peaks = excluded.affect_peaks,
                recorded_affect_summary = excluded.recorded_affect_summary,
                artifact_paths_touched = excluded.artifact_paths_touched,
                commitments_opened = excluded.commitments_opened,
                commitments_closed = excluded.commitments_closed,
                identity_mutagen_episode_ids = excluded.identity_mutagen_episode_ids,
                compaction_record_ids = excluded.compaction_record_ids,
                narrative_bridge_message_ids = excluded.narrative_bridge_message_ids,
                continuity_breadcrumb_ids = excluded.continuity_breadcrumb_ids,
                evidence_episode_ids = excluded.evidence_episode_ids,
                recall_failure_episode_ids = excluded.recall_failure_episode_ids,
                recall_recovery_episode_ids = excluded.recall_recovery_episode_ids,
                evidence_hash = excluded.evidence_hash,
                evidence_strength = excluded.evidence_strength,
                gist = CASE
                    WHEN session_anchors.evidence_hash IS NOT excluded.evidence_hash THEN NULL
                    ELSE COALESCE(session_anchors.gist, excluded.gist)
                END,
                gist_version = CASE
                    WHEN session_anchors.evidence_hash IS NOT excluded.evidence_hash THEN 0
                    ELSE session_anchors.gist_version
                END,
                confidence = CASE
                    WHEN session_anchors.evidence_hash IS NOT excluded.evidence_hash THEN NULL
                    ELSE session_anchors.confidence
                END,
                gaps_noted = excluded.gaps_noted,
                consolidation_run_id_at_time = excluded.consolidation_run_id_at_time,
                version = excluded.version
            """,
            params,
        )
        await self.db.commit()

    async def get(self, session_id: str) -> Optional[SessionAnchor]:
        cursor = await self.db.execute(
            "SELECT * FROM session_anchors WHERE session_id = ?",
            (session_id,),
        )
        row = await cursor.fetchone()
        return self._row_to_anchor(row) if row is not None else None

    async def list_recent(
        self,
        since: datetime,
        limit: int = 50,
    ) -> list[SessionAnchor]:
        cursor = await self.db.execute(
            """
            SELECT * FROM session_anchors
            WHERE created_at >= ?
            ORDER BY created_at DESC
            LIMIT ?
            """,
            (since.isoformat(), limit),
        )
        rows = await cursor.fetchall()
        return [self._row_to_anchor(row) for row in rows]

    async def list_missing_for_sessions(self, session_ids: list[str]) -> list[str]:
        missing: list[str] = []
        for session_id in session_ids:
            if await self.get(session_id) is None:
                missing.append(session_id)
        return missing

    async def update_gist(
        self,
        session_id: str,
        gist: str,
        confidence: str,
        evidence_hash: str,
        *,
        gist_version: int = 1,
    ) -> None:
        await self.db.execute(
            """
            UPDATE session_anchors
            SET gist = ?, gist_version = ?, confidence = ?, evidence_hash = ?
            WHERE session_id = ?
            """,
            (gist, gist_version, confidence, evidence_hash, session_id),
        )
        await self.db.commit()

    async def list_skeletons(self, limit: int = 100) -> list[SessionAnchor]:
        cursor = await self.db.execute(
            """
            SELECT * FROM session_anchors
            WHERE gist_version = 0
            ORDER BY created_at DESC
            LIMIT ?
            """,
            (limit,),
        )
        rows = await cursor.fetchall()
        return [self._row_to_anchor(row) for row in rows]

    def _params(self, anchor: SessionAnchor) -> dict[str, Any]:
        raw = anchor.model_dump()
        for key, value in list(raw.items()):
            if key in _JSON_FIELDS:
                raw[key] = json.dumps(value)
            elif isinstance(value, datetime):
                raw[key] = value.isoformat()
        return raw

    def _values(self, anchor: SessionAnchor) -> tuple[Any, ...]:
        raw = self._params(anchor)
        return tuple(raw[column] for column in _COLUMNS)

    def _row_to_anchor(self, row: aiosqlite.Row) -> SessionAnchor:
        data = dict(row)
        for key in _JSON_FIELDS:
            data[key] = self._load_json(data.get(key), [] if key != "episode_kind_tally" else {})
        for key in ("created_at", "ended_at"):
            if data.get(key):
                data[key] = datetime.fromisoformat(data[key])
        return SessionAnchor(**data)

    @staticmethod
    def _load_json(value: Any, default: Any) -> Any:
        if value is None or value == "":
            return default
        if not isinstance(value, str):
            return value
        try:
            return json.loads(value)
        except json.JSONDecodeError:
            return default
