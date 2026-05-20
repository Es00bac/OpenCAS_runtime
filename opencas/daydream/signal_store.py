"""SQLite persistence for daydream possibility signals."""

from __future__ import annotations

import json
from datetime import datetime, timedelta, timezone
from typing import Any, List, Optional
from uuid import uuid4

import aiosqlite

from .signals import PossibilitySignal, PossibilitySignalRoute, SelfWorkReceipt
from .sqlite_base import SqliteBackedStore


_SIGNAL_SCHEMA = """
CREATE TABLE IF NOT EXISTS daydream_signals (
    signal_id TEXT PRIMARY KEY,
    created_at TEXT NOT NULL,
    updated_at TEXT NOT NULL,
    source_reflection_id TEXT NOT NULL,
    source_thought_index INTEGER NOT NULL DEFAULT 0,
    source_mode TEXT NOT NULL DEFAULT 'waking_daydream',
    summary TEXT NOT NULL,
    imaginative_branch TEXT NOT NULL DEFAULT '',
    practical_branch TEXT NOT NULL DEFAULT '',
    bridge TEXT NOT NULL DEFAULT '',
    novelty REAL NOT NULL DEFAULT 0.5,
    usefulness REAL NOT NULL DEFAULT 0.5,
    confidence REAL NOT NULL DEFAULT 0.5,
    risk REAL NOT NULL DEFAULT 0.0,
    suggested_route TEXT NOT NULL DEFAULT 'incubate',
    contact_posture TEXT NOT NULL DEFAULT 'silent',
    self_work_kind TEXT NOT NULL DEFAULT 'none',
    route_status TEXT NOT NULL DEFAULT 'pending',
    route_reason TEXT NOT NULL DEFAULT '',
    routed_at TEXT,
    artifact_paths TEXT NOT NULL DEFAULT '[]',
    raw TEXT NOT NULL DEFAULT '{}'
);

CREATE INDEX IF NOT EXISTS idx_daydream_signals_created_at ON daydream_signals(created_at);
CREATE INDEX IF NOT EXISTS idx_daydream_signals_route ON daydream_signals(suggested_route);
CREATE INDEX IF NOT EXISTS idx_daydream_signals_status ON daydream_signals(route_status);

CREATE TABLE IF NOT EXISTS daydream_signal_routes (
    route_event_id TEXT PRIMARY KEY,
    signal_id TEXT NOT NULL,
    created_at TEXT NOT NULL,
    route TEXT NOT NULL,
    status TEXT NOT NULL,
    reason TEXT NOT NULL DEFAULT '',
    artifact_paths TEXT NOT NULL DEFAULT '[]',
    raw TEXT NOT NULL DEFAULT '{}'
);

CREATE INDEX IF NOT EXISTS idx_daydream_signal_routes_signal_id ON daydream_signal_routes(signal_id);
CREATE INDEX IF NOT EXISTS idx_daydream_signal_routes_created_at ON daydream_signal_routes(created_at);

CREATE TABLE IF NOT EXISTS daydream_self_work_receipts (
    receipt_id TEXT PRIMARY KEY,
    signal_id TEXT NOT NULL,
    created_at TEXT NOT NULL,
    route TEXT NOT NULL,
    kind TEXT NOT NULL DEFAULT 'none',
    outcome TEXT NOT NULL DEFAULT '',
    summary TEXT NOT NULL DEFAULT '',
    artifact_paths TEXT NOT NULL DEFAULT '[]',
    raw TEXT NOT NULL DEFAULT '{}'
);

CREATE INDEX IF NOT EXISTS idx_daydream_self_work_receipts_signal_id ON daydream_self_work_receipts(signal_id);
CREATE INDEX IF NOT EXISTS idx_daydream_self_work_receipts_created_at ON daydream_self_work_receipts(created_at);
"""


class DaydreamSignalStore(SqliteBackedStore):
    """Persist possibility signals, route events, and self-work receipts."""

    SCHEMA = _SIGNAL_SCHEMA

    async def save_signal(self, signal: PossibilitySignal) -> None:
        signal.updated_at = datetime.now(timezone.utc)
        await self.db.execute(
            """
            INSERT INTO daydream_signals (
                signal_id, created_at, updated_at, source_reflection_id,
                source_thought_index, source_mode, summary, imaginative_branch,
                practical_branch, bridge, novelty, usefulness, confidence, risk,
                suggested_route, contact_posture, self_work_kind, route_status,
                route_reason, routed_at, artifact_paths, raw
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            ON CONFLICT(signal_id) DO UPDATE SET
                updated_at = excluded.updated_at,
                source_reflection_id = excluded.source_reflection_id,
                source_thought_index = excluded.source_thought_index,
                source_mode = excluded.source_mode,
                summary = excluded.summary,
                imaginative_branch = excluded.imaginative_branch,
                practical_branch = excluded.practical_branch,
                bridge = excluded.bridge,
                novelty = excluded.novelty,
                usefulness = excluded.usefulness,
                confidence = excluded.confidence,
                risk = excluded.risk,
                suggested_route = excluded.suggested_route,
                contact_posture = excluded.contact_posture,
                self_work_kind = excluded.self_work_kind,
                route_status = excluded.route_status,
                route_reason = excluded.route_reason,
                routed_at = excluded.routed_at,
                artifact_paths = excluded.artifact_paths,
                raw = excluded.raw
            """,
            self._signal_params(signal),
        )
        await self.db.commit()

    async def get_signal(self, signal_id: str) -> Optional[PossibilitySignal]:
        cursor = await self.db.execute(
            "SELECT * FROM daydream_signals WHERE signal_id = ?",
            (signal_id,),
        )
        row = await cursor.fetchone()
        return self._row_to_signal(row) if row else None

    async def list_recent(self, limit: int = 50) -> List[PossibilitySignal]:
        cursor = await self.db.execute(
            "SELECT * FROM daydream_signals ORDER BY created_at DESC LIMIT ?",
            (max(1, int(limit)),),
        )
        return [self._row_to_signal(row) for row in await cursor.fetchall()]

    async def record_route(
        self,
        signal_id: str,
        *,
        route: PossibilitySignalRoute | str,
        status: str,
        reason: str = "",
        artifact_paths: Optional[List[str]] = None,
        raw: Optional[dict[str, Any]] = None,
    ) -> dict[str, Any]:
        now = datetime.now(timezone.utc)
        route_value = route.value if isinstance(route, PossibilitySignalRoute) else str(route)
        paths = list(artifact_paths or [])
        event = {
            "route_event_id": str(uuid4()),
            "signal_id": signal_id,
            "created_at": now.isoformat(),
            "route": route_value,
            "status": str(status),
            "reason": str(reason or ""),
            "artifact_paths": paths,
            "raw": dict(raw or {}),
        }
        await self.db.execute(
            """
            INSERT INTO daydream_signal_routes (
                route_event_id, signal_id, created_at, route, status, reason,
                artifact_paths, raw
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                event["route_event_id"],
                signal_id,
                event["created_at"],
                route_value,
                str(status),
                str(reason or ""),
                json.dumps(paths),
                json.dumps(dict(raw or {})),
            ),
        )
        loaded = await self.get_signal(signal_id)
        if loaded is not None:
            loaded.suggested_route = PossibilitySignalRoute(route_value)
            loaded.route_status = str(status)
            loaded.route_reason = str(reason or "")
            loaded.routed_at = now
            loaded.artifact_paths = paths
            await self.save_signal(loaded)
        await self.db.commit()
        return event

    async def save_receipt(self, receipt: SelfWorkReceipt) -> None:
        await self.db.execute(
            """
            INSERT INTO daydream_self_work_receipts (
                receipt_id, signal_id, created_at, route, kind, outcome,
                summary, artifact_paths, raw
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            ON CONFLICT(receipt_id) DO UPDATE SET
                signal_id = excluded.signal_id,
                created_at = excluded.created_at,
                route = excluded.route,
                kind = excluded.kind,
                outcome = excluded.outcome,
                summary = excluded.summary,
                artifact_paths = excluded.artifact_paths,
                raw = excluded.raw
            """,
            (
                receipt.receipt_id,
                receipt.signal_id,
                receipt.created_at.isoformat(),
                receipt.route.value,
                receipt.kind.value,
                receipt.outcome,
                receipt.summary,
                json.dumps(receipt.artifact_paths),
                json.dumps(receipt.raw),
            ),
        )
        await self.db.commit()

    async def list_receipts(self, limit: int = 50) -> List[SelfWorkReceipt]:
        cursor = await self.db.execute(
            "SELECT * FROM daydream_self_work_receipts ORDER BY created_at DESC LIMIT ?",
            (max(1, int(limit)),),
        )
        return [self._row_to_receipt(row) for row in await cursor.fetchall()]

    async def get_summary(self, window_days: int = 7) -> dict[str, Any]:
        cutoff = (datetime.now(timezone.utc) - timedelta(days=max(1, int(window_days)))).isoformat()
        total_cursor = await self.db.execute("SELECT COUNT(*) AS total FROM daydream_signals")
        window_cursor = await self.db.execute(
            "SELECT COUNT(*) AS total FROM daydream_signals WHERE created_at >= ?",
            (cutoff,),
        )
        route_cursor = await self.db.execute(
            """
            SELECT suggested_route, COUNT(*) AS total
            FROM daydream_signals
            GROUP BY suggested_route
            """
        )
        status_cursor = await self.db.execute(
            """
            SELECT route_status, COUNT(*) AS total
            FROM daydream_signals
            GROUP BY route_status
            """
        )
        receipt_cursor = await self.db.execute(
            "SELECT COUNT(*) AS total FROM daydream_self_work_receipts WHERE created_at >= ?",
            (cutoff,),
        )
        total = await total_cursor.fetchone()
        window = await window_cursor.fetchone()
        receipts = await receipt_cursor.fetchone()
        routes = await route_cursor.fetchall()
        statuses = await status_cursor.fetchall()
        return {
            "total_signals": int(total["total"] or 0),
            "window_days": max(1, int(window_days)),
            "window_signals": int(window["total"] or 0),
            "route_counts": {str(row["suggested_route"]): int(row["total"] or 0) for row in routes},
            "status_counts": {str(row["route_status"]): int(row["total"] or 0) for row in statuses},
            "window_self_work_receipts": int(receipts["total"] or 0),
        }

    @staticmethod
    def _signal_params(signal: PossibilitySignal) -> tuple[Any, ...]:
        raw = signal.model_dump(mode="json")
        return (
            signal.signal_id,
            signal.created_at.isoformat(),
            signal.updated_at.isoformat(),
            signal.source_reflection_id,
            signal.source_thought_index,
            signal.source_mode,
            signal.summary,
            signal.imaginative_branch,
            signal.practical_branch,
            signal.bridge,
            signal.novelty,
            signal.usefulness,
            signal.confidence,
            signal.risk,
            signal.suggested_route.value,
            signal.contact_posture.value,
            signal.self_work_kind.value,
            signal.route_status,
            signal.route_reason,
            signal.routed_at.isoformat() if signal.routed_at else None,
            json.dumps(signal.artifact_paths),
            json.dumps(raw),
        )

    @staticmethod
    def _row_to_signal(row: aiosqlite.Row) -> PossibilitySignal:
        raw = json.loads(row["raw"]) if row["raw"] else {}
        raw.update(
            {
                "signal_id": row["signal_id"],
                "created_at": row["created_at"],
                "updated_at": row["updated_at"],
                "source_reflection_id": row["source_reflection_id"],
                "source_thought_index": row["source_thought_index"],
                "source_mode": row["source_mode"],
                "summary": row["summary"],
                "imaginative_branch": row["imaginative_branch"],
                "practical_branch": row["practical_branch"],
                "bridge": row["bridge"],
                "novelty": row["novelty"],
                "usefulness": row["usefulness"],
                "confidence": row["confidence"],
                "risk": row["risk"],
                "suggested_route": row["suggested_route"],
                "contact_posture": row["contact_posture"],
                "self_work_kind": row["self_work_kind"],
                "route_status": row["route_status"],
                "route_reason": row["route_reason"],
                "routed_at": row["routed_at"],
                "artifact_paths": json.loads(row["artifact_paths"]) if row["artifact_paths"] else [],
            }
        )
        return PossibilitySignal.model_validate(raw)

    @staticmethod
    def _row_to_receipt(row: aiosqlite.Row) -> SelfWorkReceipt:
        return SelfWorkReceipt(
            receipt_id=row["receipt_id"],
            signal_id=row["signal_id"],
            created_at=datetime.fromisoformat(row["created_at"]),
            route=row["route"],
            kind=row["kind"],
            outcome=row["outcome"],
            summary=row["summary"],
            artifact_paths=json.loads(row["artifact_paths"]) if row["artifact_paths"] else [],
            raw=json.loads(row["raw"]) if row["raw"] else {},
        )
