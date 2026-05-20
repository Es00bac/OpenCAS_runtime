"""Repair legacy daydream signal route statuses."""

from __future__ import annotations

import json
import sqlite3
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


_STATUS_REPAIRS: dict[str, str] = {
    "ask_user": "deferred",
    "incubate": "incubated",
    "research": "deferred",
}


@dataclass(slots=True)
class DaydreamStatusRepairReport:
    db_path: Path
    applied: bool
    signal_updates: dict[str, int] = field(default_factory=dict)
    route_event_updates: dict[str, int] = field(default_factory=dict)

    @property
    def total_updates(self) -> int:
        return sum(self.signal_updates.values()) + sum(self.route_event_updates.values())

    def as_dict(self) -> dict[str, Any]:
        return {
            "db_path": str(self.db_path),
            "applied": self.applied,
            "signal_updates": dict(sorted(self.signal_updates.items())),
            "route_event_updates": dict(sorted(self.route_event_updates.items())),
            "total_updates": self.total_updates,
        }


def repair_legacy_daydream_signal_statuses(
    db_path: Path | str,
    *,
    apply: bool = False,
) -> DaydreamStatusRepairReport:
    """Reclassify old terminal ``held`` daydream routes into current statuses."""
    path = Path(db_path).expanduser()
    report = DaydreamStatusRepairReport(db_path=path, applied=apply)
    if not path.exists():
        return report

    with sqlite3.connect(path) as conn:
        conn.row_factory = sqlite3.Row
        signal_rows = conn.execute(
            """
            SELECT signal_id, suggested_route, route_status, route_reason, raw
            FROM daydream_signals
            WHERE route_status = 'held'
            """
        ).fetchall()
        route_rows = conn.execute(
            """
            SELECT route_event_id, route, status, reason, raw
            FROM daydream_signal_routes
            WHERE status = 'held'
            """
        ).fetchall()

        now = datetime.now(timezone.utc).isoformat()
        for row in signal_rows:
            new_status = _repair_status(row["suggested_route"], row["route_status"])
            if not new_status:
                continue
            key = f"{row['suggested_route']}:held->{new_status}"
            report.signal_updates[key] = report.signal_updates.get(key, 0) + 1
            if not apply:
                continue
            raw = _mark_repaired_raw(
                row["raw"],
                route=row["suggested_route"],
                from_status="held",
                to_status=new_status,
                reason=row["route_reason"],
            )
            conn.execute(
                """
                UPDATE daydream_signals
                SET route_status = ?, raw = ?, updated_at = ?
                WHERE signal_id = ?
                """,
                (new_status, json.dumps(raw), now, row["signal_id"]),
            )

        for row in route_rows:
            new_status = _repair_status(row["route"], row["status"])
            if not new_status:
                continue
            key = f"{row['route']}:held->{new_status}"
            report.route_event_updates[key] = report.route_event_updates.get(key, 0) + 1
            if not apply:
                continue
            raw = _mark_repaired_raw(
                row["raw"],
                route=row["route"],
                from_status="held",
                to_status=new_status,
                reason=row["reason"],
            )
            conn.execute(
                """
                UPDATE daydream_signal_routes
                SET status = ?, raw = ?
                WHERE route_event_id = ?
                """,
                (new_status, json.dumps(raw), row["route_event_id"]),
            )

        if apply:
            conn.commit()

    return report


def _repair_status(route: str, status: str) -> str | None:
    if status != "held":
        return None
    return _STATUS_REPAIRS.get(str(route))


def _mark_repaired_raw(
    raw_value: str | None,
    *,
    route: str,
    from_status: str,
    to_status: str,
    reason: str | None,
) -> dict[str, Any]:
    raw = _decode_json_object(raw_value)
    meta = raw.get("meta")
    if not isinstance(meta, dict):
        meta = {}
    meta["route_schema_version"] = max(2, int(meta.get("route_schema_version") or 1))
    meta["legacy_status_repair"] = {
        "route": route,
        "from_status": from_status,
        "to_status": to_status,
        "reason": reason or "",
    }
    raw["meta"] = meta
    return raw


def _decode_json_object(value: str | None) -> dict[str, Any]:
    if not value:
        return {}
    try:
        decoded = json.loads(value)
    except json.JSONDecodeError:
        return {}
    return decoded if isinstance(decoded, dict) else {}
