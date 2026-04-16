"""Archive and index large OpenBulma executive event logs."""

from __future__ import annotations

import gzip
import json
import sqlite3
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional


_SCHEMA = """
CREATE TABLE IF NOT EXISTS bulma_executive_events (
    source_line INTEGER PRIMARY KEY,
    ts TEXT NOT NULL,
    event_type TEXT NOT NULL,
    entity TEXT,
    status TEXT,
    label TEXT,
    goal_thread_id TEXT,
    goal_id TEXT,
    task_id TEXT,
    raw TEXT NOT NULL
);

CREATE INDEX IF NOT EXISTS idx_bulma_exec_ts ON bulma_executive_events(ts);
CREATE INDEX IF NOT EXISTS idx_bulma_exec_type ON bulma_executive_events(event_type);
CREATE INDEX IF NOT EXISTS idx_bulma_exec_entity ON bulma_executive_events(entity);
CREATE INDEX IF NOT EXISTS idx_bulma_exec_goal_thread ON bulma_executive_events(goal_thread_id);

CREATE TABLE IF NOT EXISTS bulma_executive_event_meta (
    key TEXT PRIMARY KEY,
    value TEXT NOT NULL
);
"""


def archive_and_index_executive_events(
    events_path: Path,
    state_dir: Path,
    *,
    archive_name: str = "executive_events.jsonl.gz",
) -> Dict[str, Any]:
    """Archive a Bulma executive event log and build a compact SQLite index."""
    events_path = Path(events_path)
    migration_dir = Path(state_dir) / "migration" / "bulma"
    migration_dir.mkdir(parents=True, exist_ok=True)
    archive_path = migration_dir / archive_name
    index_path = migration_dir / "executive_events.db"

    if not events_path.exists():
        return {
            "count": 0,
            "archive_path": str(archive_path),
            "index_path": str(index_path),
            "missing": True,
        }

    conn = sqlite3.connect(str(index_path))
    conn.execute("PRAGMA journal_mode=WAL")
    conn.executescript(_SCHEMA)
    conn.execute("DELETE FROM bulma_executive_events")
    conn.execute("DELETE FROM bulma_executive_event_meta")

    type_counts: Counter[str] = Counter()
    entity_counts: Counter[str] = Counter()
    day_counts: Counter[str] = Counter()
    first_ts: Optional[str] = None
    last_ts: Optional[str] = None
    count = 0
    batch: List[tuple[Any, ...]] = []

    with events_path.open("rb") as source, gzip.open(archive_path, "wb") as archive:
        for line_no, raw_bytes in enumerate(source, start=1):
            archive.write(raw_bytes)
            try:
                raw_text = raw_bytes.decode("utf-8")
                event = json.loads(raw_text)
            except (UnicodeDecodeError, json.JSONDecodeError):
                continue

            details = event.get("details") if isinstance(event.get("details"), dict) else {}
            ts = _normalize_ts(event.get("ts"))
            event_type = str(event.get("type") or "unknown")
            entity = _string_or_none(details.get("entity"))
            status = _string_or_none(details.get("status"))
            label = _string_or_none(details.get("label"))
            goal_thread_id = _string_or_none(details.get("goalThreadId") or details.get("goal_thread_id"))
            goal_id = _string_or_none(details.get("goalId") or details.get("goal_id"))
            task_id = _string_or_none(details.get("taskId") or details.get("task_id"))

            first_ts = first_ts or ts
            last_ts = ts
            type_counts[event_type] += 1
            if entity:
                entity_counts[entity] += 1
            day_counts[ts[:10]] += 1
            count += 1
            batch.append(
                (
                    line_no,
                    ts,
                    event_type,
                    entity,
                    status,
                    label,
                    goal_thread_id,
                    goal_id,
                    task_id,
                    raw_text.rstrip("\n"),
                )
            )
            if len(batch) >= 1000:
                _insert_batch(conn, batch)
                batch.clear()

    if batch:
        _insert_batch(conn, batch)

    meta = {
        "source_path": str(events_path),
        "archive_path": str(archive_path),
        "indexed_at": datetime.now(timezone.utc).isoformat(),
        "count": count,
        "first_ts": first_ts,
        "last_ts": last_ts,
        "top_types": type_counts.most_common(25),
        "top_entities": entity_counts.most_common(25),
        "day_count": len(day_counts),
    }
    for key, value in meta.items():
        conn.execute(
            "INSERT INTO bulma_executive_event_meta (key, value) VALUES (?, ?)",
            (key, json.dumps(value)),
        )
    conn.commit()
    conn.close()
    return {
        "count": count,
        "archive_path": str(archive_path),
        "index_path": str(index_path),
        "first_ts": first_ts,
        "last_ts": last_ts,
        "top_types": type_counts.most_common(25),
        "top_entities": entity_counts.most_common(25),
    }


def load_executive_event_summary(state_dir: Path) -> Dict[str, Any]:
    index_path = Path(state_dir) / "migration" / "bulma" / "executive_events.db"
    if not index_path.exists():
        return {"count": 0, "available": False}
    conn = sqlite3.connect(str(index_path))
    try:
        rows = conn.execute("SELECT key, value FROM bulma_executive_event_meta").fetchall()
        meta = {key: json.loads(value) for key, value in rows}
        return {"available": True, **meta}
    finally:
        conn.close()


def search_executive_events(
    state_dir: Path,
    *,
    event_type: Optional[str] = None,
    query: Optional[str] = None,
    limit: int = 50,
) -> List[Dict[str, Any]]:
    index_path = Path(state_dir) / "migration" / "bulma" / "executive_events.db"
    if not index_path.exists():
        return []
    limit = max(1, min(limit, 500))
    sql = """
        SELECT source_line, ts, event_type, entity, status, label,
               goal_thread_id, goal_id, task_id
        FROM bulma_executive_events
    """
    clauses: List[str] = []
    params: List[Any] = []
    if event_type:
        clauses.append("event_type = ?")
        params.append(event_type)
    if query:
        clauses.append("(label LIKE ? OR raw LIKE ?)")
        params.extend([f"%{query}%", f"%{query}%"])
    if clauses:
        sql += " WHERE " + " AND ".join(clauses)
    sql += " ORDER BY ts DESC LIMIT ?"
    params.append(limit)
    conn = sqlite3.connect(str(index_path))
    conn.row_factory = sqlite3.Row
    try:
        return [dict(row) for row in conn.execute(sql, params).fetchall()]
    finally:
        conn.close()


def build_executive_event_summary_episodes(summary: Dict[str, Any]) -> Iterable[Dict[str, Any]]:
    """Return compact summary dicts suitable for synthetic memory episodes."""
    if not summary.get("count"):
        return []
    top_types = ", ".join(f"{kind}={count}" for kind, count in summary.get("top_types", [])[:8])
    return [
        {
            "content": (
                "Bulma executive event archive indexed: "
                f"{summary.get('count')} events from {summary.get('first_ts')} to {summary.get('last_ts')}. "
                f"Top event types: {top_types}."
            ),
            "payload": {
                "bulma_event_archive": summary.get("archive_path"),
                "bulma_event_index": summary.get("index_path"),
                "bulma_event_count": summary.get("count"),
                "bulma_event_top_types": summary.get("top_types", []),
            },
        }
    ]


def _insert_batch(conn: sqlite3.Connection, batch: List[tuple[Any, ...]]) -> None:
    conn.executemany(
        """
        INSERT OR REPLACE INTO bulma_executive_events (
            source_line, ts, event_type, entity, status, label,
            goal_thread_id, goal_id, task_id, raw
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """,
        batch,
    )
    conn.commit()


def _normalize_ts(value: Any) -> str:
    if isinstance(value, (int, float)):
        return datetime.fromtimestamp(value / 1000.0, tz=timezone.utc).isoformat()
    if isinstance(value, str):
        try:
            return datetime.fromisoformat(value.replace("Z", "+00:00")).isoformat()
        except ValueError:
            return value
    return datetime.now(timezone.utc).isoformat()


def _string_or_none(value: Any) -> Optional[str]:
    if value is None:
        return None
    return str(value)
