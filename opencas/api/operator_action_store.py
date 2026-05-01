"""Durable registry store for operator-action provenance lines."""

from __future__ import annotations

import json
import os
import sqlite3
from dataclasses import replace
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, List, Mapping, Optional

from .registry_provenance import (
    ProvenanceRecordV1,
    format_registry_entry,
    parse_registry_entry,
    read_registry_entries,
)

__all__ = ["OperatorActionRegistryStore", "append_event", "read_events"]

_SCHEMA = """
PRAGMA journal_mode=WAL;
PRAGMA synchronous=FULL;
PRAGMA foreign_keys=ON;
PRAGMA busy_timeout=5000;

CREATE TABLE IF NOT EXISTS operator_action_registry (
    seq INTEGER PRIMARY KEY AUTOINCREMENT,
    registry_line TEXT NOT NULL,
    session_id TEXT NOT NULL,
    artifact TEXT NOT NULL,
    action TEXT NOT NULL,
    why TEXT NOT NULL,
    risk TEXT NOT NULL,
    created_at TEXT NOT NULL,
    source_trace TEXT DEFAULT NULL
);

CREATE INDEX IF NOT EXISTS idx_operator_action_registry_seq
    ON operator_action_registry(seq);
"""

_MIGRATE_ADD_SOURCE_TRACE = """
ALTER TABLE operator_action_registry ADD COLUMN source_trace TEXT DEFAULT NULL;
"""


class OperatorActionRegistryStore:
    """SQLite-backed append-only registry with legacy JSONL replay support."""

    def __init__(self, path: Path | str) -> None:
        raw_path = Path(path)
        if raw_path.suffix == ".jsonl":
            self.path = raw_path.with_suffix(".db")
            self.legacy_path = raw_path
        else:
            self.path = raw_path
            self.legacy_path = raw_path.with_suffix(".jsonl")

    def append(self, entry: str | ProvenanceRecordV1 | Mapping[str, Any]) -> ProvenanceRecordV1:
        """Append one canonical registry entry durably."""

        return self._append_event(entry)

    def list_recent(self, limit: int | None = 10, offset: int = 0) -> List[ProvenanceRecordV1]:
        """Return registry entries in the exact append order."""

        return self._read_events(limit=limit, offset=offset)

    def append_event(self, entry: str | ProvenanceRecordV1 | Mapping[str, Any]) -> ProvenanceRecordV1:
        """Compatibility alias for append_event-style call sites."""

        return self._append_event(entry)

    def read_events(self, limit: int | None = 10, offset: int = 0) -> List[ProvenanceRecordV1]:
        """Compatibility alias for read_event-style call sites."""

        return self._read_events(limit=limit, offset=offset)

    def _connect(self) -> sqlite3.Connection:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        conn = sqlite3.connect(str(self.path))
        conn.row_factory = sqlite3.Row
        conn.executescript(_SCHEMA)
        try:
            conn.execute(_MIGRATE_ADD_SOURCE_TRACE)
            conn.commit()
        except sqlite3.OperationalError:
            pass  # column already exists
        return conn

    @staticmethod
    def _extract_source_trace(entry: str | ProvenanceRecordV1 | Mapping[str, Any]) -> Optional[str]:
        """Extract source_trace JSON from the original entry before normalization."""
        if isinstance(entry, ProvenanceRecordV1) and entry.source_trace is not None:
            return json.dumps(entry.source_trace)
        if isinstance(entry, Mapping):
            st = entry.get("source_trace")
            if isinstance(st, dict):
                return json.dumps(st)
            if isinstance(st, str):
                return st
        return None

    def _append_event(self, entry: str | ProvenanceRecordV1 | Mapping[str, Any]) -> ProvenanceRecordV1:
        source_trace_json = self._extract_source_trace(entry)
        if isinstance(entry, str):
            record = parse_registry_entry(entry)
        else:
            serialized = format_registry_entry(entry)
            record = parse_registry_entry(serialized)
        serialized = format_registry_entry(record)

        with self._connect() as conn:
            self._bootstrap_legacy_log_if_needed(conn)
            conn.execute(
                """
                INSERT INTO operator_action_registry (
                    registry_line, session_id, artifact, action, why, risk, created_at, source_trace
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    serialized,
                    record.session_id,
                    record.artifact,
                    record.action.value,
                    record.why,
                    record.risk.value,
                    datetime.now(timezone.utc).isoformat(),
                    source_trace_json,
                ),
            )
            conn.commit()

        self._mirror_to_legacy_log(serialized)
        if source_trace_json is not None:
            try:
                record = replace(record, source_trace=json.loads(source_trace_json))
            except Exception:
                pass
        return record

    def _read_events(self, limit: int | None = 10, offset: int = 0) -> List[ProvenanceRecordV1]:
        with self._connect() as conn:
            self._bootstrap_legacy_log_if_needed(conn)
            if limit is None:
                cursor = conn.execute(
                    """
                    SELECT registry_line, source_trace
                    FROM operator_action_registry
                    ORDER BY seq ASC
                    LIMIT -1 OFFSET ?
                    """,
                    (offset,),
                )
            else:
                cursor = conn.execute(
                    """
                    SELECT registry_line, source_trace
                    FROM operator_action_registry
                    ORDER BY seq ASC
                    LIMIT ? OFFSET ?
                    """,
                    (limit, offset),
                )
            rows = cursor.fetchall()

        records = []
        for row in rows:
            record = parse_registry_entry(row["registry_line"])
            st_raw = row["source_trace"]
            if st_raw:
                try:
                    record = replace(record, source_trace=json.loads(st_raw))
                except Exception:
                    pass
            records.append(record)
        return records

    def _bootstrap_legacy_log_if_needed(self, conn: sqlite3.Connection) -> None:
        cursor = conn.execute("SELECT COUNT(*) AS count FROM operator_action_registry")
        if int(cursor.fetchone()["count"]) > 0:
            return
        if not self.legacy_path.exists():
            return

        try:
            raw_text = self.legacy_path.read_text(encoding="utf-8")
        except Exception:
            return

        legacy_entries = read_registry_entries(raw_text)
        if not legacy_entries:
            return

        now = datetime.now(timezone.utc).isoformat()
        conn.executemany(
            """
            INSERT INTO operator_action_registry (
                registry_line, session_id, artifact, action, why, risk, created_at
            ) VALUES (?, ?, ?, ?, ?, ?, ?)
            """,
            [
                (
                    format_registry_entry(entry),
                    entry.session_id,
                    entry.artifact,
                    entry.action.value,
                    entry.why,
                    entry.risk.value,
                    now,
                )
                for entry in legacy_entries
            ],
        )
        conn.commit()

    def _mirror_to_legacy_log(self, line: str) -> None:
        try:
            self.legacy_path.parent.mkdir(parents=True, exist_ok=True)
            with self.legacy_path.open("a", encoding="utf-8") as handle:
                handle.write(line + "\n")
                handle.flush()
                os.fsync(handle.fileno())
        except Exception:
            # The SQLite registry is authoritative; the mirrored log is a compatibility aid.
            return


def append_event(
    target: OperatorActionRegistryStore | Path | str | os.PathLike[str] | Any,
    entry: str | ProvenanceRecordV1 | Mapping[str, Any],
) -> ProvenanceRecordV1:
    """Append one canonical registry event to a store, path, or sink."""

    if isinstance(target, OperatorActionRegistryStore):
        return target._append_event(entry)
    if isinstance(target, (str, Path, os.PathLike)):
        return OperatorActionRegistryStore(target)._append_event(entry)

    if isinstance(entry, str):
        record = parse_registry_entry(entry)
    else:
        serialized = format_registry_entry(entry)
        record = parse_registry_entry(serialized)

    line = format_registry_entry(record)
    append = getattr(target, "append", None)
    if callable(append):
        result = append(line)
        if result is False:
            raise RuntimeError("registry sink rejected the entry")
        return record

    raise RuntimeError("registry sink is unavailable or unsupported")


def read_events(
    target: OperatorActionRegistryStore | Path | str | os.PathLike[str] | Any,
    *,
    limit: int | None = 10,
    offset: int = 0,
) -> List[ProvenanceRecordV1]:
    """Read stored registry events in stable append order."""

    if isinstance(target, OperatorActionRegistryStore):
        return target._read_events(limit=limit, offset=offset)
    if isinstance(target, (str, Path, os.PathLike)):
        return OperatorActionRegistryStore(target)._read_events(limit=limit, offset=offset)

    list_recent = getattr(target, "list_recent", None)
    if callable(list_recent):
        recent_limit = limit if limit is not None else 2**31 - 1
        recent = list_recent(limit=recent_limit, offset=offset)
        return list(recent or [])

    raise RuntimeError("registry sink is unavailable or unsupported")
