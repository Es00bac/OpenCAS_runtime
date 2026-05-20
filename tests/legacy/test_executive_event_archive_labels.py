"""Historical executive-event archives must not masquerade as live authority."""

from __future__ import annotations

import json
import sqlite3
from pathlib import Path

from opencas.legacy.executive_event_index import (
    load_executive_event_summary,
    search_executive_events,
)


def test_missing_executive_archive_is_labeled_historical(tmp_path: Path) -> None:
    summary = load_executive_event_summary(tmp_path)

    assert summary["available"] is False
    assert summary["archive_kind"] == "historical_openbulma_executive_events"
    assert summary["runtime_authority"] == "historical_archive_only"


def test_executive_archive_search_rows_are_labeled_historical(tmp_path: Path) -> None:
    migration_dir = tmp_path / "migration" / "bulma"
    migration_dir.mkdir(parents=True)
    db_path = migration_dir / "executive_events.db"
    conn = sqlite3.connect(db_path)
    conn.executescript(
        """
        CREATE TABLE bulma_executive_events (
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
        CREATE TABLE bulma_executive_event_meta (
            key TEXT PRIMARY KEY,
            value TEXT NOT NULL
        );
        """
    )
    conn.execute(
        "INSERT INTO bulma_executive_event_meta (key, value) VALUES (?, ?)",
        ("count", json.dumps(1)),
    )
    conn.execute(
        """
        INSERT INTO bulma_executive_events (
            source_line, ts, event_type, entity, status, label,
            goal_thread_id, goal_id, task_id, raw
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """,
        (
            1,
            "2026-04-01T00:00:00+00:00",
            "legacy_goal",
            "goal",
            "done",
            "historical goal",
            None,
            None,
            None,
            "{}",
        ),
    )
    conn.commit()
    conn.close()

    summary = load_executive_event_summary(tmp_path)
    rows = search_executive_events(tmp_path, query="historical")

    assert summary["available"] is True
    assert summary["runtime_authority"] == "historical_archive_only"
    assert rows[0]["archive_kind"] == "historical_openbulma_executive_events"
    assert rows[0]["runtime_authority"] == "historical_archive_only"
