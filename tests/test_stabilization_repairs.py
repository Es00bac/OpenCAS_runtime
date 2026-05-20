from __future__ import annotations

import json
import sqlite3
from pathlib import Path

from opencas.maintenance.executive_reframes import (
    reactivate_archived_low_divergence_reframes,
)
from opencas.maintenance.musubi_history import archive_zero_delta_musubi_history


def test_archive_zero_delta_musubi_history_moves_neutral_rows(tmp_path: Path) -> None:
    db_path = tmp_path / "relational.db"
    with sqlite3.connect(db_path) as conn:
        conn.execute(
            """
            CREATE TABLE musubi_history (
                record_id TEXT PRIMARY KEY,
                created_at TEXT NOT NULL,
                delta REAL NOT NULL,
                trigger_event TEXT NOT NULL
            )
            """
        )
        conn.execute(
            "INSERT INTO musubi_history VALUES ('zero', datetime('now'), 0.0, 'cycle_burst_started')"
        )
        conn.execute(
            "INSERT INTO musubi_history VALUES ('nonzero', datetime('now'), 0.2, 'interaction')"
        )
        conn.commit()

    preview = archive_zero_delta_musubi_history(db_path)
    assert preview.archived_rows == 1

    applied = archive_zero_delta_musubi_history(db_path, apply=True, backup_dir=tmp_path)

    assert applied.archived_rows == 1
    assert applied.backup_path is not None and applied.backup_path.exists()
    with sqlite3.connect(db_path) as conn:
        remaining = conn.execute("SELECT record_id FROM musubi_history").fetchall()
        archived = conn.execute(
            "SELECT record_id, archive_reason FROM musubi_history_neutral_archive"
        ).fetchall()
    assert remaining == [("nonzero",)]
    assert archived == [("zero", "zero_delta_non_affective_history")]


def test_reactivate_archived_low_divergence_reframes(tmp_path: Path) -> None:
    snapshot = tmp_path / "executive.json"
    snapshot.write_text(
        json.dumps(
            {
                "active_goals": [],
                "parked_goals": [],
                "parked_goal_metadata": {},
                "archived_parked_goals": [
                    "Define finished-enough naming operation",
                    "repair /package",
                ],
                "archived_parked_goal_metadata": {
                    "Define finished-enough naming operation": {
                        "reason": "low_divergence_reframe",
                        "archive_reason": "low_divergence_reframe_limit",
                    },
                    "repair /package": {
                        "reason": "low_divergence_reframe",
                    },
                },
            }
        ),
        encoding="utf-8",
    )

    preview = reactivate_archived_low_divergence_reframes(snapshot)
    assert preview.reactivated == 1
    assert preview.skipped == 1

    applied = reactivate_archived_low_divergence_reframes(
        snapshot,
        apply=True,
        backup_dir=tmp_path,
    )
    payload = json.loads(snapshot.read_text(encoding="utf-8"))

    assert applied.reactivated == 1
    assert payload["parked_goals"] == ["Define finished-enough naming operation"]
    assert payload["parked_goal_metadata"]["Define finished-enough naming operation"][
        "reason"
    ] == "reactivated_insight_reframe"
    assert payload["archived_parked_goals"] == ["repair /package"]
