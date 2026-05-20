import json
import sqlite3
import subprocess
import sys
from pathlib import Path

from scripts.quarantine_audit_beliefs import quarantine_audit_beliefs


def _create_tom_fixture(db_path):
    with sqlite3.connect(db_path) as conn:
        conn.execute(
            """
            CREATE TABLE beliefs (
                belief_id TEXT PRIMARY KEY,
                timestamp TEXT NOT NULL,
                subject TEXT NOT NULL,
                predicate TEXT NOT NULL,
                confidence REAL NOT NULL,
                evidence_ids TEXT NOT NULL DEFAULT '[]',
                meta TEXT NOT NULL DEFAULT '{}',
                belief_revision_score REAL NOT NULL DEFAULT 0.0,
                reinforcement_count INTEGER NOT NULL DEFAULT 0,
                last_reinforced TEXT
            )
            """
        )
        conn.executemany(
            """
            INSERT INTO beliefs (
                belief_id, timestamp, subject, predicate, confidence,
                evidence_ids, meta, belief_revision_score, reinforcement_count,
                last_reinforced
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            [
                (
                    "belief-audit",
                    "2026-05-07T02:21:57+00:00",
                    "world",
                    "[e16 audit-only turn 12/15] hypothetical: active model is claude",
                    0.68,
                    json.dumps(["episode:1"]),
                    json.dumps({"source": "conversation_turn", "extractor": "rule_tier_a"}),
                    0.0,
                    1,
                    None,
                ),
                (
                    "belief-live",
                    "2026-05-07T03:00:00+00:00",
                    "user",
                    "prefers grounded evidence",
                    0.8,
                    json.dumps(["episode:2"]),
                    json.dumps({"source": "conversation_turn"}),
                    0.0,
                    1,
                    None,
                ),
            ],
        )


def test_quarantine_audit_beliefs_dry_run_does_not_mutate(tmp_path):
    db_path = tmp_path / "tom.db"
    _create_tom_fixture(db_path)

    result = quarantine_audit_beliefs(db_path, apply=False)

    assert result.matched == 1
    assert result.archived == 0
    assert result.removed_from_beliefs == 0
    with sqlite3.connect(db_path) as conn:
        assert conn.execute("SELECT COUNT(*) FROM beliefs").fetchone()[0] == 2
        assert not _table_exists(conn, "audit_beliefs")


def test_quarantine_audit_beliefs_archives_then_removes_from_beliefs(tmp_path):
    db_path = tmp_path / "tom.db"
    backup_path = tmp_path / "tom.db.backup"
    _create_tom_fixture(db_path)

    result = quarantine_audit_beliefs(db_path, apply=True, backup_path=backup_path)

    assert result.matched == 1
    assert result.archived == 1
    assert result.removed_from_beliefs == 1
    assert backup_path.exists()

    with sqlite3.connect(db_path) as conn:
        remaining = conn.execute("SELECT belief_id, predicate FROM beliefs").fetchall()
        archived = conn.execute(
            "SELECT belief_id, predicate, quarantine_reason FROM audit_beliefs"
        ).fetchall()

    assert remaining == [("belief-live", "prefers grounded evidence")]
    assert archived == [
        (
            "belief-audit",
            "[e16 audit-only turn 12/15] hypothetical: active model is claude",
            "audit_only_marker",
        )
    ]


def test_quarantine_audit_beliefs_cli_runs_from_repo_root(tmp_path):
    db_path = tmp_path / "tom.db"
    _create_tom_fixture(db_path)

    result = subprocess.run(
        [sys.executable, "scripts/quarantine_audit_beliefs.py", str(db_path)],
        cwd=Path(__file__).resolve().parents[1],
        capture_output=True,
        text=True,
        check=False,
    )

    assert result.returncode == 0
    assert "matched=1 archived=0 removed_from_beliefs=0" in result.stdout


def _table_exists(conn, table_name):
    return (
        conn.execute(
            "SELECT COUNT(*) FROM sqlite_master WHERE type='table' AND name=?",
            (table_name,),
        ).fetchone()[0]
        > 0
    )
