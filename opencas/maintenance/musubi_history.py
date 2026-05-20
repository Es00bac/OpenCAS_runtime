"""Maintenance repairs for relational musubi history."""

from __future__ import annotations

import shutil
import sqlite3
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


@dataclass(frozen=True, slots=True)
class MusubiNeutralArchiveReport:
    db_path: Path
    applied: bool
    archived_rows: int
    backup_path: Path | None = None

    def as_dict(self) -> dict[str, Any]:
        return {
            "db_path": str(self.db_path),
            "applied": self.applied,
            "archived_rows": self.archived_rows,
            "backup_path": str(self.backup_path) if self.backup_path else None,
        }


def archive_zero_delta_musubi_history(
    db_path: Path | str,
    *,
    apply: bool = False,
    backup_dir: Path | str | None = None,
) -> MusubiNeutralArchiveReport:
    """Archive neutral musubi history rows that should not count as affective deltas."""
    path = Path(db_path).expanduser()
    if not path.exists():
        return MusubiNeutralArchiveReport(db_path=path, applied=apply, archived_rows=0)

    with sqlite3.connect(path) as conn:
        count = int(
            conn.execute(
                "SELECT COUNT(*) FROM musubi_history WHERE ABS(delta) <= 0.000001"
            ).fetchone()[0]
            or 0
        )
        backup_path = None
        if not apply or count == 0:
            return MusubiNeutralArchiveReport(
                db_path=path,
                applied=apply,
                archived_rows=count,
                backup_path=None,
            )

    backup_path = _backup_db(path, backup_dir=backup_dir)
    archived_at = datetime.now(timezone.utc).isoformat()
    with sqlite3.connect(path) as conn:
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS musubi_history_neutral_archive AS
            SELECT *, '' AS archived_at, '' AS archive_reason
            FROM musubi_history
            WHERE 0
            """
        )
        conn.execute(
            """
            INSERT INTO musubi_history_neutral_archive
            SELECT *, ?, ?
            FROM musubi_history
            WHERE ABS(delta) <= 0.000001
            """,
            (archived_at, "zero_delta_non_affective_history"),
        )
        conn.execute("DELETE FROM musubi_history WHERE ABS(delta) <= 0.000001")
        conn.commit()

    return MusubiNeutralArchiveReport(
        db_path=path,
        applied=True,
        archived_rows=count,
        backup_path=backup_path,
    )


def _backup_db(db_path: Path, *, backup_dir: Path | str | None = None) -> Path:
    root = Path(backup_dir).expanduser() if backup_dir else db_path.parent / "backups"
    root.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    target = root / f"{db_path.name}.pre-neutral-musubi-archive-{stamp}"
    shutil.copy2(db_path, target)
    return target
