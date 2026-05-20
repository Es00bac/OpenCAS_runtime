"""Repair stale workspace path references stored in SQLite-backed state.

The cleanup program moved agent-created Writing Project artifacts under the managed
`workspace/` root. These helpers normalize older root-level and legacy-workspace
references without guessing at unrelated file locations.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import re
import sqlite3
from typing import Dict, Iterable, List, Tuple


@dataclass
class WorkspaceReferenceRepairSummary:
    """Summary of reference rewrites applied to one SQLite database."""

    db_path: str
    scanned_fields: int = 0
    updated_fields: int = 0
    tables_touched: List[str] | None = None

    def __post_init__(self) -> None:
        if self.tables_touched is None:
            self.tables_touched = []

    def to_dict(self) -> Dict[str, object]:
        return {
            "db_path": self.db_path,
            "scanned_fields": self.scanned_fields,
            "updated_fields": self.updated_fields,
            "tables_touched": list(self.tables_touched or []),
        }


def normalize_workspace_reference_text(
    text: str,
    *,
    repo_root: Path,
    managed_root: Path,
) -> str:
    """Rewrite known stale Writing Project path patterns to the managed workspace root."""
    repo_root_str = str(repo_root.resolve())
    managed_root_str = str(managed_root.resolve())
    writing_root = f"{managed_root_str}/writing"
    historical_root_name = "Ch" + "ronicles"
    historical_file_prefix = "ch" + "ronicle"
    historical_root = f"{managed_root_str}/{historical_root_name}"

    normalized = text
    prefix_rules: Tuple[Tuple[str, str], ...] = (
        (f"{repo_root_str}/.opencas/legacy-workspace/writing", writing_root),
        (f"{repo_root_str}/writing", writing_root),
        (f"{repo_root_str}/.opencas/legacy-workspace/{historical_root_name}", historical_root),
        (f"{repo_root_str}/{historical_root_name}", historical_root),
        (f"{repo_root_str}/{historical_root_name.lower()}", historical_root),
    )
    for source, replacement in prefix_rules:
        normalized = normalized.replace(source, replacement)

    # Some earlier workflows wrote creative_writing markdown artifacts directly into the
    # repo root. Normalize only the clearly writing-project-scoped filenames here.
    normalized = re.sub(
        rf"{re.escape(repo_root_str)}/((?:story|writing_project)[^/\s`\"']+\.md)",
        rf"{writing_root}/\1",
        normalized,
    )
    normalized = re.sub(
        rf"{re.escape(repo_root_str)}/({historical_file_prefix}[^/\s`\"']+\.md)",
        rf"{historical_root}/\1",
        normalized,
    )
    return normalized


def repair_workspace_references_in_sqlite(
    db_path: Path,
    *,
    repo_root: Path,
    managed_root: Path,
    dry_run: bool = False,
) -> WorkspaceReferenceRepairSummary:
    """Rewrite stale Writing Project path references in one SQLite database."""
    summary = WorkspaceReferenceRepairSummary(db_path=str(db_path))
    if not db_path.exists():
        return summary

    conn = sqlite3.connect(str(db_path))
    try:
        cursor = conn.cursor()
        tables = _list_repairable_tables(cursor)
        for table in tables:
            text_columns = _list_text_columns(cursor, table)
            if not text_columns:
                continue
            for column in text_columns:
                for rowid, original in _iter_candidate_rows(cursor, table, column):
                    summary.scanned_fields += 1
                    updated = normalize_workspace_reference_text(
                        original,
                        repo_root=repo_root,
                        managed_root=managed_root,
                    )
                    if updated == original:
                        continue
                    summary.updated_fields += 1
                    if table not in summary.tables_touched:
                        summary.tables_touched.append(table)
                    if not dry_run:
                        cursor.execute(
                            f'UPDATE "{table}" SET "{column}" = ? WHERE rowid = ?',
                            (updated, rowid),
                        )
        if dry_run:
            conn.rollback()
        else:
            conn.commit()
    finally:
        conn.close()
    return summary


def _list_repairable_tables(cursor: sqlite3.Cursor) -> List[str]:
    rows = cursor.execute(
        """
        SELECT name
        FROM sqlite_master
        WHERE type = 'table'
          AND name NOT LIKE 'sqlite_%'
          AND name NOT LIKE '%fts%'
        ORDER BY name
        """
    ).fetchall()
    return [str(row[0]) for row in rows]


def _list_text_columns(cursor: sqlite3.Cursor, table: str) -> List[str]:
    rows = cursor.execute(f'PRAGMA table_info("{table}")').fetchall()
    return [
        str(row[1])
        for row in rows
        if "TEXT" in str(row[2]).upper()
    ]


def _iter_candidate_rows(
    cursor: sqlite3.Cursor,
    table: str,
    column: str,
) -> Iterable[Tuple[int, str]]:
    historical_root_name = "Ch" + "ronicles"
    historical_file_prefix = "ch" + "ronicle"
    rows = cursor.execute(
        f'''
        SELECT rowid, "{column}"
        FROM "{table}"
        WHERE typeof("{column}") = 'text'
          AND (
                "{column}" LIKE '%Writing Projects%'
             OR "{column}" LIKE '%writing%'
             OR "{column}" LIKE '%story_%'
             OR "{column}" LIKE '%.opencas/legacy-workspace%'
             OR "{column}" LIKE ?
             OR "{column}" LIKE ?
             OR "{column}" LIKE ?
          )
        ''',
        (
            f"%{historical_root_name}%",
            f"%{historical_root_name.lower()}%",
            f"%{historical_file_prefix}_%",
        ),
    ).fetchall()
    for rowid, value in rows:
        if isinstance(value, str):
            yield int(rowid), value
