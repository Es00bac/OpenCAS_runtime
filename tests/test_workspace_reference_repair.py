"""Tests for stale workspace reference repair helpers."""

from pathlib import Path
import sqlite3

from opencas.maintenance import (
    normalize_workspace_reference_text,
    repair_workspace_references_in_sqlite,
)


def test_normalize_workspace_reference_text_moves_legacy_writing_into_managed_root() -> None:
    repo_root = Path("/tmp/opencas-public-fixture")
    managed_root = repo_root / "workspace"

    original = (
        'tool fs_read_file: {"file_path": "/tmp/opencas-public-fixture/.opencas/legacy-workspace/writing/4246/story_4246.md"} '
        'and "/tmp/opencas-public-fixture/writing/2146/story_2146_outline.md" '
        'plus "/tmp/opencas-public-fixture/story_4246_review_notes.md"'
    )

    normalized = normalize_workspace_reference_text(
        original,
        repo_root=repo_root,
        managed_root=managed_root,
    )

    assert "/tmp/opencas-public-fixture/.opencas/legacy-workspace/writing" not in normalized
    assert "/tmp/opencas-public-fixture/writing/2146" not in normalized
    assert "/tmp/opencas-public-fixture/workspace/writing/4246/story_4246.md" in normalized
    assert "/tmp/opencas-public-fixture/workspace/writing/2146/story_2146_outline.md" in normalized
    assert "/tmp/opencas-public-fixture/workspace/writing/story_4246_review_notes.md" in normalized


def test_normalize_workspace_reference_text_moves_historical_series_paths() -> None:
    repo_root = Path("/tmp/opencas-public-fixture")
    managed_root = repo_root / "workspace"
    old_dir = "Ch" + "ronicles"
    file_prefix = "ch" + "ronicle"
    original = f'"{repo_root}/{old_dir}/2146/{file_prefix}_2146_outline.md"'

    normalized = normalize_workspace_reference_text(
        original,
        repo_root=repo_root,
        managed_root=managed_root,
    )

    assert f"{repo_root}/{old_dir}/2146" not in normalized
    assert f"{managed_root}/{old_dir}/2146/{file_prefix}_2146_outline.md" in normalized


def test_repair_workspace_references_in_sqlite_updates_text_columns(tmp_path: Path) -> None:
    db_path = tmp_path / "memory.db"
    conn = sqlite3.connect(db_path)
    try:
        conn.execute("create table episodes (content text)")
        conn.execute("create table work_objects (meta text)")
        conn.execute(
            "insert into episodes(content) values (?)",
            ('tool fs_write_file: {"file_path": "/tmp/opencas-public-fixture/writing/2146/story_2146_outline.md"}',),
        )
        conn.execute(
            "insert into work_objects(meta) values (?)",
            ('{"artifact": "/tmp/opencas-public-fixture/.opencas/legacy-workspace/writing/4246/story_4246.md"}',),
        )
        old_dir = "Ch" + "ronicles"
        file_prefix = "ch" + "ronicle"
        conn.execute(
            "insert into episodes(content) values (?)",
            (f'{{"artifact": "/tmp/opencas-public-fixture/{old_dir}/2146/{file_prefix}_2146_outline.md"}}',),
        )
        conn.commit()
    finally:
        conn.close()

    summary = repair_workspace_references_in_sqlite(
        db_path,
        repo_root=Path("/tmp/opencas-public-fixture"),
        managed_root=Path("/tmp/opencas-public-fixture/workspace"),
    )

    assert summary.updated_fields == 3
    assert set(summary.tables_touched or []) == {"episodes", "work_objects"}

    conn = sqlite3.connect(db_path)
    try:
        rows = conn.execute("select content from episodes").fetchall()
        meta_rows = conn.execute("select meta from work_objects").fetchall()
    finally:
        conn.close()

    old_dir = "Ch" + "ronicles"
    file_prefix = "ch" + "ronicle"
    assert "/tmp/opencas-public-fixture/workspace/writing/2146/story_2146_outline.md" in rows[0][0]
    assert f"/tmp/opencas-public-fixture/workspace/{old_dir}/2146/{file_prefix}_2146_outline.md" in rows[1][0]
    assert "/tmp/opencas-public-fixture/workspace/writing/4246/story_4246.md" in meta_rows[0][0]
