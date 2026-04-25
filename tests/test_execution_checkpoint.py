"""Tests for CheckpointManager snapshot/rollback."""

import subprocess
from pathlib import Path

import pytest

from opencas.execution.checkpoint import FileCopyCheckpointManager
from opencas.execution.git_checkpoint import GitCheckpointManager


# --- FileCopyCheckpointManager tests ---

def test_snapshot_and_restore_file(tmp_path):
    scratch = tmp_path / "scratch"
    target = tmp_path / "workspace" / "file.txt"
    target.parent.mkdir(parents=True)
    target.write_text("original")

    cp = FileCopyCheckpointManager(scratch)
    cp.snapshot([str(target)])

    # Modify the file
    target.write_text("modified")
    assert target.read_text() == "modified"

    # Restore
    cp.restore()
    assert target.read_text() == "original"


def test_snapshot_and_discard(tmp_path):
    scratch = tmp_path / "scratch"
    target = tmp_path / "workspace" / "file.txt"
    target.parent.mkdir(parents=True)
    target.write_text("original")

    cp = FileCopyCheckpointManager(scratch)
    cp.snapshot([str(target)])
    # The snapshot stores the full absolute path under __ROOT__
    rel = FileCopyCheckpointManager._safe_rel(target)
    assert (cp.snapshot_dir / rel).exists()

    cp.discard()
    assert not cp.snapshot_dir.exists()


def test_restore_specific_files(tmp_path):
    scratch = tmp_path / "scratch"
    f1 = tmp_path / "a.txt"
    f2 = tmp_path / "b.txt"
    f1.write_text("a")
    f2.write_text("b")

    cp = FileCopyCheckpointManager(scratch)
    cp.snapshot([str(f1), str(f2)])
    f1.write_text("a-mod")
    f2.write_text("b-mod")

    cp.restore([str(f1)])
    assert f1.read_text() == "a"
    assert f2.read_text() == "b-mod"


# --- GitCheckpointManager tests ---

def test_git_checkpoint_inside_repo(tmp_path: Path) -> None:
    subprocess.run(["git", "init"], cwd=tmp_path, check=True, capture_output=True)
    subprocess.run(
        ["git", "config", "user.email", "test@test.com"],
        cwd=tmp_path,
        check=True,
        capture_output=True,
    )
    subprocess.run(
        ["git", "config", "user.name", "Test"],
        cwd=tmp_path,
        check=True,
        capture_output=True,
    )

    file_path = tmp_path / "file.txt"
    file_path.write_text("original")
    subprocess.run(["git", "add", "file.txt"], cwd=tmp_path, check=True, capture_output=True)
    subprocess.run(["git", "commit", "-m", "initial"], cwd=tmp_path, check=True, capture_output=True)

    mgr = GitCheckpointManager(str(tmp_path))
    commit = mgr.snapshot([str(file_path)], message="checkpoint")
    assert commit

    file_path.write_text("modified")
    mgr.restore()
    assert file_path.read_text() == "original"

    mgr.discard()
    assert file_path.read_text() == "original"


def test_git_checkpoint_outside_repo(tmp_path: Path) -> None:
    scratch = tmp_path / "scratch"
    scratch.mkdir()

    mgr = GitCheckpointManager(str(scratch))
    file_path = scratch / "data.txt"
    file_path.write_text("version1")

    commit = mgr.snapshot([str(file_path)])
    assert commit

    file_path.write_text("version2")
    mgr.restore()
    assert file_path.read_text() == "version1"

    mgr.discard()
    assert file_path.read_text() == "version1"
