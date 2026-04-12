"""Git-based checkpoint manager for snapshot/rollback during task execution."""

from __future__ import annotations

import subprocess
from pathlib import Path
from typing import List, Optional


class GitCheckpointManager:
    """Uses git commits + tags to snapshot and restore file state.

    If the workspace is not already a git repository, a detached local repo
    is initialized inside *scratch_dir / "git_snapshots"*.
    """

    def __init__(self, scratch_dir: Path | str) -> None:
        self.scratch_dir = Path(scratch_dir)
        self.git_dir = self.scratch_dir / "git_snapshots"
        self._repo_root: Optional[Path] = None

    def _repo_root_path(self) -> Path:
        if self._repo_root is None:
            self._repo_root = self._discover_repo_root()
        return self._repo_root

    def _discover_repo_root(self) -> Path:
        self.scratch_dir.mkdir(parents=True, exist_ok=True)
        try:
            result = subprocess.run(
                ["git", "rev-parse", "--show-toplevel"],
                cwd=self.scratch_dir,
                capture_output=True,
                text=True,
                check=True,
            )
            return Path(result.stdout.strip())
        except (subprocess.CalledProcessError, FileNotFoundError):
            # Initialize a detached local repo directly in scratch_dir
            subprocess.run(
                ["git", "init"],
                cwd=str(self.scratch_dir),
                capture_output=True,
                check=False,
            )
            return self.scratch_dir

    def _run_git(self, args: List[str], cwd: Optional[Path] = None) -> str:
        cwd = cwd or self._repo_root_path()
        result = subprocess.run(
            ["git", *args],
            cwd=str(cwd),
            capture_output=True,
            text=True,
        )
        return result.stdout.strip()

    def snapshot(self, file_paths: List[str], message: str = "auto-checkpoint") -> str:
        """Commit the given files and return the commit hash."""
        root = self._repo_root_path()

        # Ensure files are tracked
        for fp in file_paths:
            src = Path(fp).resolve()
            if src.exists():
                rel = src.relative_to(root) if src.is_relative_to(root) else src.name
                self._run_git(["add", str(rel)], cwd=root)

        self._run_git(["commit", "-m", message, "--allow-empty"], cwd=root)
        commit_hash = self._run_git(["rev-parse", "HEAD"], cwd=root)
        tag = f"opencas-checkpoint-{commit_hash[:12]}"
        self._run_git(["tag", "-f", tag, commit_hash], cwd=root)
        return commit_hash

    def restore(self, commit_hash: Optional[str] = None) -> None:
        """Restore files to *commit_hash* or the latest checkpoint tag."""
        root = self._repo_root_path()
        if commit_hash is None:
            commit_hash = self._latest_checkpoint(root)
        if not commit_hash:
            return
        self._run_git(["checkout", commit_hash, "--", "."], cwd=root)
        self._run_git(["reset", "--mixed", commit_hash], cwd=root)

    def discard(self, commit_hash: Optional[str] = None) -> None:
        """Reset to HEAD and remove the checkpoint tag."""
        root = self._repo_root_path()
        self._run_git(["reset", "--hard", "HEAD"], cwd=root)
        if commit_hash is None:
            commit_hash = self._latest_checkpoint(root)
        if commit_hash:
            tag = f"opencas-checkpoint-{commit_hash[:12]}"
            self._run_git(["tag", "-d", tag], cwd=root)

    @staticmethod
    def _latest_checkpoint(root: Path) -> Optional[str]:
        try:
            result = subprocess.run(
                ["git", "tag", "-l", "opencas-checkpoint-*"],
                cwd=str(root),
                capture_output=True,
                text=True,
                check=True,
            )
            tags = result.stdout.strip().splitlines()
            if not tags:
                return None
            # Get the most recently created tag by resolving it to a commit
            latest_result = subprocess.run(
                ["git", "rev-list", "--tags=opencas-checkpoint-*", "--max-count=1"],
                cwd=str(root),
                capture_output=True,
                text=True,
                check=True,
            )
            return latest_result.stdout.strip() or None
        except (subprocess.CalledProcessError, FileNotFoundError):
            return None
