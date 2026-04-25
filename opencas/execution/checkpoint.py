"""Checkpoint manager for snapshot/rollback during task execution."""

from __future__ import annotations

import shutil
from pathlib import Path
from typing import List, Optional


class FileCopyCheckpointManager:
    """Copies files to a snapshot directory before execution and restores on failure."""

    def __init__(self, scratch_dir: Path | str) -> None:
        self.scratch_dir = Path(scratch_dir)
        self.snapshot_dir = self.scratch_dir / "snapshot"

    def snapshot(self, file_paths: List[str]) -> None:
        """Copy *file_paths* into the snapshot directory, preserving directory structure."""
        self.discard()
        for fp in file_paths:
            src = Path(fp)
            if not src.exists():
                continue
            # Store under snapshot with absolute path as relative to root
            # Use a safe relative representation
            rel = self._safe_rel(src)
            dst = self.snapshot_dir / rel
            dst.parent.mkdir(parents=True, exist_ok=True)
            if src.is_dir():
                shutil.copytree(src, dst)
            else:
                shutil.copy2(src, dst)

    def restore(self, file_paths: Optional[List[str]] = None) -> None:
        """Copy files from the snapshot directory back to their original locations."""
        if not self.snapshot_dir.exists():
            return
        if file_paths:
            rels = [self._safe_rel(Path(fp)) for fp in file_paths]
            for rel in rels:
                src = self.snapshot_dir / rel
                if src.exists():
                    dst = Path("/" + str(rel).replace("__ROOT__", ""))
                    dst.parent.mkdir(parents=True, exist_ok=True)
                    self._copy_back(src, dst)
        else:
            for src in self.snapshot_dir.rglob("*"):
                if src.is_dir():
                    continue
                rel = src.relative_to(self.snapshot_dir)
                dst = Path("/" + str(rel).replace("__ROOT__", ""))
                dst.parent.mkdir(parents=True, exist_ok=True)
                self._copy_back(src, dst)

    def discard(self) -> None:
        """Remove the snapshot directory."""
        if self.snapshot_dir.exists():
            shutil.rmtree(self.snapshot_dir)

    @staticmethod
    def _safe_rel(path: Path) -> Path:
        """Convert an absolute path into a relative path safe for snapshot storage."""
        p = str(path.resolve())
        if p.startswith("/"):
            p = "__ROOT__" + p
        return Path(p.lstrip("/"))

    @staticmethod
    def _copy_back(src: Path, dst: Path) -> None:
        if src.is_dir():
            if dst.exists():
                shutil.rmtree(dst)
            shutil.copytree(src, dst)
        else:
            shutil.copy2(src, dst)
