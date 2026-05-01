"""Helpers for reading the current live objective from repo docs."""

from __future__ import annotations

import re
from pathlib import Path
from typing import Optional

from .task_beacon import build_live_objective_from_task_beacon

_TASK_ENTRY_RE = re.compile(r"^- `(?:PR|TASK)-[A-Z0-9-]+` (?P<title>.+)$")


def read_tasklist_live_objective(workspace_root: Path | str | None) -> Optional[str]:
    """Return the first now-state task title from ``TaskList.md``, if present."""
    if workspace_root is None:
        return None
    tasklist_path = Path(workspace_root) / "TaskList.md"
    if not tasklist_path.exists():
        return None
    beacon_title = build_live_objective_from_task_beacon(workspace_root)
    if beacon_title:
        return beacon_title

    in_progress = False
    for raw_line in tasklist_path.read_text(encoding="utf-8").splitlines():
        line = raw_line.rstrip()
        if line.startswith("## "):
            in_progress = line.strip() == "## In Progress"
            continue
        if not in_progress:
            continue
        if not line.strip():
            continue
        if not line.startswith("- "):
            continue
        match = _TASK_ENTRY_RE.match(line.strip())
        if match:
            return match.group("title").strip()
    return None
