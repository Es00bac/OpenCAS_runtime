"""Maintenance repairs for archived executive insight reframes."""

from __future__ import annotations

import json
import shutil
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from opencas.autonomy.goal_hygiene import hard_reject_goal_reason


@dataclass(frozen=True, slots=True)
class ReframeReactivationReport:
    snapshot_path: Path
    applied: bool
    reactivated: int
    skipped: int
    backup_path: Path | None = None

    def as_dict(self) -> dict[str, Any]:
        return {
            "snapshot_path": str(self.snapshot_path),
            "applied": self.applied,
            "reactivated": self.reactivated,
            "skipped": self.skipped,
            "backup_path": str(self.backup_path) if self.backup_path else None,
        }


def reactivate_archived_low_divergence_reframes(
    snapshot_path: Path | str,
    *,
    apply: bool = False,
    backup_dir: Path | str | None = None,
) -> ReframeReactivationReport:
    """Move archived low-divergence insight reframes back to the parked surface."""
    path = Path(snapshot_path).expanduser()
    payload = _read_json(path)
    archived = list(payload.get("archived_parked_goals") or [])
    archived_meta = dict(payload.get("archived_parked_goal_metadata") or {})
    parked = list(payload.get("parked_goals") or [])
    parked_meta = dict(payload.get("parked_goal_metadata") or {})

    candidates: list[tuple[str, dict[str, Any]]] = []
    skipped = 0
    for goal in archived:
        meta = archived_meta.get(goal)
        if not isinstance(meta, dict):
            continue
        if not _is_low_divergence_reframe(meta):
            continue
        if hard_reject_goal_reason(goal):
            skipped += 1
            continue
        candidates.append((goal, meta))

    if not apply or not candidates:
        return ReframeReactivationReport(
            snapshot_path=path,
            applied=apply,
            reactivated=len(candidates),
            skipped=skipped,
            backup_path=None,
        )

    backup_path = _backup_snapshot(path, backup_dir=backup_dir)
    reactivated_at = datetime.now(timezone.utc).isoformat()
    for goal, meta in candidates:
        if goal not in parked:
            parked.append(goal)
        parked_meta[goal] = {
            "parked_at": reactivated_at,
            "reason": "reactivated_insight_reframe",
            "wake_trigger": "fresh evidence, materially different framing, or direct user request",
            "source_artifact": meta.get("source_artifact") or goal,
            "reactivated_at": reactivated_at,
            "previous_reason": meta.get("reason"),
            "previous_archive_reason": meta.get("archive_reason"),
        }
        if goal in archived:
            archived.remove(goal)
        archived_meta.pop(goal, None)

    payload["parked_goals"] = parked
    payload["parked_goal_metadata"] = parked_meta
    payload["parked_goal_count"] = len(parked)
    payload["archived_parked_goals"] = archived
    payload["archived_parked_goal_metadata"] = archived_meta
    payload["archived_parked_goal_count"] = len(archived)
    payload["updated_at"] = reactivated_at
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    return ReframeReactivationReport(
        snapshot_path=path,
        applied=True,
        reactivated=len(candidates),
        skipped=skipped,
        backup_path=backup_path,
    )


def _is_low_divergence_reframe(meta: dict[str, Any]) -> bool:
    return (
        meta.get("reason") == "low_divergence_reframe"
        or meta.get("archive_reason") == "low_divergence_reframe"
        or meta.get("archive_reason") == "low_divergence_reframe_limit"
    )


def _read_json(path: Path) -> dict[str, Any]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}
    return payload if isinstance(payload, dict) else {}


def _backup_snapshot(path: Path, *, backup_dir: Path | str | None = None) -> Path:
    root = Path(backup_dir).expanduser() if backup_dir else path.parent / "backups"
    root.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    target = root / f"{path.name}.pre-reframe-reactivation-{stamp}"
    shutil.copy2(path, target)
    return target
