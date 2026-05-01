"""Persisted scheduler state for nightly consolidation timing."""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Optional


_CONSOLIDATION_RUNTIME_STATE = "consolidation_runtime_state.json"


def consolidation_runtime_state_path(state_dir: Path) -> Path:
    """Return the persisted scheduler state path for consolidation timing."""
    return Path(state_dir) / _CONSOLIDATION_RUNTIME_STATE


def load_consolidation_runtime_state(state_dir: Path) -> Dict[str, Any]:
    """Load persisted consolidation timing state from the runtime state directory."""
    path = consolidation_runtime_state_path(state_dir)
    try:
        raw = json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError:
        return {}
    except Exception:
        backup_path = path.with_suffix(".corrupt")
        try:
            path.replace(backup_path)
        except Exception:
            pass
        return {}
    return raw if isinstance(raw, dict) else {}


def persist_consolidation_runtime_state(state_dir: Path, payload: Dict[str, Any]) -> None:
    """Persist consolidation timing state atomically."""
    path = consolidation_runtime_state_path(state_dir)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = path.with_suffix(".tmp")
    tmp_path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    tmp_path.replace(path)


def consolidation_delay_until_due(
    state_dir: Path,
    consolidation_interval: int,
    *,
    now: Optional[datetime] = None,
) -> float:
    """Return seconds until consolidation is due.

    Missing or unreadable state means consolidation has never completed and is
    therefore due immediately instead of waiting a full new uptime window.
    """
    current_time = now or datetime.now(timezone.utc)
    state = load_consolidation_runtime_state(state_dir)
    last_result_id = str(state.get("last_result_id", "") or "").strip().lower()
    if last_result_id.startswith(
        (
            "worker-timeout-",
            "worker-start-failed-",
            "worker-failed-",
            "worker-no-result-",
        )
    ):
        return 0.0
    raw_last_run = str(state.get("last_run_at", "") or "").strip()
    if not raw_last_run:
        return 0.0
    try:
        last_run_at = datetime.fromisoformat(raw_last_run.replace("Z", "+00:00"))
    except Exception:
        return 0.0
    if last_run_at.tzinfo is None:
        last_run_at = last_run_at.replace(tzinfo=timezone.utc)
    due_at = last_run_at.astimezone(timezone.utc).timestamp() + float(consolidation_interval)
    return max(0.0, due_at - current_time.astimezone(timezone.utc).timestamp())
