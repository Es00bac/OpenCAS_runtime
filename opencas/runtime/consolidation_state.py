"""Persisted scheduler state for nightly consolidation timing."""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Optional


_CONSOLIDATION_RUNTIME_STATE = "consolidation_runtime_state.json"
CONSOLIDATION_RUNTIME_SUMMARY_KEYS = (
    "clusters_formed",
    "memories_created",
    "commitments_consolidated",
    "commitment_clusters_formed",
    "commitment_work_objects_created",
    "commitments_extracted_from_chat",
    "episodes_pruned",
    "budget_exhausted",
    "budget_reason",
)


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


def consolidation_runtime_state_payload(
    payload: Dict[str, Any],
    *,
    fallback_timestamp: Optional[str] = None,
) -> Dict[str, Any]:
    """Return the persisted restart-safe summary for a consolidation result."""
    timestamp = str(
        payload.get("timestamp")
        or fallback_timestamp
        or datetime.now(timezone.utc).isoformat()
    )
    state: Dict[str, Any] = {
        "last_run_at": timestamp,
        "last_result_id": payload.get("result_id"),
    }
    for key in CONSOLIDATION_RUNTIME_SUMMARY_KEYS:
        if key in payload:
            state[key] = payload.get(key)
    return state


def _parse_state_datetime(value: Any) -> Optional[datetime]:
    raw = str(value or "").strip()
    if not raw:
        return None
    try:
        parsed = datetime.fromisoformat(raw.replace("Z", "+00:00"))
    except Exception:
        return None
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=timezone.utc)
    return parsed.astimezone(timezone.utc)


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
    current_time = current_time.astimezone(timezone.utc)
    state = load_consolidation_runtime_state(state_dir)
    retry_after = _parse_state_datetime(state.get("next_retry_after"))
    if retry_after is not None:
        return max(0.0, retry_after.timestamp() - current_time.timestamp())
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
    last_run_at = _parse_state_datetime(raw_last_run)
    if last_run_at is None:
        return 0.0
    due_at = last_run_at.timestamp() + float(consolidation_interval)
    return max(0.0, due_at - current_time.timestamp())
