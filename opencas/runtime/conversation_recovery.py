"""Durable markers for recovering interrupted conversation turns."""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional
from uuid import uuid4

from opencas.context import MessageRole

from .lane_metadata import build_assistant_message_meta


_MARKER_ROOT = "conversation_turns"
_ACTIVE_DIR = "active"
_COMPLETED_DIR = "completed"
_PREVIEW_LIMIT = 240


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _marker_dirs(state_dir: Path | str) -> tuple[Path, Path]:
    root = Path(state_dir) / _MARKER_ROOT
    return root / _ACTIVE_DIR, root / _COMPLETED_DIR


def _write_marker(path: Path, payload: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = path.with_suffix(path.suffix + ".tmp")
    tmp_path.write_text(
        json.dumps(payload, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    tmp_path.replace(path)


def start_conversation_turn_marker(
    state_dir: Path | str,
    *,
    session_id: str,
    user_input: str,
    user_meta: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """Create an active marker before beginning a user-visible turn."""
    active_dir, _completed_dir = _marker_dirs(state_dir)
    marker_id = str(uuid4())
    started_at = _now_iso()
    payload: Dict[str, Any] = {
        "marker_id": marker_id,
        "session_id": session_id,
        "phase": "started",
        "user_input": user_input,
        "user_input_preview": user_input[:_PREVIEW_LIMIT],
        "user_meta": dict(user_meta or {}),
        "started_at": started_at,
        "updated_at": started_at,
    }
    _write_marker(active_dir / f"{marker_id}.json", payload)
    return payload


def load_pending_conversation_turn_markers(state_dir: Path | str) -> List[Dict[str, Any]]:
    """Load active markers left behind by an interrupted turn."""
    active_dir, _completed_dir = _marker_dirs(state_dir)
    if not active_dir.exists():
        return []
    markers: List[Dict[str, Any]] = []
    for path in sorted(active_dir.glob("*.json")):
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            continue
        if isinstance(payload, dict):
            markers.append(payload)
    markers.sort(key=lambda item: str(item.get("started_at") or ""))
    return markers


def complete_conversation_turn_marker(
    state_dir: Path | str,
    marker_id: str,
    *,
    outcome: str,
) -> Optional[Dict[str, Any]]:
    """Move an active marker to completed state after the turn is safely visible."""
    active_dir, completed_dir = _marker_dirs(state_dir)
    active_path = active_dir / f"{marker_id}.json"
    if not active_path.exists():
        return None
    try:
        payload = json.loads(active_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        payload = {"marker_id": marker_id}
    completed_at = _now_iso()
    payload.update(
        {
            "phase": "completed",
            "outcome": outcome,
            "completed_at": completed_at,
            "updated_at": completed_at,
        }
    )
    completed_path = completed_dir / f"{marker_id}.json"
    _write_marker(completed_path, payload)
    try:
        active_path.unlink()
    except OSError:
        pass
    return payload


async def recover_interrupted_conversation_turns(runtime: Any) -> int:
    """Surface interrupted turn markers before background work resumes."""
    config = getattr(getattr(runtime, "ctx", None), "config", None)
    state_dir = getattr(config, "state_dir", None)
    context_store = getattr(getattr(runtime, "ctx", None), "context_store", None)
    if state_dir is None or context_store is None:
        return 0

    recovered = 0
    for marker in load_pending_conversation_turn_markers(state_dir):
        marker_id = str(marker.get("marker_id") or "")
        session_id = str(marker.get("session_id") or getattr(config, "session_id", "default") or "default")
        preview = str(marker.get("user_input_preview") or marker.get("user_input") or "").strip()
        if not marker_id:
            continue
        content = (
            "I was interrupted while working on your last request before I could send "
            "a response. The interrupted request was:"
        )
        if preview:
            content += f' "{preview}"'
        content += ". I can pick it back up from here."
        meta = build_assistant_message_meta(
            runtime,
            extra={
                "recovered_interrupted_turn": True,
                "conversation_marker_id": marker_id,
                "recovered_at": _now_iso(),
            },
        )
        await context_store.append(
            session_id,
            MessageRole.ASSISTANT,
            content,
            meta=meta,
        )
        complete_conversation_turn_marker(
            state_dir,
            marker_id,
            outcome="recovered_after_restart",
        )
        trace = getattr(runtime, "_trace", None)
        if callable(trace):
            trace(
                "conversation_turn_recovered",
                {"session_id": session_id, "marker_id": marker_id},
            )
        recovered += 1
    return recovered
