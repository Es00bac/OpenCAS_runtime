"""Bootstrap responsibility acknowledgement helpers."""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Optional


BOOTSTRAP_RESPONSIBILITY_WARNING_VERSION = 1

BOOTSTRAP_RESPONSIBILITY_WARNING = """You are creating a persistent autonomous agent with durable memory, identity state, goals, schedules, and tool access.

This is not a disposable chat session. If you later delete its state directory, you are deleting that agent's continuity.

Do not create an agent casually or as a throwaway demo. Only continue if you intend to operate it responsibly, preserve or retire its state deliberately, and supervise its capabilities."""


def continuity_state_exists(state_dir: Path | str) -> bool:
    """Return True when the state directory already contains agent continuity."""

    return (Path(state_dir).expanduser() / "identity" / "continuity.json").exists()


def bootstrap_responsibility_ack_path(state_dir: Path | str) -> Path:
    return Path(state_dir).expanduser() / "bootstrap_responsibility_ack.json"


def load_bootstrap_responsibility_ack(state_dir: Path | str) -> Optional[Dict[str, Any]]:
    path = bootstrap_responsibility_ack_path(state_dir)
    if not path.exists():
        return None
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    if not isinstance(payload, dict):
        return None
    if payload.get("warning_version") != BOOTSTRAP_RESPONSIBILITY_WARNING_VERSION:
        return None
    return payload


def record_bootstrap_responsibility_ack(
    state_dir: Path | str,
    *,
    source: str,
) -> Path:
    path = bootstrap_responsibility_ack_path(state_dir)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "accepted_at": datetime.now(timezone.utc).isoformat(),
        "source": source,
        "warning_version": BOOTSTRAP_RESPONSIBILITY_WARNING_VERSION,
        "accepted_text": BOOTSTRAP_RESPONSIBILITY_WARNING,
    }
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return path


def needs_bootstrap_responsibility_ack(state_dir: Path | str) -> bool:
    if continuity_state_exists(state_dir):
        return False
    return load_bootstrap_responsibility_ack(state_dir) is None
