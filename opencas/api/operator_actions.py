"""Operator-action persistence helpers for the operations API."""

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any, Dict, List, Optional
from uuid import uuid4


def resolve_operator_actions_path(runtime: Any, default_path: Path) -> Path:
    config = getattr(getattr(runtime, "ctx", None), "config", None)
    state_dir = getattr(config, "state_dir", None)
    if isinstance(state_dir, (str, Path)):
        return Path(state_dir) / "operator_action_history.jsonl"
    return default_path


def truncate_operator_text(value: Optional[str], limit: int = 160) -> str:
    text = str(value or "")
    if len(text) <= limit:
        return text
    return text[: limit - 3] + "..."


def append_operator_action(runtime: Any, entry: Dict[str, Any], *, default_path: Path) -> Dict[str, Any]:
    path = resolve_operator_actions_path(runtime, default_path)
    payload = {
        "event_id": uuid4().hex,
        "timestamp": time.time(),
        **entry,
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(payload, ensure_ascii=True) + "\n")
    return payload


def load_recent_operator_actions(
    runtime: Any,
    *,
    target_kind: str,
    target_id: str,
    default_path: Path,
    scope_key: Optional[str] = None,
    limit: int = 10,
) -> List[Dict[str, Any]]:
    path = resolve_operator_actions_path(runtime, default_path)
    if not path.exists():
        return []

    items: List[Dict[str, Any]] = []
    try:
        lines = path.read_text(encoding="utf-8").splitlines()
    except Exception:
        return []

    for line in reversed(lines):
        if not line.strip():
            continue
        try:
            payload = json.loads(line)
        except Exception:
            continue
        if str(payload.get("target_kind", "") or "") != target_kind:
            continue
        if str(payload.get("target_id", "") or "") != target_id:
            continue
        if scope_key is not None and str(payload.get("scope_key", "") or "default") != str(scope_key):
            continue
        items.append(payload)
        if len(items) >= limit:
            break
    return items
