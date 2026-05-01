"""Operator-action persistence helpers for the operations API."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from opencas.api.provenance_entry import (
    Action,
    ProvenanceRecordV1,
    build_registry_entry_from_event_context,
    parse_action,
    parse_risk,
    select_registry_sink,
)
from opencas.api.operator_action_store import append_event, read_events

CANONICAL_SCOPE_DELIM = "|"
DEFAULT_SCOPE_KEY = "default"


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


def _split_artifact(value: str) -> Tuple[str, str, str]:
    parts = str(value or "").split(CANONICAL_SCOPE_DELIM)
    if len(parts) == 3:
        target_kind, scope_key, target_id = parts
        return (
            str(target_kind or "unknown"),
            str(scope_key or DEFAULT_SCOPE_KEY),
            str(target_id or "unknown"),
        )
    if len(parts) == 2:
        return (
            str(parts[0] or "unknown"),
            DEFAULT_SCOPE_KEY,
            str(parts[1] or "unknown"),
        )
    if parts:
        return str(parts[0] or "unknown"), "unknown", DEFAULT_SCOPE_KEY
    return "unknown", "unknown", DEFAULT_SCOPE_KEY


# Context fields that may be stored in source_trace and surfaced in projections.
# Only these keys are promoted — arbitrary diagnostic fields are intentionally excluded.
_CONTEXT_PROJECTION_KEYS: frozenset[str] = frozenset({
    "action",
    "input_preview",
    "input_length",
    "observe",
    "url",
    "selector",
    "wait_until",
    "timeout_ms",
})


def _project_operator_action(record: ProvenanceRecordV1) -> Dict[str, Any]:
    target_kind, scope_key, target_id = _split_artifact(record.artifact)
    projected: Dict[str, Any] = {
        "session_id": record.session_id,
        "artifact": record.artifact,
        "action": record.action.value,
        "why": record.why,
        "risk": record.risk.value,
        "target_kind": target_kind,
        "target_id": target_id,
        "scope_key": scope_key,
    }
    if record.source_trace:
        for key in _CONTEXT_PROJECTION_KEYS:
            if key in record.source_trace:
                projected[key] = record.source_trace[key]
    return projected


def append_operator_action(runtime: Any, entry: Dict[str, Any], *, default_path: Path) -> Dict[str, Any]:
    path = resolve_operator_actions_path(runtime, default_path)
    registry_entry = build_registry_entry_from_event_context(
        entry,
        default_action=parse_action(entry.get("action", Action.UPDATE.value), strict=False),
        default_risk=parse_risk(entry.get("risk", "LOW"), strict=False),
    )
    sink = select_registry_sink(runtime, path)
    append_event(sink, registry_entry)
    return _project_operator_action(registry_entry)


def _load_recent_from_registry_store(runtime: Any, *, limit: int) -> Optional[List[Dict[str, Any]]]:
    ctx = getattr(runtime, "ctx", None)
    if ctx is None:
        return None

    for candidate_name in (
        "registry_store",
        "operator_action_store",
        "operator_action_sink",
    ):
        candidate = getattr(ctx, candidate_name, None)
        if candidate is None:
            continue
        list_recent = getattr(candidate, "list_recent", None)
        if not callable(list_recent):
            continue
        recent = list_recent(limit=limit)
        if recent is None:
            return []
        return [_project_operator_action(item) for item in recent if isinstance(item, ProvenanceRecordV1)]
    return None


def load_recent_operator_actions(
    runtime: Any,
    *,
    target_kind: str,
    target_id: str,
    default_path: Path,
    scope_key: Optional[str] = None,
    limit: int = 10,
) -> List[Dict[str, Any]]:
    store_recent = _load_recent_from_registry_store(runtime, limit=limit)
    if store_recent is not None:
        items = store_recent
    else:
        path = resolve_operator_actions_path(runtime, default_path)
        items = []
        try:
            records = read_events(path, limit=None, offset=0)
        except Exception:
            return []
        for record in reversed(records):
            projected = _project_operator_action(record)
            if projected["target_kind"] != target_kind:
                continue
            if projected["target_id"] != target_id:
                continue
            if scope_key is not None and projected["scope_key"] != str(scope_key):
                continue
            items.append(projected)
            if len(items) >= limit:
                break

        return items

    filtered: List[Dict[str, Any]] = []
    for projected in reversed(items):
        if projected["target_kind"] != target_kind:
            continue
        if projected["target_id"] != target_id:
            continue
        if scope_key is not None and projected["scope_key"] != str(scope_key):
            continue
        filtered.append(projected)
        if len(filtered) >= limit:
            break
    return filtered
