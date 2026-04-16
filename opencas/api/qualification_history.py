"""Rerun-history persistence helpers for qualification artifacts."""

from __future__ import annotations

import json
from typing import Any, Dict, List, Optional

from opencas.api.qualification_models import QualificationArtifactsPaths


def append_qualification_rerun_history(paths: QualificationArtifactsPaths, entry: Dict[str, Any]) -> None:
    path = paths.rerun_history_path
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(entry, ensure_ascii=True) + "\n")


def load_latest_rerun_history_by_label(
    paths: QualificationArtifactsPaths,
    limit: int = 200,
) -> tuple[Dict[str, Dict[str, Any]], Dict[str, Dict[str, Any]]]:
    path = paths.rerun_history_path
    if not path.exists():
        return {}, {}

    latest_requests: Dict[str, Dict[str, Any]] = {}
    latest_completions: Dict[str, Dict[str, Any]] = {}
    try:
        lines = path.read_text(encoding="utf-8").splitlines()
    except Exception:
        return {}, {}

    for line in reversed(lines[-limit:]):
        if not line.strip():
            continue
        try:
            payload = json.loads(line)
        except Exception:
            continue
        event = str(payload.get("event", "requested") or "requested")
        if event == "completed":
            labels = [str(item) for item in payload.get("labels", []) if str(item)]
            for label in labels:
                if label not in latest_completions:
                    latest_completions[label] = payload
            continue

        label = str(payload.get("label", "") or "")
        if not label or label in latest_requests:
            continue
        latest_requests[label] = payload
    return latest_requests, latest_completions


def load_recent_rerun_history(paths: QualificationArtifactsPaths, limit: int = 10) -> List[Dict[str, Any]]:
    path = paths.rerun_history_path
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
        event = str(payload.get("event", "requested") or "requested")
        timestamp = payload.get("completed_at") if event == "completed" else payload.get("requested_at")
        labels = payload.get("labels")
        if labels is None:
            label = payload.get("label")
            labels = [label] if label else []
        items.append(
            {
                **payload,
                "event": event,
                "event_time": timestamp,
                "labels": [str(item) for item in labels if str(item)],
            }
        )
        if len(items) >= limit:
            break
    return items


def load_all_rerun_history(paths: QualificationArtifactsPaths, limit: int = 400) -> List[Dict[str, Any]]:
    path = paths.rerun_history_path
    if not path.exists():
        return []

    items: List[Dict[str, Any]] = []
    try:
        lines = path.read_text(encoding="utf-8").splitlines()
    except Exception:
        return []

    for line in reversed(lines[-limit:]):
        if not line.strip():
            continue
        try:
            payload = json.loads(line)
        except Exception:
            continue
        event = str(payload.get("event", "requested") or "requested")
        labels = payload.get("labels")
        if labels is None:
            label = payload.get("label")
            labels = [label] if label else []
        items.append({**payload, "event": event, "labels": [str(item) for item in labels if str(item)]})
    return items


def find_rerun_history_by_request_id(
    paths: QualificationArtifactsPaths,
    request_id: Optional[str],
) -> tuple[Optional[Dict[str, Any]], Optional[Dict[str, Any]]]:
    if not request_id:
        return None, None
    request_entry: Optional[Dict[str, Any]] = None
    completion_entry: Optional[Dict[str, Any]] = None
    for item in load_all_rerun_history(paths):
        if str(item.get("request_id", "") or "") != str(request_id):
            continue
        if item.get("event") == "completed" and completion_entry is None:
            completion_entry = item
        elif item.get("event") == "requested" and request_entry is None:
            request_entry = item
        if request_entry is not None and completion_entry is not None:
            break
    return request_entry, completion_entry


def find_rerun_history_for_run(
    paths: QualificationArtifactsPaths,
    run_id: str,
    label: Optional[str] = None,
) -> tuple[Optional[Dict[str, Any]], Optional[Dict[str, Any]]]:
    request_entry: Optional[Dict[str, Any]] = None
    completion_entry: Optional[Dict[str, Any]] = None
    for item in load_all_rerun_history(paths):
        if item.get("event") != "completed":
            continue
        generated_ids = [str(value) for value in item.get("generated_run_ids", []) if str(value)]
        latest_run_id = str(item.get("latest_run_id", "") or "")
        labels = [str(value) for value in item.get("labels", []) if str(value)]
        if run_id not in generated_ids and run_id != latest_run_id:
            continue
        if label and labels and label not in labels:
            continue
        completion_entry = item
        request_entry, _ = find_rerun_history_by_request_id(paths, str(item.get("request_id", "") or ""))
        break
    return request_entry, completion_entry
