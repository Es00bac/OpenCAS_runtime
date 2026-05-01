"""Quiet task-beacon reducer for build/test fragments.

The beacon is intentionally read-only: it parses ``TaskList.md`` and groups
matching fragments into ``now``, ``next``, and ``later`` with stable rules so
the dashboard can show one compact operator-facing summary instead of a long
fragment feed.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path
from urllib.parse import quote
from typing import Any, Dict, List, Mapping, Optional, Sequence

_TASK_ENTRY_RE = re.compile(r"^- `(?P<task_id>(?:PR|TASK)-[A-Z0-9-]+)` (?P<title>.+)$")
_SECTION_ORDER = {
    "In Progress": 0,
    "Background Context": 1,
    "Next Up / Backlog": 2,
    "Recently Completed": 3,
}
_LIVE_SECTIONS = {
    "In Progress",
    "Next Up / Backlog",
}
_CONTEXT_ONLY_SECTIONS = {
    "Background Context",
}
_HISTORICAL_SECTIONS = {
    "Recently Completed",
    "Completed 2026-04-15 Continuation Slices",
    "Additional Completed Readiness Slices",
    "Earlier Completed Readiness And Capability Slices",
    "Archived Completions",
}
_ACTIVE_STATUSES = {
    "active",
    "executing",
    "in progress",
    "planning",
    "recovering",
    "running",
    "verifying",
}
_PENDING_STATUSES = {
    "blocked",
    "deferred",
    "pending",
    "waiting",
    "needs approval",
    "needs clarification",
}
_COMPLETED_STATUSES = {
    "abandoned",
    "completed",
    "done",
    "failed",
}
_UNKNOWN_STATUSES = {
    "",
    "unknown",
    "unspecified",
    "unset",
}
_STALE_TOKENS = (
    "stale",
    "outdated",
    "superseded",
)
_BLOCKED_STATUSES = {
    "blocked",
    "deferred",
    "waiting",
    "waiting on",
    "waiting for",
    "stalled",
    "on hold",
    "paused",
    "needs approval",
    "needs clarification",
}
_BLOCKED_TOKENS = (
    "blocked by",
    "blocked on",
    "waiting on",
    "waiting for",
    "on hold",
    "stalled",
)
_BLOCKED_PATTERNS = tuple(
    re.compile(rf"\b{re.escape(token)}\b")
    for token in _BLOCKED_TOKENS
)
_BUILD_TEST_TOKENS = (
    "build",
    "build/test",
    "ci",
    "compile",
    "coverage",
    "e2e",
    "lint",
    "pytest",
    "qa",
    "regression",
    "smoke",
    "test",
    "validation",
    "verify",
)
_TRI_STATE_BUCKETS = ("now", "next", "later")
_TRI_STATE_PRIORITY = {bucket: index for index, bucket in enumerate(_TRI_STATE_BUCKETS)}
_TRI_STATE_MODEL = {
    "states": list(_TRI_STATE_BUCKETS),
    "priority_order": list(_TRI_STATE_BUCKETS),
    "mapping_rules": [
        {
            "state": "now",
            "when": [
                "fragment text matches build/test tokens",
                "status is active, executing, in progress, planning, recovering, running, or verifying",
                "section is In Progress",
            ],
        },
        {
            "state": "next",
            "when": [
                "fragment text matches build/test tokens",
                "status is blocked, deferred, pending, waiting, needs approval, or needs clarification",
                "section is Next Up / Backlog or another actionable live section",
                "text says blocked by, blocked on, waiting on, waiting for, on hold, or stalled",
            ],
        },
        {
            "state": "later",
            "when": [
                "fragment text matches build/test tokens",
                "status is queued, completed, abandoned, done, or failed",
                "section is Recently Completed or another historical section",
                "section is Background Context, which is context-only unless the task also has a live actionable duplicate",
            ],
        },
    ],
}

_TRI_STATE_RULES = (
    "build/test fragments only are eligible for the quiet task beacon",
    "now = explicit active fragments that remain consistent after duplicate reconciliation",
    "next = blocked, pending, waiting, or conflicting fragments that still need attention",
    "later = queued, completed, archived, stale, unknown, historical, or background-context-only fragments",
    "one authoritative fragment per task id wins the merge",
    "live sections outrank background and historical duplicates before recency breaks ties",
    "unknown and stale fragments never promote themselves into now",
    "conflicting actionable live states demote to next instead of leaking a false active signal",
    "blocked fragments outrank less severe live duplicates before recency breaks ties",
    "recency = newer duplicate fragments win ties and appear first inside each bucket",
)


@dataclass(frozen=True)
class ParsedTaskFragment:
    task_id: str
    title: str
    section: str
    status: str
    content: str
    order: int
    owner: str = ""


def _coerce_fragment(fragment: Any, *, order: int) -> Optional[ParsedTaskFragment]:
    """Normalize parsed fragments or dict-shaped live fragments into one schema."""
    if isinstance(fragment, ParsedTaskFragment):
        return ParsedTaskFragment(
            task_id=fragment.task_id,
            title=fragment.title,
            section=fragment.section,
            status=fragment.status,
            content=fragment.content,
            order=order,
            owner=fragment.owner,
        )
    if not isinstance(fragment, Mapping):
        return None

    task_id = str(fragment.get("task_id") or fragment.get("stable_id") or "").strip()
    title = str(fragment.get("title") or fragment.get("objective") or fragment.get("summary") or task_id or "Untitled").strip()
    state_hint = str(fragment.get("state") or fragment.get("bucket") or "").strip().lower()
    section = str(fragment.get("section") or fragment.get("bucket") or "").strip()
    if not section and state_hint in _TRI_STATE_BUCKETS:
        section = {
            "now": "In Progress",
            "next": "Next Up / Backlog",
            "later": "Recently Completed",
        }[state_hint]
    if not section:
        section = "In Progress"
    status = str(fragment.get("status") or "").strip().lower()
    if not status:
        status = {
            "now": "in progress",
            "next": "pending",
            "later": "completed",
        }.get(state_hint, "unknown")
    if not status:
        status = "unknown"
    content = str(fragment.get("content") or fragment.get("result") or fragment.get("summary") or title).strip()
    owner = str(fragment.get("owner") or "").strip()
    if not task_id:
        task_id = title
    return ParsedTaskFragment(
        task_id=task_id,
        title=title,
        section=section,
        status=status,
        content=content,
        order=order,
        owner=owner,
    )


def _normalize_fragments(fragments: Sequence[Any], *, starting_order: int = 0) -> List[ParsedTaskFragment]:
    normalized: List[ParsedTaskFragment] = []
    for offset, fragment in enumerate(fragments):
        parsed = _coerce_fragment(fragment, order=starting_order + offset)
        if parsed is not None:
            normalized.append(parsed)
    return normalized


def _empty_beacon(*, source: str | None, error: str | None = None) -> Dict[str, Any]:
    beacon: Dict[str, Any] = {
        "available": False,
        "source": source,
        "bucket_signature": _bucket_signature_for_counts({"now": 0, "next": 0, "later": 0}),
        "headline": "now 0 • next 0 • later 0",
        "counts": {"matched": 0, "now": 0, "next": 0, "later": 0, "total": 0},
        "states": {bucket: [] for bucket in _TRI_STATE_BUCKETS},
        "details": {bucket: [] for bucket in _TRI_STATE_BUCKETS},
        "summary": {
            bucket: {"count": 0, "items": []}
            for bucket in _TRI_STATE_BUCKETS
        },
        "view_model": {
            "buckets": [
                {
                    "state": bucket,
                    "count": 0,
                    "item": None,
                    "items": [],
                }
                for bucket in _TRI_STATE_BUCKETS
            ]
        },
        "rules": list(_TRI_STATE_RULES),
        "model": _TRI_STATE_MODEL,
    }
    if error:
        beacon["error"] = error
    return beacon


def _normalize_text(value: str) -> str:
    return " ".join(str(value or "").strip().lower().split())


def _section_order(section: str) -> int:
    return _SECTION_ORDER.get(section, len(_SECTION_ORDER))


def _fragment_text(fragment: ParsedTaskFragment) -> str:
    return _normalize_text(" ".join([fragment.task_id, fragment.title, fragment.section, fragment.status, fragment.content]))


def _fragment_owner(fragment: ParsedTaskFragment) -> str:
    return str(fragment.owner or "").strip()


def _is_build_test_fragment(fragment: ParsedTaskFragment) -> bool:
    text = _fragment_text(fragment)
    return any(token in text for token in _BUILD_TEST_TOKENS)


def _is_blocked(fragment: ParsedTaskFragment) -> bool:
    status = _normalize_text(fragment.status)
    if status in _BLOCKED_STATUSES:
        return True
    text = _fragment_text(fragment)
    return any(pattern.search(text) for pattern in _BLOCKED_PATTERNS)


def _is_stale(fragment: ParsedTaskFragment) -> bool:
    status = _normalize_text(fragment.status)
    if status in {"stale", "outdated", "superseded"}:
        return True
    text = _fragment_text(fragment)
    return any(re.search(rf"\b{re.escape(token)}\b", text) for token in _STALE_TOKENS)


def _bucket_order(bucket: str) -> int:
    return _TRI_STATE_PRIORITY[bucket]


def _fragment_section_rank(fragment: ParsedTaskFragment) -> int:
    """Prefer live sections over historical sections before other tie-breakers."""
    if fragment.section in _LIVE_SECTIONS:
        return 0
    if fragment.section in _CONTEXT_ONLY_SECTIONS:
        return 1
    if fragment.section in _HISTORICAL_SECTIONS:
        return 2
    return 1


def _fragment_severity_rank(fragment: ParsedTaskFragment) -> int:
    """Rank fragments by actionable severity, with blocked items first."""
    status = _normalize_text(fragment.status)
    # Only explicit blocked statuses should dominate duplicate selection.
    # Free-text blocker phrases still map the fragment into `next`, but they
    # should not outrank a newer duplicate that has the same explicit status.
    if status in _BLOCKED_STATUSES:
        return 0
    if _is_stale(fragment):
        return 4
    if status in _ACTIVE_STATUSES and _is_blocked(fragment):
        return 1
    if status in _ACTIVE_STATUSES:
        return 1
    if status in _PENDING_STATUSES:
        return 2
    if status == "queued":
        return 3
    if status in _COMPLETED_STATUSES:
        return 3
    if status in _UNKNOWN_STATUSES:
        return 4
    return 4


def _fragment_dependency_rank(fragment: ParsedTaskFragment) -> int:
    """Return the dependency urgency used to order fragments inside a bucket."""
    return _fragment_severity_rank(fragment)


def _fragment_bucket(fragment: ParsedTaskFragment) -> str:
    section = fragment.section
    status = _normalize_text(fragment.status)

    if section in _CONTEXT_ONLY_SECTIONS:
        return "later"
    if _is_blocked(fragment):
        return "next"
    if _is_stale(fragment):
        return "later"
    if section in _HISTORICAL_SECTIONS:
        return "later"
    if status in _COMPLETED_STATUSES:
        return "later"
    if status == "queued":
        return "later"
    if status in _ACTIVE_STATUSES:
        return "now"
    if status in _PENDING_STATUSES:
        return "next"
    return "later"


def _fragment_bucket_for_beacon(fragment: ParsedTaskFragment) -> Optional[str]:
    """Return the quiet bucket for publication."""
    section = fragment.section
    status = _normalize_text(fragment.status)

    if section in _CONTEXT_ONLY_SECTIONS:
        return "later"
    if _is_blocked(fragment):
        return "next"
    if _is_stale(fragment):
        return "later"
    if section in _HISTORICAL_SECTIONS:
        return "later"
    if status in _COMPLETED_STATUSES:
        return "later"
    if status == "queued":
        return "later"
    if status in _ACTIVE_STATUSES:
        return "now"
    if status in _PENDING_STATUSES:
        return "next"
    return "later"


def _fragment_priority(fragment: ParsedTaskFragment) -> tuple[int, int, int, int, str]:
    """Return a stable precedence tuple for duplicate task fragments."""
    return (
        _fragment_dependency_rank(fragment),
        _fragment_section_rank(fragment),
        -fragment.order,
        _section_order(fragment.section),
        fragment.task_id,
    )


def _merged_fragment_display_priority(entry: Mapping[str, Any]) -> tuple[int, int, int, int, int, str]:
    """Keep the visible list ordered by final bucket, signal strength, and recency."""
    fragment = entry["fragment"]
    return (
        _bucket_order(str(entry.get("state") or "later")),
        _fragment_dependency_rank(fragment),
        _fragment_section_rank(fragment),
        -fragment.order,
        _section_order(fragment.section),
        fragment.task_id,
    )


def _fragment_state(fragment: ParsedTaskFragment) -> str:
    return _fragment_bucket(fragment)


def _fragment_excerpt(fragment: ParsedTaskFragment) -> str:
    lines = [line.strip() for line in fragment.content.splitlines() if line.strip()]
    for line in lines:
        if line.startswith("- owner:") or line.startswith("- status:") or line.startswith("- result:"):
            continue
        if line.startswith("- "):
            candidate = line[2:].strip()
            if candidate and candidate != fragment.title and not candidate.startswith(f"`{fragment.task_id}`"):
                return candidate[:180]
            continue
        if line:
            return line[:180]
    return fragment.title[:180]


def _fragment_link(fragment: ParsedTaskFragment) -> str:
    return f"/api/operations/tasks/{quote(fragment.task_id, safe='')}"


def _compact_item(fragment: ParsedTaskFragment) -> Dict[str, Any]:
    owner = _fragment_owner(fragment)
    return {
        "task_id": fragment.task_id,
        "stable_id": fragment.task_id,
        "title": fragment.title,
        "section": fragment.section,
        "status": fragment.status,
        "owner": owner,
        "severity": _fragment_severity_rank(fragment),
        "recency": fragment.order,
        "state": _fragment_state(fragment),
        "excerpt": _fragment_excerpt(fragment),
        "link": _fragment_link(fragment),
    }


def _normalized_item(
    fragment: ParsedTaskFragment,
    *,
    merged_count: int,
    state: str,
) -> Dict[str, Any]:
    item = _compact_item(fragment)
    item["state"] = state
    item["merged_count"] = merged_count
    return item


def _merged_fragment_state(fragments: List[ParsedTaskFragment], selected: ParsedTaskFragment) -> str:
    """Return the quiet bucket for a merged task fragment group."""
    states = {_fragment_state(fragment) for fragment in fragments}
    if "next" in states:
        return "next"
    if "now" in states:
        return "now"
    return "later"


def _merge_task_fragments(fragments: List[ParsedTaskFragment]) -> List[Dict[str, Any]]:
    """Group build/test fragments by task id and keep the authoritative fragment first."""
    matched = []
    for fragment in fragments:
        if not _is_build_test_fragment(fragment):
            continue
        matched.append(fragment)
    if not matched:
        return []

    grouped: Dict[str, List[ParsedTaskFragment]] = {}
    for fragment in matched:
        grouped.setdefault(fragment.task_id, []).append(fragment)

    merged: List[Dict[str, Any]] = []
    for group in grouped.values():
        selected = min(group, key=_fragment_priority)
        state = _merged_fragment_state(group, selected)
        merged.append(
            {
                "fragment": selected,
                "state": state,
                "merged_count": len(group),
                "fragments": [
                    _compact_item(fragment)
                    for fragment in sorted(group, key=_fragment_priority)
                ],
            }
        )

    merged.sort(key=_merged_fragment_display_priority)
    return merged


def _aggregate_task_fragments(
    fragments: List[ParsedTaskFragment],
    live_fragments: Sequence[Any] | None = None,
) -> tuple[List[Dict[str, Any]], Dict[str, List[Dict[str, Any]]], Dict[str, List[Dict[str, Any]]]]:
    """Build the merged task-beacon buckets once for all callers."""
    combined = list(fragments)
    if live_fragments:
        combined.extend(_normalize_fragments(live_fragments, starting_order=len(combined)))

    merged_fragments = _merge_task_fragments(combined)
    states: Dict[str, List[Dict[str, Any]]] = {bucket: [] for bucket in _TRI_STATE_BUCKETS}
    details: Dict[str, List[Dict[str, Any]]] = {bucket: [] for bucket in _TRI_STATE_BUCKETS}

    for entry in merged_fragments:
        fragment = entry["fragment"]
        state = entry["state"]
        states[state].append(_normalized_item(fragment, merged_count=entry["merged_count"], state=state))
        details[state].append(
            {
                **_compact_item(fragment),
                "state": state,
                "merged_count": entry["merged_count"],
                "fragments": list(entry["fragments"]),
            }
        )

    return merged_fragments, states, details


def collapse_task_fragments(
    fragments: List[ParsedTaskFragment],
    live_fragments: Sequence[Any] | None = None,
) -> Dict[str, List[Dict[str, Any]]]:
    """Collapse build/test fragments into the quiet now/next/later state buckets."""
    _, states, _ = _aggregate_task_fragments(fragments, live_fragments=live_fragments)
    return states


def _reduce_task_fragments(fragments: List[ParsedTaskFragment]) -> List[ParsedTaskFragment]:
    """Keep one authoritative build/test fragment per task id using stable precedence."""
    return [entry["fragment"] for entry in _merge_task_fragments(fragments)]


def _build_summary(states: Dict[str, List[Dict[str, Any]]], *, limit_per_state: int) -> Dict[str, Dict[str, Any]]:
    limit = max(0, int(limit_per_state))
    summary: Dict[str, Dict[str, Any]] = {}
    for state in ("now", "next", "later"):
        items = states.get(state, [])
        summary[state] = {"count": len(items), "items": items[:limit]}
    return summary


def _build_view_model(states: Dict[str, List[Dict[str, Any]]]) -> Dict[str, Any]:
    """Return the quiet ordered bucket view the dashboard should render."""
    return {
        "buckets": [
            {
                "state": state,
                "count": len(states.get(state, [])),
                "item": (states.get(state) or [None])[0],
                "items": list(states.get(state, [])),
            }
            for state in _TRI_STATE_BUCKETS
        ]
    }


def _headline_for_counts(counts: Dict[str, int]) -> str:
    return " • ".join(f"{state} {counts.get(state, 0)}" for state in _TRI_STATE_BUCKETS)


def _bucket_signature_for_counts(counts: Mapping[str, Any]) -> str:
    return "|".join(f"{state}:{int(counts.get(state, 0) or 0)}" for state in _TRI_STATE_BUCKETS)


def _safe_list(value: Any) -> List[Any]:
    if isinstance(value, list):
        return list(value)
    return []


def _bucket_signature_for_item(item: Any) -> str:
    if not isinstance(item, Mapping) or not item:
        return "none"
    parts = [
        f"stable_id={item.get('stable_id') or item.get('task_id') or ''}",
        f"task_id={item.get('task_id') or ''}",
        f"title={item.get('title') or ''}",
    ]
    return "|".join(parts)


def _bucket_signature_for_view_model(view_model: Mapping[str, Any]) -> str:
    buckets = view_model.get("buckets", []) if isinstance(view_model, Mapping) else []
    bucket_parts = []
    for state, bucket in zip(_TRI_STATE_BUCKETS, list(buckets)[: len(_TRI_STATE_BUCKETS)]):
        if not isinstance(bucket, Mapping):
            bucket_parts.append(f"{state}:0:none")
            continue
        bucket_parts.append(
            f"{state}:{int(bucket.get('count') or 0)}:{_bucket_signature_for_item(bucket.get('item'))}"
        )
    return "|".join(bucket_parts)


def _bucket_signature_for_public_view_model(view_model: Mapping[str, Any]) -> str:
    """Return the signature the public summary renderer uses for merge detection."""
    return _bucket_signature_for_view_model(view_model)


def _read_tasklist_fragments(tasklist_path: Path | str) -> tuple[List[ParsedTaskFragment], Optional[str]]:
    path = Path(tasklist_path)
    if not path.exists():
        return [], None

    fragments: List[ParsedTaskFragment] = []
    current_section = ""
    current_fragment: Optional[dict[str, Any]] = None
    block_lines: List[str] = []

    def flush() -> None:
        nonlocal current_fragment, block_lines
        if not current_fragment:
            block_lines = []
            return
        content = "\n".join(block_lines).strip()
        fragments.append(
            ParsedTaskFragment(
                task_id=current_fragment["task_id"],
                title=current_fragment["title"],
                section=current_section,
                status=str(current_fragment.get("status") or "unknown"),
                content=content,
                order=len(fragments),
                owner=str(current_fragment.get("owner") or "").strip(),
            )
        )
        current_fragment = None
        block_lines = []

    try:
        raw_lines = path.read_text(encoding="utf-8").splitlines()
    except (OSError, UnicodeError) as exc:
        return [], str(exc)

    for raw_line in raw_lines:
        line = raw_line.rstrip()
        if line.startswith("## "):
            flush()
            current_section = line[3:].strip()
            continue

        match = _TASK_ENTRY_RE.match(line.strip())
        if match:
            flush()
            current_fragment = {
                "task_id": match.group("task_id"),
                "title": match.group("title").strip(),
                "status": "unknown",
                "owner": "",
            }
            block_lines = [line.strip()]
            continue

        if current_fragment is None:
            continue

        stripped = line.strip()
        if stripped.startswith("- owner:") or stripped.startswith("- status:") or stripped.startswith("- result:"):
            label, _, value = stripped[2:].partition(":")
            if label == "owner":
                current_fragment["owner"] = value.strip()
                block_lines.append(stripped)
                continue
            if label == "status":
                current_fragment["status"] = value.strip() or "unknown"
            elif label == "result" and value.strip():
                block_lines.append(f"- result: {value.strip()}")
            else:
                block_lines.append(stripped)
            continue

        if stripped.startswith("- "):
            block_lines.append(stripped)
            continue

        if stripped:
            block_lines.append(stripped)

    flush()
    return fragments, None


def parse_tasklist_fragments(tasklist_path: Path | str) -> List[ParsedTaskFragment]:
    """Parse dashboard-parseable task fragments from ``TaskList.md``."""
    fragments, _ = _read_tasklist_fragments(tasklist_path)
    return fragments


def build_task_beacon(
    workspace_root: Path | str | None,
    *,
    limit_per_state: int = 5,
    live_fragments: Sequence[Any] | None = None,
) -> Dict[str, Any]:
    """Return a compact now/next/later beacon for build/test fragments."""
    if workspace_root is None:
        return _empty_beacon(source=None)

    tasklist_path = Path(workspace_root) / "TaskList.md"
    fragments, error = _read_tasklist_fragments(tasklist_path)
    if error:
        return _empty_beacon(source=str(tasklist_path), error=error)
    merged_fragments, states_full, raw_details = _aggregate_task_fragments(
        fragments,
        live_fragments=live_fragments,
    )

    states = {
        key: items[: max(0, int(limit_per_state))]
        for key, items in states_full.items()
    }
    details = {key: list(items) for key, items in raw_details.items()}

    counts = {
        "matched": len(merged_fragments),
        "now": len(states_full["now"]),
        "next": len(states_full["next"]),
        "later": len(states_full["later"]),
        "total": len(merged_fragments),
    }

    return {
        "available": bool(merged_fragments),
        "source": str(tasklist_path),
        "matched_only": bool(merged_fragments),
        "bucket_signature": _bucket_signature_for_view_model(_build_view_model(states_full)),
        "headline": _headline_for_counts(counts),
        "counts": counts,
        "states": states,
        "details": details,
        "summary": _build_summary(states_full, limit_per_state=limit_per_state),
        "view_model": _build_view_model(states_full),
        "rules": list(_TRI_STATE_RULES),
        "model": _TRI_STATE_MODEL,
    }


def public_task_beacon_payload(
    task_beacon: Dict[str, Any],
    *,
    include_details: bool = False,
    include_items: bool = False,
) -> Dict[str, Any]:
    """Return the quiet public task-beacon surface for dashboard consumers."""
    counts = dict(task_beacon.get("counts") or {})
    source_buckets = task_beacon.get("view_model", {}).get("buckets", []) or []

    def _public_item(item: Any) -> Any:
        if not isinstance(item, dict):
            return item
        compact = dict(item)
        compact.pop("excerpt", None)
        compact.pop("fragments", None)
        return compact

    view_model = {
        "buckets": [
            {
                "state": bucket.get("state"),
                "count": int(bucket.get("count") or 0),
                "item": _public_item(bucket.get("item")),
            }
            for bucket in source_buckets
        ]
    }
    payload = {
        "available": bool(task_beacon.get("available")),
        "source": task_beacon.get("source"),
        "matched_only": bool(task_beacon.get("matched_only")),
        "error": task_beacon.get("error"),
        "headline": str(task_beacon.get("headline") or ""),
        "counts": counts,
        "bucket_signature": _bucket_signature_for_public_view_model(view_model),
        "view_model": view_model,
        "rules": list(task_beacon.get("rules") or []),
        "model": dict(task_beacon.get("model") or {}),
    }
    if include_details:
        payload["details"] = dict(task_beacon.get("details") or {})
    return payload


def runtime_task_beacon_fragments(runtime: Any) -> List[Any]:
    """Return all live task-fragment hints surfaced by runtime state."""
    if runtime is None:
        return []
    ctx = getattr(runtime, "ctx", None)
    candidates = (
        getattr(ctx, "task_beacon_fragments", None),
        getattr(ctx, "live_task_fragments", None),
        getattr(ctx, "activity_fragments", None),
        getattr(runtime, "task_beacon_fragments", None),
        getattr(runtime, "live_task_fragments", None),
        getattr(runtime, "activity_fragments", None),
    )
    collected: List[Any] = []
    seen: set[str] = set()

    def _fingerprint(fragment: Any) -> str:
        if isinstance(fragment, ParsedTaskFragment):
            return "|".join(
                [
                    fragment.task_id,
                    fragment.title,
                    fragment.section,
                    fragment.status,
                    fragment.content,
                    fragment.owner,
                ]
            )
        if isinstance(fragment, Mapping):
            return "|".join(
                [
                    str(fragment.get("task_id") or fragment.get("stable_id") or ""),
                    str(fragment.get("title") or fragment.get("objective") or fragment.get("summary") or ""),
                    str(fragment.get("section") or fragment.get("bucket") or ""),
                    str(fragment.get("status") or ""),
                    str(fragment.get("content") or fragment.get("result") or fragment.get("summary") or ""),
                    str(fragment.get("owner") or ""),
                ]
            )
        return repr(fragment)

    for candidate in candidates:
        if candidate:
            if isinstance(candidate, Mapping):
                candidates_to_add = [candidate]
            if isinstance(candidate, (str, bytes)):
                continue
            if isinstance(candidate, (list, tuple)):
                candidates_to_add = list(candidate)
            elif isinstance(candidate, Mapping):
                candidates_to_add = [candidate]
            else:
                candidates_to_add = list(candidate)
            for fragment in candidates_to_add:
                fingerprint = _fingerprint(fragment)
                if fingerprint in seen:
                    continue
                seen.add(fingerprint)
                collected.append(fragment)
    return collected


def build_live_objective_from_task_beacon(
    workspace_root: Path | str | None,
    *,
    live_fragments: Sequence[Any] | None = None,
) -> Optional[str]:
    """Return the first now-state fragment title, if any."""
    beacon = build_task_beacon(workspace_root, limit_per_state=1, live_fragments=live_fragments)
    now_items = beacon.get("states", {}).get("now", [])
    if not now_items:
        return None
    title = str(now_items[0].get("title") or "").strip()
    return title or None
