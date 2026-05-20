"""Schedule intent canonicalization and duplicate merge helpers."""

from __future__ import annotations

import re
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any, Iterable

from .models import ScheduleItem, ScheduleStatus

DEDUPE_META_KEYS = {
    "dedupe_action",
    "survivor_schedule_id",
    "merged_schedule_ids",
    "duplicate_schedule_ids",
}

_WORD_RE = re.compile(r"[^a-z0-9]+")


@dataclass(frozen=True)
class ScheduleIntentKey:
    action: str
    title: str
    recurrence: str
    interval: str
    next_run_bucket: str
    gmail_query: str


def canonical_schedule_intent(item: ScheduleItem) -> ScheduleIntentKey:
    """Build a stable duplicate key from behaviorally material schedule fields."""

    next_run = item.next_run_at or item.start_at
    return ScheduleIntentKey(
        action=item.action.value,
        title=_normalize_text(item.title),
        recurrence=item.recurrence.value,
        interval=_normalize_interval(item.interval_hours),
        next_run_bucket=_hour_bucket(next_run),
        gmail_query=_normalize_text(str(item.meta.get("gmail_query") or "")),
    )


def schedules_are_duplicates(left: ScheduleItem, right: ScheduleItem) -> bool:
    if canonical_schedule_intent(left) == canonical_schedule_intent(right):
        return True
    return _dataannotations_gmail_alerts_overlap(left, right)


def choose_survivor(items: Iterable[ScheduleItem]) -> ScheduleItem:
    return max(items, key=_completeness_score)


def merge_schedule_details(survivor: ScheduleItem, incoming: ScheduleItem) -> bool:
    """Merge richer descriptive fields into the survivor without losing run state."""

    changed = False
    if _is_richer(incoming.objective, survivor.objective):
        survivor.objective = incoming.objective
        changed = True
    if _is_richer(incoming.description, survivor.description):
        survivor.description = incoming.description
        changed = True

    merged_tags = sorted({str(tag) for tag in [*survivor.tags, *incoming.tags] if str(tag).strip()})
    if merged_tags != survivor.tags:
        survivor.tags = merged_tags
        changed = True

    existing_material_meta = _material_meta(survivor.meta)
    merged_meta = _merge_meta(survivor.meta, incoming.meta)
    if merged_meta != existing_material_meta:
        survivor.meta = _with_existing_dedupe_meta(survivor.meta, merged_meta)
        changed = True
    return changed


def attach_dedupe_metadata(
    item: ScheduleItem,
    *,
    dedupe_action: str,
    survivor_schedule_id: str,
    merged_schedule_ids: Iterable[str] = (),
    duplicate_schedule_ids: Iterable[str] = (),
) -> None:
    meta = dict(item.meta)
    meta["dedupe_action"] = dedupe_action
    meta["survivor_schedule_id"] = survivor_schedule_id
    meta["merged_schedule_ids"] = list(merged_schedule_ids)
    meta["duplicate_schedule_ids"] = list(duplicate_schedule_ids)
    item.meta = meta


def is_dataannotations_gmail_alert(item: ScheduleItem) -> bool:
    if item.status != ScheduleStatus.ACTIVE or item.action.value != "gmail_alert":
        return False
    material = " ".join(
        [
            item.title,
            item.description or "",
            item.objective or "",
            " ".join(str(tag) for tag in item.tags),
            str(item.meta.get("gmail_query") or ""),
        ]
    ).lower()
    return (
        "dataannotation" in material
        or "data annotation" in material
        or "data annotations" in material
    )


def _dataannotations_gmail_alerts_overlap(left: ScheduleItem, right: ScheduleItem) -> bool:
    if not (is_dataannotations_gmail_alert(left) and is_dataannotations_gmail_alert(right)):
        return False
    left_key = canonical_schedule_intent(left)
    right_key = canonical_schedule_intent(right)
    if (
        left_key.action != right_key.action
        or left_key.recurrence != right_key.recurrence
        or left_key.interval != right_key.interval
        or left_key.next_run_bucket != right_key.next_run_bucket
    ):
        return False
    return (
        not left_key.gmail_query
        or not right_key.gmail_query
        or left_key.gmail_query == right_key.gmail_query
    )


def _normalize_text(value: str) -> str:
    return _WORD_RE.sub("", value.lower())


def _normalize_interval(value: float | None) -> str:
    if value is None:
        return ""
    return f"{float(value):g}"


def _hour_bucket(value: datetime) -> str:
    bucket = value.astimezone(timezone.utc).replace(minute=0, second=0, microsecond=0)
    return bucket.isoformat()


def _is_richer(candidate: str | None, existing: str | None) -> bool:
    candidate_text = str(candidate or "").strip()
    existing_text = str(existing or "").strip()
    return bool(candidate_text) and len(candidate_text) > len(existing_text)


def _merge_meta(existing: dict[str, Any], incoming: dict[str, Any]) -> dict[str, Any]:
    merged = _material_meta(existing)
    incoming_clean = _material_meta(incoming)

    for key, value in incoming_clean.items():
        if key == "seen_message_ids":
            merged[key] = _merge_seen_ids(merged.get(key), value)
        elif key not in merged or _is_empty(merged[key]):
            merged[key] = value

    if "seen_message_ids" in existing or "seen_message_ids" in incoming:
        merged["seen_message_ids"] = _merge_seen_ids(existing.get("seen_message_ids"), incoming.get("seen_message_ids"))
    return merged


def _material_meta(meta: dict[str, Any]) -> dict[str, Any]:
    return {key: value for key, value in meta.items() if key not in DEDUPE_META_KEYS}


def _with_existing_dedupe_meta(existing: dict[str, Any], material: dict[str, Any]) -> dict[str, Any]:
    merged = dict(material)
    for key in DEDUPE_META_KEYS:
        if key in existing:
            merged[key] = existing[key]
    return merged


def _merge_seen_ids(existing: Any, incoming: Any) -> list[str]:
    seen: list[str] = []
    for value in _as_list(existing) + _as_list(incoming):
        text = str(value)
        if text and text not in seen:
            seen.append(text)
    return seen


def _as_list(value: Any) -> list[Any]:
    if value is None:
        return []
    if isinstance(value, list):
        return value
    if isinstance(value, tuple):
        return list(value)
    return [value]


def _is_empty(value: Any) -> bool:
    return value is None or value == "" or value == [] or value == {}


def _completeness_score(item: ScheduleItem) -> tuple[int, datetime]:
    meta = {key: value for key, value in item.meta.items() if key not in DEDUPE_META_KEYS}
    score = 0
    score += min(len(str(item.objective or "").strip()), 300)
    score += min(len(str(item.description or "").strip()), 200) // 2
    score += len([tag for tag in item.tags if str(tag).strip()]) * 5
    score += len([key for key, value in meta.items() if not _is_empty(value)]) * 8
    if meta.get("gmail_query"):
        score += 25
    if meta.get("seen_message_ids"):
        score += 20 + len(_as_list(meta.get("seen_message_ids")))
    return score, item.created_at
