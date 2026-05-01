"""Canonical provenance event shape for runtime traceability."""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from typing import Any, Dict, Mapping, MutableMapping
from urllib.parse import quote

SCHEMA_VERSION = "1"

_TOP_LEVEL_FIELDS = frozenset(
    {
        "v",
        "event_type",
        "triggering_artifact",
        "triggering_action",
        "source_link",
        "recorded_at",
        "parent_link_id",
        "linked_link_ids",
        "details",
    }
)


class ProvenanceEventError(ValueError):
    """Base error for provenance event validation."""


class ProvenanceEventParseError(ProvenanceEventError):
    """Raised when a persisted provenance event cannot be parsed."""


class ProvenanceEventValidationError(ProvenanceEventError):
    """Raised when a provenance event violates the canonical schema."""


class ProvenanceEventType(str, Enum):
    CHECK = "CHECK"
    MUTATION = "MUTATION"
    BLOCKED = "BLOCKED"


def now_iso8601_ts() -> str:
    """Return a UTC ISO-8601 timestamp with millisecond precision."""

    current = datetime.now(timezone.utc)
    rounded = current.replace(microsecond=int(current.microsecond / 1000) * 1000)
    return rounded.isoformat()


def _require_text(value: Any, *, label: str) -> str:
    text = "" if value is None else str(value).strip()
    if not text:
        raise ProvenanceEventValidationError(f"{label} must be a non-empty string")
    return text


def _require_mapping(value: Any, *, label: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise ProvenanceEventValidationError(f"{label} must be an object")
    return value


def _normalize_linked_ids(value: Any) -> tuple[str, ...]:
    linked: list[str] = []
    seen: set[str] = set()
    if value is None:
        return ()
    candidates = value if isinstance(value, (list, tuple, set, frozenset)) else (value,)
    for candidate in candidates:
        text = str(candidate or "").strip()
        if not text or text in seen:
            continue
        seen.add(text)
        linked.append(text)
    return tuple(linked)


def _require_unknown_free(payload: Mapping[str, Any]) -> None:
    unknown_fields = sorted(set(payload) - _TOP_LEVEL_FIELDS)
    if unknown_fields:
        raise ProvenanceEventValidationError(f"unknown provenance event fields: {', '.join(unknown_fields)}")


def _parse_iso8601(value: Any, *, label: str) -> str:
    if isinstance(value, (int, float)) and not isinstance(value, bool):
        value = datetime.fromtimestamp(float(value), tz=timezone.utc).isoformat()
    if not isinstance(value, str):
        raise ProvenanceEventValidationError(f"{label} must be an ISO-8601 timestamp")
    text = value.strip()
    if not text:
        raise ProvenanceEventValidationError(f"{label} must be a non-empty string")
    try:
        parsed = datetime.fromisoformat(text)
    except ValueError as exc:  # pragma: no cover - defensive
        raise ProvenanceEventValidationError(f"{label} must be an ISO-8601 timestamp") from exc
    if parsed.tzinfo is None or parsed.tzinfo.utcoffset(parsed) is None:
        raise ProvenanceEventValidationError(f"{label} must include a timezone offset")
    return text


def _build_source_link(event_type: ProvenanceEventType, triggering_artifact: str, triggering_action: str) -> str:
    artifact = quote(triggering_artifact, safe="")
    action = quote(triggering_action, safe="")
    return f"opencas://provenance/{event_type.value.lower()}/{artifact}?action={action}"


@dataclass(frozen=True)
class ProvenanceEvent:
    event_type: ProvenanceEventType
    triggering_artifact: str
    triggering_action: str
    source_link: str
    recorded_at: str
    parent_link_id: str | None = None
    linked_link_ids: tuple[str, ...] = field(default_factory=tuple)
    details: Dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_mapping(cls, payload: Mapping[str, Any]) -> "ProvenanceEvent":
        payload = _require_mapping(payload, label="provenance_event")
        _require_unknown_free(payload)
        event_type = ProvenanceEventType(_require_text(payload.get("event_type"), label="event_type").upper())
        triggering_artifact = _require_text(payload.get("triggering_artifact"), label="triggering_artifact")
        triggering_action = _require_text(payload.get("triggering_action"), label="triggering_action")
        source_link_raw = payload.get("source_link")
        source_link = (
            _require_text(source_link_raw, label="source_link")
            if source_link_raw is not None
            else _build_source_link(event_type, triggering_artifact, triggering_action)
        )
        recorded_at = _parse_iso8601(payload.get("recorded_at", now_iso8601_ts()), label="recorded_at")
        parent_link_id_raw = payload.get("parent_link_id")
        parent_link_id = _require_text(parent_link_id_raw, label="parent_link_id") if parent_link_id_raw is not None else None
        linked_link_ids = _normalize_linked_ids(payload.get("linked_link_ids"))
        raw_details = payload.get("details")
        if raw_details is None:
            details: Dict[str, Any] = {}
        else:
            details = dict(_require_mapping(raw_details, label="details"))
        return cls(
            event_type=event_type,
            triggering_artifact=triggering_artifact,
            triggering_action=triggering_action,
            source_link=source_link,
            recorded_at=recorded_at,
            parent_link_id=parent_link_id,
            linked_link_ids=linked_link_ids,
            details=details,
        )

    def to_dict(self) -> Dict[str, Any]:
        payload: Dict[str, Any] = {
            "v": SCHEMA_VERSION,
            "event_type": self.event_type.value,
            "triggering_artifact": self.triggering_artifact,
            "triggering_action": self.triggering_action,
            "source_link": self.source_link,
            "recorded_at": self.recorded_at,
        }
        if self.parent_link_id is not None:
            payload["parent_link_id"] = self.parent_link_id
        if self.linked_link_ids:
            payload["linked_link_ids"] = list(self.linked_link_ids)
        if self.details:
            payload["details"] = dict(self.details)
        return payload


def build_provenance_event(
    *,
    event_type: ProvenanceEventType | str,
    triggering_artifact: str,
    triggering_action: str,
    source_link: str | None = None,
    recorded_at: str | None = None,
    parent_link_id: str | None = None,
    linked_link_ids: Any = None,
    details: Mapping[str, Any] | None = None,
) -> ProvenanceEvent:
    """Build a canonical provenance event with a stable source link."""

    normalized_type = event_type if isinstance(event_type, ProvenanceEventType) else ProvenanceEventType(str(event_type).strip().upper())
    artifact = _require_text(triggering_artifact, label="triggering_artifact")
    action = _require_text(triggering_action, label="triggering_action")
    link = _require_text(source_link, label="source_link") if source_link is not None else _build_source_link(normalized_type, artifact, action)
    payload: Dict[str, Any] = {
        "event_type": normalized_type.value,
        "triggering_artifact": artifact,
        "triggering_action": action,
        "source_link": link,
        "recorded_at": recorded_at or now_iso8601_ts(),
    }
    if parent_link_id is not None:
        payload["parent_link_id"] = _require_text(parent_link_id, label="parent_link_id")
    linked_ids = _normalize_linked_ids(linked_link_ids)
    if linked_ids:
        payload["linked_link_ids"] = list(linked_ids)
    if details is not None:
        payload["details"] = dict(_require_mapping(details, label="details"))
    return ProvenanceEvent.from_mapping(payload)


def validate_provenance_event(event: ProvenanceEvent) -> ProvenanceEvent:
    if not isinstance(event, ProvenanceEvent):
        raise ProvenanceEventValidationError("provenance event must be a ProvenanceEvent")
    return event


def provenance_event_to_dict(event: ProvenanceEvent | Mapping[str, Any]) -> Dict[str, Any]:
    if isinstance(event, ProvenanceEvent):
        return validate_provenance_event(event).to_dict()
    if not isinstance(event, Mapping):
        raise ProvenanceEventValidationError("provenance event must be a mapping or ProvenanceEvent")
    return ProvenanceEvent.from_mapping(event).to_dict()


def serialize_provenance_event(event: ProvenanceEvent | Mapping[str, Any]) -> str:
    """Render one canonical provenance event as newline-safe JSON."""

    return json.dumps(provenance_event_to_dict(event), ensure_ascii=True, separators=(",", ":"))


def parse_provenance_event(raw_line: str) -> ProvenanceEvent:
    """Parse one canonical provenance event from JSON."""

    if not isinstance(raw_line, str):
        raise ProvenanceEventParseError("provenance event must be a string")
    try:
        payload = json.loads(raw_line)
    except json.JSONDecodeError as exc:
        raise ProvenanceEventParseError("invalid provenance event json") from exc
    if not isinstance(payload, dict):
        raise ProvenanceEventParseError("provenance event must be a json object")
    version = str(payload.get("v", SCHEMA_VERSION)).strip()
    if version != SCHEMA_VERSION:
        raise ProvenanceEventParseError(f"unsupported provenance event version: {version}")
    return ProvenanceEvent.from_mapping(payload)


def append_provenance_event(
    record: MutableMapping[str, Any] | None,
    event: ProvenanceEvent | Mapping[str, Any],
    *,
    field: str = "provenance_events",
) -> Dict[str, Any]:
    """Append one canonical provenance event projection to a mutable record."""

    if record is None:
        record = {}
    if not isinstance(record, MutableMapping):
        raise ProvenanceEventValidationError("record must be a mutable mapping")
    projected = dict(record)
    events = list(projected.get(field) or [])
    events.append(provenance_event_to_dict(event))
    projected[field] = events
    return projected


def emit_provenance_event(
    record: MutableMapping[str, Any] | None,
    *,
    event_type: ProvenanceEventType | str,
    triggering_artifact: str,
    triggering_action: str,
    source_link: str | None = None,
    recorded_at: str | None = None,
    parent_link_id: str | None = None,
    linked_link_ids: Any = None,
    details: Mapping[str, Any] | None = None,
    field: str = "provenance_events",
) -> ProvenanceEvent:
    """Build one event and append it to *record* if provided."""

    event = build_provenance_event(
        event_type=event_type,
        triggering_artifact=triggering_artifact,
        triggering_action=triggering_action,
        source_link=source_link,
        recorded_at=recorded_at,
        parent_link_id=parent_link_id,
        linked_link_ids=linked_link_ids,
        details=details,
    )
    if record is not None:
        projected = append_provenance_event(record, event, field=field)
        record.clear()
        record.update(projected)
    return event
