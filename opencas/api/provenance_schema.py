"""Typed provenance record schema for trust-critical registry state."""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from typing import Any, Dict, Iterable, Mapping, Sequence, Tuple

SCHEMA_VERSION = "2"

_TOP_LEVEL_FIELDS = frozenset(
    {
        "v",
        "checked_items",
        "changes",
        "pending_work",
        "actor_identity",
        "timestamps",
        "sources",
        "verification_status",
    }
)
_ACTOR_FIELDS = frozenset({"actor_id", "kind", "display_name", "session_id"})
_SOURCE_FIELDS = frozenset({"source_id", "kind", "label", "uri"})
_TIMESTAMP_FIELDS = frozenset({"recorded_at", "checked_at", "verified_at", "updated_at"})
_CHECKED_ITEM_FIELDS = frozenset({"item_id", "status", "source_ids", "checked_at", "label", "notes"})
_CHANGE_FIELDS = frozenset({"change_id", "kind", "target", "source_ids", "changed_at", "summary", "before", "after"})
_PENDING_WORK_FIELDS = frozenset({"work_id", "status", "source_ids", "summary", "owner", "due_at"})


class ProvenanceSchemaError(ValueError):
    """Base error for canonical provenance schema validation."""


class ProvenanceParseError(ProvenanceSchemaError):
    """Raised when a persisted record cannot be parsed."""


class ProvenanceValidationError(ProvenanceSchemaError):
    """Raised when a record violates schema invariants."""


class ActorKind(str, Enum):
    HUMAN = "HUMAN"
    AGENT = "AGENT"
    SYSTEM = "SYSTEM"
    REVIEWER = "REVIEWER"


class SourceKind(str, Enum):
    EVENT = "EVENT"
    FILE = "FILE"
    URL = "URL"
    LOG = "LOG"
    MEMORY = "MEMORY"


class CheckedItemStatus(str, Enum):
    PASS = "PASS"
    FAIL = "FAIL"
    WARN = "WARN"


class ChangeKind(str, Enum):
    ADD = "ADD"
    UPDATE = "UPDATE"
    REMOVE = "REMOVE"
    RECONCILE = "RECONCILE"


class PendingWorkStatus(str, Enum):
    OPEN = "OPEN"
    IN_PROGRESS = "IN_PROGRESS"
    BLOCKED = "BLOCKED"
    DONE = "DONE"


class VerificationStatus(str, Enum):
    PENDING = "PENDING"
    CHECKED = "CHECKED"
    NEEDS_REVIEW = "NEEDS_REVIEW"
    VERIFIED = "VERIFIED"
    BLOCKED = "BLOCKED"
    REJECTED = "REJECTED"


_VERIFICATION_TRANSITIONS: Dict[VerificationStatus, frozenset[VerificationStatus]] = {
    VerificationStatus.PENDING: frozenset(
        {
            VerificationStatus.CHECKED,
            VerificationStatus.NEEDS_REVIEW,
            VerificationStatus.BLOCKED,
            VerificationStatus.REJECTED,
        }
    ),
    VerificationStatus.CHECKED: frozenset(
        {
            VerificationStatus.NEEDS_REVIEW,
            VerificationStatus.VERIFIED,
            VerificationStatus.BLOCKED,
            VerificationStatus.REJECTED,
        }
    ),
    VerificationStatus.NEEDS_REVIEW: frozenset(
        {
            VerificationStatus.CHECKED,
            VerificationStatus.VERIFIED,
            VerificationStatus.BLOCKED,
            VerificationStatus.REJECTED,
        }
    ),
    VerificationStatus.VERIFIED: frozenset({VerificationStatus.VERIFIED}),
    VerificationStatus.BLOCKED: frozenset({VerificationStatus.NEEDS_REVIEW, VerificationStatus.REJECTED}),
    VerificationStatus.REJECTED: frozenset({VerificationStatus.REJECTED}),
}


def _ensure_nonempty_text(value: Any, *, label: str, max_length: int | None = None) -> str:
    text = "" if value is None else str(value).strip()
    if not text:
        raise ProvenanceValidationError(f"{label} must be a non-empty string")
    if max_length is not None and len(text) > max_length:
        raise ProvenanceValidationError(f"{label} exceeds the maximum length of {max_length}")
    return text


def _parse_iso8601(value: Any, *, label: str) -> str:
    if isinstance(value, (int, float)) and not isinstance(value, bool):
        value = datetime.fromtimestamp(float(value), tz=timezone.utc).isoformat()
    if not isinstance(value, str):
        raise ProvenanceValidationError(f"{label} must be an ISO-8601 timestamp")
    text = value.strip()
    if not text:
        raise ProvenanceValidationError(f"{label} must be a non-empty string")
    try:
        parsed = datetime.fromisoformat(text)
    except ValueError as exc:  # pragma: no cover - defensive; tests cover invalid strings
        raise ProvenanceValidationError(f"{label} must be an ISO-8601 timestamp") from exc
    if parsed.tzinfo is None or parsed.tzinfo.utcoffset(parsed) is None:
        raise ProvenanceValidationError(f"{label} must include a timezone offset")
    return text


def _coerce_enum(enum_type: type[Enum], value: Any, *, label: str) -> Enum:
    if isinstance(value, enum_type):
        return value
    text = _ensure_nonempty_text(value, label=label)
    try:
        return enum_type(text)
    except ValueError as exc:
        try:
            return enum_type(text.upper())
        except ValueError:
            raise ProvenanceValidationError(f"unsupported {label}: {value}") from exc


def _coerce_sequence(value: Any, *, label: str) -> Tuple[Any, ...]:
    if value is None:
        raise ProvenanceValidationError(f"{label} must be a list")
    if isinstance(value, tuple):
        return value
    if isinstance(value, list):
        return tuple(value)
    raise ProvenanceValidationError(f"{label} must be a list")


def _require_mapping(value: Any, *, label: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise ProvenanceValidationError(f"{label} must be an object")
    return value


def _require_mapping_fields(payload: Mapping[str, Any], *, label: str, allowed: frozenset[str]) -> None:
    unknown_fields = sorted(set(payload) - allowed)
    if unknown_fields:
        raise ProvenanceValidationError(f"unknown {label} fields: {', '.join(unknown_fields)}")


def _require_sequence_field(payload: Mapping[str, Any], field: str) -> Tuple[Any, ...]:
    if field not in payload:
        raise ProvenanceValidationError(f"{field} must be a list")
    return _coerce_sequence(payload.get(field), label=field)


def _require_object_field(payload: Mapping[str, Any], field: str) -> Mapping[str, Any]:
    if field not in payload:
        raise ProvenanceValidationError(f"{field} must be an object")
    return _require_mapping(payload.get(field), label=field)


def _require_scalar_field(payload: Mapping[str, Any], field: str) -> Any:
    if field not in payload:
        raise ProvenanceValidationError(f"{field} is required")
    return payload.get(field)


def _transition_targets(status: VerificationStatus) -> frozenset[VerificationStatus]:
    return _VERIFICATION_TRANSITIONS[status]


def can_transition_verification_status(
    current: VerificationStatus | str,
    next_status: VerificationStatus | str,
) -> bool:
    current_status = _coerce_enum(VerificationStatus, current, label="verification_status")
    next_value = _coerce_enum(VerificationStatus, next_status, label="verification_status")
    return next_value in _transition_targets(current_status)


def transition_verification_status(
    current: VerificationStatus | str,
    next_status: VerificationStatus | str,
) -> VerificationStatus:
    current_status = _coerce_enum(VerificationStatus, current, label="verification_status")
    next_value = _coerce_enum(VerificationStatus, next_status, label="verification_status")
    if next_value not in _transition_targets(current_status):
        raise ProvenanceValidationError(
            f"illegal verification status transition: {current_status.value} -> {next_value.value}"
        )
    return next_value


@dataclass(frozen=True)
class SourceReference:
    source_id: str
    kind: SourceKind
    label: str | None = None
    uri: str | None = None

    @classmethod
    def from_mapping(cls, payload: Mapping[str, Any]) -> "SourceReference":
        payload = _require_mapping(payload, label="source")
        _require_mapping_fields(payload, label="source", allowed=_SOURCE_FIELDS)
        return cls(
            source_id=_ensure_nonempty_text(payload.get("source_id"), label="source_id"),
            kind=_coerce_enum(SourceKind, payload.get("kind"), label="kind"),
            label=_ensure_nonempty_text(payload.get("label"), label="label") if payload.get("label") is not None else None,
            uri=_ensure_nonempty_text(payload.get("uri"), label="uri") if payload.get("uri") is not None else None,
        )

    def to_dict(self) -> Dict[str, Any]:
        payload: Dict[str, Any] = {"source_id": self.source_id, "kind": self.kind.value}
        if self.label is not None:
            payload["label"] = self.label
        if self.uri is not None:
            payload["uri"] = self.uri
        return payload


@dataclass(frozen=True)
class ActorIdentity:
    actor_id: str
    kind: ActorKind
    display_name: str | None = None
    session_id: str | None = None

    @classmethod
    def from_mapping(cls, payload: Mapping[str, Any]) -> "ActorIdentity":
        payload = _require_mapping(payload, label="actor_identity")
        _require_mapping_fields(payload, label="actor_identity", allowed=_ACTOR_FIELDS)
        return cls(
            actor_id=_ensure_nonempty_text(payload.get("actor_id"), label="actor_id"),
            kind=_coerce_enum(ActorKind, payload.get("kind"), label="kind"),
            display_name=_ensure_nonempty_text(payload.get("display_name"), label="display_name")
            if payload.get("display_name") is not None
            else None,
            session_id=_ensure_nonempty_text(payload.get("session_id"), label="session_id")
            if payload.get("session_id") is not None
            else None,
        )

    def to_dict(self) -> Dict[str, Any]:
        payload: Dict[str, Any] = {"actor_id": self.actor_id, "kind": self.kind.value}
        if self.display_name is not None:
            payload["display_name"] = self.display_name
        if self.session_id is not None:
            payload["session_id"] = self.session_id
        return payload


@dataclass(frozen=True)
class TimestampBundle:
    recorded_at: str
    checked_at: str | None = None
    verified_at: str | None = None
    updated_at: str | None = None

    @classmethod
    def from_mapping(cls, payload: Mapping[str, Any]) -> "TimestampBundle":
        payload = _require_mapping(payload, label="timestamps")
        _require_mapping_fields(payload, label="timestamps", allowed=_TIMESTAMP_FIELDS)
        return cls(
            recorded_at=_parse_iso8601(payload.get("recorded_at"), label="recorded_at"),
            checked_at=_parse_iso8601(payload.get("checked_at"), label="checked_at")
            if payload.get("checked_at") is not None
            else None,
            verified_at=_parse_iso8601(payload.get("verified_at"), label="verified_at")
            if payload.get("verified_at") is not None
            else None,
            updated_at=_parse_iso8601(payload.get("updated_at"), label="updated_at")
            if payload.get("updated_at") is not None
            else None,
        )

    def to_dict(self) -> Dict[str, Any]:
        payload: Dict[str, Any] = {"recorded_at": self.recorded_at}
        if self.checked_at is not None:
            payload["checked_at"] = self.checked_at
        if self.verified_at is not None:
            payload["verified_at"] = self.verified_at
        if self.updated_at is not None:
            payload["updated_at"] = self.updated_at
        return payload


@dataclass(frozen=True)
class CheckedItem:
    item_id: str
    status: CheckedItemStatus
    source_ids: Tuple[str, ...]
    checked_at: str
    label: str | None = None
    notes: str | None = None

    @classmethod
    def from_mapping(cls, payload: Mapping[str, Any]) -> "CheckedItem":
        payload = _require_mapping(payload, label="checked_item")
        _require_mapping_fields(payload, label="checked_item", allowed=_CHECKED_ITEM_FIELDS)
        source_ids = tuple(_ensure_nonempty_text(item, label="source_id") for item in _require_sequence_field(payload, "source_ids"))
        if not source_ids:
            raise ProvenanceValidationError("checked item must reference at least one source")
        return cls(
            item_id=_ensure_nonempty_text(payload.get("item_id"), label="item_id"),
            status=_coerce_enum(CheckedItemStatus, payload.get("status"), label="status"),
            source_ids=source_ids,
            checked_at=_parse_iso8601(payload.get("checked_at"), label="checked_at"),
            label=_ensure_nonempty_text(payload.get("label"), label="label") if payload.get("label") is not None else None,
            notes=_ensure_nonempty_text(payload.get("notes"), label="notes") if payload.get("notes") is not None else None,
        )

    def to_dict(self) -> Dict[str, Any]:
        payload: Dict[str, Any] = {
            "item_id": self.item_id,
            "status": self.status.value,
            "source_ids": list(self.source_ids),
            "checked_at": self.checked_at,
        }
        if self.label is not None:
            payload["label"] = self.label
        if self.notes is not None:
            payload["notes"] = self.notes
        return payload


@dataclass(frozen=True)
class ChangeRecord:
    change_id: str
    kind: ChangeKind
    target: str
    source_ids: Tuple[str, ...]
    changed_at: str
    summary: str | None = None
    before: str | None = None
    after: str | None = None

    @classmethod
    def from_mapping(cls, payload: Mapping[str, Any]) -> "ChangeRecord":
        payload = _require_mapping(payload, label="change")
        _require_mapping_fields(payload, label="change", allowed=_CHANGE_FIELDS)
        source_ids = tuple(_ensure_nonempty_text(item, label="source_id") for item in _require_sequence_field(payload, "source_ids"))
        if not source_ids:
            raise ProvenanceValidationError("change must reference at least one source")
        return cls(
            change_id=_ensure_nonempty_text(payload.get("change_id"), label="change_id"),
            kind=_coerce_enum(ChangeKind, payload.get("kind"), label="kind"),
            target=_ensure_nonempty_text(payload.get("target"), label="target"),
            source_ids=source_ids,
            changed_at=_parse_iso8601(payload.get("changed_at"), label="changed_at"),
            summary=_ensure_nonempty_text(payload.get("summary"), label="summary") if payload.get("summary") is not None else None,
            before=_ensure_nonempty_text(payload.get("before"), label="before") if payload.get("before") is not None else None,
            after=_ensure_nonempty_text(payload.get("after"), label="after") if payload.get("after") is not None else None,
        )

    def to_dict(self) -> Dict[str, Any]:
        payload: Dict[str, Any] = {
            "change_id": self.change_id,
            "kind": self.kind.value,
            "target": self.target,
            "source_ids": list(self.source_ids),
            "changed_at": self.changed_at,
        }
        if self.summary is not None:
            payload["summary"] = self.summary
        if self.before is not None:
            payload["before"] = self.before
        if self.after is not None:
            payload["after"] = self.after
        return payload


@dataclass(frozen=True)
class PendingWorkItem:
    work_id: str
    status: PendingWorkStatus
    source_ids: Tuple[str, ...]
    summary: str | None = None
    owner: str | None = None
    due_at: str | None = None

    @classmethod
    def from_mapping(cls, payload: Mapping[str, Any]) -> "PendingWorkItem":
        payload = _require_mapping(payload, label="pending_work")
        _require_mapping_fields(payload, label="pending_work", allowed=_PENDING_WORK_FIELDS)
        source_ids = tuple(_ensure_nonempty_text(item, label="source_id") for item in _require_sequence_field(payload, "source_ids"))
        if not source_ids:
            raise ProvenanceValidationError("pending work must reference at least one source")
        return cls(
            work_id=_ensure_nonempty_text(payload.get("work_id"), label="work_id"),
            status=_coerce_enum(PendingWorkStatus, payload.get("status"), label="status"),
            source_ids=source_ids,
            summary=_ensure_nonempty_text(payload.get("summary"), label="summary") if payload.get("summary") is not None else None,
            owner=_ensure_nonempty_text(payload.get("owner"), label="owner") if payload.get("owner") is not None else None,
            due_at=_parse_iso8601(payload.get("due_at"), label="due_at") if payload.get("due_at") is not None else None,
        )

    def to_dict(self) -> Dict[str, Any]:
        payload: Dict[str, Any] = {
            "work_id": self.work_id,
            "status": self.status.value,
            "source_ids": list(self.source_ids),
        }
        if self.summary is not None:
            payload["summary"] = self.summary
        if self.owner is not None:
            payload["owner"] = self.owner
        if self.due_at is not None:
            payload["due_at"] = self.due_at
        return payload


@dataclass(frozen=True)
class ProvenanceRecord:
    checked_items: Tuple[CheckedItem, ...]
    changes: Tuple[ChangeRecord, ...]
    pending_work: Tuple[PendingWorkItem, ...]
    actor_identity: ActorIdentity
    timestamps: TimestampBundle
    sources: Tuple[SourceReference, ...]
    verification_status: VerificationStatus

    @classmethod
    def from_mapping(cls, payload: Mapping[str, Any]) -> "ProvenanceRecord":
        if not isinstance(payload, Mapping):
            raise ProvenanceValidationError("provenance payload must be an object")
        _require_mapping_fields(payload, label="provenance", allowed=_TOP_LEVEL_FIELDS)

        checked_items = tuple(CheckedItem.from_mapping(item) for item in _require_sequence_field(payload, "checked_items"))
        changes = tuple(ChangeRecord.from_mapping(item) for item in _require_sequence_field(payload, "changes"))
        pending_work = tuple(PendingWorkItem.from_mapping(item) for item in _require_sequence_field(payload, "pending_work"))
        actor_identity = ActorIdentity.from_mapping(_require_object_field(payload, "actor_identity"))
        timestamps = TimestampBundle.from_mapping(_require_object_field(payload, "timestamps"))
        sources = tuple(SourceReference.from_mapping(item) for item in _require_sequence_field(payload, "sources"))
        verification_status = _coerce_enum(
            VerificationStatus,
            _require_scalar_field(payload, "verification_status"),
            label="verification_status",
        )
        record = cls(
            checked_items=checked_items,
            changes=changes,
            pending_work=pending_work,
            actor_identity=actor_identity,
            timestamps=timestamps,
            sources=sources,
            verification_status=verification_status,
        )
        return validate_provenance_record(record)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "v": SCHEMA_VERSION,
            "checked_items": [item.to_dict() for item in self.checked_items],
            "changes": [item.to_dict() for item in self.changes],
            "pending_work": [item.to_dict() for item in self.pending_work],
            "actor_identity": self.actor_identity.to_dict(),
            "timestamps": self.timestamps.to_dict(),
            "sources": [item.to_dict() for item in self.sources],
            "verification_status": self.verification_status.value,
        }

    @property
    def source_index(self) -> Dict[str, SourceReference]:
        return {source.source_id: source for source in self.sources}


_TRANSITION_TOP_LEVEL_FIELDS = frozenset(
    {
        "v",
        "transition_id",
        "session_id",
        "entity_id",
        "kind",
        "status",
        "recorded_at",
        "details",
    }
)


class ProvenanceTransitionKind(str, Enum):
    CHECK = "CHECK"
    MUTATION = "MUTATION"
    WAITING = "WAITING"


@dataclass(frozen=True)
class ProvenanceTransitionRecord:
    transition_id: str
    session_id: str
    entity_id: str
    kind: ProvenanceTransitionKind
    status: str
    recorded_at: str
    details: Dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_mapping(cls, payload: Mapping[str, Any]) -> "ProvenanceTransitionRecord":
        if not isinstance(payload, Mapping):
            raise ProvenanceValidationError("provenance transition payload must be an object")
        _require_mapping_fields(payload, label="provenance transition", allowed=_TRANSITION_TOP_LEVEL_FIELDS)

        raw_details = payload.get("details")
        if raw_details is None:
            details: Dict[str, Any] = {}
        else:
            details = dict(_require_mapping(raw_details, label="details"))

        return cls(
            transition_id=_ensure_nonempty_text(payload.get("transition_id"), label="transition_id"),
            session_id=_ensure_nonempty_text(payload.get("session_id"), label="session_id"),
            entity_id=_ensure_nonempty_text(payload.get("entity_id"), label="entity_id"),
            kind=_coerce_enum(ProvenanceTransitionKind, payload.get("kind"), label="kind"),
            status=_ensure_nonempty_text(payload.get("status"), label="status"),
            recorded_at=_parse_iso8601(payload.get("recorded_at"), label="recorded_at"),
            details=details,
        )

    def to_dict(self) -> Dict[str, Any]:
        payload: Dict[str, Any] = {
            "v": "1",
            "transition_id": self.transition_id,
            "session_id": self.session_id,
            "entity_id": self.entity_id,
            "kind": self.kind.value,
            "status": self.status,
            "recorded_at": self.recorded_at,
        }
        if self.details:
            payload["details"] = dict(self.details)
        return payload


def validate_provenance_transition(record: ProvenanceTransitionRecord) -> ProvenanceTransitionRecord:
    if not isinstance(record, ProvenanceTransitionRecord):
        raise ProvenanceValidationError("provenance transition must be a ProvenanceTransitionRecord")
    return record


def provenance_transition_from_json(raw_line: str) -> ProvenanceTransitionRecord:
    if not isinstance(raw_line, str):
        raise ProvenanceParseError("provenance transition must be a string")
    try:
        payload = json.loads(raw_line)
    except json.JSONDecodeError as exc:
        raise ProvenanceParseError("invalid provenance transition json") from exc
    if not isinstance(payload, dict):
        raise ProvenanceParseError("provenance transition must be a json object")
    version = str(payload.get("v", "1")).strip()
    if version != "1":
        raise ProvenanceParseError(f"unsupported provenance transition version: {version}")
    return validate_provenance_transition(ProvenanceTransitionRecord.from_mapping(payload))


def serialize_provenance_transition(record: ProvenanceTransitionRecord | Mapping[str, Any]) -> str:
    if isinstance(record, ProvenanceTransitionRecord):
        normalized = validate_provenance_transition(record)
    elif isinstance(record, Mapping):
        normalized = ProvenanceTransitionRecord.from_mapping(record)
    else:
        raise ProvenanceValidationError("provenance transition entry must be a mapping or ProvenanceTransitionRecord")
    return json.dumps(normalized.to_dict(), ensure_ascii=True, separators=(",", ":"))


def parse_provenance_transition(raw_line: str) -> ProvenanceTransitionRecord:
    return provenance_transition_from_json(raw_line)


def provenance_transition_to_dict(record: ProvenanceTransitionRecord | Mapping[str, Any]) -> Dict[str, Any]:
    if isinstance(record, ProvenanceTransitionRecord):
        return validate_provenance_transition(record).to_dict()
    if not isinstance(record, Mapping):
        raise ProvenanceValidationError("provenance transition entry must be a mapping or ProvenanceTransitionRecord")
    return ProvenanceTransitionRecord.from_mapping(record).to_dict()


def _validate_source_refs(record: ProvenanceRecord) -> None:
    source_ids = list(record.source_index.keys())
    if len(source_ids) != len(record.sources):
        raise ProvenanceValidationError("source ids must be unique")
    if not source_ids and (record.checked_items or record.changes or record.pending_work):
        raise ProvenanceValidationError("checked items, changes, and pending work require at least one source")

    for item in record.checked_items:
        for source_id in item.source_ids:
            if source_id not in record.source_index:
                raise ProvenanceValidationError(f"checked item {item.item_id} references unknown source {source_id}")
    for change in record.changes:
        for source_id in change.source_ids:
            if source_id not in record.source_index:
                raise ProvenanceValidationError(f"change {change.change_id} references unknown source {source_id}")
    for work in record.pending_work:
        for source_id in work.source_ids:
            if source_id not in record.source_index:
                raise ProvenanceValidationError(f"pending work {work.work_id} references unknown source {source_id}")


def _validate_timestamp_actor_consistency(record: ProvenanceRecord) -> None:
    recorded_at = datetime.fromisoformat(record.timestamps.recorded_at)
    if record.verification_status == VerificationStatus.PENDING:
        if record.timestamps.checked_at is not None:
            raise ProvenanceValidationError("checked_at cannot be set while verification_status is PENDING")
        if record.timestamps.verified_at is not None:
            raise ProvenanceValidationError("verified_at cannot be set while verification_status is PENDING")
    if record.timestamps.checked_at is not None:
        checked_at = datetime.fromisoformat(record.timestamps.checked_at)
        if checked_at < recorded_at:
            raise ProvenanceValidationError("checked_at cannot be earlier than recorded_at")
    if record.verification_status != VerificationStatus.PENDING and record.timestamps.checked_at is None:
        raise ProvenanceValidationError("checked_at is required once the record has been checked")
    if record.timestamps.verified_at is not None:
        if record.verification_status != VerificationStatus.VERIFIED:
            raise ProvenanceValidationError("verified_at can only be set when verification_status is VERIFIED")
        if record.timestamps.checked_at is None:
            raise ProvenanceValidationError("verified_at requires checked_at")
        verified_at = datetime.fromisoformat(record.timestamps.verified_at)
        checked_at = datetime.fromisoformat(record.timestamps.checked_at)
        if verified_at < checked_at:
            raise ProvenanceValidationError("verified_at cannot be earlier than checked_at")
    if record.timestamps.updated_at is not None:
        updated_at = datetime.fromisoformat(record.timestamps.updated_at)
        if updated_at < recorded_at:
            raise ProvenanceValidationError("updated_at cannot be earlier than recorded_at")
    if not record.actor_identity.actor_id:
        raise ProvenanceValidationError("actor identity is required")


def validate_provenance_record(record: ProvenanceRecord) -> ProvenanceRecord:
    if not isinstance(record, ProvenanceRecord):
        raise ProvenanceValidationError("provenance record must be a ProvenanceRecord")
    _validate_source_refs(record)
    _validate_timestamp_actor_consistency(record)
    if record.verification_status == VerificationStatus.VERIFIED and record.timestamps.verified_at is None:
        raise ProvenanceValidationError("verified_at is required when verification_status is VERIFIED")
    if record.verification_status != VerificationStatus.VERIFIED and record.timestamps.verified_at is not None:
        raise ProvenanceValidationError("verified_at can only be set when verification_status is VERIFIED")
    return record


def provenance_record_from_json(raw_line: str) -> ProvenanceRecord:
    if not isinstance(raw_line, str):
        raise ProvenanceParseError("provenance record must be a string")
    try:
        payload = json.loads(raw_line)
    except json.JSONDecodeError as exc:
        raise ProvenanceParseError("invalid provenance json") from exc
    if not isinstance(payload, dict):
        raise ProvenanceParseError("provenance record must be a json object")
    version = str(payload.get("v", SCHEMA_VERSION)).strip()
    if version not in {SCHEMA_VERSION, "1"}:
        raise ProvenanceParseError(f"unsupported provenance version: {version}")
    if version == "1":
        return _upgrade_legacy_payload(payload)
    return ProvenanceRecord.from_mapping(payload)


def _upgrade_legacy_payload(payload: Mapping[str, Any]) -> ProvenanceRecord:
    session_id = _ensure_nonempty_text(payload.get("session_id"), label="session_id")
    artifact = _ensure_nonempty_text(payload.get("artifact"), label="artifact")
    action = _ensure_nonempty_text(payload.get("action"), label="action")
    why = _ensure_nonempty_text(payload.get("why"), label="why")
    risk = _ensure_nonempty_text(payload.get("risk"), label="risk")
    ts = _parse_iso8601(payload.get("ts") or datetime.now(timezone.utc).isoformat(), label="ts")
    actor = _ensure_nonempty_text(payload.get("actor"), label="actor") if payload.get("actor") is not None else session_id
    source_trace = payload.get("source_trace")
    source = SourceReference(
        source_id="legacy-source",
        kind=SourceKind.EVENT,
        label="legacy source_trace",
        uri=json.dumps(source_trace, sort_keys=True) if source_trace is not None else None,
    )
    checked_item = CheckedItem(
        item_id=artifact,
        status=CheckedItemStatus.PASS,
        source_ids=(source.source_id,),
        checked_at=ts,
        label=artifact,
        notes=why,
    )
    change = ChangeRecord(
        change_id=f"{artifact}:change",
        kind=ChangeKind.UPDATE if action.upper() == "UPDATE" else ChangeKind.RECONCILE,
        target=artifact,
        source_ids=(source.source_id,),
        changed_at=ts,
        summary=why,
        after=risk,
    )
    return validate_provenance_record(
        ProvenanceRecord(
            checked_items=(checked_item,),
            changes=(change,),
            pending_work=(),
            actor_identity=ActorIdentity(actor_id=actor, kind=ActorKind.SYSTEM, display_name=actor, session_id=session_id),
            timestamps=TimestampBundle(recorded_at=ts, checked_at=ts, verified_at=ts),
            sources=(source,),
            verification_status=VerificationStatus.VERIFIED,
        )
    )


def serialize_provenance_record(record: ProvenanceRecord | Mapping[str, Any]) -> str:
    return json.dumps(_normalize_record(record).to_dict(), ensure_ascii=True, separators=(",", ":"))


def parse_provenance_record(raw_line: str) -> ProvenanceRecord:
    return provenance_record_from_json(raw_line)


def _normalize_record(record: ProvenanceRecord | Mapping[str, Any]) -> ProvenanceRecord:
    if isinstance(record, ProvenanceRecord):
        return validate_provenance_record(record)
    if not isinstance(record, Mapping):
        raise ProvenanceValidationError("provenance entry must be a mapping or ProvenanceRecord")
    return ProvenanceRecord.from_mapping(record)


def provenance_record_to_dict(record: ProvenanceRecord | Mapping[str, Any]) -> Dict[str, Any]:
    return _normalize_record(record).to_dict()
