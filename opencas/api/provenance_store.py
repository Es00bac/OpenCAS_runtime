"""File-backed storage for the canonical trust-critical provenance schema."""

from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path
from typing import Any, List, Mapping, Sequence

from .provenance_schema import (
    ActorIdentity,
    ActorKind,
    ChangeKind,
    ChangeRecord,
    CheckedItem,
    CheckedItemStatus,
    can_transition_verification_status,
    PendingWorkItem,
    PendingWorkStatus,
    ProvenanceParseError,
    ProvenanceRecord,
    ProvenanceTransitionKind,
    ProvenanceTransitionRecord,
    ProvenanceValidationError,
    SourceKind,
    SourceReference,
    TimestampBundle,
    VerificationStatus,
    parse_provenance_record,
    parse_provenance_transition,
    provenance_record_from_json,
    provenance_record_to_dict,
    serialize_provenance_record,
    serialize_provenance_transition,
    transition_verification_status,
    validate_provenance_record,
    validate_provenance_transition,
)

__all__ = [
    "ActorIdentity",
    "ActorKind",
    "ChangeKind",
    "ChangeRecord",
    "CheckedItem",
    "CheckedItemStatus",
    "can_transition_verification_status",
    "PendingWorkItem",
    "PendingWorkStatus",
    "ProvenanceEntryStore",
    "ProvenanceParseError",
    "ProvenanceRecord",
    "ProvenanceTransitionKind",
    "ProvenanceTransitionRecord",
    "ProvenanceValidationError",
    "SourceKind",
    "SourceReference",
    "TimestampBundle",
    "VerificationStatus",
    "format_provenance_entry",
    "parse_provenance_entry",
    "format_provenance_transition",
    "parse_provenance_transition",
    "record_provenance_transition",
    "provenance_record_from_json",
    "provenance_record_to_dict",
    "provenance_transition_to_dict",
    "serialize_provenance_record",
    "serialize_provenance_transition",
    "transition_verification_status",
    "validate_provenance_record",
    "validate_provenance_transition",
]


def format_provenance_entry(entry: ProvenanceRecord | Mapping[str, Any]) -> str:
    """Render one canonical provenance transport record."""

    return serialize_provenance_record(entry)


def parse_provenance_entry(raw_line: str) -> ProvenanceRecord:
    """Parse one canonical provenance transport record."""

    return parse_provenance_record(raw_line)


def format_provenance_transition(entry: ProvenanceTransitionRecord | Mapping[str, Any]) -> str:
    """Render one immutable provenance transition record."""

    return serialize_provenance_transition(entry)


def parse_provenance_transition_entry(raw_line: str) -> ProvenanceTransitionRecord:
    """Parse one immutable provenance transition record."""

    return parse_provenance_transition(raw_line)


def _normalize_linked_ids(*values: Any) -> List[str]:
    """Return a de-duplicated list of non-empty linkage ids."""

    linked: List[str] = []
    seen: set[str] = set()
    for value in values:
        if value is None:
            continue
        if isinstance(value, (list, tuple, set, frozenset)):
            candidates = value
        else:
            candidates = (value,)
        for candidate in candidates:
            text = str(candidate or "").strip()
            if not text or text in seen:
                continue
            seen.add(text)
            linked.append(text)
    return linked


class ProvenanceEntryStore:
    """Persist canonical provenance records as newline-delimited JSON."""

    def __init__(self, path: Path | str) -> None:
        self.path = Path(path)
        self.transition_path = self._derive_transition_path()

    def append(self, entry: ProvenanceRecord | Mapping[str, Any]) -> ProvenanceRecord:
        record = self._coerce_entry(entry)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self.path.open("a", encoding="utf-8") as handle:
            handle.write(format_provenance_entry(record) + "\n")
        return record

    def list_recent(self, limit: int = 10, offset: int = 0) -> List[ProvenanceRecord]:
        if not self.path.exists():
            return []

        with self.path.open("r", encoding="utf-8") as handle:
            items = [
                parse_provenance_entry(line)
                for line in handle.read().splitlines()
                if line.strip()
            ]
        return items[offset : offset + limit]

    def append_transition(
        self,
        entry: ProvenanceTransitionRecord | Mapping[str, Any],
    ) -> ProvenanceTransitionRecord:
        """Append one immutable provenance transition record."""

        record = self._coerce_transition(entry)
        self._append_line(self.transition_path, format_provenance_transition(record))
        return record

    def record_transition(
        self,
        entry: ProvenanceTransitionRecord | Mapping[str, Any],
    ) -> ProvenanceTransitionRecord:
        """Compatibility alias for transition append call sites."""

        return self.append_transition(entry)

    def record_check(
        self,
        *,
        session_id: str,
        entity_id: str,
        status: str = "checked",
        trigger_artifact: str | None = None,
        source_artifact: str | None = None,
        trigger_action: str | None = None,
        parent_transition_id: str | None = None,
        linked_transition_ids: Sequence[str] | None = None,
        target_entity: str | None = None,
        origin_action_id: str | None = None,
        details: Mapping[str, Any] | None = None,
        recorded_at: str | None = None,
        transition_id: str | None = None,
    ) -> ProvenanceTransitionRecord:
        return self._record_transition(
            kind=ProvenanceTransitionKind.CHECK,
            session_id=session_id,
            entity_id=entity_id,
            status=status,
            trigger_artifact=trigger_artifact,
            source_artifact=source_artifact,
            trigger_action=trigger_action,
            parent_transition_id=parent_transition_id,
            linked_transition_ids=linked_transition_ids,
            target_entity=target_entity,
            origin_action_id=origin_action_id,
            details=details,
            recorded_at=recorded_at,
            transition_id=transition_id,
        )

    def record_mutation(
        self,
        *,
        session_id: str,
        entity_id: str,
        status: str = "mutated",
        trigger_artifact: str | None = None,
        source_artifact: str | None = None,
        trigger_action: str | None = None,
        parent_transition_id: str | None = None,
        linked_transition_ids: Sequence[str] | None = None,
        target_entity: str | None = None,
        origin_action_id: str | None = None,
        details: Mapping[str, Any] | None = None,
        recorded_at: str | None = None,
        transition_id: str | None = None,
    ) -> ProvenanceTransitionRecord:
        return self._record_transition(
            kind=ProvenanceTransitionKind.MUTATION,
            session_id=session_id,
            entity_id=entity_id,
            status=status,
            trigger_artifact=trigger_artifact,
            source_artifact=source_artifact,
            trigger_action=trigger_action,
            parent_transition_id=parent_transition_id,
            linked_transition_ids=linked_transition_ids,
            target_entity=target_entity,
            origin_action_id=origin_action_id,
            details=details,
            recorded_at=recorded_at,
            transition_id=transition_id,
        )

    def record_waiting(
        self,
        *,
        session_id: str,
        entity_id: str,
        status: str = "waiting",
        trigger_artifact: str | None = None,
        source_artifact: str | None = None,
        trigger_action: str | None = None,
        parent_transition_id: str | None = None,
        linked_transition_ids: Sequence[str] | None = None,
        target_entity: str | None = None,
        origin_action_id: str | None = None,
        details: Mapping[str, Any] | None = None,
        recorded_at: str | None = None,
        transition_id: str | None = None,
    ) -> ProvenanceTransitionRecord:
        return self._record_transition(
            kind=ProvenanceTransitionKind.WAITING,
            session_id=session_id,
            entity_id=entity_id,
            status=status,
            trigger_artifact=trigger_artifact,
            source_artifact=source_artifact,
            trigger_action=trigger_action,
            parent_transition_id=parent_transition_id,
            linked_transition_ids=linked_transition_ids,
            target_entity=target_entity,
            origin_action_id=origin_action_id,
            details=details,
            recorded_at=recorded_at,
            transition_id=transition_id,
        )

    def list_transition_history(
        self,
        limit: int | None = 10,
        offset: int = 0,
    ) -> List[ProvenanceTransitionRecord]:
        """Return immutable transition history in append order."""

        if not self.transition_path.exists():
            return []

        with self.transition_path.open("r", encoding="utf-8") as handle:
            items = [
                parse_provenance_transition_entry(line)
                for line in handle.read().splitlines()
                if line.strip()
            ]
        if limit is None:
            return items[offset:]
        return items[offset : offset + limit]

    def list_current_status(
        self,
        limit: int | None = 10,
        offset: int = 0,
        *,
        session_id: str | None = None,
        entity_id: str | None = None,
    ) -> List[ProvenanceTransitionRecord]:
        """Return the latest effective state per session/entity pair."""

        latest = self._current_status_index()
        records = [item[1] for item in sorted(latest.values(), key=lambda item: item[0], reverse=True)]
        if session_id is not None:
            records = [record for record in records if record.session_id == session_id]
        if entity_id is not None:
            records = [record for record in records if record.entity_id == entity_id]
        if limit is None:
            return records[offset:]
        return records[offset : offset + limit]

    def get_current_status(
        self,
        *,
        session_id: str,
        entity_id: str,
    ) -> ProvenanceTransitionRecord | None:
        """Return the current state for one session/entity pair."""

        items = self.list_current_status(limit=1, session_id=session_id, entity_id=entity_id)
        return items[0] if items else None

    def current_status(
        self,
        *,
        session_id: str,
        entity_id: str,
    ) -> ProvenanceTransitionRecord | None:
        """Compatibility alias for current-status queries."""

        return self.get_current_status(session_id=session_id, entity_id=entity_id)

    @staticmethod
    def _coerce_entry(entry: ProvenanceRecord | Mapping[str, Any]) -> ProvenanceRecord:
        if isinstance(entry, ProvenanceRecord):
            return validate_provenance_record(entry)
        if not isinstance(entry, Mapping):
            raise ProvenanceValidationError("provenance entry must be a mapping or ProvenanceRecord")
        return validate_provenance_record(ProvenanceRecord.from_mapping(entry))

    @staticmethod
    def _coerce_transition(
        entry: ProvenanceTransitionRecord | Mapping[str, Any],
    ) -> ProvenanceTransitionRecord:
        if isinstance(entry, ProvenanceTransitionRecord):
            return validate_provenance_transition(entry)
        if not isinstance(entry, Mapping):
            raise ProvenanceValidationError("provenance transition must be a mapping or ProvenanceTransitionRecord")
        return validate_provenance_transition(ProvenanceTransitionRecord.from_mapping(entry))

    def _record_transition(
        self,
        *,
        kind: ProvenanceTransitionKind,
        session_id: str,
        entity_id: str,
        status: str,
        trigger_artifact: str | None,
        source_artifact: str | None,
        trigger_action: str | None,
        parent_transition_id: str | None,
        linked_transition_ids: Sequence[str] | None,
        target_entity: str | None,
        origin_action_id: str | None,
        details: Mapping[str, Any] | None,
        recorded_at: str | None,
        transition_id: str | None,
    ) -> ProvenanceTransitionRecord:
        merged_details = dict(details or {})
        normalized_trigger_artifact = str(trigger_artifact or source_artifact or "").strip()
        normalized_parent_transition_id = str(parent_transition_id or origin_action_id or "").strip()
        normalized_linked_ids = _normalize_linked_ids(
            linked_transition_ids,
            normalized_parent_transition_id,
            target_entity,
        )
        if normalized_trigger_artifact:
            merged_details.setdefault("trigger_artifact", normalized_trigger_artifact)
            merged_details.setdefault("source_artifact", normalized_trigger_artifact)
        if trigger_action:
            merged_details.setdefault("trigger_action", trigger_action)
        if normalized_parent_transition_id:
            merged_details.setdefault("parent_transition_id", normalized_parent_transition_id)
            merged_details.setdefault("origin_action_id", normalized_parent_transition_id)
        if normalized_linked_ids:
            merged_details.setdefault("linked_transition_ids", normalized_linked_ids)
        if target_entity:
            merged_details.setdefault("target_entity", target_entity)
        payload = {
            "transition_id": transition_id
            or normalized_parent_transition_id
            or self._default_transition_id(kind, session_id, entity_id, recorded_at),
            "session_id": session_id,
            "entity_id": entity_id,
            "kind": kind,
            "status": status,
            "recorded_at": recorded_at or self._now_iso8601(),
            "details": merged_details,
        }
        return self.append_transition(payload)

    def _current_status_index(self) -> dict[tuple[str, str], tuple[int, ProvenanceTransitionRecord]]:
        latest: dict[tuple[str, str], tuple[int, ProvenanceTransitionRecord]] = {}
        for index, record in enumerate(self.list_transition_history(limit=None)):
            latest[(record.session_id, record.entity_id)] = (index, record)
        return latest

    def _derive_transition_path(self) -> Path:
        suffix = self.path.suffix or ".jsonl"
        return self.path.with_name(f"{self.path.stem}.transitions{suffix}")

    def _default_transition_id(
        self,
        kind: ProvenanceTransitionKind,
        session_id: str,
        entity_id: str,
        recorded_at: str | None,
    ) -> str:
        stamp = recorded_at or self._now_iso8601()
        return f"{session_id}:{entity_id}:{kind.value}:{stamp}".replace(" ", "_")

    @staticmethod
    def _now_iso8601() -> str:
        return datetime.now(timezone.utc).isoformat()

    @staticmethod
    def _append_line(path: Path, line: str) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("a", encoding="utf-8") as handle:
            handle.write(line + "\n")
            handle.flush()


def record_provenance_transition(
    *,
    state_dir: Path | str,
    kind: ProvenanceTransitionKind,
    session_id: str,
    entity_id: str,
    status: str,
    trigger_artifact: str,
    source_artifact: str,
    trigger_action: str,
    parent_transition_id: str | None = None,
    linked_transition_ids: Sequence[str] | None = None,
    target_entity: str,
    origin_action_id: str | None = None,
    details: Mapping[str, Any] | None = None,
    recorded_at: str | None = None,
    transition_id: str | None = None,
) -> ProvenanceTransitionRecord:
    """Persist one provenance transition with explicit source/target linkage."""

    state_path = Path(state_dir)
    store_path = state_path if state_path.suffix else state_path / "provenance.jsonl"
    store = ProvenanceEntryStore(store_path)
    recorders = {
        ProvenanceTransitionKind.CHECK: store.record_check,
        ProvenanceTransitionKind.MUTATION: store.record_mutation,
        ProvenanceTransitionKind.WAITING: store.record_waiting,
    }
    recorder = recorders[kind]
    return recorder(
        session_id=session_id,
        entity_id=entity_id,
        status=status,
        trigger_artifact=trigger_artifact,
        source_artifact=source_artifact,
        trigger_action=trigger_action,
        parent_transition_id=parent_transition_id,
        linked_transition_ids=linked_transition_ids,
        target_entity=target_entity,
        origin_action_id=origin_action_id,
        details=details,
        recorded_at=recorded_at,
        transition_id=transition_id,
    )
