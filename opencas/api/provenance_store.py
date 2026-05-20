"""File-backed storage for the canonical trust-critical provenance schema."""

from __future__ import annotations

import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterator, List, Mapping, Sequence

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
    provenance_transition_to_dict,
    serialize_provenance_record,
    serialize_provenance_transition,
    transition_verification_status,
    validate_provenance_record,
    validate_provenance_transition,
)

DEFAULT_TRANSITION_MAX_BYTES = 128 * 1024 * 1024
DEFAULT_TRANSITION_ARCHIVE_LIMIT = 4

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

    def __init__(
        self,
        path: Path | str,
        *,
        transition_max_bytes: int | None = None,
        transition_archive_limit: int | None = None,
    ) -> None:
        self.path = Path(path)
        self.transition_path = self._derive_transition_path()
        self.transition_max_bytes = _env_int(
            "OPENCAS_PROVENANCE_TRANSITION_MAX_BYTES",
            DEFAULT_TRANSITION_MAX_BYTES if transition_max_bytes is None else transition_max_bytes,
        )
        self.transition_archive_limit = _env_int(
            "OPENCAS_PROVENANCE_TRANSITION_ARCHIVE_LIMIT",
            DEFAULT_TRANSITION_ARCHIVE_LIMIT
            if transition_archive_limit is None
            else transition_archive_limit,
        )

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
        self.rotate_transition_log()
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

        items = [
            record
            for line in self._iter_transition_lines()
            if line.strip()
            for record in [self._parse_transition_line(line)]
            if record is not None
        ]
        if limit is None:
            return items[offset:]
        return items[offset : offset + limit]

    def rotate_transition_log(self, *, force: bool = False) -> dict[str, Any]:
        """Archive an oversized transition log and seed the active log with checkpoints."""

        if not self.transition_path.exists():
            return {"rotated": False, "reason": "missing"}
        max_bytes = int(self.transition_max_bytes or 0)
        current_bytes = self.transition_path.stat().st_size
        if not force and (max_bytes <= 0 or current_bytes <= max_bytes):
            return {
                "rotated": False,
                "reason": "below_threshold",
                "size_bytes": current_bytes,
                "max_bytes": max_bytes,
            }

        current_status = self.list_current_status(limit=None)
        skipped_invalid = getattr(self, "_last_transition_parse_errors", 0)
        archive_path = self._next_transition_archive_path()
        self.transition_path.rename(archive_path)
        checkpointed = self._write_rotation_checkpoints(current_status, archive_path)
        removed_archives = self._prune_transition_archives()
        return {
            "rotated": True,
            "archive_path": str(archive_path),
            "size_bytes": current_bytes,
            "max_bytes": max_bytes,
            "checkpointed": checkpointed,
            "skipped_invalid_lines": skipped_invalid,
            "removed_archives": [str(path) for path in removed_archives],
        }

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
        skipped_invalid = 0
        for index, line in enumerate(self._iter_transition_lines()):
            if not line.strip():
                continue
            record = self._parse_transition_line(line)
            if record is None:
                skipped_invalid += 1
                continue
            latest[(record.session_id, record.entity_id)] = (index, record)
        self._last_transition_parse_errors = skipped_invalid
        return latest

    @staticmethod
    def _parse_transition_line(line: str) -> ProvenanceTransitionRecord | None:
        try:
            return parse_provenance_transition_entry(line)
        except ProvenanceParseError:
            return None

    def _derive_transition_path(self) -> Path:
        suffix = self.path.suffix or ".jsonl"
        return self.path.with_name(f"{self.path.stem}.transitions{suffix}")

    def _iter_transition_lines(self) -> Iterator[str]:
        for path in self._transition_history_paths():
            with path.open("r", encoding="utf-8") as handle:
                yield from handle

    def _transition_history_paths(self) -> list[Path]:
        paths = self._transition_archive_paths()
        if self.transition_path.exists():
            paths.append(self.transition_path)
        return paths

    def _transition_archive_paths(self) -> list[Path]:
        pattern = f"{self.transition_path.stem}.*{self.transition_path.suffix}"
        return sorted(
            path
            for path in self.transition_path.parent.glob(pattern)
            if path.is_file() and path != self.transition_path
        )

    def _next_transition_archive_path(self) -> Path:
        stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
        base = self.transition_path.with_name(
            f"{self.transition_path.stem}.{stamp}{self.transition_path.suffix}"
        )
        if not base.exists():
            return base
        counter = 1
        while True:
            candidate = self.transition_path.with_name(
                f"{self.transition_path.stem}.{stamp}.{counter}{self.transition_path.suffix}"
            )
            if not candidate.exists():
                return candidate
            counter += 1

    def _write_rotation_checkpoints(
        self,
        records: Sequence[ProvenanceTransitionRecord],
        archive_path: Path,
    ) -> int:
        if not records:
            return 0
        checkpointed_at = self._now_iso8601()
        ordered = sorted(records, key=lambda record: record.recorded_at)
        for record in ordered:
            details = dict(record.details)
            details.update(
                {
                    "rotation_checkpoint": True,
                    "checkpoint_source_archive": archive_path.name,
                    "checkpoint_source_transition_id": record.transition_id,
                    "checkpoint_source_recorded_at": record.recorded_at,
                }
            )
            checkpoint = ProvenanceTransitionRecord(
                transition_id=(
                    f"{record.transition_id}:rotation_checkpoint:{archive_path.stem}"
                ),
                session_id=record.session_id,
                entity_id=record.entity_id,
                kind=record.kind,
                status=record.status,
                recorded_at=checkpointed_at,
                details=details,
            )
            self._append_line(self.transition_path, format_provenance_transition(checkpoint))
        return len(ordered)

    def _prune_transition_archives(self) -> list[Path]:
        limit = int(self.transition_archive_limit or 0)
        if limit <= 0:
            return []
        archives = self._transition_archive_paths()
        removable = archives[: max(0, len(archives) - limit)]
        removed: list[Path] = []
        for path in removable:
            path.unlink(missing_ok=True)
            removed.append(path)
        return removed

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


def _env_int(name: str, default: int) -> int:
    value = os.environ.get(name)
    if value is None:
        return int(default)
    try:
        return int(value)
    except ValueError:
        return int(default)


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
