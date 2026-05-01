"""Registry-bound provenance helpers for the canonical five-field contract."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any, Mapping

from .provenance_entry import (
    Action,
    InvalidProvenancePayload,
    ProvenanceRecordV1,
    Risk,
    create_registry_entry as _create_registry_entry,
    parse_registry_entry as _parse_registry_entry,
    read_registry_entries as _read_registry_entries,
    serialize_registry_entry as _serialize_registry_entry,
)

__all__ = [
    "Action",
    "InvalidProvenancePayload",
    "ProvenanceRecordV1",
    "Risk",
    "create_registry_entry",
    "format_registry_entry",
    "serialize_registry_entry",
    "parse_registry_entry",
    "persist_registry_entry",
    "read_registry_entries",
]


def create_registry_entry(
    *,
    session_id: str,
    artifact: str,
    action: Any,
    why: str,
    risk: Any,
) -> ProvenanceRecordV1:
    """Create a canonical five-field provenance entry."""

    return _create_registry_entry(
        session_id=session_id,
        artifact=artifact,
        action=action,
        why=why,
        risk=risk,
    )


def format_registry_entry(entry: ProvenanceRecordV1 | Mapping[str, Any]) -> str:
    """Render one canonical registry line."""

    record = _coerce_registry_entry(entry)
    return _serialize_registry_entry(record)


def serialize_registry_entry(entry: ProvenanceRecordV1 | Mapping[str, Any]) -> str:
    """Compatibility alias for callers that expect serialize/parse naming."""

    return format_registry_entry(entry)


def parse_registry_entry(raw_line: str) -> ProvenanceRecordV1:
    """Parse one canonical registry line."""

    return _parse_registry_entry(raw_line)


def persist_registry_entry(sink: Any, entry: ProvenanceRecordV1 | Mapping[str, Any]) -> ProvenanceRecordV1:
    """Append a canonical registry entry to an append sink or file path."""

    record = _coerce_registry_entry(entry)
    line = format_registry_entry(record)

    if isinstance(sink, (str, Path)):
        path = Path(sink)
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("a", encoding="utf-8") as handle:
            handle.write(line + "\n")
            handle.flush()
            os.fsync(handle.fileno())
        return record

    append = getattr(sink, "append", None)
    if callable(append):
        append(line)
        return record

    raise InvalidProvenancePayload("registry sink is unavailable or unsupported")


def read_registry_entries(raw_text: str) -> list[ProvenanceRecordV1]:
    """Recover canonical registry entries from noisy multi-line text."""

    return _read_registry_entries(raw_text)


def _coerce_registry_entry(entry: ProvenanceRecordV1 | Mapping[str, Any]) -> ProvenanceRecordV1:
    if isinstance(entry, ProvenanceRecordV1):
        return entry
    if not isinstance(entry, Mapping):
        raise TypeError("provenance entry must be a mapping")
    try:
        return _create_registry_entry(
            session_id=entry["session_id"],
            artifact=entry["artifact"],
            action=entry["action"],
            why=entry["why"],
            risk=entry["risk"],
        )
    except KeyError as exc:
        raise InvalidProvenancePayload(f"missing provenance field: {exc.args[0]}") from exc
