"""Versioned provenance entry contract for operator action history.

The registry-line source of truth lives in ``docs/provenance-entry-contract-v1.md``.
"""

from __future__ import annotations

import json
import os
import re
import string
from dataclasses import dataclass
from datetime import datetime, timezone
from enum import Enum
from pathlib import Path
from typing import Any, Callable, Dict, List, Mapping, Optional, Sequence

CANONICAL_REGISTRY_SCHEMA: Dict[str, Dict[str, Any]] = {
    "session_id": {
        "required": True,
        "shape": "opaque identifier",
        "validation": "non-empty, single-line, reviewable text",
    },
    "artifact": {
        "required": True,
        "shape": "opaque identifier or short label",
        "validation": "non-empty, single-line, reviewable text",
    },
    "action": {
        "required": True,
        "shape": "Action enum token",
        "validation": "canonical uppercase enum name",
    },
    "why": {
        "required": True,
        "shape": "short rationale",
        "validation": "non-empty review text",
    },
    "risk": {
        "required": True,
        "shape": "Risk enum token",
        "validation": "canonical uppercase enum name",
    },
}

CANONICAL_JSONL_TRANSPORT_SCHEMA: Dict[str, Dict[str, Any]] = {
    **CANONICAL_REGISTRY_SCHEMA,
    "ts": {
        "required": False,
        "shape": "ISO8601 timestamp",
        "validation": "timezone-aware when present",
    },
    "actor": {
        "required": False,
        "shape": "optional string",
        "validation": "omit when absent",
    },
    "source_trace": {
        "required": False,
        "shape": "optional JSON object",
        "validation": "omit when absent",
    },
}

# Backwards-compatible alias for older call sites that reference the broader transport schema name.
CANONICAL_RECORD_SCHEMA = CANONICAL_JSONL_TRANSPORT_SCHEMA

CANONICAL_SOURCE_MAP: Dict[str, Dict[str, Any]] = {
    "session_id": {
        "from_runtime": ["ctx.config.session_id", "ctx.session_id", "session_id"],
        "fallback": "default",
        "required": True,
    },
    "artifact": {
        "from_source": ["artifact", "artifact_id", "target_id"],
        "from_event": {
            "target_kind": "target_kind",
            "target_id": "target_id",
            "scope_key": "scope_key",
        },
        "from_state": ["id", "artifact_id", "session_id"],
        "required": True,
    },
    "action": {
        "from_event": {
            "tool": "tool",
            "function": "function",
            "changed_entity": "changed_entity",
            "target_kind": "target_kind",
            "legacy_action": "legacy_action",
            "event": "event",
        },
        "from_legacy": ["action"],
    },
    "why": {
        "from_source": ["why", "goal", "rationale", "trigger", "plan", "subplan", "approval_reason", "notes", "message"],
    },
    "risk": {
        "from_source": ["risk", "risk_level", "risk_state"],
        "from_signals": ["risk_context", "build_status", "test_status", "safety_state", "self_approval_level", "uncertainty", "escalation_required"],
        "from_event": {
            "safety": "safety",
            "tool": "tool",
        },
    },
}


CURRENT_PROVENANCE_VERSION = "1"
SUPPORTED_PROVENANCE_VERSIONS = {CURRENT_PROVENANCE_VERSION}
CANONICAL_REGISTRY_DELIMITER = " | "
CANONICAL_REGISTRY_FIELD_ORDER: Sequence[str] = (
    "session_id",
    "artifact",
    "action",
    "why",
    "risk",
)
CANONICAL_REGISTRY_EXAMPLE_LINE = (
    "session:daily:αβ | process:default:abc\\|child\\\\leaf | UPDATE | line one\\nline two | MEDIUM"
)
CANONICAL_FIELD_ORDER_JSONL: Sequence[str] = (
    "session_id",
    "artifact",
    "action",
    "why",
    "risk",
    "ts",
    "actor",
    "source_trace",
)
CANONICAL_FIELD_ORDER_CSV: Sequence[str] = (
    "v",
    "session_id",
    "artifact",
    "action",
    "why",
    "risk",
    "ts",
    "actor",
    "source_trace",
)
CANONICAL_REQUIRED_FIELDS = frozenset(
    {
        "session_id",
        "artifact",
        "action",
        "why",
        "risk",
    }
)
CANONICAL_FIELDS = frozenset(
    {
        "session_id",
        "artifact",
        "action",
        "why",
        "risk",
        "ts",
        "actor",
        "source_trace",
    }
)
MAX_WHY_LENGTH = 512
MAX_REGISTRY_SESSION_ID_LENGTH = 128
MAX_REGISTRY_ARTIFACT_LENGTH = 128
MAX_REGISTRY_WHY_LENGTH = 512
_ANSI_ESCAPE_RE = re.compile(r"\x1b\[[0-?]*[ -/]*[@-~]")


class ProvenanceError(ValueError):
    """Base provenance module error."""


class UnsupportedProvenanceVersion(ProvenanceError):
    """Raised when a provenance record version is not supported."""


class InvalidProvenancePayload(ProvenanceError):
    """Raised when a provenance record is malformed."""


class RegistrySinkError(ProvenanceError):
    """Raised when a registry entry cannot be written to the chosen sink."""


class Action(str, Enum):
    CREATE = "CREATE"
    READ = "READ"
    UPDATE = "UPDATE"
    DELETE = "DELETE"
    DECIDE = "DECIDE"
    TOOL_CALL = "TOOL_CALL"
    REFLECT = "REFLECT"
    CONSOLIDATE = "CONSOLIDATE"
    COMMIT = "COMMIT"
    ROLLBACK = "ROLLBACK"


class Risk(str, Enum):
    LOW = "LOW"
    MEDIUM = "MEDIUM"
    HIGH = "HIGH"
    CRITICAL = "CRITICAL"


# Keep compatibility with existing operator-facing action labels where possible.
LEGACY_ACTION_TO_ACTION: Dict[str, Action] = {
    "create": Action.CREATE,
    "read": Action.READ,
    "update": Action.UPDATE,
    "kill_process": Action.DELETE,
    "tool_call": Action.TOOL_CALL,
    "browser_navigate": Action.TOOL_CALL,
    "browser_click": Action.TOOL_CALL,
    "browser_type": Action.TOOL_CALL,
    "browser_press": Action.TOOL_CALL,
    "browser_wait": Action.TOOL_CALL,
    "browser_capture": Action.TOOL_CALL,
    "close_browser": Action.DELETE,
    "pty_input": Action.TOOL_CALL,
    "decide": Action.DECIDE,
    "reflect": Action.REFLECT,
    "consolidate": Action.CONSOLIDATE,
    "commit": Action.COMMIT,
    "rollback": Action.ROLLBACK,
}


DEFAULT_RISK_BY_ACTION: Dict[Action, Risk] = {
    Action.CREATE: Risk.LOW,
    Action.READ: Risk.LOW,
    Action.UPDATE: Risk.MEDIUM,
    Action.DELETE: Risk.HIGH,
    Action.DECIDE: Risk.MEDIUM,
    Action.TOOL_CALL: Risk.MEDIUM,
    Action.REFLECT: Risk.LOW,
    Action.CONSOLIDATE: Risk.MEDIUM,
    Action.COMMIT: Risk.MEDIUM,
    Action.ROLLBACK: Risk.HIGH,
}


class ProvenanceEntryError(Exception):
    """Compatibility alias for older module imports."""


@dataclass(frozen=True)
class ProvenanceRecordV1:
    """Canonical provenance record schema."""

    session_id: str
    artifact: str
    action: Action
    why: str
    risk: Risk
    ts: Optional[str] = None
    actor: Optional[str] = None
    source_trace: Optional[Dict[str, Any]] = None

    def to_dict(self, *, include_v: bool = True) -> Dict[str, Any]:
        payload: Dict[str, Any] = {
            "session_id": self.session_id,
            "artifact": self.artifact,
            "action": self.action.value,
            "why": self.why,
            "risk": self.risk.value,
        }
        if self.ts is not None:
            payload["ts"] = self.ts
        if self.actor is not None:
            payload["actor"] = self.actor
        if self.source_trace is not None:
            payload["source_trace"] = self.source_trace
        if include_v:
            payload["v"] = CURRENT_PROVENANCE_VERSION
        return payload


# Compatibility alias for older imports in this repo.
ProvenanceEntry = ProvenanceRecordV1


def now_iso8601_ts() -> str:
    """Return UTC ISO-8601 timestamp with millisecond precision."""

    current = datetime.now(timezone.utc)
    rounded = current.replace(microsecond=int(current.microsecond / 1000) * 1000)
    return rounded.isoformat()


def _parse_str(value: Any, *, label: str, required: bool = True) -> str:
    text = "" if value is None else str(value)
    text = text.strip()
    if required and not text:
        raise InvalidProvenancePayload(f"{label} must be a non-empty string")
    return text


def _normalize_why(value: Any) -> str:
    why = _parse_str(value, label="why", required=True)
    if len(why) > MAX_WHY_LENGTH:
        return why[: MAX_WHY_LENGTH - 3] + "..."
    return why


def _parse_iso_ts(value: Any) -> str:
    if isinstance(value, (int, float)):
        value = datetime.fromtimestamp(value, tz=timezone.utc).isoformat()
    if not isinstance(value, str):
        raise InvalidProvenancePayload("ts must be an ISO-8601 timestamp")
    text = value.strip()
    if not text:
        raise InvalidProvenancePayload("ts must not be empty")
    try:
        parsed = datetime.fromisoformat(text)
    except ValueError as exc:
        raise InvalidProvenancePayload("ts must be an ISO-8601 timestamp") from exc
    if parsed.tzinfo is None:
        raise InvalidProvenancePayload("ts must include UTC offset")
    if parsed.tzinfo.utcoffset(parsed) is None:
        raise InvalidProvenancePayload("ts must include UTC offset")
    return text


def _is_placeholder(value: str) -> bool:
    return value.strip().lower() in {"-", "n/a", "null", "unknown", "tbd"}


def _normalize_registry_text(
    value: Any,
    *,
    label: str,
    required: bool = True,
    max_length: Optional[int] = None,
) -> str:
    text = _parse_str(value, label=label, required=required)
    if required and _is_placeholder(text):
        raise InvalidProvenancePayload(f"{label} must not be a placeholder")
    if max_length is not None and len(text) > max_length:
        raise InvalidProvenancePayload(f"{label} exceeds maximum length of {max_length}")
    return text


def _compose_artifact_from_context(payload: Dict[str, Any]) -> str:
    artifact = str(payload.get("artifact", "") or "").strip()
    if artifact:
        return artifact
    target_kind = str(payload.get("target_kind", "") or "").strip()
    target_id = str(payload.get("target_id", "") or "").strip()
    scope_key = str(payload.get("scope_key", "") or "default").strip() or "default"
    if target_kind and target_id:
        return f"{target_kind}|{scope_key}|{target_id}"
    if target_id:
        return target_id
    return _normalize_registry_text(payload.get("artifact"), label="artifact", required=True)


def _compose_session_id_from_context(payload: Dict[str, Any]) -> str:
    session_id = str(payload.get("session_id", "") or "").strip()
    if session_id:
        return session_id
    target_kind = str(payload.get("target_kind", "") or "").strip()
    target_id = str(payload.get("target_id", "") or "").strip()
    scope_key = str(payload.get("scope_key", "") or "default").strip() or "default"
    if target_kind and target_id:
        return f"{target_kind}|{scope_key}|{target_id}"
    return _normalize_registry_text(payload.get("session_id"), label="session_id", required=True)


def _compose_why_from_context(payload: Dict[str, Any], *, action: Action, artifact: str) -> str:
    for key in (
        "why",
        "reason",
        "message",
        "notes",
        "summary",
        "approval_reason",
        "trigger",
        "url",
        "selector",
        "input_preview",
    ):
        raw_value = str(payload.get(key, "") or "")
        if raw_value.strip():
            return raw_value.strip()
    target_kind = str(payload.get("target_kind", "") or "").strip()
    target_id = str(payload.get("target_id", "") or "").strip()
    if target_kind and target_id:
        return f"{action.value.lower()} {target_kind} {target_id}"
    return f"{action.value.lower()} {artifact}"


def _compose_risk_from_context(payload: Dict[str, Any], *, action: Action) -> Risk:
    raw_risk = payload.get("risk")
    if raw_risk is None:
        raw_risk = payload.get("risk_level", payload.get("risk_state"))
    if raw_risk is None:
        return DEFAULT_RISK_BY_ACTION[action]
    return _normalize_risk_value(raw_risk)


def _parse_source_trace(value: Any) -> Optional[Dict[str, Any]]:
    if value is None:
        return None
    if isinstance(value, str):
        try:
            parsed = json.loads(value)
        except json.JSONDecodeError:
            return {"raw": value}
        if isinstance(parsed, dict):
            return parsed
        return {"value": parsed}
    if isinstance(value, dict):
        return value
    raise InvalidProvenancePayload("source_trace must be an object or JSON string")


def _normalize_risk_value(value: Any) -> Risk:
    if isinstance(value, Risk):
        return value
    if not isinstance(value, str):
        raise InvalidProvenancePayload("risk must be a string")

    normalized = value.strip()
    if not normalized:
        raise InvalidProvenancePayload("risk must be a non-empty string")

    try:
        return Risk(normalized.upper())
    except ValueError as exc:
        raise InvalidProvenancePayload(f"unsupported risk: {value}") from exc


def _parse_version(raw: Any) -> str:
    if isinstance(raw, int) and not isinstance(raw, bool):
        return str(raw)
    if isinstance(raw, str):
        normalized = raw.strip()
        if normalized:
            return normalized
    raise InvalidProvenancePayload("v must be a supported version")


def _validate_registry_raw_text(raw_text: str) -> None:
    if not isinstance(raw_text, str):
        raise InvalidProvenancePayload("registry line must be a string")
    if not raw_text.isprintable():
        raise InvalidProvenancePayload("registry line must contain printable text only")


def _encode_registry_field(value: str) -> str:
    encoded: List[str] = []
    for char in value:
        if char == "\\":
            encoded.append("\\\\")
        elif char == "|":
            encoded.append("\\|")
        elif char == "\n":
            encoded.append("\\n")
        elif char == "\r":
            encoded.append("\\r")
        elif not char.isprintable():
            encoded.append(f"\\u{ord(char):04x}")
        else:
            encoded.append(char)
    return "".join(encoded)


def _decode_registry_field(value: str) -> str:
    decoded: List[str] = []
    i = 0
    while i < len(value):
        char = value[i]
        if char != "\\":
            decoded.append(char)
            i += 1
            continue

        if i + 1 >= len(value):
            raise InvalidProvenancePayload("registry field contains an incomplete escape sequence")

        escape = value[i + 1]
        if escape == "\\":
            decoded.append("\\")
            i += 2
            continue
        if escape == "|":
            decoded.append("|")
            i += 2
            continue
        if escape == "n":
            decoded.append("\n")
            i += 2
            continue
        if escape == "r":
            decoded.append("\r")
            i += 2
            continue
        if escape == "u":
            if i + 6 > len(value):
                raise InvalidProvenancePayload("registry field contains an incomplete unicode escape")
            digits = value[i + 2 : i + 6]
            if any(digit not in string.hexdigits for digit in digits):
                raise InvalidProvenancePayload("registry field contains an invalid unicode escape")
            decoded.append(chr(int(digits, 16)))
            i += 6
            continue
        raise InvalidProvenancePayload("registry field contains an invalid escape sequence")
    return "".join(decoded)


def _normalize_registry_entry(entry: Any) -> ProvenanceRecordV1:
    record = _coerce_registry_entry(entry)
    return ProvenanceRecordV1(
        session_id=_normalize_registry_text(
            record.session_id,
            label="session_id",
            required=True,
            max_length=MAX_REGISTRY_SESSION_ID_LENGTH,
        ),
        artifact=_normalize_registry_text(
            record.artifact,
            label="artifact",
            required=True,
            max_length=MAX_REGISTRY_ARTIFACT_LENGTH,
        ),
        action=parse_action(record.action, strict=True),
        why=_normalize_registry_text(record.why, label="why", required=True, max_length=MAX_REGISTRY_WHY_LENGTH),
        risk=parse_risk(record.risk, strict=True),
        ts=_parse_iso_ts(record.ts) if record.ts is not None else None,
        actor=_parse_str(record.actor, label="actor", required=False) or None,
        source_trace=_parse_source_trace(record.source_trace),
    )


def validate_registry_entry(entry: Any) -> ProvenanceRecordV1:
    """Normalize a registry entry into the canonical record contract."""

    return _normalize_registry_entry(entry)


def create_registry_entry(
    *,
    session_id: str,
    artifact: str,
    action: Any,
    why: str,
    risk: Any,
) -> ProvenanceRecordV1:
    """Create a canonical registry entry from the five-field registry shape."""

    return validate_registry_entry(
        ProvenanceRecordV1(
            session_id=session_id,
            artifact=artifact,
            action=parse_action(action, strict=False),
            why=why,
            risk=_normalize_risk_value(risk),
        )
    )


def parse_action(value: Any, *, strict: bool = True) -> Action:
    if isinstance(value, Action):
        return value
    if not isinstance(value, str):
        raise InvalidProvenancePayload("action must be a string")

    normalized = value.strip()
    if not normalized:
        raise InvalidProvenancePayload("action must be a non-empty string")

    if not strict:
        legacy = LEGACY_ACTION_TO_ACTION.get(normalized)
        if legacy is not None:
            return legacy
        legacy = LEGACY_ACTION_TO_ACTION.get(normalized.lower())
        if legacy is not None:
            return legacy

    try:
        return Action(normalized)
    except ValueError as exc:
        raise InvalidProvenancePayload(f"unsupported action: {value}") from exc


def parse_risk(value: Any, *, strict: bool = True) -> Risk:
    if isinstance(value, Risk):
        return value
    if not isinstance(value, str):
        raise InvalidProvenancePayload("risk must be a string")

    normalized = value.strip()
    if not normalized:
        raise InvalidProvenancePayload("risk must be a non-empty string")

    try:
        return Risk(normalized)
    except ValueError as exc:
        if strict:
            raise InvalidProvenancePayload(f"unsupported risk: {value}") from exc
        return Risk.LOW


def _build_record_from_payload(payload: Dict[str, Any]) -> ProvenanceRecordV1:
    all_keys = set(payload.keys())
    unknown_fields = all_keys - CANONICAL_FIELDS - {"v"}
    if unknown_fields:
        raise InvalidProvenancePayload(f"unknown provenance fields: {', '.join(sorted(unknown_fields))}")

    missing_required = CANONICAL_REQUIRED_FIELDS - all_keys
    if missing_required:
        raise InvalidProvenancePayload(f"missing provenance fields: {', '.join(sorted(missing_required))}")

    session_id = _parse_str(payload.get("session_id"), label="session_id", required=True)
    artifact = _parse_str(payload.get("artifact"), label="artifact", required=True)
    action = parse_action(payload.get("action"), strict=True)
    why = _normalize_why(payload.get("why"))
    risk = parse_risk(payload.get("risk"), strict=True)
    ts = _parse_iso_ts(payload.get("ts")) if payload.get("ts") is not None else None
    actor = _parse_str(payload.get("actor"), label="actor", required=False)
    if not actor:
        actor = None
    source_trace = _parse_source_trace(payload.get("source_trace"))

    return ProvenanceRecordV1(
        session_id=session_id,
        artifact=artifact,
        action=action,
        why=why,
        risk=risk,
        ts=ts,
        actor=actor,
        source_trace=source_trace,
    )


def build_entry_from_mapping(
    payload: Dict[str, Any],
    *,
    default_action: Action = Action.UPDATE,
    default_risk: Risk = Risk.LOW,
) -> ProvenanceRecordV1:
    return upgrade_to_v1(payload, default_action=default_action, default_risk=default_risk)


def build_entry(
    *,
    session_id: str,
    artifact: str,
    action: Any,
    why: str,
    risk: Any,
    ts: Optional[str] = None,
    actor: Optional[str] = None,
    source_trace: Optional[Dict[str, Any]] = None,
) -> ProvenanceRecordV1:
    normalized_session = _parse_str(session_id, label="session_id", required=True)
    normalized_artifact = _parse_str(artifact, label="artifact", required=True)
    normalized_action = parse_action(action, strict=True)
    normalized_why = _normalize_why(why)
    normalized_risk = parse_risk(risk, strict=True) if risk is not None else DEFAULT_RISK_BY_ACTION[normalized_action]
    parsed_ts = _parse_iso_ts(ts) if ts is not None else now_iso8601_ts()

    return ProvenanceRecordV1(
        session_id=normalized_session,
        artifact=normalized_artifact,
        action=normalized_action,
        why=normalized_why,
        risk=normalized_risk,
        ts=parsed_ts,
        actor=_parse_str(actor, label="actor", required=False) or None,
        source_trace=source_trace,
    )


def build_registry_entry_from_event_context(
    payload: Dict[str, Any],
    *,
    default_action: Action = Action.UPDATE,
    default_risk: Risk = Risk.LOW,
) -> ProvenanceRecordV1:
    """Normalize event context into the canonical registry shape and optional metadata."""

    if not isinstance(payload, dict):
        raise InvalidProvenancePayload("registry event context must be an object")

    action_value = payload.get("action", payload.get("legacy_action", default_action.value))
    action = parse_action(action_value, strict=False)
    artifact = _compose_artifact_from_context(payload)
    session_id = _compose_session_id_from_context(payload)
    why = _compose_why_from_context(payload, action=action, artifact=artifact)
    risk = _compose_risk_from_context(payload, action=action)
    ts = _parse_iso_ts(payload.get("ts")) if payload.get("ts") is not None else None
    actor = _parse_str(payload.get("actor"), label="actor", required=False) or None
    source_trace = _parse_source_trace(payload.get("source_trace"))
    if risk is None:  # pragma: no cover - parse_risk(strict=False) always returns a Risk
        risk = default_risk

    return ProvenanceRecordV1(
        session_id=_normalize_registry_text(
            session_id,
            label="session_id",
            required=True,
            max_length=MAX_REGISTRY_SESSION_ID_LENGTH,
        ),
        artifact=_normalize_registry_text(
            artifact,
            label="artifact",
            required=True,
            max_length=MAX_REGISTRY_ARTIFACT_LENGTH,
        ),
        action=action,
        why=_normalize_registry_text(why, label="why", required=True, max_length=MAX_REGISTRY_WHY_LENGTH),
        risk=risk,
        ts=ts,
        actor=actor,
        source_trace=source_trace,
    )


def _split_canonical_registry_line(raw_line: str) -> List[str]:
    _validate_registry_raw_text(raw_line)
    fields: List[str] = []
    current: List[str] = []
    escape = False
    i = 0
    while i < len(raw_line):
        char = raw_line[i]
        if escape:
            if char not in {"\\", "|", "n", "r", "u"}:
                raise InvalidProvenancePayload("registry field contains an invalid escape sequence")
            current.append("\\" + char)
            escape = False
            i += 1
            continue
        if char == "\\":
            escape = True
            i += 1
            continue
        if raw_line.startswith(CANONICAL_REGISTRY_DELIMITER, i):
            fields.append("".join(current))
            current = []
            i += len(CANONICAL_REGISTRY_DELIMITER)
            continue
        if char == "|":
            raise InvalidProvenancePayload("registry field contains an unescaped pipe")
        current.append(char)
        i += 1
    if escape:
        raise InvalidProvenancePayload("registry field contains an incomplete escape sequence")
    fields.append("".join(current))
    if len(fields) != len(CANONICAL_REGISTRY_FIELD_ORDER):
        raise InvalidProvenancePayload("registry line must contain exactly five fields")
    return fields


def _normalize_canonical_registry_field(value: str, *, label: str, max_length: int) -> str:
    normalized = _normalize_registry_text(value, label=label, required=True, max_length=max_length)
    if normalized != value:
        raise InvalidProvenancePayload(f"{label} must be canonical")
    return normalized


def decode_provenance_entry(raw_line: str) -> ProvenanceRecordV1:
    fields = _split_canonical_registry_line(raw_line)
    decoded = [_decode_registry_field(field) for field in fields]
    payload = dict(zip(CANONICAL_REGISTRY_FIELD_ORDER, decoded, strict=True))
    return validate_registry_entry(
        ProvenanceRecordV1(
            session_id=_normalize_canonical_registry_field(
                payload["session_id"], label="session_id", max_length=MAX_REGISTRY_SESSION_ID_LENGTH
            ),
            artifact=_normalize_canonical_registry_field(
                payload["artifact"], label="artifact", max_length=MAX_REGISTRY_ARTIFACT_LENGTH
            ),
            action=parse_action(payload["action"], strict=True),
            why=_normalize_canonical_registry_field(payload["why"], label="why", max_length=MAX_REGISTRY_WHY_LENGTH),
            risk=parse_risk(payload["risk"], strict=True),
        )
    )


def parse_registry_line(raw_line: str) -> ProvenanceRecordV1:
    return decode_provenance_entry(raw_line)


def parse_registry_entry(raw_line: str) -> ProvenanceRecordV1:
    return decode_provenance_entry(raw_line)


def _coerce_registry_entry(entry: Any) -> ProvenanceRecordV1:
    if isinstance(entry, ProvenanceRecordV1):
        return entry
    if isinstance(entry, Mapping):
        return _build_record_from_payload(dict(entry))
    raise InvalidProvenancePayload("entry must be a ProvenanceRecordV1 or mapping")


def _coerce_registry_entry_lenient(entry: Any) -> ProvenanceRecordV1:
    if isinstance(entry, ProvenanceRecordV1):
        return entry
    if isinstance(entry, Mapping):
        payload = dict(entry)
        return ProvenanceRecordV1(
            session_id=_parse_str(payload.get("session_id"), label="session_id", required=False),
            artifact=_parse_str(payload.get("artifact"), label="artifact", required=False),
            action=parse_action(payload.get("action"), strict=True),
            why=_parse_str(payload.get("why"), label="why", required=False),
            risk=parse_risk(payload.get("risk"), strict=True),
            ts=_parse_iso_ts(payload.get("ts")) if payload.get("ts") is not None else None,
            actor=_parse_str(payload.get("actor"), label="actor", required=False) or None,
            source_trace=_parse_source_trace(payload.get("source_trace")),
        )
    raise InvalidProvenancePayload("entry must be a ProvenanceRecordV1 or mapping")


def encode_provenance_entry(entry: Any) -> str:
    record = validate_registry_entry(entry)
    fields = [
        _encode_registry_field(
            _normalize_registry_text(
                record.session_id,
                label="session_id",
                required=True,
                max_length=MAX_REGISTRY_SESSION_ID_LENGTH,
            )
        ),
        _encode_registry_field(
            _normalize_registry_text(
                record.artifact,
                label="artifact",
                required=True,
                max_length=MAX_REGISTRY_ARTIFACT_LENGTH,
            )
        ),
        record.action.value,
        _encode_registry_field(
            _normalize_registry_text(
                record.why,
                label="why",
                required=True,
                max_length=MAX_REGISTRY_WHY_LENGTH,
            )
        ),
        record.risk.value,
    ]
    return CANONICAL_REGISTRY_DELIMITER.join(fields)


def serialize_registry_line(entry: ProvenanceRecordV1) -> str:
    return encode_provenance_entry(entry)


def serialize_registry_entry(entry: ProvenanceRecordV1) -> str:
    return encode_provenance_entry(entry)


def format_provenance_entry(entry: Any) -> str:
    """Compatibility formatter that preserves empty fields for round-trips."""

    record = _coerce_registry_entry_lenient(entry)
    fields = [
        _encode_registry_field(record.session_id),
        _encode_registry_field(record.artifact),
        record.action.value,
        _encode_registry_field(record.why),
        record.risk.value,
    ]
    return CANONICAL_REGISTRY_DELIMITER.join(fields)


def parse_provenance_entry(raw_line: str) -> ProvenanceRecordV1:
    """Compatibility parser that preserves empty fields for round-trips."""

    fields = _split_canonical_registry_line(raw_line)
    decoded = [_decode_registry_field(field) for field in fields]
    payload = dict(zip(CANONICAL_REGISTRY_FIELD_ORDER, decoded, strict=True))
    return ProvenanceRecordV1(
        session_id=_parse_str(payload["session_id"], label="session_id", required=False),
        artifact=_parse_str(payload["artifact"], label="artifact", required=False),
        action=parse_action(payload["action"], strict=True),
        why=_parse_str(payload["why"], label="why", required=False),
        risk=parse_risk(payload["risk"], strict=True),
    )


def read_registry_entry(raw_text: str) -> ProvenanceRecordV1:
    """Extract one canonical registry entry from noisy multi-line text."""

    entries = read_registry_entries(raw_text)
    if len(entries) == 1:
        return entries[0]
    if not entries:
        raise InvalidProvenancePayload("no canonical registry entry found")
    raise InvalidProvenancePayload("multiple canonical registry entries found")


def read_registry_entries(raw_text: str) -> List[ProvenanceRecordV1]:
    """Return all canonical registry entries recovered from noisy multi-line text."""

    if not isinstance(raw_text, str):
        raise InvalidProvenancePayload("registry text must be a string")

    candidates: List[ProvenanceRecordV1] = []
    for raw_line in raw_text.splitlines():
        line = _ANSI_ESCAPE_RE.sub("", raw_line)
        if not line.strip():
            continue
        try:
            candidates.append(parse_registry_entry(line))
        except InvalidProvenancePayload:
            continue
    return candidates


def _parse_legacy_registry_line(raw_line: str) -> ProvenanceRecordV1:
    """Parse the historical backslash-escaped registry line format."""

    _validate_registry_raw_text(raw_line)
    fields: List[str] = []
    current: List[str] = []
    escape = False
    for char in raw_line:
        if escape:
            current.append("\\")
            current.append(char)
            escape = False
            continue
        if char == "\\":
            escape = True
            continue
        if char == "|":
            fields.append("".join(current).strip())
            current = []
            continue
        current.append(char)
    if escape:
        raise InvalidProvenancePayload("registry line ends with an incomplete escape sequence")
    fields.append("".join(current).strip())
    if len(fields) != len(CANONICAL_REGISTRY_FIELD_ORDER):
        raise InvalidProvenancePayload("registry line must contain exactly five fields")
    decoded = [_decode_registry_field(field) for field in fields]
    payload = dict(zip(CANONICAL_REGISTRY_FIELD_ORDER, decoded, strict=True))
    return ProvenanceRecordV1(
        session_id=_normalize_registry_text(
            payload["session_id"], label="session_id", required=True, max_length=MAX_REGISTRY_SESSION_ID_LENGTH
        ),
        artifact=_normalize_registry_text(
            payload["artifact"], label="artifact", required=True, max_length=MAX_REGISTRY_ARTIFACT_LENGTH
        ),
        action=parse_action(payload["action"], strict=True),
        why=_normalize_registry_text(payload["why"], label="why", required=True, max_length=MAX_REGISTRY_WHY_LENGTH),
        risk=parse_risk(payload["risk"], strict=True),
    )


def select_registry_sink(runtime: Any, default_path: Path) -> Any:
    ctx = getattr(runtime, "ctx", None)
    if ctx is None:
        return default_path

    for candidate_name in (
        "operator_action_sink",
        "operator_action_store",
        "registry_sink",
        "registry_store",
        "operator_action_log_sink",
        "registry_log_sink",
        "log_sink",
        "sink",
    ):
        candidate = getattr(ctx, candidate_name, None)
        if candidate is not None:
            return candidate

    config = getattr(ctx, "config", None)
    for candidate_name in (
        "operator_action_sink",
        "operator_action_store",
        "registry_sink",
        "registry_store",
        "operator_action_log_sink",
        "registry_log_sink",
        "log_sink",
        "sink",
    ):
        candidate = getattr(config, candidate_name, None)
        if candidate is not None:
            return candidate

    return default_path


def append_registry_entry(sink: Any, entry: ProvenanceRecordV1) -> str:
    line = serialize_registry_entry(entry)

    try:
        if isinstance(sink, (str, Path, os.PathLike)):
            path = Path(sink)
            path.parent.mkdir(parents=True, exist_ok=True)
            with path.open("a", encoding="utf-8") as handle:
                handle.write(line + "\n")
                handle.flush()
                os.fsync(handle.fileno())
            return line

        append_fn = getattr(sink, "append", None)
        if callable(append_fn):
            result = append_fn(line)
            if result is False:
                raise RegistrySinkError("registry sink rejected the entry")
            return line

        write_fn = getattr(sink, "write", None)
        if callable(write_fn):
            result = write_fn(line + "\n")
            if result is False:
                raise RegistrySinkError("registry sink rejected the entry")
            flush_fn = getattr(sink, "flush", None)
            if callable(flush_fn):
                flush_fn()
            return line
    except RegistrySinkError:
        raise
    except Exception as exc:
        raise RegistrySinkError(f"failed to write registry entry: {exc}") from exc

    raise RegistrySinkError("registry sink is unavailable or unsupported")


def append_registry_entry_from_event_context(
    runtime: Any,
    payload: Dict[str, Any],
    *,
    default_path: Path,
    default_action: Action = Action.UPDATE,
    default_risk: Risk = Risk.LOW,
) -> str:
    """Build a canonical registry entry from event context and persist it."""

    registry_entry = build_registry_entry_from_event_context(
        payload,
        default_action=default_action,
        default_risk=default_risk,
    )
    sink = select_registry_sink(runtime, default_path)
    return append_registry_entry(sink, registry_entry)


def upgrade_to_v1(
    payload: Dict[str, Any],
    *,
    default_action: Action = Action.UPDATE,
    default_risk: Risk = Risk.LOW,
) -> ProvenanceRecordV1:
    if not isinstance(payload, dict):
        raise InvalidProvenancePayload("provenance payload must be an object")

    version = payload.get("v")

    if version is None:
        # Legacy payload path
        normalized_session = _parse_str(payload.get("session_id"), label="session_id", required=True)
        normalized_artifact = _parse_str(payload.get("artifact"), label="artifact", required=True)
        raw_action = payload.get("action", default_action.value)
        raw_risk = payload.get("risk", default_risk.value)

        normalized_action = parse_action(raw_action, strict=False)
        normalized_risk = parse_risk(raw_risk, strict=False)
        normalized_ts = _parse_iso_ts(payload.get("ts")) if payload.get("ts") is not None else now_iso8601_ts()
        normalized_why = _normalize_why(payload.get("why"))
        normalized_actor = _parse_str(payload.get("actor"), label="actor", required=False) or None

        return ProvenanceRecordV1(
            session_id=normalized_session,
            artifact=normalized_artifact,
            action=normalized_action,
            why=normalized_why,
            risk=normalized_risk,
            ts=normalized_ts,
            actor=normalized_actor,
        )

    parsed_version = _parse_version(version)
    if parsed_version != CURRENT_PROVENANCE_VERSION:
        raise UnsupportedProvenanceVersion(f"Unsupported provenance version: {version}")

    return _build_record_from_payload(payload)


def downgrade_from_v1(
    entry: ProvenanceRecordV1,
    *,
    target_version: str = CURRENT_PROVENANCE_VERSION,
) -> Dict[str, Any]:
    if target_version not in SUPPORTED_PROVENANCE_VERSIONS:
        raise UnsupportedProvenanceVersion(f"Unsupported provenance version: {target_version}")
    if target_version != CURRENT_PROVENANCE_VERSION:
        raise UnsupportedProvenanceVersion(f"Unsupported provenance downgrade target: {target_version}")

    payload = entry.to_dict()
    payload["v"] = target_version
    return payload


def parse_or_upgrade(
    raw_line: str,
    *,
    compatibility_shim: Optional[Callable[[str, Dict[str, Any]], Dict[str, Any]]] = None,
) -> ProvenanceRecordV1:
    if not isinstance(raw_line, str):
        raise InvalidProvenancePayload("provenance record must be a string")

    if not raw_line.lstrip().startswith("{"):
        return decode_provenance_entry(raw_line)

    try:
        payload = json.loads(raw_line)
    except json.JSONDecodeError as exc:
        raise InvalidProvenancePayload("invalid provenance json") from exc

    if not isinstance(payload, dict):
        raise InvalidProvenancePayload("provenance record must be a json object")

    if "v" not in payload:
        return upgrade_to_v1(payload)

    version = _parse_version(payload.get("v"))
    if version == CURRENT_PROVENANCE_VERSION:
        return _build_record_from_payload(payload)

    if compatibility_shim is None:
        raise UnsupportedProvenanceVersion(f"Unsupported provenance version: {version}")

    projected = compatibility_shim(version, payload)
    if not isinstance(projected, dict):
        raise InvalidProvenancePayload("compatibility shim must produce a mapping")
    if "v" not in projected:
        projected = {**projected, "v": CURRENT_PROVENANCE_VERSION}
    if not isinstance(projected, dict):
        raise InvalidProvenancePayload("compatibility shim must return a mapping")

    return parse_or_upgrade(json.dumps(projected), compatibility_shim=None)


def serialize(entry: ProvenanceRecordV1) -> str:
    if not isinstance(entry, ProvenanceRecordV1):
        raise InvalidProvenancePayload("entry must be a ProvenanceRecordV1")
    payload = entry.to_dict()
    # Include v first and keep a stable, documented key order for JSONL lines.
    ordered_payload = {"v": CURRENT_PROVENANCE_VERSION}
    for key in CANONICAL_FIELD_ORDER_JSONL:
        if key in payload:
            ordered_payload[key] = payload[key]
    return json.dumps(ordered_payload, ensure_ascii=True, separators=(",", ":"))


def deserialize(
    raw_line: str,
    *,
    compatibility_shim: Optional[Callable[[str, Dict[str, Any]], Dict[str, Any]]] = None,
) -> ProvenanceRecordV1:
    return parse_or_upgrade(raw_line, compatibility_shim=compatibility_shim)


def parse_as_v1_records(
    raw_lines: str,
    *,
    compatibility_shim: Optional[Callable[[str, Dict[str, Any]], Dict[str, Any]]] = None,
) -> List[ProvenanceRecordV1]:
    output: List[ProvenanceRecordV1] = []
    for raw_line in raw_lines.splitlines():
        if not raw_line.strip():
            continue
        output.append(deserialize(raw_line, compatibility_shim=compatibility_shim))
    return output


def append_jsonl_record(path: Path, entry: ProvenanceRecordV1) -> str:
    line = serialize(entry)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(line + "\n")
        handle.flush()
        os.fsync(handle.fileno())
    return line


def project_provenance_entry(entry: ProvenanceRecordV1) -> Dict[str, Any]:
    """Project a canonical provenance record into plain JSON-serializable data."""

    if not isinstance(entry, ProvenanceRecordV1):
        raise InvalidProvenancePayload("entry must be a ProvenanceRecordV1")
    return entry.to_dict()


def attach_provenance(
    record: Dict[str, Any],
    entry: ProvenanceRecordV1,
    *,
    field: str = "provenance",
) -> Dict[str, Any]:
    """Attach a canonical provenance projection to an existing record mapping."""

    if not isinstance(record, dict):
        raise InvalidProvenancePayload("record must be an object")
    projected = dict(record)
    provenance = project_provenance_entry(entry)
    provenance["transport_line"] = format_provenance_entry(provenance)
    projected[field] = provenance
    return projected


def append_provenance_event(
    record: Dict[str, Any],
    entry: ProvenanceRecordV1,
    *,
    field: str = "provenance_events",
) -> Dict[str, Any]:
    """Append a canonical provenance projection to a list on an existing record."""

    if not isinstance(record, dict):
        raise InvalidProvenancePayload("record must be an object")
    projected = dict(record)
    events = list(projected.get(field) or [])
    provenance = project_provenance_entry(entry)
    provenance["transport_line"] = format_provenance_entry(provenance)
    events.append(provenance)
    projected[field] = events
    return projected


def append_provenance_record(
    record: Dict[str, Any] | None,
    *,
    session_id: str,
    artifact: str,
    action: Any,
    why: str,
    risk: Any,
    field: str = "provenance",
    ts: Optional[str] = None,
    actor: Optional[str] = None,
    source_trace: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """Build a canonical entry once and attach it to a record mapping."""

    entry = build_entry(
        session_id=session_id,
        artifact=artifact,
        action=action,
        why=why,
        risk=risk,
        ts=ts,
        actor=actor,
        source_trace=source_trace,
    )
    projected = record or {}
    if field == "provenance_events":
        return append_provenance_event(projected, entry, field=field)
    return attach_provenance(projected, entry, field=field)


# Compatibility helpers exported for legacy call sites.
def infer_session_id(*, target_kind: str, target_id: str, scope_key: str = "default") -> str:
    normalized_scope = _parse_str(scope_key, label="scope_key", required=False) or "default"
    return f"{target_kind}:{normalized_scope}:{target_id}"
