"""Helpers for advisory audit/eval turns that must not become memory."""

from __future__ import annotations

from typing import Any


AUDIT_ONLY_MARKERS: tuple[str, ...] = (
    "audit-only",
    "eval-only",
    "evaluation-only",
    "benchmark-only",
    "dry-run probe",
    "e16 audit",
    "phenomenological audit",
)


def is_audit_only_text(*values: Any) -> bool:
    """Return true when any text payload is explicitly marked audit-only."""
    haystack = "\n".join(str(value or "") for value in values).lower()
    return any(marker in haystack for marker in AUDIT_ONLY_MARKERS)


def with_audit_only_meta(meta: Any, *, audit_only: bool) -> dict[str, Any]:
    """Return a mutable metadata dict with the audit-only flag when needed."""
    payload = dict(meta or {}) if isinstance(meta, dict) else {}
    if audit_only:
        payload["audit_only"] = True
    return payload
