"""Canonical executive pause-reason normalization utilities."""

from __future__ import annotations

_EXECUTIVE_REASON_PREFIX = "executive_"
_KNOWN_EXECUTIVE_PAUSE_REASONS = frozenset(
    {
        "executive_pause",
        "executive_fatigue",
        "executive_overload",
        "executive_operator_rest",
    }
)
_KNOWN_RAW_EXECUTIVE_PAUSE_REASONS = frozenset(
    reason.removeprefix(_EXECUTIVE_REASON_PREFIX) for reason in _KNOWN_EXECUTIVE_PAUSE_REASONS
)


def _normalize_executive_pause_reason_blocked_value(reason: str) -> str:
    """Normalize any stored blocked reason into one canonical form."""
    normalized = str(reason or "").strip().lower()
    if not normalized:
        return ""
    if normalized in _KNOWN_RAW_EXECUTIVE_PAUSE_REASONS:
        return f"{_EXECUTIVE_REASON_PREFIX}{normalized}"
    return normalized


def is_known_executive_pause_reason(reason: str) -> bool:
    """Return True if the reason is one of the known canonical executive pause reasons."""
    return _normalize_executive_pause_reason_blocked_value(reason) in _KNOWN_EXECUTIVE_PAUSE_REASONS


def is_executive_pause_reason(reason: str) -> bool:
    """
    Return True when a blocked reason came from executive pause state.

    The return intentionally accepts all canonical executive_ prefixed values so
    new pause reasons can be resumed without a release-only code change.
    """
    normalized = _normalize_executive_pause_reason_blocked_value(reason)
    if not normalized:
        return False
    return normalized.startswith(_EXECUTIVE_REASON_PREFIX)
