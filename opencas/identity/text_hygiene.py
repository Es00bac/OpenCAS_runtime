"""Identity text helpers for prompt stability and stale-intention detection."""

from __future__ import annotations

import re
from typing import Any, Dict, Optional


BOOTSTRAP_PLACEHOLDER_INTENTION = "establish trust and understanding"

RECURSIVE_IDENTITY_LOOP_RE = re.compile(
    r"(?i)(?:\breturning\s+to\s+){2,}|\b(?P<term>returning|thread|drifted)\b\s+(?P=term)\b"
)

FORBIDDEN_TERM_REPLACEMENTS: Dict[str, str] = {
    "returning": "revisiting",
    "thread": "path",
    "drifted": "shifted",
}

FORBIDDEN_TERM_PATTERNS: Dict[str, str] = {
    "returning": r"\breturning\b",
    "thread": r"\bthreads?\b",
    "drifted": r"\bdrifted\b",
}


def is_bootstrap_placeholder_intention(value: Optional[str]) -> bool:
    """Return True when *value* is still the seed intention."""
    return str(value or "").strip().lower() == BOOTSTRAP_PLACEHOLDER_INTENTION


def collapse_recursive_identity_text(value: Optional[str]) -> Optional[str]:
    """Collapse obvious recursive stutter so prompts do not amplify fixation."""
    if value is None:
        return None
    text = " ".join(str(value).split())
    if not text:
        return text

    # Imported identity text can contain loops like
    # "returning to returning to returning to ...". Collapse these so the
    # runtime sees the fixation as one idea, not an instruction to recurse.
    collapsed = re.sub(
        r"(?i)(?:\breturning to\s+){2,}",
        "returning to ",
        text,
    )
    collapsed = re.sub(
        r"(?i)\b(returning|thread|drifted)\b\s+\1\b",
        r"\1",
        collapsed,
    )
    return collapsed


def sanitize_identity_text(value: Optional[str]) -> str:
    """Apply recursive-term safe normalization used by introspective surfaces."""
    if not value:
        return ""

    sanitized = collapse_recursive_identity_text(str(value))
    assert sanitized is not None

    for term, replacement in FORBIDDEN_TERM_REPLACEMENTS.items():
        pattern = FORBIDDEN_TERM_PATTERNS[term]
        sanitized = re.sub(pattern, replacement, sanitized, flags=re.IGNORECASE)

    return " ".join(sanitized.split())


def has_recursive_identity_loop(value: Optional[str]) -> bool:
    """Return True when recursive identity-token loops are present."""
    if not value:
        return False
    normalized = " ".join(str(value).split())
    if "returning to returning" in normalized.lower():
        return True
    return bool(RECURSIVE_IDENTITY_LOOP_RE.search(normalized))


def sanitize_identity_structure(value: Any) -> Any:
    """Recursively sanitize identity-bearing strings in a JSON-like object."""
    if isinstance(value, str):
        return sanitize_identity_text(value)
    if isinstance(value, dict):
        sanitized = {}
        for key, raw in value.items():
            sanitized_key = sanitize_identity_text(str(key)) if isinstance(key, str) else str(key)
            sanitized[sanitized_key] = sanitize_identity_structure(raw)
        return sanitized
    if isinstance(value, list):
        return [sanitize_identity_structure(item) for item in value]
    return value
