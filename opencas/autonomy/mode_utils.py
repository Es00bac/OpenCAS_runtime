"""Approval-mode normalization helpers."""

from __future__ import annotations

from opencas.autonomy.models import ApprovalMode


def normalize_approval_mode(value: ApprovalMode | str | None) -> ApprovalMode:
    """Normalize user/config approval-mode values into ``ApprovalMode``."""
    if isinstance(value, ApprovalMode):
        return value
    cleaned = str(value or ApprovalMode.DEFAULT.value).strip().lower().replace("-", "_")
    if cleaned == "yolo":
        cleaned = ApprovalMode.FULLY_AUTONOMOUS.value
    return ApprovalMode(cleaned)
