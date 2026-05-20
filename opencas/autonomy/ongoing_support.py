"""Evidence-backed classification for durable operator-support commitments."""

from __future__ import annotations

from typing import Any


SUPPORT_TAGS = {
    "business_model",
    "follow_through",
    "income",
    "jarrod_support",
    "non_employee",
    "operator_support",
    "routine",
    "user_support",
}

SUPPORT_MARKERS = (
    "business model",
    "cash flow",
    "cash-flow",
    "complex multi-step",
    "complex multistep",
    "high priority mission",
    "income",
    "mission",
    "non-employee",
    "ongoing support",
    "proactive support",
    "productive routine",
    "start making income",
    "start making money",
)


def is_ongoing_support_commitment_shape(
    commitment: Any,
    *,
    require_active: bool = False,
) -> bool:
    """Return true when a commitment describes durable operator support.

    The classifier is intentionally content/tag based so evidence gates can
    still recognize an ongoing-support commitment before or during a terminal
    status transition.
    """

    if require_active:
        status = getattr(commitment, "status", None)
        status_value = getattr(status, "value", status)
        if str(status_value) != "active":
            return False

    meta = dict(getattr(commitment, "meta", {}) or {})
    if str(meta.get("normalization_source", "")) == "contextual_assistant_acceptance":
        return True

    tags = {str(tag).lower().strip() for tag in (getattr(commitment, "tags", []) or [])}
    text = " ".join(
        str(value)
        for value in (
            getattr(commitment, "content", ""),
            meta.get("source_user_request", ""),
            meta.get("previous_user_turn", ""),
            meta.get("source_sentence", ""),
        )
        if value
    ).lower()
    has_support_marker = any(marker in text for marker in SUPPORT_MARKERS)
    has_support_tag = bool(tags & SUPPORT_TAGS)
    source = str(meta.get("source", ""))
    if has_support_marker and (
        has_support_tag
        or source in {"assistant_response", "workflow_create_commitment"}
    ):
        return True
    return has_support_tag and any(marker in text for marker in ("mission", "income", "support", "routine"))
