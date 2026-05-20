"""Evidence checks for operator-authorized project follow-through."""

from __future__ import annotations

from typing import Any, Iterable, Mapping


SAFE_OPERATOR_PROJECT_FOLLOWTHROUGH_EVIDENCE_KEYS = frozenset(
    {
        "source_session_id",
        "workspace_abs_path",
        "workspace_rel_path",
        "workspace_project_confidence",
        "workspace_project_evidence",
        "requested_workspace_abs_path",
        "requested_workspace_rel_path",
        "requested_workspace_kind",
        "requested_workspace_raw_text",
        "start_policy",
        "return_policy",
        "project_intent",
        "next_step",
        "creative_completion_contract",
        "target_word_count",
        "project_start_contract",
    }
)


def copy_safe_operator_project_followthrough_evidence(
    *metas: Mapping[str, Any] | None,
) -> dict[str, Any]:
    """Copy only narrow operator project-return evidence between derived tasks.

    Later metadata wins so current task evidence can refine older commitment
    evidence, but provenance fields such as ``source`` are intentionally not
    copied. Callers should set their own source for the derived object.
    """

    evidence: dict[str, Any] = {}
    for meta in metas:
        data = meta if isinstance(meta, Mapping) else {}
        for key in SAFE_OPERATOR_PROJECT_FOLLOWTHROUGH_EVIDENCE_KEYS:
            if key not in data:
                continue
            value = data.get(key)
            if _is_empty_evidence_value(value):
                continue
            evidence[key] = value
    return evidence


def _is_empty_evidence_value(value: Any) -> bool:
    if value is None:
        return True
    if isinstance(value, str):
        return not value.strip()
    if isinstance(value, (list, tuple, set, dict)):
        return len(value) == 0
    return False


def has_operator_project_followthrough_authority(
    meta: Mapping[str, Any] | None,
    *,
    tags: Iterable[str] = (),
    allow_schedule_authority: bool = False,
) -> bool:
    """Return true when a project task should bypass overload-only throttling.

    The gate is intentionally narrow: it requires a real operator session and a
    project-followthrough source, then accepts either grounded workspace evidence
    or an immediate operator work contract. This lets new project starts proceed
    before a project root exists without opening fatigue/safety pauses.
    """

    data = dict(meta or {})
    source = str(data.get("source") or "").strip().lower()
    authority = str(data.get("authority") or "").strip().lower()
    normalized_tags = {str(tag).strip().lower() for tag in tags}
    source_session_id = str(data.get("source_session_id") or "").strip()
    if not source_session_id or source_session_id == "system:automated":
        return False
    if source != "project_return_capture" and "project_return" not in normalized_tags:
        if not (allow_schedule_authority and source == "schedule" and authority == "schedule_due"):
            return False
    return (
        _has_grounded_workspace(data)
        or _has_requested_workspace(data)
        or _has_immediate_operator_work_contract(data)
    )


def _has_grounded_workspace(meta: Mapping[str, Any]) -> bool:
    has_workspace = bool(
        str(meta.get("workspace_abs_path") or "").strip()
        or str(meta.get("workspace_rel_path") or "").strip()
    )
    try:
        confidence = float(meta.get("workspace_project_confidence") or 0.0)
    except (TypeError, ValueError):
        confidence = 0.0
    return has_workspace and confidence >= 0.75


def _has_requested_workspace(meta: Mapping[str, Any]) -> bool:
    return bool(
        str(meta.get("requested_workspace_abs_path") or "").strip()
        or str(meta.get("requested_workspace_rel_path") or "").strip()
    )


def _has_immediate_operator_work_contract(meta: Mapping[str, Any]) -> bool:
    return (
        str(meta.get("start_policy") or "").strip().lower() == "immediate_operator_work_request"
        or str(meta.get("return_policy") or "").strip().lower() == "immediate_autonomous_work"
    )
