"""Completion-evidence rules for commitments that cannot close on word overlap."""

from __future__ import annotations

import re
from typing import Any

from opencas.autonomy.ongoing_support import is_ongoing_support_commitment_shape
from opencas.projects.execution_contracts import new_project_completion_rejection_reason

_MANUSCRIPT_WORD_COUNT_PATTERNS = (
    re.compile(
        r"\b(?:current|actual|total)\s+(?:manuscript\s+)?word\s+count\s*(?:is|:)?\s*"
        r"(?P<count>\d{1,3}(?:,\d{3})+|\d+)\s*words?\b",
        re.IGNORECASE,
    ),
    re.compile(
        r"\b(?P<count>\d{1,3}(?:,\d{3})+|\d+)\s*(?:manuscript|draft)\s+words?\b",
        re.IGNORECASE,
    ),
)
_ARTIFACT_REFERENCE_PATTERNS = (
    re.compile(r"\b(?:workspace|drafts|manuscripts?|artifacts?)/[^\s,;:]+\.(?:md|markdown|txt)\b", re.IGNORECASE),
    re.compile(r"\b[^\s,;:]+/(?:[^\s,;:]+/)*[^\s,;:]+\.(?:md|markdown|txt)\b", re.IGNORECASE),
    re.compile(r"\bartifacts?\s*:\s*[^\s,;:]+\.(?:md|markdown|txt)\b", re.IGNORECASE),
)
_GLOB_CHARS = frozenset("*?[]")

_EVIDENCE_REQUIRED_MARKERS = (
    "build",
    "cmake",
    "compile",
    "diagnostic",
    "launch",
    "lint",
    "proof",
    "pytest",
    "run",
    "smoke test",
    "test",
    "verify",
)
_NEGATIVE_EVIDENCE_MARKERS = (
    "cannot verify",
    "can't verify",
    "failed",
    "failure",
    "not compiled",
    "not yet",
    "not verified",
    "no shell access",
    "remaining",
    "requires shell access",
    "todo",
    "unverified",
)
_FIRST_DRAFT_ONLY_MARKERS = (
    "first draft complete",
    "first full draft complete",
    "first full manuscript draft complete",
    "full first manuscript draft complete",
)
_CLEAN_REVISION_EVIDENCE_MARKERS = (
    "clean manuscript",
    "clean revised manuscript",
    "line edit",
    "publication-clean",
    "publishable",
    "revised clean manuscript",
    "revised manuscript",
    "second draft",
)
_NAME_ORIGINALITY_EVIDENCE_MARKERS = (
    "name/place",
    "name and place",
    "name checks",
    "naming checks",
    "originality",
    "research log",
)


def requires_explicit_completion_evidence(commitment: Any) -> bool:
    """Return true when a commitment needs proof instead of heuristic closure."""

    meta = getattr(commitment, "meta", {}) or {}
    tags = {str(tag).strip().lower() for tag in (getattr(commitment, "tags", []) or [])}
    content = str(getattr(commitment, "content", "") or "").lower()
    project_type = str(meta.get("project_type", "") or "").lower()

    if is_ongoing_support_commitment_shape(commitment):
        return True
    if "project_return" in tags or meta.get("source") == "project_return_capture":
        return True
    if project_type == "software" or "software" in tags:
        return any(marker in content for marker in _EVIDENCE_REQUIRED_MARKERS)
    return any(marker in content for marker in _EVIDENCE_REQUIRED_MARKERS)


def completion_evidence_rejection_reason(commitment: Any, evidence: str | None) -> str | None:
    """Return a rejection reason when completion evidence is missing or negative."""

    if not requires_explicit_completion_evidence(commitment):
        return None
    evidence_text = str(evidence or "").strip()
    if not evidence_text:
        return (
            "Completion requires explicit evidence for this commitment. Provide "
            "completion_evidence naming the command, receipt, artifact, or runtime "
            "proof that satisfied the ongoing-support/build/test/verify requirement."
        )
    lowered = evidence_text.lower()
    for marker in _NEGATIVE_EVIDENCE_MARKERS:
        if marker in lowered:
            return (
                "Completion evidence says the work is still unverified or failed. "
                "Keep the commitment active or blocked until the proof is positive."
            )
    root_reason = _workspace_root_rejection_reason(commitment, evidence_text)
    if root_reason is not None:
        return root_reason
    new_project_reason = new_project_completion_rejection_reason(
        getattr(commitment, "meta", {}) or {},
        evidence_text,
    )
    if new_project_reason is not None:
        return new_project_reason
    target_word_count = _target_word_count(commitment)
    if target_word_count is not None:
        if max(_manuscript_word_counts(evidence_text), default=0) < target_word_count:
            return (
                "Completion evidence must include a current manuscript word count that meets or exceeds "
                f"the {target_word_count:,}-word target."
            )
        if not _names_manuscript_artifact(evidence_text):
            return "Completion evidence must name the manuscript artifact or workspace draft path."
    if _requires_clean_revision(commitment):
        if any(marker in lowered for marker in _FIRST_DRAFT_ONLY_MARKERS):
            return (
                "Completion evidence only describes first-draft completion, but this project's "
                "completion contract requires a revised clean manuscript."
            )
        if not _has_marker(lowered, _CLEAN_REVISION_EVIDENCE_MARKERS):
            return (
                "Completion evidence must name the revised/clean manuscript or second-draft artifact "
                "when the project contract requires a revision pass."
            )
    if _requires_name_originality_research(commitment) and not _has_marker(
        lowered,
        _NAME_ORIGINALITY_EVIDENCE_MARKERS,
    ):
        return (
            "Completion evidence must mention the name/place/originality research pass when the "
            "project contract requires naming checks."
        )
    return None


def _workspace_root_rejection_reason(commitment: Any, evidence_text: str) -> str | None:
    meta = getattr(commitment, "meta", {}) or {}
    expected_values = [
        str(meta.get("workspace_rel_path") or "").strip(),
        str(meta.get("workspace_abs_path") or "").strip(),
    ]
    expected_values = [value.rstrip("/") for value in expected_values if value.strip()]
    if not expected_values:
        return None
    normalized_evidence = evidence_text.replace("\\", "/")
    for expected in expected_values:
        normalized_expected = expected.replace("\\", "/")
        if normalized_expected and normalized_expected in normalized_evidence:
            return None
    return (
        "Completion evidence must name an artifact under the commitment's canonical workspace project root. "
        "Evidence from a sibling or different project cannot complete this commitment."
    )


def _target_word_count(commitment: Any) -> int | None:
    meta = getattr(commitment, "meta", {}) or {}
    raw = meta.get("target_word_count")
    if raw is None and isinstance(meta.get("creative_completion_contract"), dict):
        raw = meta["creative_completion_contract"].get("target_word_count")
    try:
        value = int(raw)
    except (TypeError, ValueError):
        return None
    return value if value > 0 else None


def _manuscript_word_counts(evidence_text: str) -> list[int]:
    counts: list[int] = []
    for pattern in _MANUSCRIPT_WORD_COUNT_PATTERNS:
        counts.extend(
            int(match.group("count").replace(",", ""))
            for match in pattern.finditer(evidence_text or "")
        )
    return counts


def _names_manuscript_artifact(evidence_text: str) -> bool:
    for pattern in _ARTIFACT_REFERENCE_PATTERNS:
        for match in pattern.finditer(evidence_text or ""):
            if not any(char in match.group(0) for char in _GLOB_CHARS):
                return True
    return False


def _requires_clean_revision(commitment: Any) -> bool:
    contract = _creative_contract(commitment)
    return bool(contract.get("requires_clean_revision"))


def _requires_name_originality_research(commitment: Any) -> bool:
    contract = _creative_contract(commitment)
    return bool(contract.get("requires_name_originality_research"))


def _creative_contract(commitment: Any) -> dict[str, Any]:
    meta = getattr(commitment, "meta", {}) or {}
    contract = meta.get("creative_completion_contract")
    return contract if isinstance(contract, dict) else {}


def _has_marker(text: str, markers: tuple[str, ...]) -> bool:
    return any(marker in text for marker in markers)
