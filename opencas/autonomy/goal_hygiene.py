"""Goal-surface hygiene helpers for keeping live focus narrow."""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any, Dict, Iterable, List

_GENERIC_GOAL_PREFIXES = {
    "build",
    "fix",
    "repair",
    "refactor",
    "test",
    "verify",
}
_ABSTRACT_THEME_GOALS = {
    "assist",
    "care",
    "continuity",
    "memory",
    "persistence",
}
_NUMBERED_FRAGMENT_RE = re.compile(r"^-?\d+$")
_PROMOTED_PROJECT_FRAGMENT_RE = re.compile(
    r"^advance\s+promoted\s+work:\s*return\s+to\s+project:\s*(?P<fragment>.+)$",
    re.IGNORECASE,
)
_SELF_REFERENTIAL_SUPPRESSION_REASONS = {
    "duplicate_low_divergence_objective",
    "low_divergence_reframe",
}
_HARD_REJECT_REASONS = frozenset(
    {
        "machine_fragment_goal",
        "numbered_fragment_goal",
        "path_fragment_goal",
        "stale_promoted_work_fragment",
    }
)


@dataclass(frozen=True)
class GoalSurface:
    """Split goal text into live and parked surfaces."""

    active_goals: List[str] = field(default_factory=list)
    parked_goals: List[str] = field(default_factory=list)
    parked_reasons: Dict[str, str] = field(default_factory=dict)


def normalize_goal_text(goal: str | None) -> str:
    """Return a compact, stable goal string."""
    return " ".join(str(goal or "").split())


def classify_goal_residue(goal: str | None) -> str | None:
    """Return a park reason when *goal* looks like residue instead of live focus."""
    normalized = normalize_goal_text(goal)
    if not normalized:
        return "empty_goal"

    lowered = normalized.lower()
    if lowered in _ABSTRACT_THEME_GOALS:
        return "abstract_theme_goal"

    promoted_match = _PROMOTED_PROJECT_FRAGMENT_RE.match(normalized)
    if promoted_match:
        fragment = normalize_goal_text(promoted_match.group("fragment")).strip(" .")
        words = [word for word in re.split(r"\s+", fragment) if word]
        has_binding = any(marker in fragment for marker in ('"', "'", "/", "\\")) or len(words) > 5
        if not has_binding:
            return "stale_promoted_work_fragment"

    prefix, _, remainder = lowered.partition(" ")
    remainder = remainder.strip()

    if prefix not in _GENERIC_GOAL_PREFIXES:
        return None
    if not remainder:
        return "generic_verb_without_binding"
    if _NUMBERED_FRAGMENT_RE.fullmatch(remainder):
        return "numbered_fragment_goal"
    if remainder.startswith(("-", "/", ":", ".")):
        return "machine_fragment_goal"
    if "/" in remainder or "\\" in remainder:
        return "path_fragment_goal"
    return None


def hard_reject_goal_reason(goal: str | None) -> str | None:
    """Return a rejection reason when *goal* is machine/path residue."""
    reason = classify_goal_residue(goal)
    return reason if reason in _HARD_REJECT_REASONS else None


def is_hard_reject_goal_reason(reason: str | None) -> bool:
    """Return true when a classified residue reason should be dropped, not parked."""
    return str(reason or "").strip() in _HARD_REJECT_REASONS


def is_self_referential_suppression_metadata(
    goal: str | None,
    *,
    reason: str | None = None,
    source_artifact: str | None = None,
    details: Dict[str, Any] | None = None,
) -> bool:
    """Return true when suppression metadata repeats the goal as its own evidence."""
    normalized_goal = normalize_goal_text(goal).lower()
    if not normalized_goal:
        return False
    if normalize_goal_text(reason).lower() not in _SELF_REFERENTIAL_SUPPRESSION_REASONS:
        return False

    normalized_source = normalize_goal_text(source_artifact).lower()
    if normalized_source != normalized_goal:
        return False

    metadata = details if isinstance(details, dict) else {}
    failed_framings = metadata.get("failed_framings")
    if not isinstance(failed_framings, list):
        return False
    return any(normalize_goal_text(item).lower() == normalized_goal for item in failed_framings)


def split_live_and_parked_goals(goals: Iterable[str]) -> GoalSurface:
    """Split *goals* into live and parked lists while preserving input order."""
    active_goals: List[str] = []
    parked_goals: List[str] = []
    parked_reasons: Dict[str, str] = {}
    seen_active: set[str] = set()
    seen_parked: set[str] = set()

    for raw_goal in goals:
        goal = normalize_goal_text(raw_goal)
        if not goal:
            continue
        park_reason = classify_goal_residue(goal)
        if park_reason:
            if goal not in seen_parked:
                parked_goals.append(goal)
                parked_reasons[goal] = park_reason
                seen_parked.add(goal)
            continue
        if goal not in seen_active:
            active_goals.append(goal)
            seen_active.add(goal)

    return GoalSurface(
        active_goals=active_goals,
        parked_goals=parked_goals,
        parked_reasons=parked_reasons,
    )
