"""Goal-surface hygiene helpers for keeping live focus narrow."""

from __future__ import annotations

from dataclasses import dataclass, field
import re
from typing import Dict, Iterable, List

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
