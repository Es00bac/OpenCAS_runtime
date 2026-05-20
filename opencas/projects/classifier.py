"""Project-type classification helpers."""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any, Iterable, Mapping

PROJECT_TYPE_SOFTWARE = "software"
PROJECT_TYPE_WRITING = "writing"
PROJECT_TYPE_RESEARCH = "research"
PROJECT_TYPE_GENERAL = "general"
PROJECT_TYPES = {
    PROJECT_TYPE_SOFTWARE,
    PROJECT_TYPE_WRITING,
    PROJECT_TYPE_RESEARCH,
    PROJECT_TYPE_GENERAL,
}

_SOFTWARE_MARKERS = (
    "api",
    "app",
    "application",
    "build",
    "cmake",
    "code",
    "compile",
    "compiler",
    "debug",
    "email client",
    "executable",
    "gui",
    "library",
    "native qt",
    "program",
    "pytest",
    "qt",
    "qt6",
    "repository",
    "repo",
    "run instructions",
    "source file",
    "source files",
    "software",
    "test suite",
    "ui",
)
_SOFTWARE_SUFFIX_RE = re.compile(
    r"\b[\w./~-]+\.(?:c|cc|cpp|cxx|h|hpp|py|js|jsx|ts|tsx|qml|rs|go|java|kt|swift|cs|cmake|toml)\b"
)
_WRITING_MARKERS = (
    "article",
    "book",
    "character bible",
    "chapter",
    "creative_writing",
    "creative writing",
    "draft prose",
    "essay",
    "fiction",
    "manuscript",
    "novel",
    "poem",
    "prose",
    "revision",
    "scene",
    "story",
    "writing task",
)
_RESEARCH_MARKERS = (
    "audit",
    "compare",
    "deep dive",
    "fact check",
    "investigate",
    "research",
    "source backed",
    "source-backed",
    "study",
)


@dataclass(frozen=True)
class ProjectClassification:
    """A compact project-type classification result."""

    project_type: str
    confidence: float
    evidence: tuple[str, ...]


def classify_project_type(
    *,
    current_turn_text: str = "",
    context_text: str = "",
    metadata: Mapping[str, Any] | None = None,
    fallback: str = PROJECT_TYPE_GENERAL,
) -> ProjectClassification:
    """Classify the kind of project being discussed.

    Current-turn evidence is weighted ahead of wider context so stale retrieved
    material cannot override the user's immediate project request.
    """

    meta_type = str((metadata or {}).get("project_type", "") or "").strip().lower()
    if meta_type in PROJECT_TYPES and meta_type != PROJECT_TYPE_GENERAL:
        return ProjectClassification(meta_type, 1.0, (f"metadata:{meta_type}",))

    current = str(current_turn_text or "")
    context = str(context_text or "")
    current_scores = _score_text(current)
    current_type = _dominant_type(current_scores)
    if current_type is not None:
        return _result(current_type, current_scores[current_type], current_scores)

    combined_scores = _score_text(f"{current}\n{context}")
    project_type = _dominant_type(combined_scores)
    if project_type is None:
        fallback_type = fallback if fallback in PROJECT_TYPES else PROJECT_TYPE_GENERAL
        return ProjectClassification(fallback_type, 0.0, ())
    if combined_scores[project_type]:
        return _result(project_type, combined_scores[project_type], combined_scores)

    fallback_type = fallback if fallback in PROJECT_TYPES else PROJECT_TYPE_GENERAL
    return ProjectClassification(fallback_type, 0.0, ())


def marker_in_text(text: str, marker: str) -> bool:
    """Return true when *marker* appears as a phrase or token, not a substring accident."""

    marker_text = str(marker or "").lower().strip()
    if not marker_text:
        return False
    haystack = str(text or "").lower()
    if re.search(r"[^a-z0-9_]", marker_text):
        return marker_text in haystack
    return re.search(rf"\b{re.escape(marker_text)}\b", haystack) is not None


def any_marker_in_text(text: str, markers: Iterable[str]) -> bool:
    return any(marker_in_text(text, marker) for marker in markers)


def _score_text(text: str) -> dict[str, list[str]]:
    return {
        PROJECT_TYPE_SOFTWARE: _hits(text, _SOFTWARE_MARKERS, suffix_re=_SOFTWARE_SUFFIX_RE),
        PROJECT_TYPE_WRITING: _hits(text, _WRITING_MARKERS),
        PROJECT_TYPE_RESEARCH: _hits(text, _RESEARCH_MARKERS),
    }


def _dominant_type(scores: Mapping[str, Iterable[str]]) -> str | None:
    priority = {
        PROJECT_TYPE_SOFTWARE: 3,
        PROJECT_TYPE_WRITING: 2,
        PROJECT_TYPE_RESEARCH: 1,
    }
    ranked = sorted(
        (
            (len(tuple(hits)), priority.get(project_type, 0), project_type)
            for project_type, hits in scores.items()
        ),
        reverse=True,
    )
    if not ranked or ranked[0][0] <= 0:
        return None
    return ranked[0][2]


def _hits(text: str, markers: Iterable[str], *, suffix_re: re.Pattern[str] | None = None) -> list[str]:
    found = [marker for marker in markers if marker_in_text(text, marker)]
    if suffix_re is not None:
        found.extend(match.group(0) for match in suffix_re.finditer(str(text or "").lower()))
    return tuple(dict.fromkeys(found))


def _result(project_type: str, score: Iterable[str], scores: Mapping[str, Iterable[str]]) -> ProjectClassification:
    evidence = tuple(score)
    confidence = min(1.0, 0.45 + (0.15 * len(evidence)))
    if len([kind for kind, hits in scores.items() if hits]) > 1:
        confidence = max(0.5, confidence - 0.1)
    return ProjectClassification(project_type, round(confidence, 3), evidence)
