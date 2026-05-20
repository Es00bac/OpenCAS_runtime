"""Resolve managed-workspace project roots from grounded project identity."""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Optional, Sequence

_NON_KEY_RE = re.compile(r"[^a-z0-9]+")
_WORKSPACE_PATH_RE = re.compile(
    r"(?P<path>(?:~|/)[^\s`'\"<>]+|(?:\.?/)?workspace/[^\s`'\"<>]+)"
)
_GENERIC_PROJECT_KEYS = {
    "",
    "app",
    "application",
    "book",
    "build",
    "client",
    "code",
    "conversation-project",
    "dist",
    "draft",
    "manuscript",
    "node-modules",
    "out",
    "program",
    "project",
    "software",
    "software-project",
    "story",
    "tool",
    "writing-project",
}
_IGNORED_WORKSPACE_DIRS = {
    ".git",
    ".opencas",
    ".venv",
    "__pycache__",
    "_compost",
    "build",
    "chat_uploads",
    "dist",
    "node_modules",
    "out",
    "target",
    "venv",
}
_PROJECT_MARKERS = (
    "CMakeLists.txt",
    "pyproject.toml",
    "package.json",
    "Cargo.toml",
    "go.mod",
    "Makefile",
    "README.md",
    "BUILD_PROOF.md",
    "src",
    "app",
    "PROJECT.md",
    "drafts",
    "bible",
    "notes",
    "research",
    "review",
    "bible/world_bible.md",
    "bible/character_bible.md",
)


@dataclass(frozen=True)
class WorkspaceProjectCandidate:
    """A grounded project root found under the managed OpenCAS workspace."""

    project_key: str
    project_title: str
    path: Path
    workspace_rel_path: Path
    confidence: float
    evidence: tuple[str, ...]


@dataclass(frozen=True)
class WorkspacePathRequest:
    """A managed-workspace path explicitly requested by the operator."""

    path: Path
    workspace_rel_path: Path
    kind: str
    raw_text: str


def resolve_workspace_project(
    runtime: Any,
    *,
    project_key: str = "",
    project_title: str = "",
) -> Optional[WorkspaceProjectCandidate]:
    """Resolve a project identity to a grounded root in the managed workspace.

    The resolver is intentionally conservative. It requires a non-generic
    project identity and an exact normalized directory-name match, so labels
    such as "project" or "software project" never drift into an unrelated
    workspace directory.
    """

    workspace_root = _agent_workspace_root(runtime)
    if workspace_root is None or not workspace_root.exists():
        return None
    requested_keys = _requested_project_keys(project_key=project_key, project_title=project_title)
    if not requested_keys or requested_keys.issubset(_GENERIC_PROJECT_KEYS):
        return None

    candidates: list[WorkspaceProjectCandidate] = []
    for project_root in _workspace_project_roots(workspace_root):
        display_title = _project_display_title(project_root)
        project_key = _project_key(display_title)
        directory_key = _project_key(project_root.name)
        if project_key not in requested_keys and directory_key not in requested_keys:
            continue
        evidence = _project_evidence(project_root)
        score = 80 + min(len(evidence), 5) * 4
        candidates.append(
            WorkspaceProjectCandidate(
                project_key=project_key,
                project_title=display_title,
                path=project_root.resolve(),
                workspace_rel_path=_workspace_relative_path(runtime, workspace_root, project_root),
                confidence=min(1.0, score / 100.0),
                evidence=tuple(evidence),
            )
        )
    if not candidates:
        return None
    candidates.sort(key=lambda item: (-item.confidence, len(str(item.path)), item.project_title.lower()))
    return candidates[0]


def resolve_requested_workspace_path(runtime: Any, text: str) -> Optional[WorkspacePathRequest]:
    """Resolve an operator-mentioned workspace path even before a project exists."""

    workspace_root = _agent_workspace_root(runtime)
    if workspace_root is None:
        return None
    primary_root = _primary_workspace_root(runtime)
    haystack = str(text or "")
    if not haystack:
        return None
    candidates: list[WorkspacePathRequest] = []
    for match in _WORKSPACE_PATH_RE.finditer(haystack):
        raw = _clean_path_token(match.group("path"))
        if not raw:
            continue
        path = _resolve_requested_path(raw, workspace_root=workspace_root, primary_root=primary_root)
        if path is None:
            continue
        try:
            path.relative_to(workspace_root)
        except ValueError:
            continue
        kind = _requested_path_kind(path, raw)
        candidates.append(
            WorkspacePathRequest(
                path=path,
                workspace_rel_path=_workspace_relative_path(runtime, workspace_root, path),
                kind=kind,
                raw_text=raw,
            )
        )
    if not candidates:
        return None
    candidates.sort(key=lambda item: (item.kind != "project_root", -len(str(item.path))))
    return candidates[0]


def resolve_recent_workspace_project(
    runtime: Any,
    *,
    project_type: str = "",
    excluded_project_keys: Sequence[str] = (),
) -> Optional[WorkspaceProjectCandidate]:
    """Return the newest unfinished-looking managed project root."""

    workspace_root = _agent_workspace_root(runtime)
    if workspace_root is None or not workspace_root.exists():
        return None
    excluded = {_project_key(value) for value in excluded_project_keys if _project_key(value)}
    candidates: list[tuple[float, int, WorkspaceProjectCandidate]] = []
    for project_root in _workspace_project_roots(workspace_root):
        evidence = _project_evidence(project_root)
        if not evidence:
            continue
        display_title = _project_display_title(project_root)
        project_key = _project_key(display_title)
        directory_key = _project_key(project_root.name)
        if project_key in _GENERIC_PROJECT_KEYS or directory_key in _GENERIC_PROJECT_KEYS:
            continue
        if project_key in excluded or directory_key in excluded:
            continue
        if project_type == "writing" and not _has_writing_evidence(evidence):
            continue
        candidate = WorkspaceProjectCandidate(
            project_key=project_key,
            project_title=display_title,
            path=project_root.resolve(),
            workspace_rel_path=_workspace_relative_path(runtime, workspace_root, project_root),
            confidence=min(1.0, (80 + min(len(evidence), 5) * 4) / 100.0),
            evidence=tuple(evidence),
        )
        candidates.append((_latest_project_mtime(project_root), len(evidence), candidate))
    if not candidates:
        return None
    candidates.sort(key=lambda item: (-item[0], -item[1], len(str(item[2].path)), item[2].project_title.lower()))
    return candidates[0][2]


def resolve_workspace_project_from_text(runtime: Any, text: str) -> Optional[WorkspaceProjectCandidate]:
    """Resolve a managed project when a turn names its workspace path or project-file title."""

    workspace_root = _agent_workspace_root(runtime)
    if workspace_root is None or not workspace_root.exists():
        return None
    haystack = str(text or "").lower()
    if not haystack:
        return None
    candidates: list[WorkspaceProjectCandidate] = []
    for project_root in _workspace_project_roots(workspace_root):
        evidence = _project_evidence(project_root)
        if not evidence:
            continue
        candidate = WorkspaceProjectCandidate(
            project_key=_project_key(_project_display_title(project_root)),
            project_title=_project_display_title(project_root),
            path=project_root.resolve(),
            workspace_rel_path=_workspace_relative_path(runtime, workspace_root, project_root),
            confidence=min(1.0, (82 + min(len(evidence), 5) * 4) / 100.0),
            evidence=tuple(evidence),
        )
        names = {
            candidate.project_title.lower(),
            project_root.name.lower(),
        }
        path_names = {
            str(candidate.workspace_rel_path).lower(),
            str(candidate.path).lower(),
        }
        directory_key = _project_key(project_root.name)
        key_is_generic = candidate.project_key in _GENERIC_PROJECT_KEYS or directory_key in _GENERIC_PROJECT_KEYS
        path_mentioned = any(name and name in haystack for name in path_names)
        name_mentioned = (not key_is_generic) and any(_mentions_name(haystack, name) for name in names)
        if path_mentioned or name_mentioned:
            candidates.append(candidate)
    if not candidates:
        return None
    candidates.sort(key=lambda item: (-item.confidence, len(str(item.path)), item.project_title.lower()))
    return candidates[0]


def _has_writing_evidence(evidence: Sequence[str]) -> bool:
    return bool(
        set(evidence)
        & {"PROJECT.md", "drafts", "bible", "review", "bible/world_bible.md", "bible/character_bible.md"}
    )


def _latest_project_mtime(project_root: Path) -> float:
    latest = 0.0
    try:
        for path in project_root.rglob("*"):
            if path.is_file() and not path.is_symlink():
                latest = max(latest, path.stat().st_mtime)
    except OSError:
        pass
    try:
        latest = max(latest, project_root.stat().st_mtime)
    except OSError:
        pass
    return latest


def _agent_workspace_root(runtime: Any) -> Optional[Path]:
    config = getattr(getattr(runtime, "ctx", None), "config", None)
    getter = getattr(config, "agent_workspace_root", None)
    if not callable(getter):
        return None
    try:
        return Path(getter()).expanduser().resolve()
    except Exception:
        return None


def _primary_workspace_root(runtime: Any) -> Optional[Path]:
    config = getattr(getattr(runtime, "ctx", None), "config", None)
    getter = getattr(config, "primary_workspace_root", None)
    if not callable(getter):
        return None
    try:
        return Path(getter()).expanduser().resolve()
    except Exception:
        return None


def _requested_project_keys(*, project_key: str, project_title: str) -> set[str]:
    values = {project_key, project_title}
    for value in list(values):
        if not value:
            continue
        path_name = Path(str(value).strip()).name
        if path_name:
            values.add(path_name)
    return {_project_key(value) for value in values if _project_key(value)}


def _project_display_title(project_root: Path) -> str:
    project_file = project_root / "PROJECT.md"
    try:
        for line in project_file.read_text(encoding="utf-8").splitlines():
            stripped = line.strip()
            if stripped.startswith("#"):
                title = stripped.lstrip("#").strip()
                if title:
                    return title
    except OSError:
        pass
    return project_root.name


def _workspace_project_roots(workspace_root: Path) -> Iterable[Path]:
    try:
        resolved_workspace = workspace_root.resolve()
        children = list(workspace_root.iterdir())
    except Exception:
        return []
    roots: list[Path] = []
    for child in children:
        if not _is_visible_workspace_dir(child, resolved_workspace):
            continue
        roots.append(child)
        try:
            descendants = child.rglob("*")
            roots.extend(path for path in descendants if _is_visible_workspace_dir(path, resolved_workspace))
        except Exception:
            continue
    return roots


def _is_visible_workspace_dir(path: Path, workspace_root: Path) -> bool:
    if path.name in _IGNORED_WORKSPACE_DIRS or path.name.startswith(".") or path.is_symlink():
        return False
    try:
        path.resolve().relative_to(workspace_root)
    except (OSError, ValueError):
        return False
    return path.is_dir()


def _project_evidence(path: Path) -> list[str]:
    evidence: list[str] = []
    for marker in _PROJECT_MARKERS:
        if (path / marker).exists():
            evidence.append(marker)
    return evidence


def _workspace_relative_path(runtime: Any, workspace_root: Path, project_root: Path) -> Path:
    primary_root = _primary_workspace_root(runtime)
    if primary_root is not None:
        try:
            return project_root.resolve().relative_to(primary_root)
        except ValueError:
            pass
    try:
        return Path(workspace_root.name) / project_root.resolve().relative_to(workspace_root)
    except ValueError:
        return Path(project_root.name)


def _clean_path_token(raw: str) -> str:
    token = str(raw or "").strip()
    token = token.rstrip(".,;:!?)\"'")
    return token


def _resolve_requested_path(
    raw: str,
    *,
    workspace_root: Path,
    primary_root: Optional[Path],
) -> Optional[Path]:
    try:
        candidate = Path(raw).expanduser()
        if not candidate.is_absolute():
            if raw.startswith("./"):
                candidate = (primary_root or workspace_root.parent) / raw[2:]
            elif raw.startswith("workspace/") or raw.startswith("workspace"):
                candidate = (primary_root or workspace_root.parent) / raw
            else:
                candidate = workspace_root / raw
        return candidate.resolve(strict=False)
    except Exception:
        return None


def _requested_path_kind(path: Path, raw: str) -> str:
    if path.exists() and path.is_dir() and _project_evidence(path):
        return "project_root"
    if raw.endswith(("/", "\\")):
        return "parent"
    if path.exists() and path.is_dir() and not _project_evidence(path):
        return "parent"
    return "target"


def _project_key(value: str) -> str:
    normalized = _NON_KEY_RE.sub("-", str(value or "").lower()).strip("-")
    return normalized


def _mentions_name(haystack: str, name: str) -> bool:
    name = str(name or "").lower().strip()
    if not name:
        return False
    return re.search(rf"(?<![a-z0-9]){re.escape(name)}(?![a-z0-9])", haystack) is not None
