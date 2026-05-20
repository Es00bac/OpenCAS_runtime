"""State-directory hygiene checks for live OpenCAS stores."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


_SQLITE_SUFFIXES = (".db", ".sqlite", ".sqlite3")


@dataclass(frozen=True, slots=True)
class StatePathViolation:
    path: Path
    reason: str
    size_bytes: int


def sqlite_path_hygiene_violations(state_dir: Path | str) -> list[StatePathViolation]:
    """Return SQLite-like state files whose names violate store-path policy."""
    root = Path(state_dir).expanduser()
    if not root.exists():
        return []
    violations: list[StatePathViolation] = []
    for path in sorted(root.iterdir(), key=lambda item: item.name):
        if not path.is_file():
            continue
        name = path.name
        if not _looks_sqlite_file(name):
            continue
        if any(char.isspace() for char in name):
            violations.append(
                StatePathViolation(
                    path=path,
                    reason="sqlite_filename_contains_whitespace",
                    size_bytes=path.stat().st_size,
                )
            )
    return violations


def remove_empty_whitespace_sqlite_files(state_dir: Path | str) -> list[Path]:
    """Delete zero-byte whitespace SQLite artifacts after callers snapshot state."""
    removed: list[Path] = []
    for violation in sqlite_path_hygiene_violations(state_dir):
        if violation.size_bytes != 0:
            continue
        violation.path.unlink()
        removed.append(violation.path)
    return removed


def _looks_sqlite_file(name: str) -> bool:
    if name.endswith(_SQLITE_SUFFIXES):
        return True
    return ".db " in name or ".sqlite " in name or ".sqlite3 " in name
