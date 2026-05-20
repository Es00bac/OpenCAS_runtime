"""Environment helpers for local tool execution.

OpenCAS often runs as a user service rather than from an interactive terminal.
User-installed CLI tools may live in bins that the service manager does not
include in PATH, so execution adapters share one conservative PATH expansion.
"""

from __future__ import annotations

import os
import shutil
from pathlib import Path
from typing import Mapping

_COMMON_USER_BIN_TEMPLATES = (
    "~/.npm-global/bin",
    "~/.local/bin",
    "~/.cargo/bin",
    "~/.bun/bin",
    "~/.opencode/bin",
    "/home/linuxbrew/.linuxbrew/bin",
    "/home/linuxbrew/.linuxbrew/sbin",
)


def build_tool_execution_env(
    base_env: Mapping[str, str] | None = None,
) -> dict[str, str]:
    """Return an environment suitable for OpenCAS-owned local tools."""

    env = dict(os.environ if base_env is None else base_env)
    path = build_tool_execution_path(env.get("PATH", ""), home=env.get("HOME"))
    env["PATH"] = path
    return env


def build_tool_execution_path(
    current_path: str | None = None,
    *,
    home: str | None = None,
) -> str:
    """Return PATH plus common per-user tool directories that exist."""

    parts: list[str] = []
    for item in str(current_path or "").split(os.pathsep):
        if item:
            parts.append(item)

    extra = os.getenv("OPENCAS_EXTRA_TOOL_PATHS", "")
    for item in extra.split(os.pathsep):
        if item:
            parts.append(_expand_path(item, home=home))

    for template in _COMMON_USER_BIN_TEMPLATES:
        expanded = _expand_path(template, home=home)
        if Path(expanded).is_dir():
            parts.append(expanded)

    return os.pathsep.join(_dedupe(parts))


def resolve_executable(
    command: str,
    *,
    env: Mapping[str, str] | None = None,
) -> str | None:
    """Resolve *command* using the OpenCAS tool execution PATH."""

    candidate = str(command or "").strip()
    if not candidate:
        return None
    expanded = _expand_path(candidate, home=(env or {}).get("HOME"))
    if os.path.sep in expanded:
        return expanded if os.access(expanded, os.X_OK) else None
    lookup_env = build_tool_execution_env(env)
    return shutil.which(expanded, path=lookup_env.get("PATH"))


def _expand_path(value: str, *, home: str | None) -> str:
    if value.startswith("~/") and home:
        return str(Path(home) / value[2:])
    return str(Path(value).expanduser())


def _dedupe(values: list[str]) -> list[str]:
    seen: set[str] = set()
    result: list[str] = []
    for value in values:
        key = value.rstrip(os.path.sep) or value
        if key in seen:
            continue
        seen.add(key)
        result.append(value)
    return result
