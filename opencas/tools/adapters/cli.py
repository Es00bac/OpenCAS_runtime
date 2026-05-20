"""CLI discovery adapter for learning local command-line tools."""

from __future__ import annotations

import json
import os
import shlex
import subprocess
from pathlib import Path
from typing import Any, Dict

from opencas.tools.environment import build_tool_execution_env, resolve_executable

from ..models import ToolResult

_DEFAULT_PROBES: tuple[tuple[str, ...], ...] = (
    ("--help",),
    ("-h",),
    ("help",),
    ("--version",),
    ("version",),
)
_MAX_PROBE_CHARS = 4000


class CliDiscoveryToolAdapter:
    """Resolve a CLI and collect bounded help/version/manual evidence."""

    def __init__(self, cwd: str) -> None:
        self.cwd = str(cwd)

    def __call__(self, name: str, args: Dict[str, Any]) -> ToolResult:
        if name != "cli_discover_command":
            return ToolResult(False, f"Unknown CLI discovery tool: {name}", {})

        command = str(args.get("command", "") or "").strip()
        valid, reason = _valid_command_name(command)
        if not valid:
            return ToolResult(False, reason, {"command": command})

        env = build_tool_execution_env()
        max_chars = _coerce_int(args.get("max_chars"), default=_MAX_PROBE_CHARS, lower=500, upper=12000)
        timeout = _coerce_int(args.get("timeout_seconds"), default=5, lower=1, upper=30)
        resolved = resolve_executable(command, env=env)
        matches = _all_matches(command, env=env)
        if resolved is None:
            payload = {
                "ok": False,
                "command": command,
                "code": 127,
                "stderr": f"{command}: command not found",
                "matches": matches,
                "path": env.get("PATH", ""),
                "next_steps": [
                    "Check spelling.",
                    "Inspect PATH and common user-local bin directories.",
                    "If the operator says it exists, ask for or search the install path before claiming absence.",
                ],
            }
            return ToolResult(
                False,
                json.dumps(payload, indent=2),
                {
                    "command": command,
                    "missing_command": True,
                    "path": env.get("PATH", ""),
                },
            )

        probes = [
            _run_probe([resolved, *probe_args], env=env, cwd=self.cwd, timeout=timeout, max_chars=max_chars)
            for probe_args in _probe_args(args.get("probe_args"))
        ]
        manual = None
        if bool(args.get("include_man", False)):
            manual = _run_manual_probe(command, env=env, cwd=self.cwd, timeout=timeout, max_chars=max_chars)

        payload = {
            "ok": True,
            "command": command,
            "resolved_path": resolved,
            "matches": matches,
            "path": env.get("PATH", ""),
            "probes": probes,
        }
        if manual is not None:
            payload["manual"] = manual
        return ToolResult(
            True,
            json.dumps(payload, indent=2),
            {
                "command": command,
                "resolved_path": resolved,
                "matches": matches,
                "probe_count": len(probes),
                "path": env.get("PATH", ""),
            },
        )


def _valid_command_name(command: str) -> tuple[bool, str]:
    if not command:
        return False, "command is required"
    try:
        parts = shlex.split(command)
    except ValueError as exc:
        return False, f"invalid command: {exc}"
    if len(parts) != 1:
        return False, "command must be a single executable name or path; do not include arguments"
    if any(ch in command for ch in "|&;<>()$`*?[]{}\n"):
        return False, "command must not include shell metacharacters"
    return True, ""


def _probe_args(raw: Any) -> list[tuple[str, ...]]:
    if not isinstance(raw, list) or not raw:
        return list(_DEFAULT_PROBES)
    probes: list[tuple[str, ...]] = []
    for item in raw[:8]:
        if isinstance(item, str):
            try:
                parts = tuple(shlex.split(item))
            except ValueError:
                continue
        elif isinstance(item, list):
            parts = tuple(str(part) for part in item if str(part).strip())
        else:
            continue
        if parts:
            probes.append(parts)
    return probes or list(_DEFAULT_PROBES)


def _run_probe(
    argv: list[str],
    *,
    env: dict[str, str],
    cwd: str,
    timeout: int,
    max_chars: int,
) -> dict[str, Any]:
    try:
        result = subprocess.run(
            argv,
            cwd=cwd,
            env=env,
            capture_output=True,
            text=True,
            timeout=timeout,
        )
        return {
            "argv": [Path(argv[0]).name, *argv[1:]],
            "code": result.returncode,
            "stdout": _truncate(result.stdout, max_chars),
            "stderr": _truncate(result.stderr, max_chars),
            "stdout_truncated": len(result.stdout) > max_chars,
            "stderr_truncated": len(result.stderr) > max_chars,
        }
    except subprocess.TimeoutExpired as exc:
        return {
            "argv": [Path(argv[0]).name, *argv[1:]],
            "code": None,
            "timed_out": True,
            "stdout": _truncate(exc.stdout or "", max_chars),
            "stderr": _truncate(exc.stderr or "", max_chars),
        }
    except Exception as exc:
        return {
            "argv": [Path(argv[0]).name, *argv[1:]],
            "code": None,
            "error": str(exc),
            "error_type": type(exc).__name__,
        }


def _run_manual_probe(
    command: str,
    *,
    env: dict[str, str],
    cwd: str,
    timeout: int,
    max_chars: int,
) -> dict[str, Any]:
    man_target = shlex.quote(Path(command).name)
    script = f"MANWIDTH=100 man {man_target} 2>/dev/null | col -b | sed -n '1,160p'"
    return _run_probe(["/bin/sh", "-lc", script], env=env, cwd=cwd, timeout=timeout, max_chars=max_chars)


def _all_matches(command: str, *, env: dict[str, str]) -> list[str]:
    if os.path.sep in command:
        candidate = Path(command).expanduser()
        return [str(candidate)] if candidate.exists() else []
    matches: list[str] = []
    seen: set[str] = set()
    for directory in env.get("PATH", "").split(os.pathsep):
        if not directory:
            continue
        candidate = Path(directory) / command
        if not candidate.is_file() or not os.access(candidate, os.X_OK):
            continue
        resolved = str(candidate)
        if resolved in seen:
            continue
        seen.add(resolved)
        matches.append(resolved)
    return matches


def _coerce_int(raw: Any, *, default: int, lower: int, upper: int) -> int:
    try:
        return max(lower, min(int(raw), upper))
    except Exception:
        return default


def _truncate(value: Any, max_chars: int) -> str:
    text = str(value or "")
    if len(text) <= max_chars:
        return text
    return text[:max_chars].rstrip()
