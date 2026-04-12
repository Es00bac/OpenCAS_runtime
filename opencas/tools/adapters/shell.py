"""Shell tool adapter for OpenCAS."""

from __future__ import annotations

import dataclasses
import json
import shlex
import subprocess
from typing import Any, Dict, Optional

from ..models import ToolResult


@dataclasses.dataclass
class _SafetyResult:
    ok: bool
    reason: str


_BLOCKED_PATTERNS = [
    "rm -rf /",
    "> /dev/sda",
    "dd if=",
    ":(){ :|:& };:",
    "mkfs.",
]

_SHELL_META_CHARS = set("|&;<>()$`*?[]{}~\n")


class ShellToolAdapter:
    """Adapter for executing shell commands with basic safety checks."""

    def __init__(
        self,
        cwd: str,
        timeout: float = 30.0,
        docker_sandbox: Optional[Any] = None,
    ) -> None:
        self.cwd = str(cwd)
        self.timeout = timeout
        self.docker_sandbox = docker_sandbox

    def __call__(self, name: str, args: Dict[str, Any]) -> ToolResult:
        if name != "bash_run_command":
            return ToolResult(
                success=False,
                output=f"Unknown shell tool: {name}",
                metadata={},
            )
        command = str(args.get("command", ""))
        safety = self._validate_command(command)
        if not safety.ok:
            return ToolResult(
                success=False,
                output=f"Command blocked by safety policy: {safety.reason}",
                metadata={"command": command},
            )
        try:
            if self.docker_sandbox is not None:
                result = self.docker_sandbox.run_command(
                    command,
                    cwd=self.cwd,
                )
                return ToolResult(
                    success=result.get("ok", False),
                    output=json.dumps(result),
                    metadata={
                        "command": command,
                        "sandboxed": True,
                        "returncode": result.get("code"),
                    },
                )
            prepared_command, use_shell = _prepare_subprocess_command(command)
            result = subprocess.run(
                prepared_command,
                cwd=self.cwd,
                shell=use_shell,
                capture_output=True,
                text=True,
                timeout=self.timeout,
            )
            return ToolResult(
                success=result.returncode == 0,
                output=json.dumps(
                    {
                        "ok": result.returncode == 0,
                        "code": result.returncode,
                        "stdout": result.stdout,
                        "stderr": result.stderr,
                    }
                ),
                metadata={
                    "command": command,
                    "returncode": result.returncode,
                    "used_shell": use_shell,
                },
            )
        except subprocess.TimeoutExpired:
            return ToolResult(
                success=False,
                output=json.dumps(
                    {"ok": False, "error": f"Command timed out after {self.timeout}s"}
                ),
                metadata={"command": command, "timed_out": True},
            )
        except Exception as exc:
            return ToolResult(
                success=False,
                output=str(exc),
                metadata={"command": command, "error_type": type(exc).__name__},
            )

    def _validate_command(self, command: str):
        for pattern in _BLOCKED_PATTERNS:
            if pattern in command:
                return _SafetyResult(ok=False, reason=f"blocked pattern: {pattern}")
        return _SafetyResult(ok=True, reason="")


def _prepare_subprocess_command(command: str) -> tuple[str | list[str], bool]:
    """Use shell=False for ordinary commands, shell=True only when required."""
    if any(ch in command for ch in _SHELL_META_CHARS):
        return command, True
    try:
        return shlex.split(command), False
    except ValueError:
        return command, True
