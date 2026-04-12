"""Test skill for OpenCAS.

Registers /test as a skill tool.
"""

from __future__ import annotations

import subprocess
import sys
from typing import Any, Dict

from opencas.autonomy.models import ActionRiskTier
from opencas.tools import ToolRegistry
from opencas.tools.models import ToolResult


def _run_test(args: Dict[str, Any]) -> ToolResult:
    target = str(args.get("target", ""))
    cwd = str(args.get("cwd", "."))
    verbosity = str(args.get("verbosity", "-q"))
    cmd = [sys.executable, "-m", "pytest", verbosity]
    if target:
        cmd.append(target)
    try:
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            cwd=cwd,
            timeout=120.0,
        )
        output = result.stdout.strip()
        if result.stderr.strip():
            output += "\n" + result.stderr.strip()
        if result.returncode == 0:
            return ToolResult(success=True, output=output, metadata={"returncode": 0})
        return ToolResult(
            success=False,
            output=output,
            metadata={"returncode": result.returncode},
        )
    except Exception as exc:
        return ToolResult(success=False, output=str(exc), metadata={"error_type": type(exc).__name__})


def register_skills(tools: ToolRegistry) -> None:
    """Register test skill tools."""
    tools.register(
        "run_tests",
        "Run pytest against a target file, directory, or test path.",
        lambda name, args: _run_test(args),
        ActionRiskTier.SHELL_LOCAL,
        {
            "type": "object",
            "properties": {
                "target": {
                    "type": "string",
                    "description": "Target file, directory, or test selector.",
                },
                "cwd": {"type": "string", "description": "Working directory."},
                "verbosity": {
                    "type": "string",
                    "enum": ["-q", "-v", "-vv"],
                    "description": "Verbosity level.",
                },
            },
            "required": [],
        },
    )
