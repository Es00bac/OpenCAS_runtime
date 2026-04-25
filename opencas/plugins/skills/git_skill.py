"""Git skill for OpenCAS.

Registers /commit and /review-pr as skill tools.
"""

from __future__ import annotations

import json
import subprocess
from typing import Any, Dict

from opencas.autonomy.models import ActionRiskTier
from opencas.tools import ToolRegistry
from opencas.tools.models import ToolResult


def _run_git_command(args: list[str], cwd: str | None = None) -> ToolResult:
    try:
        result = subprocess.run(
            ["git", *args],
            capture_output=True,
            text=True,
            cwd=cwd,
            timeout=30.0,
        )
        if result.returncode == 0:
            return ToolResult(success=True, output=result.stdout.strip(), metadata={})
        return ToolResult(
            success=False,
            output=result.stderr.strip() or result.stdout.strip(),
            metadata={"returncode": result.returncode},
        )
    except Exception as exc:
        return ToolResult(success=False, output=str(exc), metadata={"error_type": type(exc).__name__})


def _git_commit(args: Dict[str, Any]) -> ToolResult:
    message = str(args.get("message", ""))
    files = args.get("files", [])
    cwd = str(args.get("cwd", "."))
    if not message:
        return ToolResult(success=False, output="message is required", metadata={})

    # Stage files
    if files:
        stage_result = _run_git_command(["add", *files], cwd=cwd)
        if not stage_result.success:
            return stage_result
    else:
        stage_result = _run_git_command(["add", "-A"], cwd=cwd)
        if not stage_result.success:
            return stage_result

    # Commit
    return _run_git_command(["commit", "-m", message], cwd=cwd)


def _git_review_pr(args: Dict[str, Any]) -> ToolResult:
    pr_number = str(args.get("pr_number", ""))
    cwd = str(args.get("cwd", "."))
    if not pr_number:
        return ToolResult(success=False, output="pr_number is required", metadata={})

    # Fetch PR summary via gh CLI if available
    try:
        result = subprocess.run(
            ["gh", "pr", "view", pr_number, "--json", "title,author,body,files"],
            capture_output=True,
            text=True,
            cwd=cwd,
            timeout=30.0,
        )
        if result.returncode == 0:
            data = json.loads(result.stdout)
            summary = (
                f"Title: {data.get('title')}\n"
                f"Author: {data.get('author', {}).get('login')}\n\n"
                f"{data.get('body', '')}"
            )
            return ToolResult(success=True, output=summary, metadata={"pr_number": pr_number})
    except FileNotFoundError:
        pass
    except Exception as exc:
        return ToolResult(success=False, output=str(exc), metadata={"error_type": type(exc).__name__})

    return ToolResult(
        success=False,
        output="gh CLI is not available or PR could not be fetched.",
        metadata={"pr_number": pr_number},
    )


def register_skills(tools: ToolRegistry) -> None:
    """Register git skill tools."""
    tools.register(
        "git_commit",
        "Create a git commit with the given message and optional files.",
        lambda name, args: _git_commit(args),
        ActionRiskTier.WORKSPACE_WRITE,
        {
            "type": "object",
            "properties": {
                "message": {"type": "string", "description": "Commit message."},
                "files": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": "Files to stage (default: all).",
                },
                "cwd": {"type": "string", "description": "Working directory."},
            },
            "required": ["message"],
        },
    )
    tools.register(
        "git_review_pr",
        "Fetch and summarize a GitHub pull request by number.",
        lambda name, args: _git_review_pr(args),
        ActionRiskTier.READONLY,
        {
            "type": "object",
            "properties": {
                "pr_number": {"type": "string", "description": "Pull request number."},
                "cwd": {"type": "string", "description": "Working directory."},
            },
            "required": ["pr_number"],
        },
    )
