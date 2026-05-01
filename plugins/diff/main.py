"""Diff plugin for OpenCAS."""

from __future__ import annotations

import difflib
from pathlib import Path
from typing import Any, Dict

from opencas.autonomy.models import ActionRiskTier
from opencas.tools.models import ToolResult


def _stats(diff_lines: list[str]) -> Dict[str, int]:
    insertions = 0
    deletions = 0
    for line in diff_lines:
        if line.startswith("+") and not line.startswith("+++"):
            insertions += 1
        elif line.startswith("-") and not line.startswith("---"):
            deletions += 1
    return {"insertions": insertions, "deletions": deletions}


def _build_diff(
    a_text: str,
    b_text: str,
    *,
    a_label: str,
    b_label: str,
    context: int,
) -> tuple[str, Dict[str, int]]:
    a_lines = a_text.splitlines(keepends=True) or [""]
    b_lines = b_text.splitlines(keepends=True) or [""]
    diff_lines = list(
        difflib.unified_diff(
            a_lines,
            b_lines,
            fromfile=a_label,
            tofile=b_label,
            n=context,
        )
    )
    return ("".join(diff_lines), _stats(diff_lines))


def _diff_text(args: Dict[str, Any]) -> ToolResult:
    a = args.get("a")
    b = args.get("b")
    if not isinstance(a, str) or not isinstance(b, str):
        return ToolResult(success=False, output="'a' and 'b' must be strings", metadata={})
    context = int(args.get("context", 3))
    a_label = str(args.get("a_label", "a"))
    b_label = str(args.get("b_label", "b"))
    diff, stats = _build_diff(a, b, a_label=a_label, b_label=b_label, context=context)
    if not diff:
        return ToolResult(success=True, output="(identical)", metadata={**stats, "identical": True})
    return ToolResult(success=True, output=diff, metadata={**stats, "identical": False})


def _diff_files(args: Dict[str, Any]) -> ToolResult:
    a_path = Path(str(args.get("a", ""))).expanduser()
    b_path = Path(str(args.get("b", ""))).expanduser()
    if not a_path.is_file():
        return ToolResult(success=False, output=f"file not found: {a_path}", metadata={})
    if not b_path.is_file():
        return ToolResult(success=False, output=f"file not found: {b_path}", metadata={})
    try:
        a_text = a_path.read_text(encoding="utf-8")
        b_text = b_path.read_text(encoding="utf-8")
    except UnicodeDecodeError as exc:
        return ToolResult(success=False, output=f"non-utf8 file: {exc}", metadata={})
    context = int(args.get("context", 3))
    diff, stats = _build_diff(a_text, b_text, a_label=str(a_path), b_label=str(b_path), context=context)
    metadata = {**stats, "a": str(a_path), "b": str(b_path)}
    if not diff:
        return ToolResult(success=True, output="(identical)", metadata={**metadata, "identical": True})
    return ToolResult(success=True, output=diff, metadata={**metadata, "identical": False})


def register_skills(skill_registry, tools) -> None:
    tools.register(
        "diff_text",
        "Unified-diff between two strings; metadata includes insertion/deletion counts.",
        lambda name, args: _diff_text(args),
        ActionRiskTier.READONLY,
        {
            "type": "object",
            "properties": {
                "a": {"type": "string"},
                "b": {"type": "string"},
                "a_label": {"type": "string"},
                "b_label": {"type": "string"},
                "context": {"type": "integer"},
            },
            "required": ["a", "b"],
        },
    )
    tools.register(
        "diff_files",
        "Unified-diff between two text files on disk.",
        lambda name, args: _diff_files(args),
        ActionRiskTier.READONLY,
        {
            "type": "object",
            "properties": {
                "a": {"type": "string", "description": "Path to first file."},
                "b": {"type": "string", "description": "Path to second file."},
                "context": {"type": "integer"},
            },
            "required": ["a", "b"],
        },
    )
