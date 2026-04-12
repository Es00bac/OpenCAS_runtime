"""Documentation skill for OpenCAS.

Registers /docs as a skill tool.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict

from opencas.autonomy.models import ActionRiskTier
from opencas.tools import ToolRegistry
from opencas.tools.models import ToolResult


def _generate_docstring(module_path: str) -> str:
    """Heuristic docstring generator for a Python module."""
    path = Path(module_path)
    if not path.exists() or not path.is_file():
        return ""
    source = path.read_text(encoding="utf-8")
    lines = source.splitlines()
    docstrings_added = 0
    result_lines: list[str] = []
    i = 0
    while i < len(lines):
        line = lines[i]
        stripped = line.strip()
        # Simple heuristic: class or def line not followed by a docstring
        if stripped.startswith("def ") or stripped.startswith("class "):
            result_lines.append(line)
            indent = len(line) - len(line.lstrip())
            # Peek next non-empty line
            j = i + 1
            while j < len(lines) and lines[j].strip() == "":
                j += 1
            next_line = lines[j] if j < len(lines) else ""
            next_stripped = next_line.strip()
            if not (next_stripped.startswith('"""') or next_stripped.startswith("'''")):
                # Insert a simple docstring
                name = stripped.split("(")[0].split()[-1]
                doc_line = " " * (indent + 4) + f'"""{name}"""'
                result_lines.append(doc_line)
                docstrings_added += 1
            i += 1
        else:
            result_lines.append(line)
            i += 1
    return "\n".join(result_lines) if docstrings_added > 0 else source


def _run_docs(args: Dict[str, Any]) -> ToolResult:
    file_path = str(args.get("file_path", ""))
    write = bool(args.get("write", False))
    if not file_path:
        return ToolResult(success=False, output="file_path is required", metadata={})

    path = Path(file_path)
    if not path.exists() or not path.is_file():
        return ToolResult(success=False, output=f"File not found: {file_path}", metadata={})

    updated = _generate_docstring(file_path)
    if write:
        path.write_text(updated, encoding="utf-8")
        return ToolResult(
            success=True,
            output=f"Updated docstrings in {file_path}",
            metadata={"file_path": file_path},
        )
    return ToolResult(
        success=True,
        output=updated,
        metadata={"file_path": file_path, "preview": True},
    )


def register_skills(tools: ToolRegistry) -> None:
    """Register documentation skill tools."""
    tools.register(
        "generate_docs",
        "Generate or update module docstrings. Returns a preview unless write=True.",
        lambda name, args: _run_docs(args),
        ActionRiskTier.WORKSPACE_WRITE,
        {
            "type": "object",
            "properties": {
                "file_path": {
                    "type": "string",
                    "description": "Python file to document.",
                },
                "write": {
                    "type": "boolean",
                    "description": "If true, write the updated source back to the file.",
                },
            },
            "required": ["file_path"],
        },
    )
