"""Edit tool adapter for precise file modifications."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Optional

from ..models import ToolResult


class EditToolAdapter:
    """Adapter for precise string replacement in files."""

    def __init__(self, allowed_roots: List[str]) -> None:
        self.allowed_roots = [Path(r).expanduser().resolve() for r in allowed_roots]

    def __call__(self, name: str, args: Dict[str, Any]) -> ToolResult:
        if name != "edit_file":
            return ToolResult(success=False, output=f"Unknown edit tool: {name}", metadata={})
        try:
            return self._edit_file(args)
        except Exception as exc:
            return ToolResult(success=False, output=str(exc), metadata={"error_type": type(exc).__name__})

    def _edit_file(self, args: Dict[str, Any]) -> ToolResult:
        file_path = str(args.get("file_path", ""))
        old_string = str(args.get("old_string", ""))
        new_string = str(args.get("new_string", ""))
        occurrence_index: Optional[int] = args.get("occurrence_index")

        if not file_path:
            return ToolResult(success=False, output="file_path is required", metadata={})
        if old_string == "":
            return ToolResult(success=False, output="old_string is required", metadata={})

        path = self._resolve_path(file_path)
        original = path.read_text(encoding="utf-8")

        if occurrence_index is not None:
            parts = original.split(old_string)
            if occurrence_index < 0 or occurrence_index >= len(parts) - 1:
                return ToolResult(
                    success=False,
                    output=f"occurrence_index {occurrence_index} out of range ({len(parts) - 1} occurrences)",
                    metadata={"path": str(path)},
                )
            modified = old_string.join(parts[: occurrence_index + 1]) + new_string + old_string.join(parts[occurrence_index + 1 :])
        else:
            count = original.count(old_string)
            if count == 0:
                return ToolResult(
                    success=False,
                    output="old_string not found in file",
                    metadata={"path": str(path)},
                )
            if count > 1:
                return ToolResult(
                    success=False,
                    output=f"old_string appears {count} times; provide occurrence_index to disambiguate",
                    metadata={"path": str(path), "occurrences": count},
                )
            modified = original.replace(old_string, new_string, 1)

        path.write_text(modified, encoding="utf-8")
        return ToolResult(
            success=True,
            output=f"Edited {path}",
            metadata={"path": str(path), "replacements": 1},
        )

    def _resolve_path(self, raw: str) -> Path:
        target = Path(raw).expanduser().resolve()
        if not self.allowed_roots:
            return target
        for root in self.allowed_roots:
            try:
                target.relative_to(root)
                return target
            except ValueError:
                continue
        raise PermissionError(f"Path {target} is outside allowed roots: {self.allowed_roots}")
