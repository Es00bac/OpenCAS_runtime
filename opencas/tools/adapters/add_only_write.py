"""Add-only file write adapter for bounded assistant workspaces."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, Iterable, List

from ..models import ToolResult

_SAFE_APPEND_SUFFIXES = {
    ".md",
    ".txt",
    ".log",
    ".csv",
    ".jsonl",
}


class AddOnlyFileWriteToolAdapter:
    """Create or append text files without allowing overwrite or deletion."""

    def __init__(
        self,
        allowed_roots: List[str],
        *,
        allowed_suffixes: Iterable[str] = _SAFE_APPEND_SUFFIXES,
    ) -> None:
        self.allowed_roots = [Path(root).expanduser().resolve() for root in allowed_roots]
        self.allowed_suffixes = {suffix.lower() for suffix in allowed_suffixes}

    def __call__(self, name: str, args: Dict[str, Any]) -> ToolResult:
        try:
            if name != "fs_write_file":
                return ToolResult(success=False, output=f"Unknown add-only write tool: {name}", metadata={})
            return self._write_file(args)
        except Exception as exc:
            return ToolResult(success=False, output=str(exc), metadata={"error_type": type(exc).__name__})

    def _write_file(self, args: Dict[str, Any]) -> ToolResult:
        path = self._resolve_path(args.get("file_path", ""))
        content = str(args.get("content", "")).strip()
        if not content:
            return ToolResult(success=False, output="content is required", metadata={"path": str(path)})
        if path.suffix.lower() not in self.allowed_suffixes:
            return ToolResult(
                success=False,
                output=(
                    "add-only writes are restricted to note-like files: "
                    + ", ".join(sorted(self.allowed_suffixes))
                ),
                metadata={"path": str(path)},
            )

        created = not path.exists()
        path.parent.mkdir(parents=True, exist_ok=True)
        existing = path.read_text(encoding="utf-8") if path.exists() else ""

        prefix = ""
        if existing:
            if not existing.endswith("\n"):
                prefix += "\n"
            prefix += "\n"
        payload = prefix + content + ("\n" if not content.endswith("\n") else "")
        with path.open("a", encoding="utf-8") as handle:
            handle.write(payload)

        return ToolResult(
            success=True,
            output=json.dumps(
                {
                    "ok": True,
                    "mode": "created" if created else "appended",
                    "bytes_written": len(payload),
                }
            ),
            metadata={"path": str(path), "created": created},
        )

    def _resolve_path(self, raw: str) -> Path:
        target = Path(str(raw)).expanduser().resolve()
        if not self.allowed_roots:
            return target
        for root in self.allowed_roots:
            try:
                target.relative_to(root)
                return target
            except ValueError:
                continue
        raise PermissionError(f"Path {target} is outside allowed roots: {self.allowed_roots}")
