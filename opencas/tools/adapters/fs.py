"""Filesystem tool adapter for OpenCAS."""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any, Dict, List

from ..models import ToolResult


class FileSystemToolAdapter:
    """Adapter for filesystem read/list/write operations with path policy."""

    def __init__(self, allowed_roots: List[str]) -> None:
        self.allowed_roots = [Path(r).expanduser().resolve() for r in allowed_roots]

    def __call__(self, name: str, args: Dict[str, Any]) -> ToolResult:
        try:
            if name == "fs_read_file":
                return self._read_file(args)
            if name == "fs_list_dir":
                return self._list_dir(args)
            if name == "fs_write_file":
                return self._write_file(args)
            return ToolResult(success=False, output=f"Unknown fs tool: {name}", metadata={})
        except Exception as exc:
            return ToolResult(success=False, output=str(exc), metadata={"error_type": type(exc).__name__})

    def _read_file(self, args: Dict[str, Any]) -> ToolResult:
        path = self._resolve_path(args.get("file_path", ""))
        text = path.read_text(encoding="utf-8")
        return ToolResult(success=True, output=text, metadata={"path": str(path)})

    def _list_dir(self, args: Dict[str, Any]) -> ToolResult:
        path = self._resolve_path(args.get("dir_path", ""))
        entries = []
        for entry in path.iterdir():
            entries.append({"name": entry.name, "is_directory": entry.is_dir()})
        return ToolResult(
            success=True,
            output=json.dumps({"ok": True, "entries": entries}),
            metadata={"path": str(path)},
        )

    def _write_file(self, args: Dict[str, Any]) -> ToolResult:
        path = self._resolve_path(args.get("file_path", ""))
        content = str(args.get("content", ""))
        path.parent.mkdir(parents=True, exist_ok=True)
        temp = path.with_suffix(".tmp")
        temp.write_text(content, encoding="utf-8")
        with temp.open("rb") as handle:
            os.fsync(handle.fileno())
        temp.replace(path)
        return ToolResult(
            success=True,
            output=json.dumps({"ok": True, "bytes_written": len(content)}),
            metadata={"path": str(path)},
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
