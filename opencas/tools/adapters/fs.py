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
        self.DEFAULT_READ_LIMIT = 2000

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

        requested_pagination = "offset" in args or "limit" in args
        if not requested_pagination:
            return ToolResult(success=True, output=text, metadata={"path": str(path)})

        offset = max(0, _coerce_int(args.get("offset"), default=0))
        limit = _coerce_int(args.get("limit"), default=self.DEFAULT_READ_LIMIT)
        if limit <= 0:
            limit = self.DEFAULT_READ_LIMIT

        total_chars = len(text)
        total_lines = len(text.splitlines())
        chunk = text[offset : offset + limit]
        returned_count = len(chunk)
        end_index = offset + returned_count
        truncated = end_index < total_chars
        next_offset = end_index if truncated else None

        payload = {
            "ok": True,
            "path": str(path),
            "content": chunk,
            "pagination_unit": "chars",
            "total_chars": total_chars,
            "total_count": total_chars,
            "total_lines": total_lines,
            "returned_count": returned_count,
            "offset": offset,
            "limit": limit,
            "truncated": truncated,
            "next_offset": next_offset,
            "read_session_id": _coerce_str(args.get("read_session_id"), default=""),
            "concept_scope": _coerce_str(args.get("concept_scope"), default=""),
            "concept_label": _coerce_str(args.get("concept_label"), default=""),
        }
        metadata = {
            "path": str(path),
            "pagination_unit": "chars",
            "total_chars": total_chars,
            "total_count": total_chars,
            "total_lines": total_lines,
            "returned_count": returned_count,
            "offset": offset,
            "limit": limit,
            "truncated": truncated,
            "next_offset": next_offset,
        }
        return ToolResult(
            success=True,
            output=json.dumps(payload),
            metadata=metadata,
        )

    DEFAULT_LIST_LIMIT = 1000

    def _list_dir(self, args: Dict[str, Any]) -> ToolResult:
        path = self._resolve_path(args.get("dir_path", ""))
        all_entries = sorted(path.iterdir(), key=lambda item: item.name.lower())
        total_count = len(all_entries)

        offset = max(0, _coerce_int(args.get("offset"), default=0))
        limit_raw = args.get("limit")
        limit = _coerce_int(limit_raw, default=self.DEFAULT_LIST_LIMIT)
        if limit <= 0:
            limit = self.DEFAULT_LIST_LIMIT

        window = all_entries[offset : offset + limit]
        entries = [
            {"name": entry.name, "is_directory": entry.is_dir()} for entry in window
        ]
        returned_count = len(entries)
        end_index = offset + returned_count
        truncated = end_index < total_count
        next_offset = end_index if truncated else None

        payload = {
            "ok": True,
            "path": str(path),
            "entries": entries,
            "total_count": total_count,
            "returned_count": returned_count,
            "offset": offset,
            "limit": limit,
            "truncated": truncated,
            "next_offset": next_offset,
        }
        metadata = {
            "path": str(path),
            "total_count": total_count,
            "returned_count": returned_count,
            "offset": offset,
            "limit": limit,
            "truncated": truncated,
            "next_offset": next_offset,
        }
        return ToolResult(
            success=True,
            output=json.dumps(payload),
            metadata=metadata,
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


def _coerce_int(value: Any, *, default: int) -> int:
    if value is None:
        return default
    try:
        return int(value)
    except (TypeError, ValueError):
        return default


def _coerce_str(value: Any, *, default: str = "") -> str:
    text = str(value or "").strip()
    return text if text else default
