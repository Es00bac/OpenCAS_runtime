"""Search tool adapters for grep and glob operations."""

from __future__ import annotations

import json
import re
import subprocess
from pathlib import Path
from typing import Any, Dict, List

from ..models import ToolResult


class SearchToolAdapter:
    """Adapter for grep and glob search operations."""

    DEFAULT_RESULT_LIMIT = 500

    def __init__(self, allowed_roots: List[str]) -> None:
        self.allowed_roots = [Path(r).expanduser().resolve() for r in allowed_roots]

    def __call__(self, name: str, args: Dict[str, Any]) -> ToolResult:
        try:
            if name == "grep_search":
                return self._grep_search(args)
            if name == "glob_search":
                return self._glob_search(args)
            return ToolResult(success=False, output=f"Unknown search tool: {name}", metadata={})
        except Exception as exc:
            return ToolResult(success=False, output=str(exc), metadata={"error_type": type(exc).__name__})

    def _grep_search(self, args: Dict[str, Any]) -> ToolResult:
        pattern = str(args.get("pattern", ""))
        raw_path = args.get("path")
        output_mode = str(args.get("output_mode", "content"))
        glob = args.get("glob")
        max_count_raw = args.get("max_count")
        max_count = int(max_count_raw) if max_count_raw is not None else None
        glob_str = str(glob) if isinstance(glob, str) and glob else None

        offset = max(0, _coerce_int(args.get("offset"), default=0))
        limit = _coerce_int(args.get("limit"), default=self.DEFAULT_RESULT_LIMIT)
        if limit <= 0:
            limit = self.DEFAULT_RESULT_LIMIT

        if not pattern:
            return ToolResult(success=False, output="pattern is required", metadata={})

        search_paths = self._resolve_search_paths(raw_path)

        matches: List[Dict[str, Any]] | None = None
        files_matched: List[str] | None = None

        # Try ripgrep first if available
        try:
            rg_matches, rg_files = self._run_ripgrep(
                pattern,
                search_paths,
                output_mode,
                glob=glob_str,
                max_count=max_count,
            )
            matches, files_matched = rg_matches, rg_files
        except Exception:
            matches, files_matched = None, None

        if matches is None and files_matched is None:
            matches, files_matched = self._python_grep(
                pattern, search_paths, output_mode
            )

        if output_mode == "files_with_matches":
            unique_files = sorted(set(files_matched or []))
            return self._paginated_result(
                items=unique_files,
                payload_key="files",
                offset=offset,
                limit=limit,
            )

        sorted_matches = sorted(
            matches or [],
            key=lambda m: (m.get("path", ""), m.get("line", 0)),
        )
        return self._paginated_result(
            items=sorted_matches,
            payload_key="matches",
            offset=offset,
            limit=limit,
            indent=2,
        )

    @staticmethod
    def _paginated_result(
        *,
        items: List[Any],
        payload_key: str,
        offset: int,
        limit: int,
        indent: int | None = None,
    ) -> ToolResult:
        total_count = len(items)
        window = items[offset : offset + limit]
        returned_count = len(window)
        end_index = offset + returned_count
        truncated = end_index < total_count
        next_offset = end_index if truncated else None

        payload = {
            "ok": True,
            payload_key: window,
            "total_count": total_count,
            "returned_count": returned_count,
            "offset": offset,
            "limit": limit,
            "truncated": truncated,
            "next_offset": next_offset,
        }
        metadata = {
            "match_count": total_count,
            "total_count": total_count,
            "returned_count": returned_count,
            "offset": offset,
            "limit": limit,
            "truncated": truncated,
            "next_offset": next_offset,
        }
        return ToolResult(
            success=True,
            output=json.dumps(payload, indent=indent) if indent else json.dumps(payload),
            metadata=metadata,
        )

    def _python_grep(
        self,
        pattern: str,
        search_paths: List[Path],
        output_mode: str,
    ) -> tuple[List[Dict[str, Any]], List[str]]:
        matches: List[Dict[str, Any]] = []
        files_matched: List[str] = []
        compiled = re.compile(pattern)
        for root in search_paths:
            if root.is_file():
                files = [root]
            else:
                files = [p for p in root.rglob("*") if p.is_file()]
            for file_path in files:
                try:
                    text = file_path.read_text(encoding="utf-8", errors="ignore")
                except Exception:
                    continue
                if compiled.search(text):
                    files_matched.append(str(file_path))
                    if output_mode == "content":
                        for line_no, line in enumerate(text.splitlines(), start=1):
                            if compiled.search(line):
                                matches.append(
                                    {
                                        "path": str(file_path),
                                        "line": line_no,
                                        "content": line,
                                    }
                                )
        return matches, files_matched

    def _run_ripgrep(
        self,
        pattern: str,
        search_paths: List[Path],
        output_mode: str,
        *,
        glob: str | None = None,
        max_count: int | None = None,
    ) -> tuple[List[Dict[str, Any]] | None, List[str] | None]:
        cmd = ["rg", "-n", "--no-heading"]
        if output_mode == "files_with_matches":
            cmd.append("-l")
        if glob:
            cmd.extend(["--glob", glob])
        if max_count is not None:
            cmd.extend(["--max-count", str(max_count)])
        cmd.extend(["--", pattern])
        cmd.extend(str(p) for p in search_paths)
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=30.0)
        if result.returncode not in (0, 1):
            return None, None
        if output_mode == "files_with_matches":
            files = [line.strip() for line in result.stdout.splitlines() if line.strip()]
            return None, files
        matches: List[Dict[str, Any]] = []
        for line in result.stdout.splitlines():
            if ":" not in line:
                continue
            path_str, rest = line.split(":", 1)
            line_no_str, content = rest.split(":", 1)
            try:
                line_no = int(line_no_str)
            except ValueError:
                continue
            matches.append(
                {
                    "path": path_str,
                    "line": line_no,
                    "content": content,
                }
            )
        return matches, None

    def _glob_search(self, args: Dict[str, Any]) -> ToolResult:
        pattern = str(args.get("pattern", ""))
        raw_path = args.get("path")

        if not pattern:
            return ToolResult(success=False, output="pattern is required", metadata={})

        offset = max(0, _coerce_int(args.get("offset"), default=0))
        limit = _coerce_int(args.get("limit"), default=self.DEFAULT_RESULT_LIMIT)
        if limit <= 0:
            limit = self.DEFAULT_RESULT_LIMIT

        search_paths = self._resolve_search_paths(raw_path)
        seen: set[str] = set()
        results: List[str] = []
        for root in search_paths:
            if root.is_file():
                key = str(root)
                if key not in seen:
                    seen.add(key)
                    results.append(key)
            else:
                for p in root.rglob(pattern):
                    key = str(p)
                    if key not in seen:
                        seen.add(key)
                        results.append(key)
        results.sort(key=lambda value: (Path(value).name.lower(), value))
        return self._paginated_result(
            items=results,
            payload_key="files",
            offset=offset,
            limit=limit,
        )

    def _resolve_search_paths(self, raw_path: Any) -> List[Path]:
        if raw_path:
            paths = [Path(str(raw_path)).expanduser().resolve()]
        else:
            paths = self.allowed_roots or [Path.cwd()]
        validated: List[Path] = []
        for target in paths:
            if not self.allowed_roots:
                validated.append(target)
                continue
            for root in self.allowed_roots:
                try:
                    target.relative_to(root)
                    validated.append(target)
                    break
                except ValueError:
                    continue
            else:
                raise PermissionError(f"Path {target} is outside allowed roots: {self.allowed_roots}")
        return validated


def _coerce_int(value: Any, *, default: int) -> int:
    if value is None:
        return default
    try:
        return int(value)
    except (TypeError, ValueError):
        return default
