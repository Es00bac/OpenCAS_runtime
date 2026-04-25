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

        if not pattern:
            return ToolResult(success=False, output="pattern is required", metadata={})

        search_paths = self._resolve_search_paths(raw_path)

        # Try ripgrep first if available
        try:
            rg_results = self._run_ripgrep(pattern, search_paths, output_mode)
            if rg_results is not None:
                return rg_results
        except Exception:
            pass

        # Fallback to Python re traversal
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
                                matches.append({
                                    "path": str(file_path),
                                    "line": line_no,
                                    "content": line,
                                })

        if output_mode == "files_with_matches":
            return ToolResult(
                success=True,
                output=json.dumps({"ok": True, "files": list(set(files_matched))}),
                metadata={"match_count": len(files_matched)},
            )
        return ToolResult(
            success=True,
            output=json.dumps({"ok": True, "matches": matches[:500]}, indent=2),
            metadata={"match_count": len(matches)},
        )

    def _run_ripgrep(
        self, pattern: str, search_paths: List[Path], output_mode: str
    ) -> ToolResult | None:
        cmd = ["rg", "-n", "--no-heading"]
        if output_mode == "files_with_matches":
            cmd.append("-l")
        cmd.extend(["--", pattern])
        cmd.extend(str(p) for p in search_paths)
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=30.0)
        if result.returncode not in (0, 1):
            return None
        if output_mode == "files_with_matches":
            files = [line.strip() for line in result.stdout.splitlines() if line.strip()]
            return ToolResult(
                success=True,
                output=json.dumps({"ok": True, "files": files}),
                metadata={"match_count": len(files)},
            )
        matches: List[Dict[str, Any]] = []
        for line in result.stdout.splitlines():
            if ":" not in line:
                continue
            path_str, rest = line.split(":", 1)
            line_no_str, content = rest.split(":", 1)
            matches.append({
                "path": path_str,
                "line": int(line_no_str),
                "content": content,
            })
        return ToolResult(
            success=True,
            output=json.dumps({"ok": True, "matches": matches[:500]}, indent=2),
            metadata={"match_count": len(matches)},
        )

    def _glob_search(self, args: Dict[str, Any]) -> ToolResult:
        pattern = str(args.get("pattern", ""))
        raw_path = args.get("path")

        if not pattern:
            return ToolResult(success=False, output="pattern is required", metadata={})

        search_paths = self._resolve_search_paths(raw_path)
        results: List[str] = []
        for root in search_paths:
            if root.is_file():
                results.append(str(root))
            else:
                for p in root.rglob(pattern):
                    results.append(str(p))
        return ToolResult(
            success=True,
            output=json.dumps({"ok": True, "files": results[:500]}),
            metadata={"match_count": len(results)},
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
