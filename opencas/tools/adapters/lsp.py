"""LSP tool adapter providing lightweight code intelligence."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List

from ..models import ToolResult


class LspToolAdapter:
    """Adapter for lightweight LSP operations using jedi (optional)."""

    def __init__(self) -> None:
        pass

    def __call__(self, name: str, args: Dict[str, Any]) -> ToolResult:
        try:
            if name == "lsp_goto_definition":
                return self._goto_definition(args)
            if name == "lsp_find_references":
                return self._find_references(args)
            if name == "lsp_hover":
                return self._hover(args)
            if name == "lsp_document_symbols":
                return self._document_symbols(args)
            if name == "lsp_diagnostics":
                return self._diagnostics(args)
            return ToolResult(success=False, output=f"Unknown LSP tool: {name}", metadata={})
        except Exception as exc:
            return ToolResult(success=False, output=str(exc), metadata={"error_type": type(exc).__name__})

    def _ensure_jedi(self) -> Any:
        try:
            import jedi
        except ImportError as exc:
            raise ImportError("jedi is required for LSP tools; install it with: pip install jedi") from exc
        return jedi

    def _goto_definition(self, args: Dict[str, Any]) -> ToolResult:
        jedi = self._ensure_jedi()
        file_path = str(args.get("file_path", ""))
        line = int(args.get("line", 1))
        character = int(args.get("character", 0))
        if not file_path:
            return ToolResult(success=False, output="file_path is required", metadata={})

        script = jedi.Script(path=file_path)
        definitions = script.goto(line=line, column=character)
        results: List[Dict[str, Any]] = []
        for d in definitions:
            results.append({
                "name": d.name,
                "file_path": d.module_path,
                "line": d.line,
                "column": d.column,
                "type": d.type,
            })
        return ToolResult(
            success=True,
            output=json.dumps({"ok": True, "definitions": results}, indent=2),
            metadata={"result_count": len(results)},
        )

    def _find_references(self, args: Dict[str, Any]) -> ToolResult:
        jedi = self._ensure_jedi()
        file_path = str(args.get("file_path", ""))
        line = int(args.get("line", 1))
        character = int(args.get("character", 0))
        if not file_path:
            return ToolResult(success=False, output="file_path is required", metadata={})

        script = jedi.Script(path=file_path)
        references = script.get_references(line=line, column=character)
        results: List[Dict[str, Any]] = []
        for r in references:
            results.append({
                "name": r.name,
                "file_path": r.module_path,
                "line": r.line,
                "column": r.column,
            })
        return ToolResult(
            success=True,
            output=json.dumps({"ok": True, "references": results}, indent=2),
            metadata={"result_count": len(results)},
        )

    def _hover(self, args: Dict[str, Any]) -> ToolResult:
        jedi = self._ensure_jedi()
        file_path = str(args.get("file_path", ""))
        line = int(args.get("line", 1))
        character = int(args.get("character", 0))
        if not file_path:
            return ToolResult(success=False, output="file_path is required", metadata={})

        script = jedi.Script(path=file_path)
        signatures = script.get_signatures(line=line, column=character)
        hover_texts: List[str] = []
        for sig in signatures:
            hover_texts.append(sig.to_string())

        # If no signatures, try inferred type
        if not hover_texts:
            inferences = script.infer(line=line, column=character)
            for inf in inferences:
                desc = inf.description or inf.name or ""
                if desc:
                    hover_texts.append(desc)

        return ToolResult(
            success=True,
            output="\n".join(hover_texts) if hover_texts else "(no info)",
            metadata={"result_count": len(hover_texts)},
        )

    def _document_symbols(self, args: Dict[str, Any]) -> ToolResult:
        jedi = self._ensure_jedi()
        file_path = str(args.get("file_path", ""))
        if not file_path:
            return ToolResult(success=False, output="file_path is required", metadata={})

        script = jedi.Script(path=file_path)
        names = script.get_names(all_scopes=True, definitions=True, references=False)
        results: List[Dict[str, Any]] = []
        for n in names:
            results.append({
                "name": n.name,
                "type": n.type,
                "line": n.line,
                "column": n.column,
            })
        return ToolResult(
            success=True,
            output=json.dumps({"ok": True, "symbols": results}, indent=2),
            metadata={"result_count": len(results)},
        )

    def _diagnostics(self, args: Dict[str, Any]) -> ToolResult:
        jedi = self._ensure_jedi()
        file_path = str(args.get("file_path", ""))
        if not file_path:
            return ToolResult(success=False, output="file_path is required", metadata={})

        source = Path(file_path).read_text(encoding="utf-8", errors="ignore")
        try:
            errors = jedi.api.environment.find_syntax_errors(source, path=file_path)
        except Exception:
            # Fallback: use Script parsing
            script = jedi.Script(code=source, path=file_path)
            errors = []
            for error in script.get_syntax_errors():
                errors.append(error)

        results: List[Dict[str, Any]] = []
        for err in errors:
            results.append({
                "message": getattr(err, "message", str(err)),
                "line": getattr(err, "line", None),
                "column": getattr(err, "column", None),
            })
        return ToolResult(
            success=True,
            output=json.dumps({"ok": True, "errors": results}, indent=2),
            metadata={"error_count": len(results)},
        )
