"""REPL tool adapter for Python code execution."""

from __future__ import annotations

import io
import json
import sys
import traceback
from code import InteractiveInterpreter
from typing import Any, Dict, List

from ..models import ToolResult


class ReplToolAdapter:
    """Adapter for Python REPL execution with persistent sessions."""

    def __init__(self) -> None:
        self._sessions: Dict[str, InteractiveInterpreter] = {}

    def __call__(self, name: str, args: Dict[str, Any]) -> ToolResult:
        try:
            if name == "python_repl":
                return self._python_repl(args)
            return ToolResult(success=False, output=f"Unknown REPL tool: {name}", metadata={})
        except Exception as exc:
            return ToolResult(success=False, output=str(exc), metadata={"error_type": type(exc).__name__})

    def _python_repl(self, args: Dict[str, Any]) -> ToolResult:
        code = str(args.get("code", ""))
        session_id = str(args.get("research_session_id", "default"))
        if not code:
            return ToolResult(success=False, output="code is required", metadata={})

        interpreter = self._sessions.setdefault(session_id, InteractiveInterpreter())
        stdout_buf = io.StringIO()
        stderr_buf = io.StringIO()

        old_stdout, old_stderr = sys.stdout, sys.stderr
        sys.stdout, sys.stderr = stdout_buf, stderr_buf

        try:
            # runsource returns True if more input is expected (incomplete syntax)
            needs_more = interpreter.runsource(code, filename="<repl>", symbol="exec")
        except Exception:
            traceback.print_exc()
            success = False
        else:
            success = True
        finally:
            sys.stdout, sys.stderr = old_stdout, old_stderr

        stdout_text = stdout_buf.getvalue()
        stderr_text = stderr_buf.getvalue()

        if not success:
            return ToolResult(
                success=False,
                output=stderr_text or "Execution failed",
                metadata={"session_id": session_id, "stdout": stdout_text},
            )

        output = stdout_text
        if stderr_text:
            output += "\n" + stderr_text if output else stderr_text

        return ToolResult(
            success=True,
            output=output.strip() if output.strip() else "(no output)",
            metadata={"session_id": session_id, "needs_more": needs_more},
        )
