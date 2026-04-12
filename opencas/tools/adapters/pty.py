"""PTY tool adapter for terminal-native interactive sessions."""

from __future__ import annotations

import json
from typing import Any, Dict

from ...execution.pty_supervisor import PtySupervisor
from ..models import ToolResult


class PtyToolAdapter:
    """Adapter for managing PTY-backed interactive terminal sessions."""

    def __init__(
        self,
        supervisor: PtySupervisor,
        default_cwd: str,
    ) -> None:
        self.supervisor = supervisor
        self.default_cwd = str(default_cwd)

    def __call__(self, name: str, args: Dict[str, Any]) -> ToolResult:
        scope_key = str(args.get("scope_key", "default"))

        if name == "pty_start":
            command = str(args.get("command", ""))
            cwd = args.get("cwd") or self.default_cwd
            rows = int(args.get("rows", 24))
            cols = int(args.get("cols", 80))
            if not command:
                return ToolResult(False, "Missing required argument: command", {})
            try:
                session_id = self.supervisor.start(
                    scope_key,
                    command,
                    cwd=cwd,
                    rows=rows,
                    cols=cols,
                )
                return ToolResult(
                    True,
                    json.dumps({"session_id": session_id, "status": "started"}),
                    {
                        "scope_key": scope_key,
                        "command": command,
                        "cwd": cwd,
                        "rows": rows,
                        "cols": cols,
                    },
                )
            except Exception as exc:
                return ToolResult(False, str(exc), {"error_type": type(exc).__name__})

        if name == "pty_poll":
            session_id = str(args.get("session_id", ""))
            max_bytes = int(args.get("max_bytes", 8192))
            if not session_id:
                return ToolResult(False, "Missing required argument: session_id", {})
            result = self.supervisor.poll(scope_key, session_id, max_bytes=max_bytes)
            return ToolResult(
                success=result.get("found", False),
                output=json.dumps(result),
                metadata={"scope_key": scope_key},
            )

        if name == "pty_observe":
            session_id = str(args.get("session_id", ""))
            idle_seconds = float(args.get("idle_seconds", 0.4))
            max_wait_seconds = float(args.get("max_wait_seconds", 8.0))
            max_bytes_per_poll = int(args.get("max_bytes_per_poll", 4096))
            if not session_id:
                return ToolResult(False, "Missing required argument: session_id", {})
            result = self.supervisor.observe_until_quiet(
                scope_key,
                session_id,
                idle_seconds=idle_seconds,
                max_wait_seconds=max_wait_seconds,
                max_bytes_per_poll=max_bytes_per_poll,
            )
            return ToolResult(
                success=result.get("found", False),
                output=json.dumps(result),
                metadata={"scope_key": scope_key},
            )

        if name == "pty_interact":
            session_id = str(args.get("session_id", "")).strip() or None
            command = str(args.get("command", "")).strip() or None
            cwd = args.get("cwd") or self.default_cwd
            rows = int(args.get("rows", 24))
            cols = int(args.get("cols", 80))
            input_text = args.get("input")
            if input_text is not None:
                input_text = str(input_text)
            idle_seconds = float(args.get("idle_seconds", 0.4))
            max_wait_seconds = float(args.get("max_wait_seconds", 8.0))
            max_bytes_per_poll = int(args.get("max_bytes_per_poll", 4096))
            if session_id is None and command is None:
                return ToolResult(
                    False,
                    "Missing required argument: session_id or command",
                    {},
                )
            result = self.supervisor.interact(
                scope_key,
                session_id=session_id,
                command=command,
                cwd=cwd,
                rows=rows,
                cols=cols,
                input_text=input_text,
                idle_seconds=idle_seconds,
                max_wait_seconds=max_wait_seconds,
                max_bytes_per_poll=max_bytes_per_poll,
            )
            return ToolResult(
                success=result.get("found", False),
                output=json.dumps(result),
                metadata={"scope_key": scope_key},
            )

        if name == "pty_write":
            session_id = str(args.get("session_id", ""))
            input_text = str(args.get("input", ""))
            if not session_id:
                return ToolResult(False, "Missing required argument: session_id", {})
            ok = self.supervisor.write(scope_key, session_id, input_text)
            return ToolResult(True, json.dumps({"ok": ok}), {"scope_key": scope_key})

        if name == "pty_resize":
            session_id = str(args.get("session_id", ""))
            rows = int(args.get("rows", 24))
            cols = int(args.get("cols", 80))
            if not session_id:
                return ToolResult(False, "Missing required argument: session_id", {})
            ok = self.supervisor.resize(scope_key, session_id, rows=rows, cols=cols)
            return ToolResult(
                True,
                json.dumps({"ok": ok, "rows": rows, "cols": cols}),
                {"scope_key": scope_key},
            )

        if name == "pty_kill":
            session_id = str(args.get("session_id", ""))
            if not session_id:
                return ToolResult(False, "Missing required argument: session_id", {})
            ok = self.supervisor.kill(scope_key, session_id)
            return ToolResult(True, json.dumps({"ok": ok}), {"scope_key": scope_key})

        if name == "pty_remove":
            session_id = str(args.get("session_id", ""))
            if not session_id:
                return ToolResult(False, "Missing required argument: session_id", {})
            ok = self.supervisor.remove(scope_key, session_id)
            return ToolResult(True, json.dumps({"ok": ok}), {"scope_key": scope_key})

        if name == "pty_clear":
            count = self.supervisor.clear(scope_key)
            return ToolResult(True, json.dumps({"removed": count}), {"scope_key": scope_key})

        return ToolResult(False, f"Unknown PTY tool: {name}", {})
