"""Process tool adapter for OpenCAS."""

from __future__ import annotations

import json
from typing import Any, Dict

from ...execution.process_supervisor import ProcessSupervisor
from ..models import ToolResult


class ProcessToolAdapter:
    """Adapter for managing long-running background shell processes."""

    def __init__(
        self,
        supervisor: ProcessSupervisor,
        default_cwd: str,
    ) -> None:
        self.supervisor = supervisor
        self.default_cwd = str(default_cwd)

    def __call__(self, name: str, args: Dict[str, Any]) -> ToolResult:
        scope_key = str(args.get("scope_key", "default"))

        if name == "process_start":
            command = str(args.get("command", ""))
            cwd = args.get("cwd") or self.default_cwd
            if not command:
                return ToolResult(
                    success=False,
                    output="Missing required argument: command",
                    metadata={},
                )
            try:
                process_id = self.supervisor.start(scope_key, command, cwd=cwd)
                return ToolResult(
                    success=True,
                    output=json.dumps({"process_id": process_id, "status": "started"}),
                    metadata={"command": command, "scope_key": scope_key, "cwd": cwd},
                )
            except Exception as exc:
                return ToolResult(
                    success=False,
                    output=str(exc),
                    metadata={"error_type": type(exc).__name__},
                )

        if name == "process_poll":
            process_id = str(args.get("process_id", ""))
            if not process_id:
                return ToolResult(
                    success=False,
                    output="Missing required argument: process_id",
                    metadata={},
                )
            result = self.supervisor.poll(scope_key, process_id)
            return ToolResult(
                success=result.get("found", False),
                output=json.dumps(result),
                metadata={"scope_key": scope_key},
            )

        if name == "process_write":
            process_id = str(args.get("process_id", ""))
            input_text = str(args.get("input", ""))
            if not process_id:
                return ToolResult(
                    success=False,
                    output="Missing required argument: process_id",
                    metadata={},
                )
            ok = self.supervisor.write(scope_key, process_id, input_text)
            return ToolResult(
                success=ok,
                output=json.dumps({"ok": ok}),
                metadata={"scope_key": scope_key},
            )

        if name == "process_send_signal":
            process_id = str(args.get("process_id", ""))
            signal_num = int(args.get("signal", 15))
            if not process_id:
                return ToolResult(
                    success=False,
                    output="Missing required argument: process_id",
                    metadata={},
                )
            ok = self.supervisor.send_signal(scope_key, process_id, signal_num)
            return ToolResult(
                success=ok,
                output=json.dumps({"ok": ok}),
                metadata={"scope_key": scope_key, "signal": signal_num},
            )

        if name == "process_kill":
            process_id = str(args.get("process_id", ""))
            if not process_id:
                return ToolResult(
                    success=False,
                    output="Missing required argument: process_id",
                    metadata={},
                )
            ok = self.supervisor.kill(scope_key, process_id)
            return ToolResult(
                success=ok,
                output=json.dumps({"ok": ok}),
                metadata={"scope_key": scope_key},
            )

        if name == "process_clear":
            count = self.supervisor.clear(scope_key)
            return ToolResult(
                success=True,
                output=json.dumps({"removed": count}),
                metadata={"scope_key": scope_key},
            )

        if name == "process_remove":
            process_id = str(args.get("process_id", ""))
            if not process_id:
                return ToolResult(
                    success=False,
                    output="Missing required argument: process_id",
                    metadata={},
                )
            ok = self.supervisor.remove(scope_key, process_id)
            return ToolResult(
                success=ok,
                output=json.dumps({"ok": ok}),
                metadata={"scope_key": scope_key},
            )

        return ToolResult(
            success=False,
            output=f"Unknown process tool: {name}",
            metadata={},
        )
