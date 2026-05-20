"""Read-oriented Himalaya CLI email adapter for OpenCAS.

Himalaya is a local IMAP/SMTP CLI. This adapter deliberately exposes only
inspection commands so email triage can use the configured local account without
falling back to a general shell command.
"""

from __future__ import annotations

import asyncio
import json
import os
from typing import Any, Dict, Optional

from opencas.tools.environment import build_tool_execution_env, resolve_executable

from ..models import ToolResult

_DEFAULT_TIMEOUT_SECONDS = 30


def himalaya_cli_command() -> str:
    """Return the configured Himalaya CLI command."""
    return str(os.getenv("HIMALAYA_CLI_COMMAND", "himalaya")).strip() or "himalaya"


def himalaya_cli_available(command: Optional[str] = None) -> bool:
    """Return True when the Himalaya CLI is available on PATH."""
    command_name = command or himalaya_cli_command()
    return resolve_executable(command_name) is not None


class HimalayaEmailToolAdapter:
    """Safe read-oriented wrapper around the local ``himalaya`` CLI."""

    def __init__(self, command: Optional[str] = None) -> None:
        self.command = command or himalaya_cli_command()

    async def __call__(self, name: str, args: Dict[str, Any]) -> ToolResult:
        try:
            if name == "himalaya_email_accounts":
                return await self._accounts(args)
            if name == "himalaya_email_headlines":
                return await self._headlines(args)
            if name == "himalaya_email_read_message":
                return await self._read_message(args)
            return ToolResult(False, f"Unknown Himalaya email tool: {name}", {})
        except Exception as exc:
            return ToolResult(False, str(exc), {"error_type": type(exc).__name__})

    async def _accounts(self, args: Dict[str, Any]) -> ToolResult:
        timeout = _coerce_timeout(args)
        return await self._run_json_command(["account", "list"], timeout=timeout)

    async def _headlines(self, args: Dict[str, Any]) -> ToolResult:
        timeout = _coerce_timeout(args)
        folder = str(args.get("folder", "INBOX")).strip() or "INBOX"
        page_size = max(1, min(int(args.get("page_size", 10)), 50))
        page = max(1, int(args.get("page", 1)))
        query = str(args.get("query", "order by date desc")).strip() or "order by date desc"
        argv = [
            "envelope",
            "list",
            "--folder",
            folder,
            "--page-size",
            str(page_size),
            "--page",
            str(page),
        ]
        account = _clean_optional(args.get("account"))
        if account:
            argv.extend(["--account", account])
        argv.extend(query.split())
        result = await self._run_json_command(argv, timeout=timeout)
        if result.success:
            result.metadata.update({"folder": folder, "query": query, "page_size": page_size, "page": page})
        return result

    async def _read_message(self, args: Dict[str, Any]) -> ToolResult:
        message_id = str(args.get("message_id", "")).strip()
        if not message_id:
            return ToolResult(False, "message_id is required", {})
        timeout = _coerce_timeout(args)
        folder = str(args.get("folder", "INBOX")).strip() or "INBOX"
        argv = ["message", "read", "--folder", folder, "--preview"]
        account = _clean_optional(args.get("account"))
        if account:
            argv.extend(["--account", account])
        if bool(args.get("no_headers")):
            argv.append("--no-headers")
        argv.append(message_id)
        result = await self._run_json_command(argv, timeout=timeout)
        if result.success:
            result.metadata.update({"folder": folder, "message_id": message_id, "preview": True})
        return result

    async def _run_json_command(self, argv: list[str], *, timeout: int) -> ToolResult:
        exe = resolve_executable(self.command)
        if exe is None:
            return ToolResult(False, f"Himalaya CLI not found: {self.command}", {"command": self.command})
        proc = await asyncio.create_subprocess_exec(
            exe,
            "--output",
            "json",
            *argv,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
            env=build_tool_execution_env(),
        )
        try:
            stdout, stderr = await asyncio.wait_for(proc.communicate(), timeout=timeout)
        except asyncio.TimeoutError:
            proc.kill()
            await proc.communicate()
            return ToolResult(False, f"Himalaya CLI timed out after {timeout}s", {"argv": argv})
        out = stdout.decode("utf-8", errors="replace").strip()
        err = stderr.decode("utf-8", errors="replace").strip()
        if proc.returncode != 0:
            return ToolResult(False, err or out or f"himalaya exited {proc.returncode}", {"argv": argv, "returncode": proc.returncode})
        # Preserve valid JSON as-is; wrap plain/empty output for callers that expect JSON.
        if out:
            try:
                json.loads(out)
                payload = out
            except json.JSONDecodeError:
                payload = json.dumps({"output": out}, indent=2)
        else:
            payload = json.dumps({"output": ""}, indent=2)
        return ToolResult(True, payload, {"argv": argv})


def _clean_optional(value: Any) -> Optional[str]:
    text = str(value or "").strip()
    return text or None


def _coerce_timeout(args: Dict[str, Any]) -> int:
    try:
        value = int(args.get("timeout_seconds") or _DEFAULT_TIMEOUT_SECONDS)
    except (TypeError, ValueError):
        value = _DEFAULT_TIMEOUT_SECONDS
    return max(1, min(value, 120))
