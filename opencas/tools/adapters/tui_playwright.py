"""Tool adapter for opening PTY sessions through a Playwright-targetable page."""

from __future__ import annotations

import json
from typing import Any, Dict

from opencas.execution.tui_playwright_bridge import (
    DEFAULT_COLS,
    DEFAULT_IDLE_SECONDS,
    DEFAULT_MAX_WAIT_SECONDS,
    DEFAULT_ROWS,
    DEFAULT_SCOPE_KEY,
    default_cwd,
    key_sequence,
    observe_snapshot,
    session_payload,
)

from ..models import ToolResult


class TuiPlaywrightToolAdapter:
    """Start a PTY session and return the browser target contract for it."""

    def __init__(self, runtime: Any) -> None:
        self.runtime = runtime

    def __call__(self, name: str, args: Dict[str, Any]) -> ToolResult:
        supervisor = getattr(self.runtime, "pty_supervisor", None)
        if supervisor is None:
            return ToolResult(False, "PTY supervisor is not available", {})

        if name == "tui_playwright_open":
            return self._open(supervisor, args)
        if name == "tui_playwright_input":
            return self._input(supervisor, args)
        if name == "tui_playwright_keys":
            return self._keys(supervisor, args)
        if name == "tui_playwright_resize":
            return self._resize(supervisor, args)
        if name == "tui_playwright_snapshot":
            return self._snapshot(supervisor, args)
        if name == "tui_playwright_close":
            return self._close(supervisor, args)
        return ToolResult(False, f"Unknown TUI Playwright tool: {name}", {})

    def _open(self, supervisor: Any, args: Dict[str, Any]) -> ToolResult:
        command = str(args.get("command", "")).strip()
        if not command:
            return ToolResult(False, "Missing required argument: command", {})

        scope_key = str(args.get("scope_key") or DEFAULT_SCOPE_KEY)
        cwd = str(args.get("cwd") or default_cwd(self.runtime))
        rows = int(args.get("rows", DEFAULT_ROWS))
        cols = int(args.get("cols", DEFAULT_COLS))
        observe = bool(args.get("observe", True))
        idle_seconds = float(args.get("idle_seconds", DEFAULT_IDLE_SECONDS))
        max_wait_seconds = float(
            args.get("max_wait_seconds", DEFAULT_MAX_WAIT_SECONDS)
        )

        try:
            session_id = supervisor.start(
                scope_key,
                command,
                cwd=cwd,
                rows=rows,
                cols=cols,
            )
            snapshot: dict[str, Any] = {}
            if observe:
                snapshot = observe_snapshot(
                    supervisor,
                    scope_key=scope_key,
                    session_id=session_id,
                    idle_seconds=idle_seconds,
                    max_wait_seconds=max_wait_seconds,
                )
        except Exception as exc:
            return ToolResult(False, str(exc), {"error_type": type(exc).__name__})

        payload = session_payload(
            session_id=session_id,
            scope_key=scope_key,
            snapshot=snapshot,
            base_url=_runtime_base_url(self.runtime),
        )
        payload["next_tools"] = [
            "browser_start",
            "browser_navigate",
            "browser_snapshot",
        ]
        return ToolResult(
            True,
            json.dumps(payload),
            {
                "session_id": session_id,
                "scope_key": scope_key,
                "command": command,
                "cwd": cwd,
                "playwright_url": payload["playwright_url"],
                "browser_url": payload.get("browser_url"),
            },
        )

    def _input(self, supervisor: Any, args: Dict[str, Any]) -> ToolResult:
        session_id = str(args.get("session_id", "")).strip()
        input_text = str(args.get("input", ""))
        if not session_id:
            return ToolResult(False, "Missing required argument: session_id", {})
        scope_key = str(args.get("scope_key") or DEFAULT_SCOPE_KEY)
        ok = supervisor.write(scope_key, session_id, input_text)
        if not ok:
            return ToolResult(False, "Failed to write TUI input", {"scope_key": scope_key})
        snapshot = _observe_if_requested(supervisor, scope_key, session_id, args)
        return _json_result(
            True,
            {
                "found": True,
                "ok": True,
                "session_id": session_id,
                "scope_key": scope_key,
                "input": input_text,
                "snapshot": snapshot,
            },
            {"scope_key": scope_key, "session_id": session_id},
        )

    def _keys(self, supervisor: Any, args: Dict[str, Any]) -> ToolResult:
        session_id = str(args.get("session_id", "")).strip()
        if not session_id:
            return ToolResult(False, "Missing required argument: session_id", {})
        keys = list(args.get("keys") or [])
        if not keys:
            return ToolResult(False, "keys is required", {})
        try:
            input_text = key_sequence(keys)
        except ValueError as exc:
            return ToolResult(False, str(exc), {"error_type": type(exc).__name__})
        scope_key = str(args.get("scope_key") or DEFAULT_SCOPE_KEY)
        ok = supervisor.write(scope_key, session_id, input_text)
        if not ok:
            return ToolResult(False, "Failed to write TUI key input", {"scope_key": scope_key})
        snapshot = _observe_if_requested(supervisor, scope_key, session_id, args)
        return _json_result(
            True,
            {
                "found": True,
                "ok": True,
                "session_id": session_id,
                "scope_key": scope_key,
                "keys": keys,
                "input": input_text,
                "snapshot": snapshot,
            },
            {"scope_key": scope_key, "session_id": session_id},
        )

    def _resize(self, supervisor: Any, args: Dict[str, Any]) -> ToolResult:
        session_id = str(args.get("session_id", "")).strip()
        if not session_id:
            return ToolResult(False, "Missing required argument: session_id", {})
        scope_key = str(args.get("scope_key") or DEFAULT_SCOPE_KEY)
        rows = int(args.get("rows", DEFAULT_ROWS))
        cols = int(args.get("cols", DEFAULT_COLS))
        ok = supervisor.resize(scope_key, session_id, rows=rows, cols=cols)
        return _json_result(
            bool(ok),
            {
                "found": bool(ok),
                "ok": bool(ok),
                "session_id": session_id,
                "scope_key": scope_key,
                "rows": rows,
                "cols": cols,
            },
            {"scope_key": scope_key, "session_id": session_id},
        )

    def _snapshot(self, supervisor: Any, args: Dict[str, Any]) -> ToolResult:
        session_id = str(args.get("session_id", "")).strip()
        if not session_id:
            return ToolResult(False, "Missing required argument: session_id", {})
        scope_key = str(args.get("scope_key") or DEFAULT_SCOPE_KEY)
        snapshot = _observe_if_requested(
            supervisor,
            scope_key,
            session_id,
            {**args, "observe": True},
        )
        payload = session_payload(
            session_id=session_id,
            scope_key=scope_key,
            snapshot=snapshot,
            base_url=_runtime_base_url(self.runtime),
        )
        return _json_result(
            bool(snapshot.get("found", True)),
            payload,
            {"scope_key": scope_key, "session_id": session_id},
        )

    def _close(self, supervisor: Any, args: Dict[str, Any]) -> ToolResult:
        session_id = str(args.get("session_id", "")).strip()
        if not session_id:
            return ToolResult(False, "Missing required argument: session_id", {})
        scope_key = str(args.get("scope_key") or DEFAULT_SCOPE_KEY)
        ok = supervisor.remove(scope_key, session_id)
        return _json_result(
            bool(ok),
            {
                "found": bool(ok),
                "ok": bool(ok),
                "session_id": session_id,
                "scope_key": scope_key,
            },
            {"scope_key": scope_key, "session_id": session_id},
        )


def _runtime_base_url(runtime: Any) -> str | None:
    value = getattr(runtime, "server_base_url", None) or getattr(
        runtime,
        "_server_base_url",
        None,
    )
    if value:
        return str(value).rstrip("/")
    return None


def _observe_if_requested(
    supervisor: Any,
    scope_key: str,
    session_id: str,
    args: Dict[str, Any],
) -> dict[str, Any]:
    if args.get("observe", True) is False:
        return {}
    return observe_snapshot(
        supervisor,
        scope_key=scope_key,
        session_id=session_id,
        idle_seconds=float(args.get("idle_seconds", DEFAULT_IDLE_SECONDS)),
        max_wait_seconds=float(args.get("max_wait_seconds", DEFAULT_MAX_WAIT_SECONDS)),
    )


def _json_result(
    success: bool,
    payload: dict[str, Any],
    metadata: dict[str, Any],
) -> ToolResult:
    return ToolResult(success, json.dumps(payload), metadata)
