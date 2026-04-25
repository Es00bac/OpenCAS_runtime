"""Browser tool adapter for Playwright-backed browser sessions."""

from __future__ import annotations

import json
from typing import Any, Dict

from ...execution.browser_supervisor import BrowserSupervisor
from ..models import ToolResult


class BrowserToolAdapter:
    """Adapter for managing interactive browser sessions."""

    def __init__(self, supervisor: BrowserSupervisor) -> None:
        self.supervisor = supervisor

    async def __call__(self, name: str, args: Dict[str, Any]) -> ToolResult:
        scope_key = str(args.get("scope_key", "default"))

        if name == "browser_start":
            headless = bool(args.get("headless", True))
            viewport_width = int(args.get("viewport_width", 1280))
            viewport_height = int(args.get("viewport_height", 900))
            try:
                session_id = await self.supervisor.start(
                    scope_key=scope_key,
                    headless=headless,
                    viewport_width=viewport_width,
                    viewport_height=viewport_height,
                )
                return ToolResult(
                    success=True,
                    output=json.dumps({"session_id": session_id, "status": "started"}),
                    metadata={
                        "scope_key": scope_key,
                        "headless": headless,
                        "viewport_width": viewport_width,
                        "viewport_height": viewport_height,
                    },
                )
            except Exception as exc:
                return ToolResult(False, str(exc), {"error_type": type(exc).__name__})

        if name == "browser_navigate":
            session_id = str(args.get("session_id", ""))
            url = str(args.get("url", ""))
            wait_until = str(args.get("wait_until", "load"))
            timeout_ms = int(args.get("timeout_ms", 30000))
            if not session_id or not url:
                return ToolResult(False, "session_id and url are required", {})
            result = await self.supervisor.navigate(
                scope_key=scope_key,
                session_id=session_id,
                url=url,
                wait_until=wait_until,
                timeout_ms=timeout_ms,
            )
            return ToolResult(
                success=result.get("found", False),
                output=json.dumps(result),
                metadata={"scope_key": scope_key},
            )

        if name == "browser_click":
            session_id = str(args.get("session_id", ""))
            selector = str(args.get("selector", ""))
            timeout_ms = int(args.get("timeout_ms", 5000))
            if not session_id or not selector:
                return ToolResult(False, "session_id and selector are required", {})
            result = await self.supervisor.click(
                scope_key=scope_key,
                session_id=session_id,
                selector=selector,
                timeout_ms=timeout_ms,
            )
            return ToolResult(
                success=result.get("found", False),
                output=json.dumps(result),
                metadata={"scope_key": scope_key},
            )

        if name == "browser_type":
            session_id = str(args.get("session_id", ""))
            selector = str(args.get("selector", ""))
            text = str(args.get("text", ""))
            clear = bool(args.get("clear", True))
            timeout_ms = int(args.get("timeout_ms", 5000))
            if not session_id or not selector:
                return ToolResult(False, "session_id and selector are required", {})
            result = await self.supervisor.type_text(
                scope_key=scope_key,
                session_id=session_id,
                selector=selector,
                text=text,
                clear=clear,
                timeout_ms=timeout_ms,
            )
            return ToolResult(
                success=result.get("found", False),
                output=json.dumps(result),
                metadata={"scope_key": scope_key},
            )

        if name == "browser_press":
            session_id = str(args.get("session_id", ""))
            key = str(args.get("key", ""))
            if not session_id or not key:
                return ToolResult(False, "session_id and key are required", {})
            result = await self.supervisor.press(
                scope_key=scope_key,
                session_id=session_id,
                key=key,
            )
            return ToolResult(
                success=result.get("found", False),
                output=json.dumps(result),
                metadata={"scope_key": scope_key},
            )

        if name == "browser_wait":
            session_id = str(args.get("session_id", ""))
            timeout_ms = int(args.get("timeout_ms", 5000))
            load_state = str(args.get("load_state", "load"))
            selector = args.get("selector")
            if not session_id:
                return ToolResult(False, "session_id is required", {})
            result = await self.supervisor.wait(
                scope_key=scope_key,
                session_id=session_id,
                timeout_ms=timeout_ms,
                load_state=load_state,
                selector=str(selector) if selector is not None else None,
            )
            return ToolResult(
                success=result.get("found", False),
                output=json.dumps(result),
                metadata={"scope_key": scope_key},
            )

        if name == "browser_snapshot":
            session_id = str(args.get("session_id", ""))
            max_text_length = int(args.get("max_text_length", 4000))
            max_links = int(args.get("max_links", 20))
            capture_screenshot = bool(args.get("capture_screenshot", False))
            full_page = bool(args.get("full_page", False))
            if not session_id:
                return ToolResult(False, "session_id is required", {})
            result = await self.supervisor.snapshot_page(
                scope_key=scope_key,
                session_id=session_id,
                max_text_length=max_text_length,
                max_links=max_links,
                capture_screenshot=capture_screenshot,
                full_page=full_page,
            )
            return ToolResult(
                success=result.get("found", False),
                output=json.dumps(result),
                metadata={"scope_key": scope_key},
            )

        if name == "browser_close":
            session_id = str(args.get("session_id", ""))
            if not session_id:
                return ToolResult(False, "session_id is required", {})
            ok = await self.supervisor.close(scope_key=scope_key, session_id=session_id)
            return ToolResult(True, json.dumps({"ok": ok}), {"scope_key": scope_key})

        if name == "browser_clear":
            removed = await self.supervisor.clear(scope_key=scope_key)
            return ToolResult(
                True,
                json.dumps({"removed": removed}),
                {"scope_key": scope_key},
            )

        return ToolResult(False, f"Unknown browser tool: {name}", {})
