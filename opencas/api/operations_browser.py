"""Shared browser-session helpers for operations routes."""

from __future__ import annotations

import time
from pathlib import Path
from typing import Any, Awaitable, Callable, Dict, List, Optional

from fastapi import APIRouter
from fastapi.responses import FileResponse
from opencas.api.operations_models import (
    BrowserCaptureRequest,
    BrowserClickRequest,
    BrowserNavigateRequest,
    BrowserPressRequest,
    BrowserTypeRequest,
    BrowserWaitRequest,
)


class BrowserSessionService:
    """Centralize browser session lookup, refresh, and operator-action wiring."""

    def __init__(
        self,
        runtime: Any,
        *,
        append_operator_action: Callable[[Any, Dict[str, Any]], Dict[str, Any]],
        load_recent_operator_actions: Callable[..., List[Dict[str, Any]]],
    ) -> None:
        self.runtime = runtime
        self._append_operator_action = append_operator_action
        self._load_recent_operator_actions = load_recent_operator_actions

    @property
    def available(self) -> bool:
        return hasattr(self.runtime, "browser_supervisor")

    def missing_response(self) -> Dict[str, Any]:
        return {"found": False, "error": "Browser supervisor not available"}

    def find_session_entry(self, *, scope_key: str, session_id: str) -> Optional[Dict[str, Any]]:
        snapshot = self.runtime.browser_supervisor.snapshot(scope_key=scope_key)
        return next((item for item in snapshot.get("entries", []) if item.get("session_id") == session_id), None)

    def merge_observation(
        self,
        entry: Dict[str, Any],
        observed: Optional[Dict[str, Any]],
        *,
        url_hint: Optional[str] = None,
    ) -> Dict[str, Any]:
        if not observed or not observed.get("found", False):
            return entry
        return {
            **entry,
            "url": url_hint or observed.get("url", entry.get("url")),
            "title": observed.get("title"),
            "last_snapshot_text": observed.get("text"),
            "last_snapshot_links": observed.get("links", []),
            "last_snapshot_screenshot": observed.get("screenshot_path"),
            "last_observed_at": time.time(),
        }

    def load_recent_actions(self, *, scope_key: str, session_id: str) -> List[Dict[str, Any]]:
        return self._load_recent_operator_actions(
            self.runtime,
            target_kind="browser",
            target_id=session_id,
            scope_key=scope_key,
        )

    async def get_session(
        self,
        *,
        scope_key: str,
        session_id: str,
        refresh: bool = False,
        capture_screenshot: bool = False,
        full_page: bool = False,
    ) -> Dict[str, Any]:
        if not self.available:
            return self.missing_response()

        entry = self.find_session_entry(scope_key=scope_key, session_id=session_id)
        if entry is None:
            return {"found": False}

        observed = None
        if refresh:
            observed = await self.runtime.browser_supervisor.snapshot_page(
                scope_key=scope_key,
                session_id=session_id,
                capture_screenshot=capture_screenshot,
                full_page=full_page,
            )
            if not observed.get("found", False):
                return {"found": False, "error": observed.get("error", "Browser session not found")}
            entry = self.merge_observation(entry, observed)

        return {
            "found": True,
            "session": entry,
            "observed": observed,
            "recent_operator_actions": self.load_recent_actions(scope_key=scope_key, session_id=session_id),
        }

    async def perform_action(
        self,
        *,
        scope_key: str,
        session_id: str,
        action: Callable[[], Awaitable[Dict[str, Any]]],
        response_key: str,
        operator_action: Dict[str, Any],
        refresh: bool = True,
        refresh_kwargs: Optional[Dict[str, Any]] = None,
        response_value: Optional[Callable[[Dict[str, Any], Optional[Dict[str, Any]]], Dict[str, Any]]] = None,
        url_hint: Optional[Callable[[Dict[str, Any], Optional[Dict[str, Any]]], Optional[str]]] = None,
        merge_result_as_observation: bool = False,
    ) -> Dict[str, Any]:
        if not self.available:
            return self.missing_response()

        entry = self.find_session_entry(scope_key=scope_key, session_id=session_id)
        if entry is None:
            return {"found": False}

        result = await action()
        if not result.get("found", False):
            return {"found": False, "error": result.get("error", "Browser session not found")}

        observed = result if merge_result_as_observation else None
        if refresh:
            observed = await self.runtime.browser_supervisor.snapshot_page(
                scope_key=scope_key,
                session_id=session_id,
                **(refresh_kwargs or {}),
            )

        refreshed_entry = self.find_session_entry(scope_key=scope_key, session_id=session_id) or entry
        updated_entry = self.merge_observation(
            refreshed_entry,
            observed,
            url_hint=url_hint(result, observed) if url_hint else None,
        )
        self._append_operator_action(self.runtime, operator_action)
        payload = response_value(result, observed) if response_value else result
        return {
            "found": True,
            "session": updated_entry,
            "observed": observed,
            response_key: payload,
            "recent_operator_actions": self.load_recent_actions(scope_key=scope_key, session_id=session_id),
        }

def register_browser_routes(
    router: APIRouter,
    *,
    runtime: Any,
    browser_sessions: BrowserSessionService,
    append_operator_action: Callable[[Any, Dict[str, Any]], Dict[str, Any]],
    truncate_text: Callable[[Optional[str], int], str],
) -> None:
    """Attach browser session routes to the operations router."""

    @router.get("/sessions/browser/{session_id}")
    async def get_browser_session(
        session_id: str,
        scope_key: str = "default",
        refresh: bool = False,
        capture_screenshot: bool = False,
        full_page: bool = False,
    ) -> Dict[str, Any]:
        return await browser_sessions.get_session(
            scope_key=scope_key,
            session_id=session_id,
            refresh=refresh,
            capture_screenshot=capture_screenshot,
            full_page=full_page,
        )

    @router.delete("/sessions/browser/{session_id}")
    async def close_browser_session(
        session_id: str,
        scope_key: str = "default",
    ) -> Dict[str, Any]:
        if not hasattr(runtime, "browser_supervisor"):
            return {"ok": False, "error": "Browser supervisor not available"}
        ok = await runtime.browser_supervisor.close(scope_key=scope_key, session_id=session_id)
        append_operator_action(
            runtime,
            {
                "action": "close_browser",
                "target_kind": "browser",
                "target_id": session_id,
                "scope_key": scope_key,
                "ok": bool(ok),
            },
        )
        return {"ok": ok, "session_id": session_id}

    @router.delete("/sessions/browser")
    async def clear_browser_sessions(scope_key: str = "default") -> Dict[str, Any]:
        if not hasattr(runtime, "browser_supervisor"):
            return {"ok": False, "error": "Browser supervisor not available"}
        removed = await runtime.browser_supervisor.clear(scope_key=scope_key)
        return {"ok": True, "removed": removed, "scope_key": scope_key}

    @router.post("/sessions/browser/{session_id}/navigate")
    async def navigate_browser_session(
        session_id: str,
        payload: BrowserNavigateRequest,
        scope_key: str = "default",
    ) -> Dict[str, Any]:
        return await browser_sessions.perform_action(
            scope_key=scope_key,
            session_id=session_id,
            action=lambda: runtime.browser_supervisor.navigate(
                scope_key=scope_key,
                session_id=session_id,
                url=payload.url,
                wait_until=payload.wait_until,
                timeout_ms=payload.timeout_ms,
            ),
            response_key="navigate",
            operator_action={
                "action": "browser_navigate",
                "target_kind": "browser",
                "target_id": session_id,
                "scope_key": scope_key,
                "ok": True,
                "url": payload.url,
                "wait_until": payload.wait_until,
                "timeout_ms": payload.timeout_ms,
                "source_trace": {
                    "action": "browser_navigate",
                    "url": payload.url,
                    "wait_until": payload.wait_until,
                    "timeout_ms": payload.timeout_ms,
                },
            },
            refresh=payload.refresh,
            url_hint=lambda result, _observed: result.get("url"),
        )

    @router.post("/sessions/browser/{session_id}/click")
    async def click_browser_session(
        session_id: str,
        payload: BrowserClickRequest,
        scope_key: str = "default",
    ) -> Dict[str, Any]:
        return await browser_sessions.perform_action(
            scope_key=scope_key,
            session_id=session_id,
            action=lambda: runtime.browser_supervisor.click(
                scope_key=scope_key,
                session_id=session_id,
                selector=payload.selector,
                timeout_ms=payload.timeout_ms,
            ),
            response_key="click",
            operator_action={
                "action": "browser_click",
                "target_kind": "browser",
                "target_id": session_id,
                "scope_key": scope_key,
                "ok": True,
                "selector": payload.selector,
                "timeout_ms": payload.timeout_ms,
            },
            refresh=payload.refresh,
            url_hint=lambda result, _observed: result.get("url"),
        )

    @router.post("/sessions/browser/{session_id}/type")
    async def type_browser_session(
        session_id: str,
        payload: BrowserTypeRequest,
        scope_key: str = "default",
    ) -> Dict[str, Any]:
        return await browser_sessions.perform_action(
            scope_key=scope_key,
            session_id=session_id,
            action=lambda: runtime.browser_supervisor.type_text(
                scope_key=scope_key,
                session_id=session_id,
                selector=payload.selector,
                text=payload.text,
                clear=payload.clear,
                timeout_ms=payload.timeout_ms,
            ),
            response_key="type",
            operator_action={
                "action": "browser_type",
                "target_kind": "browser",
                "target_id": session_id,
                "scope_key": scope_key,
                "ok": True,
                "selector": payload.selector,
                "text_length": len(payload.text or ""),
                "text_preview": truncate_text(payload.text),
                "clear": bool(payload.clear),
                "timeout_ms": payload.timeout_ms,
            },
            refresh=payload.refresh,
            url_hint=lambda result, _observed: result.get("url"),
        )

    @router.post("/sessions/browser/{session_id}/press")
    async def press_browser_session(
        session_id: str,
        payload: BrowserPressRequest,
        scope_key: str = "default",
    ) -> Dict[str, Any]:
        return await browser_sessions.perform_action(
            scope_key=scope_key,
            session_id=session_id,
            action=lambda: runtime.browser_supervisor.press(
                scope_key=scope_key,
                session_id=session_id,
                key=payload.key,
            ),
            response_key="press",
            operator_action={
                "action": "browser_press",
                "target_kind": "browser",
                "target_id": session_id,
                "scope_key": scope_key,
                "ok": True,
                "key": payload.key,
            },
            refresh=payload.refresh,
            url_hint=lambda result, _observed: result.get("url"),
        )

    @router.post("/sessions/browser/{session_id}/wait")
    async def wait_browser_session(
        session_id: str,
        payload: BrowserWaitRequest,
        scope_key: str = "default",
    ) -> Dict[str, Any]:
        return await browser_sessions.perform_action(
            scope_key=scope_key,
            session_id=session_id,
            action=lambda: runtime.browser_supervisor.wait(
                scope_key=scope_key,
                session_id=session_id,
                timeout_ms=payload.timeout_ms,
                load_state=payload.load_state,
                selector=payload.selector,
            ),
            response_key="wait",
            operator_action={
                "action": "browser_wait",
                "target_kind": "browser",
                "target_id": session_id,
                "scope_key": scope_key,
                "ok": True,
                "selector": payload.selector,
                "load_state": payload.load_state,
                "timeout_ms": payload.timeout_ms,
            },
            refresh=payload.refresh,
            url_hint=lambda result, _observed: result.get("url"),
        )

    @router.post("/sessions/browser/{session_id}/capture")
    async def capture_browser_session(
        session_id: str,
        payload: BrowserCaptureRequest,
        scope_key: str = "default",
    ) -> Dict[str, Any]:
        return await browser_sessions.perform_action(
            scope_key=scope_key,
            session_id=session_id,
            action=lambda: runtime.browser_supervisor.snapshot_page(
                scope_key=scope_key,
                session_id=session_id,
                capture_screenshot=True,
                full_page=payload.full_page,
            ),
            response_key="capture",
            operator_action={
                "action": "browser_capture",
                "target_kind": "browser",
                "target_id": session_id,
                "scope_key": scope_key,
                "ok": True,
                "full_page": bool(payload.full_page),
            },
            response_value=lambda result, _observed: {
                "screenshot_path": result.get("screenshot_path"),
                "full_page": payload.full_page,
            },
            url_hint=lambda result, _observed: result.get("url"),
            refresh=False,
            merge_result_as_observation=True,
        )

    @router.get("/sessions/browser/{session_id}/screenshot")
    async def get_browser_session_screenshot(
        session_id: str,
        scope_key: str = "default",
    ) -> Any:
        if not browser_sessions.available:
            return browser_sessions.missing_response()

        entry = browser_sessions.find_session_entry(scope_key=scope_key, session_id=session_id)
        if entry is None:
            return {"found": False}

        screenshot_path = entry.get("last_snapshot_screenshot")
        if not screenshot_path:
            return {"found": False, "error": "No captured browser screenshot for this session"}
        path = Path(screenshot_path)
        if not path.exists():
            return {"found": False, "error": "Captured browser screenshot is missing"}
        return FileResponse(path, media_type="image/png", filename=path.name)
