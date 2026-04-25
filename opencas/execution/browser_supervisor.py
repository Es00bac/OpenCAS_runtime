"""Playwright-backed supervisor for real browser operator sessions."""

from __future__ import annotations

import os
import tempfile
import threading
import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional
from uuid import uuid4

try:
    from playwright.async_api import (
        Browser,
        BrowserContext,
        Page,
        Playwright,
        async_playwright,
    )
except ImportError:  # pragma: no cover - exercised through availability snapshot
    Browser = BrowserContext = Page = Playwright = Any  # type: ignore[assignment]
    async_playwright = None


@dataclass
class _ManagedBrowserSession:
    browser: Browser
    context: BrowserContext
    page: Page
    scope_key: str
    created_at: float = field(default_factory=time.time)
    headless: bool = True
    viewport_width: int = 1280
    viewport_height: int = 900
    last_observed_at: Optional[float] = None
    last_title: Optional[str] = None
    last_snapshot_text: Optional[str] = None
    last_snapshot_links: List[Dict[str, str]] = field(default_factory=list)
    last_snapshot_screenshot: Optional[str] = None


class BrowserSupervisor:
    """Manages interactive browser sessions backed by Playwright."""

    def __init__(self) -> None:
        self._sessions: Dict[str, _ManagedBrowserSession] = {}
        self._lock = threading.Lock()
        self._playwright_manager: Optional[Any] = None
        self._playwright: Optional[Playwright] = None
        self._available = async_playwright is not None

    @staticmethod
    def _cleanup_screenshot_file(path: Optional[str]) -> None:
        if not path:
            return
        try:
            os.remove(path)
        except FileNotFoundError:
            return

    @staticmethod
    def _truncate_preview(text: Optional[str], limit: int = 280) -> Optional[str]:
        if not text:
            return text
        if len(text) <= limit:
            return text
        return text[:limit] + "\n[truncated]"

    async def start(
        self,
        scope_key: str,
        headless: bool = True,
        viewport_width: int = 1280,
        viewport_height: int = 900,
    ) -> str:
        """Start a new browser session and return its session id."""
        playwright = await self._ensure_playwright()
        browser = await playwright.chromium.launch(headless=headless)
        context = await browser.new_context(
            viewport={"width": viewport_width, "height": viewport_height}
        )
        page = await context.new_page()
        session_id = str(uuid4())
        session = _ManagedBrowserSession(
            browser=browser,
            context=context,
            page=page,
            scope_key=scope_key,
            headless=headless,
            viewport_width=viewport_width,
            viewport_height=viewport_height,
        )
        with self._lock:
            self._sessions[session_id] = session
        return session_id

    async def navigate(
        self,
        scope_key: str,
        session_id: str,
        url: str,
        wait_until: str = "load",
        timeout_ms: int = 30000,
    ) -> Dict[str, Any]:
        """Navigate a browser session to a URL."""
        session = self._get_session(scope_key, session_id)
        if session is None:
            return {"found": False, "error": "Browser session not found"}
        response = await session.page.goto(url, wait_until=wait_until, timeout=timeout_ms)
        return {
            "found": True,
            "url": session.page.url,
            "title": await session.page.title(),
            "status": response.status if response is not None else None,
        }

    async def click(
        self,
        scope_key: str,
        session_id: str,
        selector: str,
        timeout_ms: int = 5000,
    ) -> Dict[str, Any]:
        """Click an element in the page."""
        session = self._get_session(scope_key, session_id)
        if session is None:
            return {"found": False, "error": "Browser session not found"}
        await session.page.locator(selector).click(timeout=timeout_ms)
        return {"found": True, "url": session.page.url, "title": await session.page.title()}

    async def type_text(
        self,
        scope_key: str,
        session_id: str,
        selector: str,
        text: str,
        clear: bool = True,
        timeout_ms: int = 5000,
    ) -> Dict[str, Any]:
        """Type into an input or editable element."""
        session = self._get_session(scope_key, session_id)
        if session is None:
            return {"found": False, "error": "Browser session not found"}
        locator = session.page.locator(selector)
        if clear:
            await locator.fill("", timeout=timeout_ms)
        await locator.type(text, timeout=timeout_ms)
        return {"found": True, "url": session.page.url, "title": await session.page.title()}

    async def press(
        self,
        scope_key: str,
        session_id: str,
        key: str,
    ) -> Dict[str, Any]:
        """Press a keyboard key in the active page."""
        session = self._get_session(scope_key, session_id)
        if session is None:
            return {"found": False, "error": "Browser session not found"}
        await session.page.keyboard.press(key)
        return {"found": True, "url": session.page.url, "title": await session.page.title()}

    async def wait(
        self,
        scope_key: str,
        session_id: str,
        timeout_ms: int = 5000,
        load_state: str = "load",
        selector: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Wait for page load or selector visibility."""
        session = self._get_session(scope_key, session_id)
        if session is None:
            return {"found": False, "error": "Browser session not found"}
        if selector:
            await session.page.locator(selector).wait_for(timeout=timeout_ms)
        else:
            await session.page.wait_for_load_state(load_state, timeout=timeout_ms)
        return {"found": True, "url": session.page.url, "title": await session.page.title()}

    async def snapshot_page(
        self,
        scope_key: str,
        session_id: str,
        max_text_length: int = 4000,
        max_links: int = 20,
        capture_screenshot: bool = False,
        full_page: bool = False,
    ) -> Dict[str, Any]:
        """Return a structured snapshot of the current page."""
        session = self._get_session(scope_key, session_id)
        if session is None:
            return {"found": False, "error": "Browser session not found"}
        body_text = (await session.page.locator("body").inner_text()).strip()
        if len(body_text) > max_text_length:
            body_text = body_text[:max_text_length] + "\n[truncated]"
        anchors = await session.page.query_selector_all("a")
        links: List[Dict[str, str]] = []
        for anchor in anchors[:max_links]:
            text = (await anchor.inner_text()).strip()
            href = await anchor.get_attribute("href") or ""
            if text or href:
                links.append({"text": text, "href": href})
        snapshot = {
            "found": True,
            "url": session.page.url,
            "title": await session.page.title(),
            "text": body_text,
            "links": links,
        }
        if capture_screenshot:
            self._cleanup_screenshot_file(session.last_snapshot_screenshot)
            with tempfile.NamedTemporaryFile(
                prefix="opencas-browser-",
                suffix=".png",
                delete=False,
            ) as tmp:
                screenshot_path = tmp.name
            await session.page.screenshot(path=screenshot_path, full_page=full_page)
            snapshot["screenshot_path"] = screenshot_path
            session.last_snapshot_screenshot = screenshot_path
        session.last_title = snapshot["title"]
        session.last_snapshot_text = snapshot["text"]
        session.last_snapshot_links = list(snapshot["links"])
        session.last_observed_at = time.time()
        return snapshot

    async def close(self, scope_key: str, session_id: str) -> bool:
        """Close and remove a browser session."""
        with self._lock:
            session = self._sessions.get(session_id)
        if session is None or session.scope_key != scope_key:
            return False
        self._cleanup_screenshot_file(session.last_snapshot_screenshot)
        await session.context.close()
        await session.browser.close()
        with self._lock:
            self._sessions.pop(session_id, None)
        return True

    async def clear(self, scope_key: str) -> int:
        """Close and remove all browser sessions in a scope."""
        with self._lock:
            session_ids = [
                session_id
                for session_id, session in self._sessions.items()
                if session.scope_key == scope_key
            ]
        removed = 0
        for session_id in session_ids:
            if await self.close(scope_key, session_id):
                removed += 1
        return removed

    async def clear_all(self) -> int:
        """Close and remove all tracked browser sessions without stopping Playwright."""
        with self._lock:
            session_items = [
                (session_id, session.scope_key)
                for session_id, session in self._sessions.items()
            ]
        removed = 0
        for session_id, scope_key in session_items:
            if await self.close(scope_key, session_id):
                removed += 1
        return removed

    async def shutdown(self) -> None:
        """Close all sessions and stop Playwright if it was started."""
        with self._lock:
            session_items = [
                (session_id, session.scope_key)
                for session_id, session in self._sessions.items()
            ]
        for session_id, scope_key in session_items:
            await self.close(scope_key, session_id)
        if self._playwright is not None:
            await self._playwright.stop()
            self._playwright = None
        self._playwright_manager = None

    def snapshot(
        self,
        scope_key: Optional[str] = None,
        sample_limit: int = 10,
    ) -> Dict[str, Any]:
        """Return a monitoring snapshot of browser availability and sessions."""
        with self._lock:
            items = list(self._sessions.items())

        filtered_items = []
        for session_id, session in items:
            if scope_key is not None and session.scope_key != scope_key:
                continue
            filtered_items.append((session_id, session))

        filtered_items.sort(key=lambda item: item[1].created_at, reverse=True)
        entries = []
        scopes = set()
        for session_id, session in filtered_items[:sample_limit]:
            scopes.add(session.scope_key)
            entries.append(
                {
                    "session_id": session_id,
                    "scope_key": session.scope_key,
                    "url": session.page.url,
                    "title": session.last_title,
                    "headless": session.headless,
                    "viewport": f"{session.viewport_width}x{session.viewport_height}",
                    "created_at": session.created_at,
                    "last_observed_at": session.last_observed_at,
                    "last_snapshot_text": self._truncate_preview(session.last_snapshot_text),
                    "last_snapshot_links": list(session.last_snapshot_links),
                    "last_snapshot_screenshot": session.last_snapshot_screenshot,
                }
            )
        for _, session in filtered_items[sample_limit:]:
            scopes.add(session.scope_key)
        return {
            "available": self._available,
            "total_count": len(filtered_items),
            "scope_count": len(scopes),
            "entries": entries,
        }

    def describe_session(
        self,
        scope_key: str,
        session_id: str,
    ) -> Optional[Dict[str, Any]]:
        """Return lightweight session metadata for policy/approval checks."""
        session = self._get_session(scope_key, session_id)
        if session is None:
            return None
        return {
            "session_id": session_id,
            "scope_key": scope_key,
            "url": session.page.url,
            "title": session.last_title,
        }

    async def _ensure_playwright(self) -> Playwright:
        if async_playwright is None:
            raise RuntimeError("playwright is not installed")
        if self._playwright is None:
            self._playwright_manager = async_playwright()
            self._playwright = await self._playwright_manager.start()
        return self._playwright

    def _get_session(
        self,
        scope_key: str,
        session_id: str,
    ) -> Optional[_ManagedBrowserSession]:
        with self._lock:
            session = self._sessions.get(session_id)
        if session is None or session.scope_key != scope_key:
            return None
        return session
