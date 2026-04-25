"""Tests for the Playwright-backed browser supervisor and tool adapter."""

import json
from pathlib import Path
from urllib.parse import quote

import pytest

from opencas.execution.browser_supervisor import BrowserSupervisor
from opencas.tools.adapters.browser import BrowserToolAdapter


def _data_url(html: str) -> str:
    return "data:text/html," + quote(html)


@pytest.mark.asyncio
async def test_browser_supervisor_navigate_and_snapshot() -> None:
    supervisor = BrowserSupervisor()
    session_id = await supervisor.start("test")
    try:
        html = """
        <html>
          <head><title>Browser Test</title></head>
          <body>
            <h1>Hello Browser</h1>
            <a href="https://example.com">Example</a>
          </body>
        </html>
        """
        result = await supervisor.navigate("test", session_id, _data_url(html))
        assert result["found"] is True
        assert result["title"] == "Browser Test"
        snapshot = await supervisor.snapshot_page("test", session_id)
        assert "Hello Browser" in snapshot["text"]
        assert snapshot["links"][0]["href"] == "https://example.com"
        runtime = supervisor.snapshot()
        assert runtime["available"] is True
        assert runtime["total_count"] == 1
    finally:
        await supervisor.close("test", session_id)
        await supervisor.shutdown()


@pytest.mark.asyncio
async def test_browser_supervisor_persists_and_cleans_screenshot_metadata() -> None:
    supervisor = BrowserSupervisor()
    session_id = await supervisor.start("test")
    screenshot_path: Path | None = None
    try:
        html = """
        <html>
          <head><title>Browser Shot</title></head>
          <body>
            <h1>Screenshot Body</h1>
          </body>
        </html>
        """
        await supervisor.navigate("test", session_id, _data_url(html))
        snapshot = await supervisor.snapshot_page("test", session_id, capture_screenshot=True)
        screenshot_path = Path(snapshot["screenshot_path"])
        assert screenshot_path.exists()
        runtime = supervisor.snapshot()
        entry = runtime["entries"][0]
        assert entry["title"] == "Browser Shot"
        assert "Screenshot Body" in entry["last_snapshot_text"]
        assert entry["last_snapshot_screenshot"] == str(screenshot_path)
    finally:
        await supervisor.close("test", session_id)
        if screenshot_path is not None:
            assert not screenshot_path.exists()
        await supervisor.shutdown()


@pytest.mark.asyncio
async def test_browser_supervisor_snapshot_filters_by_scope() -> None:
    supervisor = BrowserSupervisor()
    test_session = await supervisor.start("test")
    other_session = await supervisor.start("other")
    try:
        await supervisor.navigate("test", test_session, _data_url("<html><head><title>Test Scope</title></head><body>test</body></html>"))
        await supervisor.navigate("other", other_session, _data_url("<html><head><title>Other Scope</title></head><body>other</body></html>"))
        test_snapshot = supervisor.snapshot(scope_key="test")
        assert test_snapshot["total_count"] == 1
        assert test_snapshot["entries"][0]["session_id"] == test_session
        assert test_snapshot["entries"][0]["scope_key"] == "test"
        empty_snapshot = supervisor.snapshot(scope_key="missing")
        assert empty_snapshot["total_count"] == 0
        assert empty_snapshot["entries"] == []
    finally:
        await supervisor.close("test", test_session)
        await supervisor.close("other", other_session)
        await supervisor.shutdown()


@pytest.mark.asyncio
async def test_browser_tool_adapter_interaction() -> None:
    supervisor = BrowserSupervisor()
    adapter = BrowserToolAdapter(supervisor)
    started = await adapter("browser_start", {})
    assert started.success is True
    session_id = json.loads(started.output)["session_id"]
    try:
        html = """
        <html>
          <head><title>Interactive Browser</title></head>
          <body>
            <input id="name" />
            <button id="go" onclick="document.getElementById('result').textContent = document.getElementById('name').value;">Go</button>
            <div id="result"></div>
          </body>
        </html>
        """
        navigated = await adapter(
            "browser_navigate",
            {"session_id": session_id, "url": _data_url(html)},
        )
        assert navigated.success is True
        typed = await adapter(
            "browser_type",
            {"session_id": session_id, "selector": "#name", "text": "OpenCAS"},
        )
        assert typed.success is True
        clicked = await adapter(
            "browser_click",
            {"session_id": session_id, "selector": "#go"},
        )
        assert clicked.success is True
        snapshot = await adapter("browser_snapshot", {"session_id": session_id})
        payload = json.loads(snapshot.output)
        assert "OpenCAS" in payload["text"]
    finally:
        await adapter("browser_close", {"session_id": session_id})
        await supervisor.shutdown()
