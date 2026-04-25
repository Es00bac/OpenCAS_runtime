"""Web and browser tool registration for AgentRuntime."""

from __future__ import annotations

from typing import Any

from opencas.autonomy.models import ActionRiskTier
from opencas.tools.adapters.browser import BrowserToolAdapter
from opencas.tools.adapters.web import WebToolAdapter

from .tool_registration_specs import ToolRegistrationSpec, register_tool_specs


def register_web_and_browser_tools(runtime: Any) -> None:
    web = WebToolAdapter()
    register_tool_specs(
        runtime,
        web,
        [
            ToolRegistrationSpec(
                name="web_fetch",
                description="Fetch a URL and return extracted text.",
                risk_tier=ActionRiskTier.READONLY,
                schema={
                    "type": "object",
                    "properties": {
                        "url": {"type": "string", "description": "URL to fetch."},
                        "max_length": {"type": "integer", "description": "Maximum characters to return."},
                    },
                    "required": ["url"],
                },
            ),
            ToolRegistrationSpec(
                name="web_search",
                description="Search the live web for real-time information, breaking news, and recent developments. Returns result links and titles.",
                risk_tier=ActionRiskTier.READONLY,
                schema={
                    "type": "object",
                    "properties": {
                        "query": {"type": "string", "description": "Search query."},
                    },
                    "required": ["query"],
                },
            ),
        ],
    )

    browser = BrowserToolAdapter(supervisor=runtime.browser_supervisor)
    register_tool_specs(
        runtime,
        browser,
        [
            ToolRegistrationSpec(
                name="browser_start",
                description="Start a Playwright-backed browser session.",
                risk_tier=ActionRiskTier.READONLY,
                schema={
                    "type": "object",
                    "properties": {
                        "headless": {"type": "boolean", "description": "Run browser headlessly (default true)."},
                        "viewport_width": {"type": "integer", "description": "Browser viewport width."},
                        "viewport_height": {"type": "integer", "description": "Browser viewport height."},
                        "scope_key": {"type": "string", "description": "Scope key for browser session isolation."},
                    },
                    "required": [],
                },
            ),
            ToolRegistrationSpec(
                name="browser_navigate",
                description="Navigate a browser session to a URL.",
                risk_tier=ActionRiskTier.READONLY,
                schema={
                    "type": "object",
                    "properties": {
                        "session_id": {"type": "string", "description": "Browser session id from browser_start."},
                        "url": {"type": "string", "description": "URL to navigate to."},
                        "wait_until": {"type": "string", "description": "Playwright wait state: load, domcontentloaded, networkidle, or commit."},
                        "timeout_ms": {"type": "integer", "description": "Navigation timeout in milliseconds."},
                        "scope_key": {"type": "string", "description": "Scope key for browser session isolation."},
                    },
                    "required": ["session_id", "url"],
                },
            ),
            ToolRegistrationSpec(
                name="browser_click",
                description="Click an element inside the active browser page.",
                risk_tier=ActionRiskTier.EXTERNAL_WRITE,
                schema={
                    "type": "object",
                    "properties": {
                        "session_id": {"type": "string", "description": "Browser session id from browser_start."},
                        "selector": {"type": "string", "description": "Selector to click."},
                        "timeout_ms": {"type": "integer", "description": "Click timeout in milliseconds."},
                        "scope_key": {"type": "string", "description": "Scope key for browser session isolation."},
                    },
                    "required": ["session_id", "selector"],
                },
            ),
            ToolRegistrationSpec(
                name="browser_type",
                description="Type text into an input or editable element.",
                risk_tier=ActionRiskTier.EXTERNAL_WRITE,
                schema={
                    "type": "object",
                    "properties": {
                        "session_id": {"type": "string", "description": "Browser session id from browser_start."},
                        "selector": {"type": "string", "description": "Selector to type into."},
                        "text": {"type": "string", "description": "Text to type."},
                        "clear": {"type": "boolean", "description": "Clear the field before typing."},
                        "timeout_ms": {"type": "integer", "description": "Typing timeout in milliseconds."},
                        "scope_key": {"type": "string", "description": "Scope key for browser session isolation."},
                    },
                    "required": ["session_id", "selector", "text"],
                },
            ),
            ToolRegistrationSpec(
                name="browser_press",
                description="Press a keyboard key in the active browser page.",
                risk_tier=ActionRiskTier.EXTERNAL_WRITE,
                schema={
                    "type": "object",
                    "properties": {
                        "session_id": {"type": "string", "description": "Browser session id from browser_start."},
                        "key": {"type": "string", "description": "Keyboard key to press, e.g. Enter."},
                        "scope_key": {"type": "string", "description": "Scope key for browser session isolation."},
                    },
                    "required": ["session_id", "key"],
                },
            ),
            ToolRegistrationSpec(
                name="browser_wait",
                description="Wait for page readiness or a selector to appear.",
                risk_tier=ActionRiskTier.READONLY,
                schema={
                    "type": "object",
                    "properties": {
                        "session_id": {"type": "string", "description": "Browser session id from browser_start."},
                        "selector": {"type": "string", "description": "Optional selector to wait for."},
                        "load_state": {"type": "string", "description": "Playwright load state when selector is omitted."},
                        "timeout_ms": {"type": "integer", "description": "Wait timeout in milliseconds."},
                        "scope_key": {"type": "string", "description": "Scope key for browser session isolation."},
                    },
                    "required": ["session_id"],
                },
            ),
            ToolRegistrationSpec(
                name="browser_snapshot",
                description="Capture a text-and-link snapshot of the current browser page.",
                risk_tier=ActionRiskTier.READONLY,
                schema={
                    "type": "object",
                    "properties": {
                        "session_id": {"type": "string", "description": "Browser session id from browser_start."},
                        "max_text_length": {"type": "integer", "description": "Maximum text length to return."},
                        "max_links": {"type": "integer", "description": "Maximum number of links to include."},
                        "capture_screenshot": {"type": "boolean", "description": "Capture a screenshot to a temp file path."},
                        "full_page": {"type": "boolean", "description": "Capture the full page when taking a screenshot."},
                        "scope_key": {"type": "string", "description": "Scope key for browser session isolation."},
                    },
                    "required": ["session_id"],
                },
            ),
            ToolRegistrationSpec(
                name="browser_close",
                description="Close and remove a browser session.",
                risk_tier=ActionRiskTier.READONLY,
                schema={
                    "type": "object",
                    "properties": {
                        "session_id": {"type": "string", "description": "Browser session id from browser_start."},
                        "scope_key": {"type": "string", "description": "Scope key for browser session isolation."},
                    },
                    "required": ["session_id"],
                },
            ),
            ToolRegistrationSpec(
                name="browser_clear",
                description="Close and remove all browser sessions in a scope.",
                risk_tier=ActionRiskTier.READONLY,
                schema={
                    "type": "object",
                    "properties": {
                        "scope_key": {"type": "string", "description": "Scope key for browser session isolation."},
                    },
                    "required": [],
                },
            ),
        ],
    )
