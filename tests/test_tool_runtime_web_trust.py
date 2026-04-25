"""Focused tests for web-trust payload enrichment in runtime tool requests."""

from __future__ import annotations

from types import SimpleNamespace

from opencas.runtime.tool_runtime import _build_tool_request_payload


def test_build_tool_request_payload_marks_web_fetch_domain() -> None:
    runtime = SimpleNamespace(
        ctx=SimpleNamespace(config=SimpleNamespace()),
        browser_supervisor=SimpleNamespace(),
    )

    payload = _build_tool_request_payload(
        runtime,
        "web_fetch",
        {"url": "https://docs.python.org/3/library/pathlib.html"},
    )

    assert payload["web_action_class"] == "fetch"
    assert payload["web_domain"] == "docs.python.org"
    assert payload["web_url"] == "https://docs.python.org/3/library/pathlib.html"


def test_build_tool_request_payload_uses_browser_session_domain_for_click() -> None:
    runtime = SimpleNamespace(
        ctx=SimpleNamespace(config=SimpleNamespace()),
        browser_supervisor=SimpleNamespace(
            describe_session=lambda scope_key, session_id: {
                "url": "https://platform.openai.com/docs/overview",
                "title": "Docs",
            }
        ),
    )

    payload = _build_tool_request_payload(
        runtime,
        "browser_click",
        {"session_id": "session-1", "scope_key": "default", "selector": "a"},
    )

    assert payload["web_action_class"] == "interact"
    assert payload["web_domain"] == "platform.openai.com"
    assert payload["web_url"] == "https://platform.openai.com/docs/overview"
