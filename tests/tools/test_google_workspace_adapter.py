from __future__ import annotations

import json

import pytest

from opencas.tools.adapters.google_workspace import GoogleWorkspaceToolAdapter
from opencas.tools.models import ToolResult


@pytest.mark.asyncio
async def test_google_workspace_auth_status_reports_missing_command() -> None:
    adapter = GoogleWorkspaceToolAdapter(command="/definitely/missing/gws")

    result = await adapter("google_workspace_auth_status", {})

    assert result.success is False
    assert "not available" in result.output
    assert result.metadata["missing_command"] is True


@pytest.mark.asyncio
async def test_google_workspace_readonly_api_blocks_non_allowlisted_calls() -> None:
    adapter = GoogleWorkspaceToolAdapter(command="gws")

    result = await adapter(
        "google_workspace_readonly_api",
        {
            "service": "gmail",
            "resource": "users",
            "sub_resource": "messages",
            "method": "send",
            "params": {"userId": "me"},
        },
    )

    assert result.success is False
    assert "Unsupported Google Workspace readonly operation" in result.output
    assert result.metadata["method"] == "send"


@pytest.mark.asyncio
async def test_google_workspace_gmail_headlines_collects_headers(monkeypatch: pytest.MonkeyPatch) -> None:
    adapter = GoogleWorkspaceToolAdapter(command="gws")

    async def fake_run(argv: list[str], *, timeout: int) -> ToolResult:
        if argv[:4] == ["gmail", "users", "messages", "list"]:
            return ToolResult(
                True,
                json.dumps(
                    {
                        "messages": [{"id": "m-1"}, {"id": "m-2"}],
                        "resultSizeEstimate": 2,
                    }
                ),
                {"argv": argv, "timeout": timeout},
            )
        message_id = json.loads(argv[5])["id"]
        return ToolResult(
            True,
            json.dumps(
                {
                    "threadId": f"thread-{message_id}",
                    "labelIds": ["INBOX"],
                    "payload": {
                        "headers": [
                            {"name": "Date", "value": "Thu, 16 Apr 2026 12:00:00 -0600"},
                            {"name": "From", "value": f"Sender {message_id} <sender@example.com>"},
                            {"name": "Subject", "value": f"Subject {message_id}"},
                        ]
                    },
                }
            ),
            {"argv": argv, "timeout": timeout},
        )

    monkeypatch.setattr(adapter, "_run_json_command", fake_run)

    result = await adapter(
        "google_workspace_gmail_headlines",
        {"query": "label:inbox", "max_results": 2},
    )

    assert result.success is True
    payload = json.loads(result.output)
    assert payload["count"] == 2
    assert payload["messages"][0]["subject"] == "Subject m-1"
    assert payload["messages"][1]["from"] == "Sender m-2 <sender@example.com>"


@pytest.mark.asyncio
async def test_google_workspace_calendar_schedule_date_shortcut_expands(monkeypatch: pytest.MonkeyPatch) -> None:
    adapter = GoogleWorkspaceToolAdapter(command="gws")
    seen: dict[str, object] = {}

    async def fake_run(argv: list[str], *, timeout: int) -> ToolResult:
        seen["argv"] = argv
        seen["timeout"] = timeout
        return ToolResult(True, json.dumps({"items": []}), {"argv": argv, "timeout": timeout})

    monkeypatch.setattr(adapter, "_run_json_command", fake_run)

    result = await adapter(
        "google_workspace_calendar_schedule",
        {"date": "2026-04-16", "calendar_id": "primary"},
    )

    assert result.success is True
    params = json.loads(seen["argv"][4])
    assert params["calendarId"] == "primary"
    assert params["timeMin"].startswith("2026-04-16T")
    assert params["timeMax"].startswith("2026-04-17T")
