from __future__ import annotations

import json

import pytest

from opencas.tools.adapters.google_workspace import (
    GoogleWorkspaceToolAdapter,
    looks_like_google_workspace_auth_error,
)
from opencas.tools.models import ToolResult


@pytest.mark.asyncio
async def test_google_workspace_auth_status_reports_missing_command() -> None:
    adapter = GoogleWorkspaceToolAdapter(command="/definitely/missing/gws")

    result = await adapter("google_workspace_auth_status", {})

    assert result.success is False
    assert "not available" in result.output
    assert result.metadata["missing_command"] is True


def test_google_workspace_auth_error_detection_handles_revoked_oauth_text() -> None:
    text = (
        '{"error":{"code":401,"message":"Authentication failed: Failed to get token: '
        'Server error: invalid_grant: Token has been expired or revoked.",'
        '"reason":"authError"}}'
    )

    assert looks_like_google_workspace_auth_error(text) is True


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


@pytest.mark.asyncio
async def test_google_workspace_calendar_dedupe_dry_run_keeps_one_duplicate(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    adapter = GoogleWorkspaceToolAdapter(command="gws")
    events = [
        {
            "id": "keep",
            "summary": "Exafunction ()",
            "description": "Monthly subscription renewal",
            "start": {"dateTime": "2026-05-10T09:00:00-06:00"},
            "end": {"dateTime": "2026-05-10T10:00:00-06:00"},
            "created": "2026-01-01T00:00:00Z",
            "updated": "2026-01-02T00:00:00Z",
        },
        {
            "id": "delete-1",
            "summary": "Exafunction ()",
            "description": "Monthly subscription renewal",
            "start": {"dateTime": "2026-05-10T09:00:00-06:00"},
            "end": {"dateTime": "2026-05-10T10:00:00-06:00"},
            "created": "2026-01-03T00:00:00Z",
            "updated": "2026-01-03T00:00:00Z",
        },
    ]

    async def fake_run(argv: list[str], *, timeout: int) -> ToolResult:
        assert argv[:4] == ["calendar", "events", "list", "--params"]
        return ToolResult(True, json.dumps({"items": events}), {"argv": argv, "timeout": timeout})

    monkeypatch.setattr(adapter, "_run_json_command", fake_run)

    result = await adapter(
        "google_workspace_calendar_dedupe",
        {
            "calendar_id": "primary",
            "time_min": "2026-01-01T00:00:00Z",
            "time_max": "2027-01-01T00:00:00Z",
        },
    )

    assert result.success is True
    payload = json.loads(result.output)
    assert payload["status"] == "dry_run"
    assert payload["duplicate_group_count"] == 1
    assert payload["duplicate_candidate_count"] == 1
    assert payload["candidates"][0]["id"] == "delete-1"
    assert payload["candidates"][0]["kept_event_id"] == "keep"


@pytest.mark.asyncio
async def test_google_workspace_calendar_dedupe_apply_deletes_candidates(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    adapter = GoogleWorkspaceToolAdapter(command="gws")
    calls: list[list[str]] = []

    async def fake_run(argv: list[str], *, timeout: int) -> ToolResult:
        calls.append(argv)
        if argv[:4] == ["calendar", "events", "list", "--params"]:
            return ToolResult(
                True,
                json.dumps(
                    {
                        "items": [
                            {
                                "id": "keep",
                                "summary": "Weekly Therapy",
                                "start": {"dateTime": "2026-05-05T15:00:00-06:00"},
                                "end": {"dateTime": "2026-05-05T16:00:00-06:00"},
                                "created": "2026-01-01T00:00:00Z",
                            },
                            {
                                "id": "delete-1",
                                "summary": "Weekly Therapy",
                                "start": {"dateTime": "2026-05-05T15:00:00-06:00"},
                                "end": {"dateTime": "2026-05-05T16:00:00-06:00"},
                                "created": "2026-01-02T00:00:00Z",
                            },
                        ]
                    }
                ),
                {"argv": argv, "timeout": timeout},
            )
        assert argv[:4] == ["calendar", "events", "delete", "--params"]
        params = json.loads(argv[4])
        assert params == {
            "calendarId": "primary",
            "eventId": "delete-1",
            "sendUpdates": "none",
        }
        return ToolResult(True, "{}", {"argv": argv, "timeout": timeout})

    monkeypatch.setattr(adapter, "_run_json_command", fake_run)

    result = await adapter(
        "google_workspace_calendar_dedupe",
        {
            "calendar_id": "primary",
            "apply": True,
            "time_min": "2026-01-01T00:00:00Z",
            "time_max": "2027-01-01T00:00:00Z",
        },
    )

    assert result.success is True
    payload = json.loads(result.output)
    assert payload["status"] == "applied"
    assert payload["deleted_count"] == 1
    assert len(calls) == 2


@pytest.mark.asyncio
async def test_google_workspace_calendar_dedupe_applies_recurring_series_once(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    adapter = GoogleWorkspaceToolAdapter(command="gws")
    deleted_ids: list[str] = []

    def event(event_id: str, series_id: str, start: str, created: str) -> dict[str, object]:
        return {
            "id": event_id,
            "recurringEventId": series_id,
            "summary": "Exafunction ()",
            "description": "Monthly subscription renewal",
            "start": {"dateTime": start},
            "end": {"dateTime": start.replace("09:00:00", "10:00:00")},
            "created": created,
        }

    async def fake_run(argv: list[str], *, timeout: int) -> ToolResult:
        if argv[:4] == ["calendar", "events", "list", "--params"]:
            return ToolResult(
                True,
                json.dumps(
                    {
                        "items": [
                            event("keep-1", "keep-series", "2026-05-10T09:00:00-06:00", "2026-01-01T00:00:00Z"),
                            event("delete-1a", "delete-series", "2026-05-10T09:00:00-06:00", "2026-01-02T00:00:00Z"),
                            event("keep-2", "keep-series", "2026-06-10T09:00:00-06:00", "2026-01-01T00:00:00Z"),
                            event("delete-1b", "delete-series", "2026-06-10T09:00:00-06:00", "2026-01-02T00:00:00Z"),
                        ]
                    }
                ),
                {"argv": argv, "timeout": timeout},
            )
        assert argv[:4] == ["calendar", "events", "delete", "--params"]
        deleted_ids.append(json.loads(argv[4])["eventId"])
        return ToolResult(True, "{}", {"argv": argv, "timeout": timeout})

    monkeypatch.setattr(adapter, "_run_json_command", fake_run)

    result = await adapter(
        "google_workspace_calendar_dedupe",
        {
            "calendar_id": "primary",
            "apply": True,
            "time_min": "2026-01-01T00:00:00Z",
            "time_max": "2027-01-01T00:00:00Z",
        },
    )

    assert result.success is True
    payload = json.loads(result.output)
    assert payload["duplicate_candidate_count"] == 1
    assert payload["deleted_count"] == 1
    assert payload["deleted"][0]["delete_scope"] == "recurring_series"
    assert payload["deleted"][0]["duplicate_occurrence_count"] == 2
    assert deleted_ids == ["delete-series"]
