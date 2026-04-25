"""Bounded Google Workspace CLI adapters for OpenCAS.

These tools wrap the local ``gws`` CLI instead of exposing a general shell
escape. The surface is intentionally read-only by default so the agent can use
Google Workspace safely for inspection tasks like inbox triage, calendar
lookups, and Drive search.
"""

from __future__ import annotations

import asyncio
import json
import os
import shutil
from datetime import UTC, date, datetime, time, timedelta
from typing import Any, Dict, Iterable, Optional

from ..models import ToolResult

_DEFAULT_TIMEOUT_SECONDS = 30
_READONLY_ALLOWLIST: set[tuple[str, str, Optional[str], str]] = {
    ("gmail", "users", "messages", "list"),
    ("gmail", "users", "messages", "get"),
    ("gmail", "users", "threads", "list"),
    ("gmail", "users", "threads", "get"),
    ("gmail", "users", "labels", "list"),
    ("gmail", "users", "labels", "get"),
    ("calendar", "calendarList", None, "list"),
    ("calendar", "calendarList", None, "get"),
    ("calendar", "calendars", None, "get"),
    ("calendar", "events", None, "list"),
    ("calendar", "events", None, "get"),
    ("drive", "files", None, "list"),
    ("drive", "files", None, "get"),
    ("drive", "comments", None, "list"),
    ("drive", "comments", None, "get"),
    ("docs", "documents", None, "get"),
    ("sheets", "spreadsheets", None, "get"),
    ("sheets", "spreadsheets", "values", "get"),
    ("sheets", "spreadsheets", "values", "batchGet"),
    ("slides", "presentations", None, "get"),
    ("people", "people", None, "get"),
    ("people", "people", "connections", "list"),
}


def google_workspace_cli_command() -> str:
    """Return the configured Google Workspace CLI command."""
    return str(os.getenv("GOOGLE_WORKSPACE_CLI_COMMAND", "gws")).strip() or "gws"


def google_workspace_cli_available(command: Optional[str] = None) -> bool:
    """Return True when the Google Workspace CLI is available on PATH."""
    return shutil.which(command or google_workspace_cli_command()) is not None


class GoogleWorkspaceToolAdapter:
    """Read-oriented wrapper around the local ``gws`` CLI."""

    def __init__(self, command: Optional[str] = None) -> None:
        self.command = command or google_workspace_cli_command()

    async def __call__(self, name: str, args: Dict[str, Any]) -> ToolResult:
        try:
            if name == "google_workspace_auth_status":
                return await self._auth_status(args)
            if name == "google_workspace_schema":
                return await self._schema(args)
            if name == "google_workspace_readonly_api":
                return await self._readonly_api(args)
            if name == "google_workspace_gmail_headlines":
                return await self._gmail_headlines(args)
            if name == "google_workspace_gmail_get_message":
                return await self._gmail_get_message(args)
            if name == "google_workspace_calendar_schedule":
                return await self._calendar_schedule(args)
            if name == "google_workspace_drive_search":
                return await self._drive_search(args)
            return ToolResult(False, f"Unknown Google Workspace tool: {name}", {})
        except Exception as exc:
            return ToolResult(False, str(exc), {"error_type": type(exc).__name__})

    async def _auth_status(self, args: Dict[str, Any]) -> ToolResult:
        timeout = _coerce_timeout(args)
        return await self._run_json_command(["auth", "status", "--format", "json"], timeout=timeout)

    async def _schema(self, args: Dict[str, Any]) -> ToolResult:
        schema_ref = str(args.get("schema_ref", "")).strip()
        if not schema_ref:
            return ToolResult(False, "schema_ref is required", {})
        timeout = _coerce_timeout(args)
        argv = ["schema", schema_ref]
        if bool(args.get("resolve_refs")):
            argv.append("--resolve-refs")
        return await self._run_json_command(argv, timeout=timeout)

    async def _readonly_api(self, args: Dict[str, Any]) -> ToolResult:
        service = str(args.get("service", "")).strip().lower()
        resource = str(args.get("resource", "")).strip()
        sub_resource = _clean_optional(args.get("sub_resource"))
        method = str(args.get("method", "")).strip().lower()
        params = args.get("params") or {}
        if not service or not resource or not method:
            return ToolResult(
                False,
                "service, resource, and method are required",
                {},
            )
        if not isinstance(params, dict):
            return ToolResult(False, "params must be an object", {})
        if not _is_readonly_allowed(service, resource, sub_resource, method):
            return ToolResult(
                False,
                (
                    "Unsupported Google Workspace readonly operation. "
                    "Use google_workspace_schema to inspect available methods."
                ),
                {
                    "service": service,
                    "resource": resource,
                    "sub_resource": sub_resource,
                    "method": method,
                },
            )

        argv = [service, resource]
        if sub_resource:
            argv.append(sub_resource)
        argv.append(method)
        argv.extend(["--params", json.dumps(params, separators=(",", ":")), "--format", "json"])
        api_version = _clean_optional(args.get("api_version"))
        if api_version:
            argv.extend(["--api-version", api_version])
        if bool(args.get("page_all")):
            argv.append("--page-all")
            page_limit = args.get("page_limit")
            if page_limit is not None:
                argv.extend(["--page-limit", str(int(page_limit))])
            page_delay = args.get("page_delay_ms")
            if page_delay is not None:
                argv.extend(["--page-delay", str(int(page_delay))])
        timeout = _coerce_timeout(args)
        result = await self._run_json_command(argv, timeout=timeout)
        if result.success:
            result.metadata.update(
                {
                    "service": service,
                    "resource": resource,
                    "sub_resource": sub_resource,
                    "method": method,
                }
            )
        return result

    async def _gmail_headlines(self, args: Dict[str, Any]) -> ToolResult:
        max_results = max(1, min(int(args.get("max_results", 10)), 25))
        query = str(args.get("query", "in:inbox")).strip() or "in:inbox"
        include_snippet = bool(args.get("include_snippet"))
        timeout = _coerce_timeout(args)
        list_params: Dict[str, Any] = {
            "userId": "me",
            "maxResults": max_results,
            "q": query,
        }
        if args.get("label_ids"):
            label_ids = args["label_ids"]
            if not isinstance(label_ids, list):
                return ToolResult(False, "label_ids must be a list of strings", {})
            list_params["labelIds"] = [str(item) for item in label_ids if str(item).strip()]
        list_result = await self._run_json_command(
            [
                "gmail",
                "users",
                "messages",
                "list",
                "--params",
                json.dumps(list_params, separators=(",", ":")),
                "--format",
                "json",
            ],
            timeout=timeout,
        )
        if not list_result.success:
            return list_result
        listing = json.loads(list_result.output)
        message_refs = list(listing.get("messages") or [])
        items = []
        for ref in message_refs:
            message_id = str(ref.get("id", "")).strip()
            if not message_id:
                continue
            get_params: Dict[str, Any] = {
                "userId": "me",
                "id": message_id,
                "format": "metadata",
                "metadataHeaders": ["Subject", "From", "Date"],
            }
            detail_result = await self._run_json_command(
                [
                    "gmail",
                    "users",
                    "messages",
                    "get",
                    "--params",
                    json.dumps(get_params, separators=(",", ":")),
                    "--format",
                    "json",
                ],
                timeout=timeout,
            )
            if not detail_result.success:
                items.append({"id": message_id, "error": detail_result.output})
                continue
            detail = json.loads(detail_result.output)
            headers = _header_map(((detail.get("payload") or {}).get("headers") or []))
            item: Dict[str, Any] = {
                "id": message_id,
                "threadId": detail.get("threadId"),
                "date": headers.get("date"),
                "from": headers.get("from"),
                "subject": headers.get("subject"),
                "labelIds": detail.get("labelIds") or [],
            }
            if include_snippet and detail.get("snippet"):
                item["snippet"] = detail.get("snippet")
            items.append(item)

        payload = {
            "count": len(items),
            "messages": items,
            "query": query,
            "nextPageToken": listing.get("nextPageToken"),
            "resultSizeEstimate": listing.get("resultSizeEstimate"),
        }
        return ToolResult(
            True,
            json.dumps(payload, indent=2),
            {
                "service": "gmail",
                "query": query,
                "result_count": len(items),
                "next_page_token": listing.get("nextPageToken"),
            },
        )

    async def _gmail_get_message(self, args: Dict[str, Any]) -> ToolResult:
        message_id = str(args.get("message_id", "")).strip()
        if not message_id:
            return ToolResult(False, "message_id is required", {})
        fmt = str(args.get("format", "metadata")).strip().lower()
        if fmt not in {"metadata", "full", "minimal"}:
            return ToolResult(False, "format must be one of: metadata, full, minimal", {})
        params: Dict[str, Any] = {"userId": "me", "id": message_id, "format": fmt}
        if fmt == "metadata":
            params["metadataHeaders"] = ["Subject", "From", "To", "Date"]
        timeout = _coerce_timeout(args)
        result = await self._run_json_command(
            [
                "gmail",
                "users",
                "messages",
                "get",
                "--params",
                json.dumps(params, separators=(",", ":")),
                "--format",
                "json",
            ],
            timeout=timeout,
        )
        if result.success:
            result.metadata.update({"service": "gmail", "message_id": message_id, "format": fmt})
        return result

    async def _calendar_schedule(self, args: Dict[str, Any]) -> ToolResult:
        calendar_id = str(args.get("calendar_id", "primary")).strip() or "primary"
        max_results = max(1, min(int(args.get("max_results", 10)), 50))
        time_min = _clean_optional(args.get("time_min"))
        time_max = _clean_optional(args.get("time_max"))
        date_value = _clean_optional(args.get("date"))
        if date_value and not (time_min or time_max):
            range_start, range_end = _day_bounds(date_value)
            time_min = range_start
            time_max = range_end
        params: Dict[str, Any] = {
            "calendarId": calendar_id,
            "singleEvents": True,
            "orderBy": "startTime",
            "maxResults": max_results,
        }
        if time_min:
            params["timeMin"] = time_min
        if time_max:
            params["timeMax"] = time_max
        timeout = _coerce_timeout(args)
        result = await self._run_json_command(
            [
                "calendar",
                "events",
                "list",
                "--params",
                json.dumps(params, separators=(",", ":")),
                "--format",
                "json",
            ],
            timeout=timeout,
        )
        if result.success:
            result.metadata.update(
                {
                    "service": "calendar",
                    "calendar_id": calendar_id,
                    "time_min": time_min,
                    "time_max": time_max,
                }
            )
        return result

    async def _drive_search(self, args: Dict[str, Any]) -> ToolResult:
        query = str(args.get("query", "trashed=false")).strip() or "trashed=false"
        page_size = max(1, min(int(args.get("page_size", 10)), 50))
        params = {
            "pageSize": page_size,
            "q": query,
            "fields": "files(id,name,mimeType,modifiedTime,webViewLink,parents),nextPageToken",
        }
        timeout = _coerce_timeout(args)
        result = await self._run_json_command(
            [
                "drive",
                "files",
                "list",
                "--params",
                json.dumps(params, separators=(",", ":")),
                "--format",
                "json",
            ],
            timeout=timeout,
        )
        if result.success:
            result.metadata.update({"service": "drive", "query": query})
        return result

    async def _run_json_command(self, argv: list[str], *, timeout: int) -> ToolResult:
        if not google_workspace_cli_available(self.command):
            return ToolResult(
                False,
                f"Google Workspace CLI not available: {self.command}",
                {"missing_command": True, "command": self.command},
            )
        proc = await asyncio.create_subprocess_exec(
            self.command,
            *argv,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )
        try:
            stdout, stderr = await asyncio.wait_for(proc.communicate(), timeout=timeout)
        except asyncio.TimeoutError:
            proc.kill()
            await proc.communicate()
            return ToolResult(
                False,
                f"Google Workspace CLI timed out after {timeout}s",
                {"command": self.command, "argv": argv, "timeout_seconds": timeout},
            )

        stdout_text = stdout.decode("utf-8", errors="replace").strip()
        stderr_text = stderr.decode("utf-8", errors="replace").strip()
        parsed = _try_parse_json(stdout_text)
        metadata = {
            "command": self.command,
            "argv": argv,
            "exit_code": proc.returncode,
        }
        if stderr_text:
            metadata["stderr"] = stderr_text

        if proc.returncode != 0:
            output = stdout_text or stderr_text or f"{self.command} exited with code {proc.returncode}"
            if parsed is not None:
                output = json.dumps(parsed, indent=2)
            metadata["error"] = True
            metadata["auth_error"] = proc.returncode == 2
            return ToolResult(False, output, metadata)

        if parsed is not None:
            return ToolResult(True, json.dumps(parsed, indent=2), metadata)
        return ToolResult(True, stdout_text or stderr_text or "{}", metadata)


def _clean_optional(value: Any) -> Optional[str]:
    text = str(value or "").strip()
    return text or None


def _coerce_timeout(args: Dict[str, Any]) -> int:
    try:
        return max(5, min(int(args.get("timeout_seconds", _DEFAULT_TIMEOUT_SECONDS)), 120))
    except Exception:
        return _DEFAULT_TIMEOUT_SECONDS


def _is_readonly_allowed(
    service: str,
    resource: str,
    sub_resource: Optional[str],
    method: str,
) -> bool:
    return (service, resource, sub_resource, method) in _READONLY_ALLOWLIST


def _try_parse_json(text: str) -> Optional[Any]:
    if not text:
        return None
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        return None


def _header_map(headers: Iterable[Dict[str, Any]]) -> Dict[str, str]:
    mapped: Dict[str, str] = {}
    for header in headers:
        name = str(header.get("name", "")).strip().lower()
        if not name:
            continue
        mapped[name] = str(header.get("value", "")).strip()
    return mapped


def _day_bounds(value: str) -> tuple[str, str]:
    local_day = date.fromisoformat(value)
    local_tz = datetime.now().astimezone().tzinfo or UTC
    start = datetime.combine(local_day, time.min, tzinfo=local_tz)
    end = start + timedelta(days=1)
    return _to_utc_rfc3339(start), _to_utc_rfc3339(end)


def _to_utc_rfc3339(value: datetime) -> str:
    return value.astimezone(UTC).isoformat().replace("+00:00", "Z")
