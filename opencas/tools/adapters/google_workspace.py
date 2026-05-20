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
from datetime import UTC, date, datetime, time, timedelta
from typing import Any, Dict, Iterable, Optional

from opencas.tools.environment import build_tool_execution_env, resolve_executable

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
    command_name = command or google_workspace_cli_command()
    return resolve_executable(command_name) is not None


def looks_like_google_workspace_auth_error(text: str) -> bool:
    """Return True for common Google Workspace OAuth/token failure text."""
    lowered = str(text or "").lower()
    return any(
        token in lowered
        for token in (
            "invalid_grant",
            "expired or revoked",
            "authentication failed",
            '"reason":"autherror"',
            '"reason": "autherror"',
            "failed to get token",
        )
    )


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
            if name == "google_workspace_calendar_dedupe":
                return await self._calendar_dedupe(args)
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

    async def _calendar_dedupe(self, args: Dict[str, Any]) -> ToolResult:
        calendar_id = str(args.get("calendar_id", "primary")).strip() or "primary"
        apply_changes = bool(args.get("apply", False))
        max_results = max(1, min(int(args.get("max_results", 2500)), 2500))
        max_deletions = max(1, min(int(args.get("max_deletions", 25)), 100))
        send_updates = str(args.get("send_updates", "none")).strip() or "none"
        if send_updates not in {"all", "externalOnly", "none"}:
            return ToolResult(False, "send_updates must be one of: all, externalOnly, none", {})
        timeout = _coerce_timeout(args)

        time_min = _clean_optional(args.get("time_min"))
        time_max = _clean_optional(args.get("time_max"))
        if not time_min:
            past_days = max(0, min(int(args.get("scan_past_days", 3650)), 3650))
            time_min = _to_utc_rfc3339(datetime.now(UTC) - timedelta(days=past_days))
        if not time_max:
            future_days = max(1, min(int(args.get("scan_future_days", 3650)), 3650))
            time_max = _to_utc_rfc3339(datetime.now(UTC) + timedelta(days=future_days))

        params: Dict[str, Any] = {
            "calendarId": calendar_id,
            "singleEvents": True,
            "orderBy": "startTime",
            "showDeleted": False,
            "maxResults": max_results,
            "timeMin": time_min,
            "timeMax": time_max,
        }
        list_result = await self._run_json_command(
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
        if not list_result.success:
            return list_result
        listing = json.loads(list_result.output)
        events = [event for event in (listing.get("items") or []) if isinstance(event, dict)]
        duplicate_groups = _calendar_duplicate_groups(events)
        candidates: list[dict[str, Any]] = []
        kept: list[dict[str, Any]] = []
        for group in duplicate_groups:
            keep = max(group, key=_calendar_keep_score)
            kept.append(_calendar_event_summary(keep))
            for event in group:
                if event is keep:
                    continue
                candidates.append(
                    {
                        **_calendar_event_summary(event),
                        "delete_event_id": str(event.get("recurringEventId") or event.get("id") or ""),
                        "delete_scope": "recurring_series" if event.get("recurringEventId") else "single_event",
                        "kept_event_id": str(keep.get("id") or ""),
                    }
                )
        candidates = _coalesce_calendar_delete_candidates(candidates)

        deletions: list[dict[str, Any]] = []
        failures: list[dict[str, Any]] = []
        if apply_changes:
            if len(candidates) > max_deletions:
                payload = {
                    "status": "blocked",
                    "reason": "candidate_count_exceeds_max_deletions",
                    "calendar_id": calendar_id,
                    "scanned_events": len(events),
                    "duplicate_group_count": len(duplicate_groups),
                    "duplicate_candidate_count": len(candidates),
                    "max_deletions": max_deletions,
                    "candidates": candidates,
                    "kept": kept,
                    "time_min": time_min,
                    "time_max": time_max,
                }
                return ToolResult(False, json.dumps(payload, indent=2), payload)
            for candidate in candidates:
                event_id = str(candidate.get("delete_event_id") or "").strip()
                if not event_id:
                    failures.append({**candidate, "error": "missing_event_id"})
                    continue
                delete_params = {
                    "calendarId": calendar_id,
                    "eventId": event_id,
                    "sendUpdates": send_updates,
                }
                delete_result = await self._run_json_command(
                    [
                        "calendar",
                        "events",
                        "delete",
                        "--params",
                        json.dumps(delete_params, separators=(",", ":")),
                        "--format",
                        "json",
                    ],
                    timeout=timeout,
                )
                if delete_result.success:
                    deletions.append(candidate)
                else:
                    failures.append({**candidate, "error": delete_result.output})

        payload = {
            "status": "applied" if apply_changes else "dry_run",
            "calendar_id": calendar_id,
            "scanned_events": len(events),
            "duplicate_group_count": len(duplicate_groups),
            "duplicate_candidate_count": len(candidates),
            "deleted_count": len(deletions),
            "failed_count": len(failures),
            "time_min": time_min,
            "time_max": time_max,
            "send_updates": send_updates if apply_changes else None,
            "candidates": candidates,
            "kept": kept,
            "deleted": deletions,
            "failures": failures,
            "next_page_token": listing.get("nextPageToken"),
        }
        return ToolResult(
            apply_changes is False or not failures,
            json.dumps(payload, indent=2),
            {
                "service": "calendar",
                "calendar_id": calendar_id,
                "apply": apply_changes,
                "duplicate_candidate_count": len(candidates),
                "deleted_count": len(deletions),
                "failed_count": len(failures),
                "time_min": time_min,
                "time_max": time_max,
            },
        )

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
        env = build_tool_execution_env()
        resolved_command = resolve_executable(self.command, env=env)
        if resolved_command is None:
            return ToolResult(
                False,
                f"Google Workspace CLI not available: {self.command}",
                {
                    "missing_command": True,
                    "command": self.command,
                    "path": env.get("PATH", ""),
                },
            )
        proc = await asyncio.create_subprocess_exec(
            resolved_command,
            *argv,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
            env=env,
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
            "resolved_command": resolved_command,
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
            metadata["auth_error"] = proc.returncode == 2 or looks_like_google_workspace_auth_error(output)
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


def _calendar_duplicate_groups(events: list[dict[str, Any]]) -> list[list[dict[str, Any]]]:
    grouped: dict[tuple[str, str, str, str, str], list[dict[str, Any]]] = {}
    for event in events:
        if str(event.get("status") or "").lower() == "cancelled":
            continue
        event_id = str(event.get("id") or "").strip()
        if not event_id:
            continue
        key = _calendar_duplicate_key(event)
        grouped.setdefault(key, []).append(event)
    return [group for group in grouped.values() if len(group) > 1]


def _calendar_duplicate_key(event: dict[str, Any]) -> tuple[str, str, str, str, str]:
    return (
        _normalize_event_text(event.get("summary")),
        _event_time_key(event.get("start")),
        _event_time_key(event.get("end")),
        _normalize_event_text(event.get("location")),
        _normalize_event_text(event.get("description")),
    )


def _calendar_keep_score(event: dict[str, Any]) -> tuple[int, int, float]:
    richness = 0
    if str(event.get("description") or "").strip():
        richness += 1
    if str(event.get("location") or "").strip():
        richness += 1
    if event.get("attendees"):
        richness += len(event.get("attendees") or [])
    if event.get("reminders"):
        richness += 1
    if event.get("conferenceData"):
        richness += 1
    if event.get("recurringEventId"):
        richness += 1
    created = _event_timestamp(event.get("created"))
    updated = _event_timestamp(event.get("updated"))
    # Prefer richer events, then older originals, then newer updates.
    created_preference = -created if created else float("-inf")
    return (richness, created_preference, updated)


def _calendar_event_summary(event: dict[str, Any]) -> dict[str, Any]:
    return {
        "id": str(event.get("id") or ""),
        "summary": str(event.get("summary") or ""),
        "start": event.get("start"),
        "end": event.get("end"),
        "description": str(event.get("description") or ""),
        "location": str(event.get("location") or ""),
        "created": event.get("created"),
        "updated": event.get("updated"),
        "recurringEventId": event.get("recurringEventId"),
        "htmlLink": event.get("htmlLink"),
    }


def _coalesce_calendar_delete_candidates(candidates: list[dict[str, Any]]) -> list[dict[str, Any]]:
    coalesced: dict[str, dict[str, Any]] = {}
    for candidate in candidates:
        delete_event_id = str(candidate.get("delete_event_id") or candidate.get("id") or "").strip()
        if not delete_event_id:
            continue
        existing = coalesced.get(delete_event_id)
        occurrence = {
            "id": candidate.get("id"),
            "start": candidate.get("start"),
            "end": candidate.get("end"),
            "kept_event_id": candidate.get("kept_event_id"),
        }
        if existing is None:
            copied = dict(candidate)
            copied["duplicate_occurrence_count"] = 1
            copied["sample_occurrences"] = [occurrence]
            coalesced[delete_event_id] = copied
            continue
        existing["duplicate_occurrence_count"] = int(existing.get("duplicate_occurrence_count", 1)) + 1
        samples = existing.setdefault("sample_occurrences", [])
        if isinstance(samples, list) and len(samples) < 5:
            samples.append(occurrence)
    return list(coalesced.values())


def _normalize_event_text(value: Any) -> str:
    return " ".join(str(value or "").split()).casefold()


def _event_time_key(value: Any) -> str:
    if not isinstance(value, dict):
        return ""
    return str(value.get("dateTime") or value.get("date") or "").strip()


def _event_timestamp(value: Any) -> float:
    text = str(value or "").strip()
    if not text:
        return 0.0
    try:
        return datetime.fromisoformat(text.replace("Z", "+00:00")).timestamp()
    except ValueError:
        return 0.0


def _day_bounds(value: str) -> tuple[str, str]:
    local_day = date.fromisoformat(value)
    local_tz = datetime.now().astimezone().tzinfo or UTC
    start = datetime.combine(local_day, time.min, tzinfo=local_tz)
    end = start + timedelta(days=1)
    return _to_utc_rfc3339(start), _to_utc_rfc3339(end)


def _to_utc_rfc3339(value: datetime) -> str:
    return value.astimezone(UTC).isoformat().replace("+00:00", "Z")
