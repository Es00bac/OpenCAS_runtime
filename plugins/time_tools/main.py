"""Time/date utilities plugin for OpenCAS."""

from __future__ import annotations

import json
import os
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Dict
from zoneinfo import ZoneInfo, ZoneInfoNotFoundError

from opencas.autonomy.models import ActionRiskTier
from opencas.tools.models import ToolResult


def _resolve_tz(tz: str | None) -> tuple[Any, str | None]:
    if not tz:
        return timezone.utc, None
    try:
        return ZoneInfo(tz), None
    except ZoneInfoNotFoundError:
        return None, f"unknown timezone: {tz}"


def _parse_iso(value: str) -> datetime | None:
    text = value.strip()
    if not text:
        return None
    if text.endswith("Z"):
        text = text[:-1] + "+00:00"
    try:
        dt = datetime.fromisoformat(text)
    except ValueError:
        return None
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    return dt


def _format_payload(dt: datetime) -> Dict[str, Any]:
    return {
        "iso": dt.isoformat(),
        "epoch": dt.timestamp(),
        "weekday": dt.strftime("%A"),
        "date": dt.strftime("%Y-%m-%d"),
        "time": dt.strftime("%H:%M:%S"),
        "tz": str(dt.tzinfo),
    }


def _time_now(args: Dict[str, Any]) -> ToolResult:
    tz_str = str(args.get("tz", "")).strip() or None
    tz, err = _resolve_tz(tz_str)
    if err:
        return ToolResult(success=False, output=err, metadata={})
    now = datetime.now(tz)
    return ToolResult(success=True, output=json.dumps(_format_payload(now), indent=2), metadata={"tz": str(tz)})


def _time_parse(args: Dict[str, Any]) -> ToolResult:
    value = str(args.get("value", "")).strip()
    if not value:
        return ToolResult(success=False, output="value is required", metadata={})
    dt = _parse_iso(value)
    if dt is None:
        return ToolResult(success=False, output=f"could not parse: {value}", metadata={"value": value})
    tz_str = str(args.get("tz", "")).strip() or None
    if tz_str:
        tz, err = _resolve_tz(tz_str)
        if err:
            return ToolResult(success=False, output=err, metadata={})
        dt = dt.astimezone(tz)
    return ToolResult(success=True, output=json.dumps(_format_payload(dt), indent=2), metadata={})


def _format_delta(td: timedelta) -> Dict[str, Any]:
    total = td.total_seconds()
    sign = -1 if total < 0 else 1
    abs_total = abs(total)
    days, rem = divmod(int(abs_total), 86_400)
    hours, rem = divmod(rem, 3_600)
    minutes, seconds = divmod(rem, 60)
    parts = []
    if days:
        parts.append(f"{days}d")
    if hours:
        parts.append(f"{hours}h")
    if minutes:
        parts.append(f"{minutes}m")
    parts.append(f"{seconds}s")
    human = ("-" if sign < 0 else "") + " ".join(parts)
    return {
        "total_seconds": total,
        "total_minutes": total / 60,
        "total_hours": total / 3600,
        "total_days": total / 86_400,
        "human": human,
    }


def _time_diff(args: Dict[str, Any]) -> ToolResult:
    a = _parse_iso(str(args.get("from", "")))
    b = _parse_iso(str(args.get("to", "")))
    if a is None:
        return ToolResult(success=False, output="'from' is required and must be ISO 8601", metadata={})
    if b is None:
        return ToolResult(success=False, output="'to' is required and must be ISO 8601", metadata={})
    delta = b - a
    return ToolResult(
        success=True,
        output=json.dumps(_format_delta(delta), indent=2),
        metadata={"from": a.isoformat(), "to": b.isoformat()},
    )


def _time_age(args: Dict[str, Any]) -> ToolResult:
    raw_path = str(args.get("path", "")).strip()
    if not raw_path:
        return ToolResult(success=False, output="path is required", metadata={})
    path = Path(raw_path)
    if not path.exists():
        return ToolResult(success=False, output=f"path not found: {raw_path}", metadata={})
    stat = path.stat()
    now = datetime.now(timezone.utc)
    mtime = datetime.fromtimestamp(stat.st_mtime, tz=timezone.utc)
    ctime = datetime.fromtimestamp(stat.st_ctime, tz=timezone.utc)
    payload = {
        "path": str(path),
        "modified_iso": mtime.isoformat(),
        "modified_age": _format_delta(now - mtime),
        "created_iso": ctime.isoformat(),
        "created_age": _format_delta(now - ctime),
        "size_bytes": stat.st_size,
    }
    return ToolResult(success=True, output=json.dumps(payload, indent=2), metadata={"path": str(path)})


def register_skills(skill_registry, tools) -> None:
    tools.register(
        "time_now",
        "Return the current time as ISO 8601 plus components. Optional 'tz' (IANA name, e.g. 'America/Denver').",
        lambda name, args: _time_now(args),
        ActionRiskTier.READONLY,
        {"type": "object", "properties": {"tz": {"type": "string"}}},
    )
    tools.register(
        "time_parse",
        "Parse an ISO 8601 datetime; optional 'tz' converts to that zone.",
        lambda name, args: _time_parse(args),
        ActionRiskTier.READONLY,
        {
            "type": "object",
            "properties": {"value": {"type": "string"}, "tz": {"type": "string"}},
            "required": ["value"],
        },
    )
    tools.register(
        "time_diff",
        "Compute the duration between two ISO 8601 datetimes (to - from).",
        lambda name, args: _time_diff(args),
        ActionRiskTier.READONLY,
        {
            "type": "object",
            "properties": {"from": {"type": "string"}, "to": {"type": "string"}},
            "required": ["from", "to"],
        },
    )
    tools.register(
        "time_age",
        "Return modified/created age and size for a filesystem path.",
        lambda name, args: _time_age(args),
        ActionRiskTier.READONLY,
        {
            "type": "object",
            "properties": {"path": {"type": "string"}},
            "required": ["path"],
        },
    )
