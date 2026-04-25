"""Usage and spend observability routes for the dashboard."""

from __future__ import annotations

import subprocess
import time
from typing import Any, Dict, List

from fastapi import APIRouter


def _window_bounds_ms(window_days: int) -> tuple[int, int]:
    end_ms = int(time.time() * 1000)
    start_ms = end_ms - (max(1, int(window_days)) * 24 * 60 * 60 * 1000)
    return start_ms, end_ms


def _safe_compact(value: Any) -> str:
    text = str(value or "").strip()
    if len(text) <= 120:
        return text
    return text[:117] + "..."


def _scan_process_hygiene() -> Dict[str, Any]:
    snapshot: Dict[str, Any] = {
        "available": True,
        "opencas_processes": [],
        "pytest_processes": [],
        "gateway_processes": [],
        "duplicate_server_count": 0,
        "orphan_pytest_count": 0,
        "notes": [],
    }
    try:
        result = subprocess.run(
            ["ps", "-eo", "pid=,ppid=,etimes=,args="],
            check=True,
            capture_output=True,
            text=True,
        )
    except Exception as exc:
        snapshot["available"] = False
        snapshot["notes"].append(f"Process scan unavailable: {exc}")
        return snapshot

    opencas_processes: List[Dict[str, Any]] = []
    pytest_processes: List[Dict[str, Any]] = []
    gateway_processes: List[Dict[str, Any]] = []

    for raw_line in result.stdout.splitlines():
        line = raw_line.strip()
        if not line:
            continue
        parts = line.split(None, 3)
        if len(parts) != 4:
            continue
        pid, ppid, etimes, command = parts
        entry = {
            "pid": int(pid),
            "ppid": int(ppid),
            "age_seconds": int(etimes),
            "command": command,
            "command_compact": _safe_compact(command),
        }
        lowered = command.lower()
        if "python -m opencas" in lowered:
            opencas_processes.append(entry)
        elif "pytest" in lowered:
            pytest_processes.append(entry)
        elif "open_llm_auth" in lowered:
            gateway_processes.append(entry)

    opencas_processes.sort(key=lambda item: item["pid"])
    pytest_processes.sort(key=lambda item: item["pid"])
    gateway_processes.sort(key=lambda item: item["pid"])

    snapshot["opencas_processes"] = opencas_processes
    snapshot["pytest_processes"] = pytest_processes
    snapshot["gateway_processes"] = gateway_processes
    snapshot["duplicate_server_count"] = max(0, len(opencas_processes) - 1)
    snapshot["orphan_pytest_count"] = sum(1 for item in pytest_processes if item["ppid"] == 1)

    if snapshot["duplicate_server_count"] > 0:
        snapshot["notes"].append(
            f"Detected {snapshot['duplicate_server_count']} extra OpenCAS server process(es)."
        )
    if snapshot["orphan_pytest_count"] > 0:
        snapshot["notes"].append(
            f"Detected {snapshot['orphan_pytest_count']} orphaned pytest process(es)."
        )
    if not snapshot["notes"]:
        snapshot["notes"].append("No obvious duplicate OpenCAS servers or orphaned pytest runners detected.")
    return snapshot


async def _build_gateway_usage_snapshot(window_days: int, recent_limit: int) -> Dict[str, Any]:
    try:
        from open_llm_auth.server.usage_api import build_usage_overview, collect_provider_telemetry
    except Exception as exc:
        return {
            "available": False,
            "overview": {},
            "provider_telemetry": [],
            "notes": [f"open_llm_auth usage APIs are unavailable: {exc}"],
        }

    try:
        overview = build_usage_overview(days=window_days, recent_limit=recent_limit)
        telemetry = await collect_provider_telemetry(days=max(1, min(window_days, 30)))
        return {
            "available": True,
            "overview": overview,
            "provider_telemetry": telemetry,
            "notes": [],
        }
    except Exception as exc:
        return {
            "available": False,
            "overview": {},
            "provider_telemetry": [],
            "notes": [f"Failed to load open_llm_auth usage data: {exc}"],
        }


def _build_opencas_usage_snapshot(
    runtime: Any,
    *,
    window_days: int,
    bucket_hours: int,
    recent_limit: int,
) -> Dict[str, Any]:
    ctx = getattr(runtime, "ctx", None)
    telemetry = getattr(ctx, "token_telemetry", None)
    session_id = getattr(getattr(ctx, "config", None), "session_id", None)
    snapshot: Dict[str, Any] = {
        "available": telemetry is not None,
        "window_days": window_days,
        "bucket_hours": bucket_hours,
        "summary": {},
        "session_summary": {},
        "daily_rollup": [],
        "time_series": [],
        "by_provider": [],
        "by_model": [],
        "by_source": [],
        "by_execution_mode": [],
        "recent_events": [],
        "top_events": [],
        "notes": [],
    }
    if telemetry is None:
        snapshot["notes"].append("Token telemetry is not available in the current runtime.")
        return snapshot

    start_ms, end_ms = _window_bounds_ms(window_days)
    bucket_ms = max(1, min(bucket_hours, 24)) * 60 * 60 * 1000
    snapshot["summary"] = telemetry.get_summary(start_ms, end_ms).to_dict()
    snapshot["session_summary"] = telemetry.get_session_summary(session_id).to_dict() if session_id else {}
    snapshot["daily_rollup"] = [item.to_dict() for item in telemetry.get_daily_rollup(start_ms, end_ms)]
    snapshot["time_series"] = [item.to_dict() for item in telemetry.get_time_series(start_ms, end_ms, bucket_ms=bucket_ms)]
    snapshot["by_provider"] = telemetry.get_breakdown(start_ms, end_ms, "provider", limit=10)
    snapshot["by_model"] = telemetry.get_breakdown(start_ms, end_ms, "model", limit=10)
    snapshot["by_source"] = telemetry.get_breakdown(start_ms, end_ms, "source", limit=10)
    snapshot["by_execution_mode"] = telemetry.get_breakdown(start_ms, end_ms, "execution_mode", limit=10)
    snapshot["recent_events"] = telemetry.get_recent_events(start_ms, end_ms, limit=recent_limit)
    snapshot["top_events"] = telemetry.get_top_events(start_ms, end_ms, limit=10)

    if int(snapshot["summary"].get("totalCalls", 0) or 0) <= 0:
        snapshot["notes"].append("No OpenCAS token usage has been recorded in the selected window.")
    return snapshot


def build_usage_router(runtime: Any) -> APIRouter:
    router = APIRouter(prefix="/api/usage", tags=["usage"])

    @router.get("/overview")
    async def get_usage_overview(
        window_days: int = 7,
        bucket_hours: int = 6,
        recent_limit: int = 25,
    ) -> Dict[str, Any]:
        clamped_days = max(1, min(window_days, 30))
        clamped_bucket = max(1, min(bucket_hours, 24))
        clamped_recent = max(5, min(recent_limit, 100))
        return {
            "window_days": clamped_days,
            "bucket_hours": clamped_bucket,
            "generated_at": int(time.time() * 1000),
            "opencas": _build_opencas_usage_snapshot(
                runtime,
                window_days=clamped_days,
                bucket_hours=clamped_bucket,
                recent_limit=clamped_recent,
            ),
            "gateway": await _build_gateway_usage_snapshot(
                clamped_days,
                clamped_recent,
            ),
            "process_hygiene": _scan_process_hygiene(),
        }

    return router
