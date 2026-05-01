"""System stats plugin for OpenCAS."""

from __future__ import annotations

import json
import os
import shutil
from typing import Any, Dict

from opencas.autonomy.models import ActionRiskTier
from opencas.tools.models import ToolResult

try:
    import psutil  # type: ignore
except Exception:  # pragma: no cover
    psutil = None  # type: ignore


def _human_bytes(n: float) -> str:
    for unit in ("B", "KB", "MB", "GB", "TB"):
        if n < 1024.0:
            return f"{n:.1f}{unit}"
        n /= 1024.0
    return f"{n:.1f}PB"


def _system_status(args: Dict[str, Any]) -> ToolResult:
    paths_arg = args.get("disk_paths") or ["/"]
    if not isinstance(paths_arg, list):
        return ToolResult(success=False, output="disk_paths must be a list of strings", metadata={})

    snapshot: Dict[str, Any] = {}

    if psutil is not None:
        snapshot["cpu_percent"] = psutil.cpu_percent(interval=0.2)
        snapshot["cpu_count"] = psutil.cpu_count(logical=True)
        vm = psutil.virtual_memory()
        snapshot["memory"] = {
            "total": vm.total,
            "available": vm.available,
            "used": vm.used,
            "percent": vm.percent,
            "human_total": _human_bytes(vm.total),
            "human_available": _human_bytes(vm.available),
        }
        try:
            snapshot["load_avg"] = list(os.getloadavg())
        except (AttributeError, OSError):
            snapshot["load_avg"] = None
        snapshot["process_count"] = len(psutil.pids())
    else:
        try:
            snapshot["load_avg"] = list(os.getloadavg())
        except (AttributeError, OSError):
            snapshot["load_avg"] = None
        snapshot["psutil_available"] = False

    disks: Dict[str, Any] = {}
    for raw_path in paths_arg:
        path = str(raw_path)
        try:
            usage = shutil.disk_usage(path)
            disks[path] = {
                "total": usage.total,
                "used": usage.used,
                "free": usage.free,
                "percent": round(usage.used / usage.total * 100, 1) if usage.total else 0.0,
                "human_free": _human_bytes(usage.free),
                "human_total": _human_bytes(usage.total),
            }
        except OSError as exc:
            disks[path] = {"error": str(exc)}
    snapshot["disks"] = disks

    return ToolResult(
        success=True,
        output=json.dumps(snapshot, indent=2),
        metadata={"keys": list(snapshot.keys())},
    )


def register_skills(skill_registry, tools) -> None:
    tools.register(
        "system_status",
        "Return CPU, memory, load, process count, and disk usage as JSON.",
        lambda name, args: _system_status(args),
        ActionRiskTier.READONLY,
        {
            "type": "object",
            "properties": {
                "disk_paths": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": "Mount points to inspect (default ['/']).",
                }
            },
        },
    )
