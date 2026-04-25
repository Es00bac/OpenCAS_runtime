"""PTY supervision workflow helpers."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, Optional

from ..models import ToolResult


def supervision_advisory(
    pty_data: Dict[str, Any],
    *,
    verification_exists: bool,
) -> Dict[str, Any]:
    screen_state = pty_data.get("screen_state", {}) or {}
    mode = str(screen_state.get("mode", "idle"))
    ready_for_input = bool(screen_state.get("ready_for_input", False))
    blocked = bool(screen_state.get("blocked", False))
    running = pty_data.get("running")

    if verification_exists:
        return {
            "action": "stop",
            "reason": "verification_satisfied",
            "next_step": "cleanup_and_verify",
            "observe_idle_seconds": 0.2,
            "observe_max_wait_seconds": 1.0,
        }
    if running is False:
        return {
            "action": "stop",
            "reason": "process_exited",
            "next_step": "inspect_exit_state",
            "observe_idle_seconds": 0.2,
            "observe_max_wait_seconds": 1.0,
        }
    if blocked or mode == "auth_required":
        return {
            "action": "stop",
            "reason": "auth_or_gate_blocked",
            "next_step": "resolve_auth_or_gate",
            "observe_idle_seconds": 0.2,
            "observe_max_wait_seconds": 1.0,
        }
    if mode in {"shell_prompt", "error_prompt"}:
        return {
            "action": "stop",
            "reason": "awaiting_operator_input",
            "next_step": "send_follow_up_input",
            "observe_idle_seconds": 0.2,
            "observe_max_wait_seconds": 1.5,
        }
    if ready_for_input:
        return {
            "action": "observe_briefly",
            "reason": "interactive_idle_ready_for_input",
            "next_step": "send_follow_up_if_stalled",
            "observe_idle_seconds": 0.35,
            "observe_max_wait_seconds": 2.5,
        }
    return {
        "action": "observe",
        "reason": "still_running",
        "next_step": "continue_observing",
        "observe_idle_seconds": 1.0,
        "observe_max_wait_seconds": 15.0,
    }


async def supervise_session(runtime: Any, args: Dict[str, Any]) -> ToolResult:
    command = str(args.get("command", "")).strip()
    session_id = args.get("session_id")
    task_input = str(args.get("task", "")).strip()
    scope_key = str(args.get("scope_key", "workflow-supervision"))
    max_wait = float(args.get("max_wait_seconds", 15.0))
    idle_seconds = float(args.get("idle_seconds", 1.0))
    startup_wait = float(args.get("startup_wait_seconds", min(max_wait, 8.0)))
    continue_wait = float(args.get("continue_wait_seconds", max_wait))
    max_rounds = max(1, int(args.get("max_rounds", 3)))
    verification_path_arg = str(args.get("verification_path", "")).strip()
    verification_path = Path(verification_path_arg) if verification_path_arg else None
    advisory: Dict[str, Any] = {}
    rounds_used = 0
    follow_up_attempted = False

    if not command and not session_id:
        return ToolResult(
            False,
            "Either command (to start new session) or session_id (to resume) is required",
            {},
        )

    start_args: Dict[str, Any] = {
        "scope_key": scope_key,
        "idle_seconds": idle_seconds,
        "max_wait_seconds": max_wait,
    }
    if session_id:
        start_args["session_id"] = str(session_id)
    if command:
        start_args["command"] = command

    active_session_id: Optional[str] = str(session_id).strip() if session_id else None
    if command and not session_id:
        start_args["max_wait_seconds"] = startup_wait
        start_result = await runtime.execute_tool("pty_interact", start_args)
        start_raw = start_result.get("output", "")
        try:
            start_data = json.loads(start_raw) if isinstance(start_raw, str) else start_raw
        except (json.JSONDecodeError, TypeError):
            start_data = {"raw": start_raw}
        active_session_id = str(start_data.get("session_id") or "").strip() or None
        rounds_used += 1
        if not task_input:
            pty_data = start_data
        elif not active_session_id:
            return ToolResult(
                False,
                "PTY session did not return a usable session_id during startup",
                {"scope_key": scope_key},
            )
        else:
            startup_advisory = supervision_advisory(
                start_data,
                verification_exists=verification_path.exists() if verification_path else False,
            )
            if startup_advisory["action"] == "stop":
                pty_data = start_data
                advisory = startup_advisory
            else:
                submit_args: Dict[str, Any] = {
                    "scope_key": scope_key,
                    "session_id": active_session_id,
                    "idle_seconds": idle_seconds,
                    "max_wait_seconds": max_wait,
                    "input": task_input + "\r",
                }
                result = await runtime.execute_tool("pty_interact", submit_args)
                raw = result.get("output", "")
                try:
                    pty_data = json.loads(raw) if isinstance(raw, str) else raw
                except (json.JSONDecodeError, TypeError):
                    pty_data = {"raw": raw}
                pty_data["session_id"] = active_session_id
                rounds_used += 1
    else:
        if task_input:
            start_args["input"] = task_input + "\r"
        result = await runtime.execute_tool("pty_interact", start_args)
        raw = result.get("output", "")
        try:
            pty_data = json.loads(raw) if isinstance(raw, str) else raw
        except (json.JSONDecodeError, TypeError):
            pty_data = {"raw": raw}
        rounds_used += 1

    active_session_id = str(pty_data.get("session_id") or active_session_id or "").strip() or None
    verification_exists = verification_path.exists() if verification_path else False
    if not advisory:
        advisory = supervision_advisory(
            pty_data,
            verification_exists=verification_exists,
        )

    for _ in range(max_rounds - 1):
        if verification_exists or not active_session_id or pty_data.get("running") is False:
            break
        if advisory.get("action") == "stop":
            break

        observe_idle_seconds = min(
            idle_seconds,
            float(advisory.get("observe_idle_seconds", idle_seconds)),
        )
        observe_max_wait_seconds = min(
            continue_wait,
            float(advisory.get("observe_max_wait_seconds", continue_wait)),
        )
        observe = await runtime.execute_tool(
            "pty_observe",
            {
                "session_id": active_session_id,
                "scope_key": scope_key,
                "idle_seconds": observe_idle_seconds,
                "max_wait_seconds": observe_max_wait_seconds,
            },
        )
        observe_raw = observe.get("output", "")
        try:
            pty_data = json.loads(observe_raw) if isinstance(observe_raw, str) else observe_raw
        except (json.JSONDecodeError, TypeError):
            pty_data = {"raw": observe_raw}
        pty_data["session_id"] = active_session_id
        rounds_used += 1
        verification_exists = verification_path.exists() if verification_path else False
        advisory = supervision_advisory(
            pty_data,
            verification_exists=verification_exists,
        )

        app = str((pty_data.get("screen_state", {}) or {}).get("app", "")).strip().lower()
        if (
            verification_path is not None
            and task_input
            and not verification_exists
            and not follow_up_attempted
            and app in {"kilocode", "kilo"}
            and advisory.get("reason") == "interactive_idle_ready_for_input"
        ):
            follow_up = await runtime.execute_tool(
                "pty_interact",
                {
                    "session_id": active_session_id,
                    "scope_key": scope_key,
                    "idle_seconds": idle_seconds,
                    "max_wait_seconds": max_wait,
                    "input": "\r",
                },
            )
            follow_up_raw = follow_up.get("output", "")
            try:
                pty_data = json.loads(follow_up_raw) if isinstance(follow_up_raw, str) else follow_up_raw
            except (json.JSONDecodeError, TypeError):
                pty_data = {"raw": follow_up_raw}
            pty_data["session_id"] = active_session_id
            rounds_used += 1
            follow_up_attempted = True
            verification_exists = verification_path.exists()
            advisory = supervision_advisory(
                pty_data,
                verification_exists=verification_exists,
            )

    return ToolResult(
        True,
        json.dumps(
            {
                "session_id": pty_data.get("session_id"),
                "running": pty_data.get("running"),
                "cleaned_output": pty_data.get("cleaned_combined_output", ""),
                "screen_state": pty_data.get("screen_state", {}),
                "supervision_advisory": advisory,
                "idle_reached": pty_data.get("idle_reached", False),
                "timed_out": pty_data.get("timed_out", False),
                "elapsed_ms": pty_data.get("elapsed_ms"),
                "verification_path": str(verification_path) if verification_path else None,
                "verification_exists": verification_exists,
                "rounds": max_rounds,
                "rounds_used": rounds_used,
            }
        ),
        {"scope_key": scope_key},
    )
