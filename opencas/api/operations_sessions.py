"""Shared process and PTY session helpers for operations routes."""

from __future__ import annotations

import time
from typing import Any, Callable, Dict, List, Optional, Tuple

from opencas.api.operations_models import ProcessDetailResponse, PtyInputRequest, SessionEntry, SessionListResponse, SessionScopeEntry


def build_session_entry(entry: Dict[str, Any]) -> SessionEntry:
    return SessionEntry(
        session_id=entry["session_id"],
        pid=entry.get("pid"),
        scope_key=entry.get("scope_key", ""),
        command=entry.get("command", ""),
        cwd=entry.get("cwd"),
        running=entry.get("running", False),
        returncode=entry.get("returncode"),
        rows=entry.get("rows"),
        cols=entry.get("cols"),
        created_at=entry.get("created_at"),
        last_observed_at=entry.get("last_observed_at"),
        last_screen_state=entry.get("last_screen_state", {}) or {},
        last_cleaned_output=entry.get("last_cleaned_output"),
    )


class SessionOperationsService:
    """Collect repeated process and PTY route behavior behind one helper seam."""

    def __init__(
        self,
        runtime: Any,
        *,
        find_rerun_history_by_request_id: Callable[[Any], Tuple[Optional[Dict[str, Any]], Optional[Dict[str, Any]]]],
        append_operator_action: Callable[[Any, Dict[str, Any]], Dict[str, Any]],
        load_recent_operator_actions: Callable[..., List[Dict[str, Any]]],
        truncate_text: Callable[[Optional[str]], str],
    ) -> None:
        self.runtime = runtime
        self._find_rerun_history_by_request_id = find_rerun_history_by_request_id
        self._append_operator_action = append_operator_action
        self._load_recent_operator_actions = load_recent_operator_actions
        self._truncate_text = truncate_text

    def list_sessions(self, *, scope_key: Optional[str] = None) -> SessionListResponse:
        process_entries: List[Dict[str, Any]] = []
        pty_entries: List[SessionEntry] = []
        browser_entries: List[Dict[str, Any]] = []
        scope_summary: Dict[str, Dict[str, int]] = {}
        total_processes = 0
        total_pty = 0
        total_browser = 0

        if hasattr(self.runtime, "process_supervisor"):
            p_snapshot = self.runtime.process_supervisor.snapshot(scope_key=scope_key)
            total_processes = p_snapshot.get("total_count", 0)
            process_entries = p_snapshot.get("entries", [])
            for entry in process_entries:
                entry_scope = str(entry.get("scope_key", "") or "default")
                scope_summary.setdefault(entry_scope, {"process_count": 0, "pty_count": 0, "browser_count": 0})["process_count"] += 1

        if hasattr(self.runtime, "pty_supervisor"):
            snapshot = self.runtime.pty_supervisor.snapshot(scope_key=scope_key)
            total_pty = snapshot.get("total_count", 0)
            for entry in snapshot.get("entries", []):
                entry_scope = str(entry.get("scope_key", "") or "default")
                scope_summary.setdefault(entry_scope, {"process_count": 0, "pty_count": 0, "browser_count": 0})["pty_count"] += 1
                pty_entries.append(build_session_entry(entry))

        if hasattr(self.runtime, "browser_supervisor"):
            b_snapshot = self.runtime.browser_supervisor.snapshot(scope_key=scope_key)
            total_browser = b_snapshot.get("total_count", 0)
            browser_entries = b_snapshot.get("entries", [])
            for entry in browser_entries:
                entry_scope = str(entry.get("scope_key", "") or "default")
                scope_summary.setdefault(entry_scope, {"process_count": 0, "pty_count": 0, "browser_count": 0})["browser_count"] += 1

        return SessionListResponse(
            processes=process_entries,
            pty=pty_entries,
            browser=browser_entries,
            scopes=[
                SessionScopeEntry(scope_key=scope, **counts)
                for scope, counts in sorted(scope_summary.items(), key=lambda item: item[0])
            ],
            current_scope=scope_key or None,
            total_processes=total_processes,
            total_pty=total_pty,
            total_browser=total_browser,
        )

    def get_process_session(
        self,
        *,
        process_id: str,
        scope_key: str = "default",
        refresh: bool = True,
    ) -> ProcessDetailResponse:
        if not hasattr(self.runtime, "process_supervisor"):
            return ProcessDetailResponse(found=False, process={"error": "Process supervisor not available"})
        snapshot = self.runtime.process_supervisor.snapshot(scope_key=scope_key)
        entry = next((item for item in snapshot.get("entries", []) if item.get("process_id") == process_id), None)
        if entry is None:
            return ProcessDetailResponse(found=False)
        polled = self.runtime.process_supervisor.poll(scope_key, process_id) if refresh else {}
        metadata = entry.get("metadata", {}) or {}
        rerun_request, rerun_completion = self._find_rerun_history_by_request_id(metadata.get("request_id"))
        return ProcessDetailResponse(
            found=True,
            process={
                **entry,
                "running": polled.get("running", entry.get("running", False)),
                "returncode": polled.get("returncode", entry.get("returncode")),
                "polled": polled,
                "rerun_request": rerun_request,
                "rerun_completion": rerun_completion,
                "recent_operator_actions": self._load_recent_operator_actions(
                    self.runtime,
                    target_kind="process",
                    target_id=process_id,
                    scope_key=scope_key,
                ),
            },
        )

    def kill_process_session(self, *, process_id: str, scope_key: str = "default") -> Dict[str, Any]:
        if not hasattr(self.runtime, "process_supervisor"):
            return {"ok": False, "error": "Process supervisor not available"}
        ok = self.runtime.process_supervisor.kill(scope_key, process_id)
        self.runtime.process_supervisor.remove(scope_key, process_id)
        self._append_operator_action(
            self.runtime,
            {
                "action": "kill_process",
                "target_kind": "process",
                "target_id": process_id,
                "scope_key": scope_key,
                "ok": bool(ok),
            },
        )
        return {"ok": ok, "process_id": process_id}

    def clear_process_sessions(self, *, scope_key: str = "default") -> Dict[str, Any]:
        if not hasattr(self.runtime, "process_supervisor"):
            return {"ok": False, "error": "Process supervisor not available"}
        removed = self.runtime.process_supervisor.clear(scope_key)
        return {"ok": True, "removed": removed, "scope_key": scope_key}

    def kill_pty_session(self, *, session_id: str, scope_key: str = "default") -> Dict[str, Any]:
        if not hasattr(self.runtime, "pty_supervisor"):
            return {"ok": False, "error": "PTY supervisor not available"}
        ok = self.runtime.pty_supervisor.kill(scope_key, session_id)
        self.runtime.pty_supervisor.remove(scope_key, session_id)
        return {"ok": ok, "session_id": session_id}

    def clear_pty_sessions(self, *, scope_key: str = "default") -> Dict[str, Any]:
        if not hasattr(self.runtime, "pty_supervisor"):
            return {"ok": False, "error": "PTY supervisor not available"}
        removed = self.runtime.pty_supervisor.clear(scope_key)
        return {"ok": True, "removed": removed, "scope_key": scope_key}

    def get_pty_session(
        self,
        *,
        session_id: str,
        scope_key: str = "default",
        refresh: bool = False,
        idle_seconds: float = 0.25,
        max_wait_seconds: float = 1.5,
    ) -> Dict[str, Any]:
        if not hasattr(self.runtime, "pty_supervisor"):
            return {"found": False, "error": "PTY supervisor not available"}

        snapshot = self.runtime.pty_supervisor.snapshot(scope_key=scope_key, sample_limit=500)
        entry = next((item for item in snapshot.get("entries", []) if item.get("session_id") == session_id), None)
        if entry is None:
            return {"found": False}

        observed = None
        if refresh and entry.get("running", False):
            observed = self.runtime.pty_supervisor.observe_until_quiet(
                scope_key,
                session_id,
                idle_seconds=idle_seconds,
                max_wait_seconds=max_wait_seconds,
            )
            entry = {
                **entry,
                "running": observed.get("running", entry.get("running", False)),
                "returncode": observed.get("returncode", entry.get("returncode")),
                "last_screen_state": observed.get("screen_state", entry.get("last_screen_state", {})),
                "last_cleaned_output": observed.get("cleaned_combined_output", entry.get("last_cleaned_output")),
                "last_observed_at": time.time() if observed.get("elapsed_ms") is not None else entry.get("last_observed_at"),
            }

        return {
            "found": True,
            "session": build_session_entry(entry).model_dump(mode="json"),
            "observed": observed,
            "recent_operator_actions": self._load_recent_operator_actions(
                self.runtime,
                target_kind="pty",
                target_id=session_id,
                scope_key=scope_key,
            ),
        }

    def send_pty_input(
        self,
        *,
        session_id: str,
        payload: PtyInputRequest,
        scope_key: str = "default",
    ) -> Dict[str, Any]:
        if not hasattr(self.runtime, "pty_supervisor"):
            return {"found": False, "error": "PTY supervisor not available"}

        snapshot = self.runtime.pty_supervisor.snapshot(scope_key=scope_key, sample_limit=500)
        entry = next((item for item in snapshot.get("entries", []) if item.get("session_id") == session_id), None)
        if entry is None:
            return {"found": False}

        ok = self.runtime.pty_supervisor.write(scope_key, session_id, payload.input)
        if not ok:
            return {"found": False, "error": "Failed to write PTY input"}

        observed = None
        if payload.observe:
            observed = self.runtime.pty_supervisor.observe_until_quiet(
                scope_key,
                session_id,
                idle_seconds=payload.idle_seconds,
                max_wait_seconds=payload.max_wait_seconds,
            )

        refreshed = self.runtime.pty_supervisor.snapshot(scope_key=scope_key, sample_limit=500)
        updated_entry = next(
            (item for item in refreshed.get("entries", []) if item.get("session_id") == session_id),
            entry,
        )
        self._append_operator_action(
            self.runtime,
            {
                "action": "pty_input",
                "target_kind": "pty",
                "target_id": session_id,
                "scope_key": scope_key,
                "ok": True,
                "input_length": len(payload.input or ""),
                "input_preview": self._truncate_text(payload.input),
                "observe": bool(payload.observe),
                "observed_mode": (observed or {}).get("screen_state", {}).get("mode"),
            },
        )
        return {
            "found": True,
            "ok": True,
            "session": build_session_entry(updated_entry).model_dump(mode="json"),
            "observed": observed,
            "recent_operator_actions": self._load_recent_operator_actions(
                self.runtime,
                target_kind="pty",
                target_id=session_id,
                scope_key=scope_key,
            ),
        }
