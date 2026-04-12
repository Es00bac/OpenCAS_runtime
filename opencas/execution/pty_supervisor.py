"""PTY-backed supervisor for terminal-native interactive sessions."""

from __future__ import annotations

import errno
import fcntl
import os
import pty
import re
import signal
import struct
import subprocess
import termios
import threading
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Optional
from uuid import uuid4

# Matches ANSI escape sequences: CSI (ESC[...), OSC (ESC]...), DCS, and bare ESC codes.
_ANSI_RE = re.compile(
    r"\x1b"           # ESC character
    r"(?:"
    r"\[[0-9;?]*[ -/]*[A-Za-z]"   # CSI sequences: ESC [ params intermediates letter
    r"|\[[><=][0-9;]*[A-Za-z]"    # DEC private CSI: ESC [ > / = / < params letter
    r"|\][^\x07\x1b]*(?:\x07|\x1b\\)"  # OSC sequences: ESC ] ... BEL/ST
    r"|P[^\x1b]*\x1b\\"           # DCS sequences: ESC P ... ST
    r"|\([0-9A-Za-z]"             # Character set selection
    r"|[>=<78DEHM]"               # Single-char ESC codes
    r")"
)


def strip_ansi(text: str) -> str:
    """Remove ANSI escape sequences from terminal output."""
    return _ANSI_RE.sub("", text)


_SHELL_PROMPT_RE = re.compile(r"(?:^|\n)[^\n]*[#$%>] ?$")


def _classify_command(command: str) -> str:
    if not command.strip():
        return "unknown"
    head = command.strip().split()[0]
    return Path(head).name.lower()


def infer_screen_state(command: str, cleaned_text: str, *, running: bool) -> Dict[str, Any]:
    """Infer a coarse PTY screen state from command context and cleaned terminal text."""
    app = _classify_command(command)
    lowered = cleaned_text.lower()
    normalized = re.sub(r"\s+", " ", lowered.replace("\b", "").replace("\r", "\n")).strip()
    indicators: list[str] = []
    mode = "idle"
    ready_for_input = False
    needs_input = False
    blocked = False

    auth_markers = ("sign in", "login", "log in", "authenticate", "api key", "token")
    if any(marker in normalized for marker in auth_markers):
        indicators.append("auth_required")
        mode = "auth_required"
        needs_input = True
        blocked = True
    elif "e212: can't open file for writing" in normalized or (
        app in {"vim", "nvim"} and "press enter or type command to continue" in normalized
    ):
        indicators.append("vim_write_error")
        mode = "error_prompt"
        ready_for_input = True
        needs_input = True
    elif "-- insert --" in normalized:
        indicators.append("vim_insert")
        mode = "insert"
        ready_for_input = True
        needs_input = True
    elif _SHELL_PROMPT_RE.search(cleaned_text.rstrip()):
        indicators.append("shell_prompt")
        mode = "shell_prompt"
        ready_for_input = True
        needs_input = True
    elif running:
        mode = "interactive"
        ready_for_input = True
        needs_input = True

    full_screen_apps = {"vim", "nvim", "less", "nano", "htop", "btop", "mc", "claude", "codex", "kilocode", "kilo"}
    if app in full_screen_apps:
        indicators.append("full_screen_tui")
    if app in {"vim", "nvim"}:
        indicators.append("editor")
        if mode == "interactive":
            mode = "normal"
    elif app in {"claude", "codex", "kilocode", "kilo"}:
        indicators.append("ai_tui")
    elif app in {"bash", "zsh", "sh", "fish"} and "shell_prompt" not in indicators:
        indicators.append("shell")

    if not running:
        indicators.append("process_exited")
        needs_input = False
        if mode == "idle":
            mode = "exited"

    return {
        "app": app,
        "mode": mode,
        "ready_for_input": ready_for_input,
        "needs_input": needs_input,
        "blocked": blocked,
        "indicators": indicators,
    }


@dataclass
class _ManagedPtySession:
    process: subprocess.Popen[Any]
    master_fd: int
    scope_key: str
    command: str
    cwd: str
    created_at: float = field(default_factory=time.time)
    rows: int = 24
    cols: int = 80
    last_screen_state: Dict[str, Any] = field(default_factory=dict)
    last_cleaned_output: str = ""
    last_observed_at: Optional[float] = None


class PtySupervisor:
    """Manages PTY sessions for interactive terminal workflows."""

    def __init__(self) -> None:
        self._sessions: Dict[str, _ManagedPtySession] = {}
        self._lock = threading.Lock()

    def start(
        self,
        scope_key: str,
        command: str,
        cwd: Optional[str] = None,
        rows: int = 24,
        cols: int = 80,
    ) -> str:
        """Start a PTY-backed session and return its session id."""
        session_id = str(uuid4())
        resolved_cwd = cwd or os.getcwd()
        master_fd, slave_fd = pty.openpty()
        self._set_winsize(slave_fd, rows=rows, cols=cols)
        self._set_nonblocking(master_fd)
        try:
            proc = subprocess.Popen(
                ["/bin/bash", "-lc", command],
                cwd=resolved_cwd,
                stdin=slave_fd,
                stdout=slave_fd,
                stderr=slave_fd,
                text=False,
                close_fds=True,
                start_new_session=True,
            )
        except Exception:
            os.close(master_fd)
            os.close(slave_fd)
            raise
        finally:
            try:
                os.close(slave_fd)
            except OSError:
                pass

        session = _ManagedPtySession(
            process=proc,
            master_fd=master_fd,
            scope_key=scope_key,
            command=command,
            cwd=resolved_cwd,
            rows=rows,
            cols=cols,
        )
        with self._lock:
            self._sessions[session_id] = session
        return session_id

    def poll(
        self,
        scope_key: str,
        session_id: str,
        max_bytes: int = 8192,
    ) -> Dict[str, Any]:
        """Return current session state and any available PTY output."""
        with self._lock:
            session = self._sessions.get(session_id)
        if session is None or session.scope_key != scope_key:
            return {"found": False, "error": "PTY session not found"}

        returncode = session.process.poll()
        output = self._drain_output(session.master_fd, max_bytes=max_bytes)
        cleaned = strip_ansi(output)
        screen_state = infer_screen_state(
            session.command,
            cleaned,
            running=returncode is None,
        )
        self._record_observation(session_id, cleaned, screen_state)
        return {
            "found": True,
            "pid": session.process.pid,
            "command": session.command,
            "running": returncode is None,
            "returncode": returncode,
            "output": output,
            "cleaned_output": cleaned,
            "screen_state": screen_state,
            "rows": session.rows,
            "cols": session.cols,
        }

    def observe_until_quiet(
        self,
        scope_key: str,
        session_id: str,
        *,
        idle_seconds: float = 0.4,
        max_wait_seconds: float = 8.0,
        max_bytes_per_poll: int = 4096,
    ) -> Dict[str, Any]:
        """Adaptively poll a PTY session until it goes quiet or exits."""
        started_at = time.time()
        output_chunks: list[str] = []
        last_activity_at = started_at
        sleep_seconds = 0.05
        iterations = 0
        seen_output = False

        while time.time() - started_at < max_wait_seconds:
            iterations += 1
            snapshot = self.poll(
                scope_key,
                session_id,
                max_bytes=max_bytes_per_poll,
            )
            if not snapshot.get("found", False):
                snapshot["elapsed_ms"] = int((time.time() - started_at) * 1000)
                snapshot["iterations"] = iterations
                combined = "".join(output_chunks)
                snapshot["combined_output"] = combined
                cleaned = strip_ansi(combined)
                screen_state = infer_screen_state(
                    snapshot.get("command", ""),
                    cleaned,
                    running=bool(snapshot.get("running")),
                )
                snapshot["cleaned_combined_output"] = cleaned
                snapshot["screen_state"] = screen_state
                self._record_observation(session_id, cleaned, screen_state)
                return snapshot

            chunk = snapshot.get("output", "")
            if chunk:
                output_chunks.append(chunk)
                last_activity_at = time.time()
                seen_output = True
                sleep_seconds = 0.05
            else:
                sleep_seconds = min(0.75, sleep_seconds * 1.6)

            if snapshot.get("running") is False:
                snapshot["elapsed_ms"] = int((time.time() - started_at) * 1000)
                snapshot["iterations"] = iterations
                combined = "".join(output_chunks)
                snapshot["combined_output"] = combined
                cleaned = strip_ansi(combined)
                screen_state = infer_screen_state(
                    snapshot.get("command", ""),
                    cleaned,
                    running=bool(snapshot.get("running")),
                )
                snapshot["cleaned_combined_output"] = cleaned
                snapshot["screen_state"] = screen_state
                self._record_observation(session_id, cleaned, screen_state)
                return snapshot

            if seen_output and (time.time() - last_activity_at) >= idle_seconds:
                snapshot["elapsed_ms"] = int((time.time() - started_at) * 1000)
                snapshot["iterations"] = iterations
                combined = "".join(output_chunks)
                cleaned = strip_ansi(combined)
                screen_state = infer_screen_state(
                    snapshot.get("command", ""),
                    cleaned,
                    running=bool(snapshot.get("running")),
                )
                snapshot["combined_output"] = combined
                snapshot["cleaned_combined_output"] = cleaned
                snapshot["screen_state"] = screen_state
                self._record_observation(session_id, cleaned, screen_state)
                snapshot["idle_reached"] = True
                return snapshot

            time.sleep(sleep_seconds)

        snapshot = self.poll(scope_key, session_id, max_bytes=max_bytes_per_poll)
        snapshot["elapsed_ms"] = int((time.time() - started_at) * 1000)
        snapshot["iterations"] = iterations
        combined = "".join(output_chunks) + snapshot.get("output", "")
        cleaned = strip_ansi(combined)
        screen_state = infer_screen_state(
            snapshot.get("command", ""),
            cleaned,
            running=bool(snapshot.get("running")),
        )
        snapshot["combined_output"] = combined
        snapshot["cleaned_combined_output"] = cleaned
        snapshot["screen_state"] = screen_state
        self._record_observation(session_id, cleaned, screen_state)
        snapshot["timed_out"] = True
        return snapshot

    def interact(
        self,
        scope_key: str,
        *,
        session_id: Optional[str] = None,
        command: Optional[str] = None,
        cwd: Optional[str] = None,
        rows: int = 24,
        cols: int = 80,
        input_text: Optional[str] = None,
        idle_seconds: float = 0.4,
        max_wait_seconds: float = 8.0,
        max_bytes_per_poll: int = 4096,
    ) -> Dict[str, Any]:
        """Start or continue a PTY session, optionally write input, then observe it."""
        started = False
        created_session_id = session_id

        if not created_session_id:
            if not command:
                return {
                    "found": False,
                    "error": "Either session_id or command is required",
                }
            created_session_id = self.start(
                scope_key,
                command,
                cwd=cwd,
                rows=rows,
                cols=cols,
            )
            started = True

        if input_text:
            wrote = self.write(scope_key, created_session_id, input_text)
            if not wrote:
                return {
                    "found": False,
                    "session_id": created_session_id,
                    "error": "Failed to write to PTY session",
                }

        observed = self.observe_until_quiet(
            scope_key,
            created_session_id,
            idle_seconds=idle_seconds,
            max_wait_seconds=max_wait_seconds,
            max_bytes_per_poll=max_bytes_per_poll,
        )
        observed["session_id"] = created_session_id
        observed["started"] = started
        return observed

    def write(self, scope_key: str, session_id: str, input_text: str) -> bool:
        """Write input to the PTY session."""
        with self._lock:
            session = self._sessions.get(session_id)
        if session is None or session.scope_key != scope_key:
            return False
        try:
            os.write(session.master_fd, input_text.encode("utf-8"))
            return True
        except OSError:
            return False

    def resize(self, scope_key: str, session_id: str, rows: int, cols: int) -> bool:
        """Resize the PTY window."""
        with self._lock:
            session = self._sessions.get(session_id)
        if session is None or session.scope_key != scope_key:
            return False
        try:
            self._set_winsize(session.master_fd, rows=rows, cols=cols)
            session.rows = rows
            session.cols = cols
            return True
        except OSError:
            return False

    def kill(self, scope_key: str, session_id: str) -> bool:
        """Terminate the PTY session process."""
        with self._lock:
            session = self._sessions.get(session_id)
        if session is None or session.scope_key != scope_key:
            return False
        try:
            if session.process.poll() is None:
                os.killpg(session.process.pid, signal.SIGKILL)
            return True
        except OSError:
            return False

    def remove(self, scope_key: str, session_id: str) -> bool:
        """Remove a PTY session from tracking, killing it first if needed."""
        with self._lock:
            session = self._sessions.get(session_id)
        if session is None or session.scope_key != scope_key:
            return False
        try:
            if session.process.poll() is None:
                os.killpg(session.process.pid, signal.SIGKILL)
        except OSError:
            pass
        try:
            os.close(session.master_fd)
        except OSError:
            pass
        with self._lock:
            self._sessions.pop(session_id, None)
        return True

    def clear(self, scope_key: str) -> int:
        """Kill and remove all PTY sessions in the given scope."""
        with self._lock:
            session_ids = [
                session_id
                for session_id, session in self._sessions.items()
                if session.scope_key == scope_key
            ]
        removed = 0
        for session_id in session_ids:
            if self.remove(scope_key, session_id):
                removed += 1
        return removed

    def clear_all(self) -> int:
        """Kill and remove all PTY sessions without disabling the supervisor."""
        with self._lock:
            session_items = [
                (session_id, session.scope_key)
                for session_id, session in self._sessions.items()
            ]
        removed = 0
        for session_id, scope_key in session_items:
            if self.remove(scope_key, session_id):
                removed += 1
        return removed

    def shutdown(self) -> None:
        """Terminate and forget all PTY sessions."""
        with self._lock:
            session_items = [
                (session_id, session.scope_key)
                for session_id, session in self._sessions.items()
            ]
        for session_id, scope_key in session_items:
            self.remove(scope_key, session_id)

    def snapshot(
        self,
        scope_key: Optional[str] = None,
        sample_limit: int = 10,
    ) -> Dict[str, Any]:
        """Return a summary of tracked PTY sessions for monitoring surfaces."""
        with self._lock:
            items = list(self._sessions.items())

        entries = []
        running_count = 0
        completed_count = 0
        scopes = set()
        filtered_items = []
        for session_id, session in items:
            if scope_key is not None and session.scope_key != scope_key:
                continue
            filtered_items.append((session_id, session))

        filtered_items.sort(key=lambda item: item[1].created_at, reverse=True)
        for session_id, session in filtered_items[:sample_limit]:
            returncode = session.process.poll()
            is_running = returncode is None
            if is_running:
                running_count += 1
            else:
                completed_count += 1
            scopes.add(session.scope_key)
            entries.append(
                {
                    "session_id": session_id,
                    "pid": session.process.pid,
                    "scope_key": session.scope_key,
                    "command": session.command,
                    "cwd": session.cwd,
                    "running": is_running,
                    "returncode": returncode,
                    "rows": session.rows,
                    "cols": session.cols,
                    "created_at": session.created_at,
                    "last_screen_state": session.last_screen_state,
                    "last_cleaned_output": session.last_cleaned_output,
                    "last_observed_at": session.last_observed_at,
                }
            )

        for _, session in filtered_items[sample_limit:]:
            returncode = session.process.poll()
            if returncode is None:
                running_count += 1
            else:
                completed_count += 1
            scopes.add(session.scope_key)

        return {
            "total_count": len(filtered_items),
            "running_count": running_count,
            "completed_count": completed_count,
            "scope_count": len(scopes),
            "entries": entries,
        }

    def _record_observation(
        self,
        session_id: str,
        cleaned_text: str,
        screen_state: Dict[str, Any],
    ) -> None:
        with self._lock:
            session = self._sessions.get(session_id)
            if session is None:
                return
            session.last_screen_state = dict(screen_state)
            if cleaned_text:
                session.last_cleaned_output = cleaned_text[-400:]
            session.last_observed_at = time.time()

    @staticmethod
    def _set_nonblocking(fd: int) -> None:
        flags = fcntl.fcntl(fd, fcntl.F_GETFL)
        fcntl.fcntl(fd, fcntl.F_SETFL, flags | os.O_NONBLOCK)

    @staticmethod
    def _set_winsize(fd: int, rows: int, cols: int) -> None:
        winsize = struct.pack("HHHH", rows, cols, 0, 0)
        fcntl.ioctl(fd, termios.TIOCSWINSZ, winsize)

    @staticmethod
    def _drain_output(master_fd: int, max_bytes: int) -> str:
        chunks: list[bytes] = []
        total = 0
        while total < max_bytes:
            try:
                chunk = os.read(master_fd, min(4096, max_bytes - total))
                if not chunk:
                    break
                chunks.append(chunk)
                total += len(chunk)
            except OSError as exc:
                if exc.errno in (errno.EAGAIN, errno.EWOULDBLOCK, errno.EIO):
                    break
                raise
        return b"".join(chunks).decode("utf-8", errors="replace")
