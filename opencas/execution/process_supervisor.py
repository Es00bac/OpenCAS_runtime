"""Process supervisor for managing background shell sessions."""

from __future__ import annotations

import logging
import os
import signal
import shlex
import subprocess
import threading
import time
from collections import deque
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional
from uuid import UUID, uuid4

logger = logging.getLogger(__name__)

_SHELL_META_CHARS = set("|&;<>()$`*?[]{}~\n")
_MAX_PROCESS_STREAM_LINES = 200
_MAX_PROCESS_STREAM_LINE_CHARS = 2048


@dataclass
class _ManagedProcess:
    process: subprocess.Popen[str]
    scope_key: str
    command: str
    cwd: str
    metadata: Dict[str, Any] = field(default_factory=dict)
    created_at: float = field(default_factory=time.time)
    stdout_buffer: deque[str] = field(
        default_factory=lambda: deque(maxlen=_MAX_PROCESS_STREAM_LINES),
    )
    stderr_buffer: deque[str] = field(
        default_factory=lambda: deque(maxlen=_MAX_PROCESS_STREAM_LINES),
    )
    last_stdout: str = ""
    last_stderr: str = ""
    last_polled_at: float = 0.0


class ProcessSupervisor:
    """Manages background shell processes with scope-key isolation."""

    def __init__(self) -> None:
        self._processes: Dict[str, _ManagedProcess] = {}
        self._lock = threading.Lock()
        self._reader_threads: List[threading.Thread] = []
        self._running = True

    def start(
        self,
        scope_key: str,
        command: str,
        cwd: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> str:
        """Start a new background process in the given scope."""
        process_id = str(uuid4())
        resolved_cwd = cwd or os.getcwd()
        try:
            prepared_command, use_shell = _prepare_subprocess_command(command)
            proc = subprocess.Popen(
                prepared_command,
                cwd=resolved_cwd,
                shell=use_shell,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                stdin=subprocess.PIPE,
                text=True,
            )
        except Exception as exc:
            logger.exception("Failed to start process")
            raise RuntimeError(f"Failed to start process: {exc}") from exc

        managed = _ManagedProcess(
            process=proc,
            scope_key=scope_key,
            command=command,
            cwd=resolved_cwd,
            metadata=dict(metadata or {}),
        )
        with self._lock:
            self._processes[process_id] = managed

        stdout_thread = threading.Thread(
            target=self._read_stream,
            args=(process_id, proc.stdout, "stdout"),
            daemon=True,
        )
        stderr_thread = threading.Thread(
            target=self._read_stream,
            args=(process_id, proc.stderr, "stderr"),
            daemon=True,
        )
        stdout_thread.start()
        stderr_thread.start()
        self._reader_threads.extend([stdout_thread, stderr_thread])

        return process_id

    def poll(self, scope_key: str, process_id: str) -> Dict[str, Any]:
        """Return the current status and recent output of a process."""
        with self._lock:
            managed = self._processes.get(process_id)
        if managed is None or managed.scope_key != scope_key:
            return {"found": False, "error": "Process not found"}

        proc = managed.process
        returncode = proc.poll()
        stdout = "".join(managed.stdout_buffer)
        stderr = "".join(managed.stderr_buffer)
        # Clear buffers after reading
        managed.stdout_buffer.clear()
        managed.stderr_buffer.clear()
        managed.last_stdout = stdout or managed.last_stdout
        managed.last_stderr = stderr or managed.last_stderr
        managed.last_polled_at = time.time()

        return {
            "found": True,
            "pid": proc.pid,
            "command": managed.command,
            "metadata": managed.metadata,
            "running": returncode is None,
            "returncode": returncode,
            "stdout": stdout,
            "stderr": stderr,
        }

    def write(self, scope_key: str, process_id: str, input_text: str) -> bool:
        """Write input to the process stdin."""
        with self._lock:
            managed = self._processes.get(process_id)
        if managed is None or managed.scope_key != scope_key:
            return False
        if managed.process.stdin is None:
            return False
        try:
            managed.process.stdin.write(input_text)
            managed.process.stdin.flush()
            return True
        except Exception:
            return False

    def send_signal(self, scope_key: str, process_id: str, signal_num: int) -> bool:
        """Send a POSIX signal to the process."""
        with self._lock:
            managed = self._processes.get(process_id)
        if managed is None or managed.scope_key != scope_key:
            return False
        try:
            managed.process.send_signal(signal_num)
            return True
        except Exception:
            return False

    def kill(self, scope_key: str, process_id: str) -> bool:
        """Forcefully terminate a process."""
        with self._lock:
            managed = self._processes.get(process_id)
        if managed is None or managed.scope_key != scope_key:
            return False
        try:
            managed.process.kill()
            return True
        except Exception:
            return False

    def clear(self, scope_key: str) -> int:
        """Kill and remove all processes in the given scope. Returns removed count."""
        to_remove: List[str] = []
        with self._lock:
            for pid, managed in self._processes.items():
                if managed.scope_key == scope_key:
                    to_remove.append(pid)
                    try:
                        managed.process.kill()
                    except Exception:
                        pass
            for pid in to_remove:
                self._processes.pop(pid, None)
        return len(to_remove)

    def clear_all(self) -> int:
        """Kill and remove all tracked processes without disabling the supervisor."""
        with self._lock:
            process_ids = list(self._processes.keys())
        removed = 0
        for process_id in process_ids:
            with self._lock:
                managed = self._processes.get(process_id)
            if managed is None:
                continue
            if self.remove(managed.scope_key, process_id):
                removed += 1
        return removed

    def remove(self, scope_key: str, process_id: str) -> bool:
        """Remove a process from tracking. Kills it first if still running."""
        with self._lock:
            managed = self._processes.get(process_id)
            if managed is None or managed.scope_key != scope_key:
                return False
            try:
                if managed.process.poll() is None:
                    managed.process.kill()
            except Exception:
                pass
            self._processes.pop(process_id, None)
        return True

    def snapshot(
        self,
        scope_key: Optional[str] = None,
        sample_limit: int = 10,
    ) -> Dict[str, Any]:
        """Return a summary of tracked processes for monitoring surfaces."""
        with self._lock:
            items = list(self._processes.items())

        entries = []
        running_count = 0
        completed_count = 0
        scopes = set()
        filtered_items = []
        for process_id, managed in items:
            if scope_key is not None and managed.scope_key != scope_key:
                continue
            filtered_items.append((process_id, managed))

        filtered_items.sort(key=lambda item: item[1].created_at, reverse=True)
        for process_id, managed in filtered_items[:sample_limit]:
            returncode = managed.process.poll()
            is_running = returncode is None
            if is_running:
                running_count += 1
            else:
                completed_count += 1
            scopes.add(managed.scope_key)
            entries.append(
                {
                    "process_id": process_id,
                    "pid": managed.process.pid,
                    "scope_key": managed.scope_key,
                    "command": managed.command,
                    "cwd": managed.cwd,
                    "metadata": managed.metadata,
                    "running": is_running,
                    "returncode": returncode,
                    "created_at": managed.created_at,
                    "stdout_preview": _preview_text("".join(managed.stdout_buffer) or managed.last_stdout),
                    "stderr_preview": _preview_text("".join(managed.stderr_buffer) or managed.last_stderr),
                    "last_polled_at": managed.last_polled_at or None,
                }
            )

        for _, managed in filtered_items[sample_limit:]:
            returncode = managed.process.poll()
            if returncode is None:
                running_count += 1
            else:
                completed_count += 1
            scopes.add(managed.scope_key)

        return {
            "total_count": len(filtered_items),
            "running_count": running_count,
            "completed_count": completed_count,
            "scope_count": len(scopes),
            "entries": entries,
        }

    def _read_stream(
        self,
        process_id: str,
        stream: Optional[Any],
        stream_name: str,
    ) -> None:
        """Background thread reading stdout/stderr lines into buffers."""
        if stream is None:
            return
        try:
            for line in iter(stream.readline, ""):
                if not self._running:
                    break
                with self._lock:
                    managed = self._processes.get(process_id)
                    if managed is None:
                        break
                    if stream_name == "stdout":
                        self._append_output_line(
                            managed.stdout_buffer,
                            line,
                            _MAX_PROCESS_STREAM_LINE_CHARS,
                        )
                    else:
                        self._append_output_line(
                            managed.stderr_buffer,
                            line,
                            _MAX_PROCESS_STREAM_LINE_CHARS,
                        )
        except Exception:
            pass
        finally:
            try:
                stream.close()
            except Exception:
                pass

    @staticmethod
    def _append_output_line(
        buffer: deque[str],
        line: str,
        max_line_chars: int,
    ) -> None:
        if max_line_chars > 0 and len(line) > max_line_chars:
            line = line[-max_line_chars:]
        buffer.append(line)

    def shutdown(self) -> None:
        """Kill all processes and stop reader threads."""
        self._running = False
        with self._lock:
            for managed in list(self._processes.values()):
                try:
                    managed.process.kill()
                except Exception:
                    pass
            self._processes.clear()


def _prepare_subprocess_command(command: str) -> tuple[str | list[str], bool]:
    """Use shell=False for ordinary commands, shell=True only when required."""
    if any(ch in command for ch in _SHELL_META_CHARS):
        return command, True
    try:
        return shlex.split(command), False
    except ValueError:
        return command, True


def _preview_text(text: str, limit: int = 240) -> str:
    text = text.strip()
    if len(text) <= limit:
        return text
    return text[-limit:]
