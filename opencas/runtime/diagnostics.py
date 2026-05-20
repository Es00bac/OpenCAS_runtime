"""Runtime diagnostics for live OpenCAS responsiveness failures."""

from __future__ import annotations

import asyncio
import faulthandler
import json
import signal
import threading
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, TextIO


DEFAULT_STACK_DUMP_NAME = "runtime_stack_dump.log"


@dataclass
class RuntimeDiagnosticsHandle:
    """Owned resources for runtime responsiveness diagnostics."""

    stack_dump_path: Path
    heartbeat_task: asyncio.Task[None]
    watchdog_stop: threading.Event
    watchdog_thread: threading.Thread
    loop: asyncio.AbstractEventLoop
    signal_file: TextIO | None = None


def _runtime_state_dir(runtime: Any) -> Path:
    config = getattr(getattr(runtime, "ctx", None), "config", None)
    state_dir = getattr(config, "state_dir", None)
    return Path(state_dir or ".opencas")


def _stack_dump_path(runtime: Any) -> Path:
    return _runtime_state_dir(runtime) / DEFAULT_STACK_DUMP_NAME


def _trace(runtime: Any, event: str, payload: dict[str, Any]) -> None:
    tracer = getattr(runtime, "_trace", None)
    if not callable(tracer):
        return
    try:
        tracer(event, payload)
    except Exception:
        return


def _queue_trace(handle: RuntimeDiagnosticsHandle, runtime: Any, event: str, payload: dict[str, Any]) -> None:
    try:
        handle.loop.call_soon_threadsafe(_trace, runtime, event, payload)
    except Exception:
        _trace(runtime, event, payload)


def _write_stack_dump(path: Path, *, reason: str, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now(timezone.utc).isoformat()
    with path.open("a", encoding="utf-8") as handle:
        handle.write(f"\n=== OpenCAS runtime stack dump {timestamp} ===\n")
        handle.write(f"reason={reason}\n")
        handle.write(json.dumps(payload, sort_keys=True, default=str))
        handle.write("\n")
        handle.flush()
        faulthandler.dump_traceback(file=handle, all_threads=True)
        handle.write("=== end OpenCAS runtime stack dump ===\n")
        handle.flush()


def _install_signal_dump(runtime: Any, path: Path) -> TextIO | None:
    signum = getattr(signal, "SIGUSR1", None)
    if signum is None:
        return None
    path.parent.mkdir(parents=True, exist_ok=True)
    signal_file = path.open("a", encoding="utf-8")
    try:
        faulthandler.register(signum, file=signal_file, all_threads=True, chain=False)
    except Exception:
        try:
            signal_file.close()
        except Exception:
            pass
        return None
    _trace(
        runtime,
        "runtime_stack_dump_signal_installed",
        {"signal": "SIGUSR1", "path": str(path)},
    )
    return signal_file


async def _heartbeat_loop(
    heartbeat: dict[str, float],
    stop_event: threading.Event,
    interval_seconds: float,
) -> None:
    try:
        while not stop_event.is_set():
            heartbeat["last"] = time.monotonic()
            await asyncio.sleep(interval_seconds)
    except asyncio.CancelledError:
        raise


def start_runtime_diagnostics(
    runtime: Any,
    *,
    heartbeat_interval_seconds: float = 1.0,
    lag_threshold_seconds: float = 2.0,
    dump_cooldown_seconds: float = 30.0,
    install_signal: bool = True,
) -> RuntimeDiagnosticsHandle:
    """Start low-overhead diagnostics for event-loop stalls and stack capture."""

    existing = getattr(runtime, "_runtime_diagnostics_handle", None)
    if isinstance(existing, RuntimeDiagnosticsHandle):
        return existing

    path = _stack_dump_path(runtime)
    path.parent.mkdir(parents=True, exist_ok=True)
    loop = asyncio.get_running_loop()
    stop_event = threading.Event()
    heartbeat = {"last": time.monotonic()}
    heartbeat_task = asyncio.create_task(
        _heartbeat_loop(heartbeat, stop_event, heartbeat_interval_seconds),
        name="opencas-runtime-diagnostics-heartbeat",
    )

    handle = RuntimeDiagnosticsHandle(
        stack_dump_path=path,
        heartbeat_task=heartbeat_task,
        watchdog_stop=stop_event,
        watchdog_thread=threading.Thread(),
        loop=loop,
        signal_file=_install_signal_dump(runtime, path) if install_signal else None,
    )

    def _watchdog() -> None:
        last_dump_at = 0.0
        while not stop_event.wait(heartbeat_interval_seconds):
            now = time.monotonic()
            heartbeat_silence = now - float(heartbeat.get("last", now))
            lag_seconds = heartbeat_silence - heartbeat_interval_seconds
            if lag_seconds < lag_threshold_seconds:
                continue
            if now - last_dump_at < dump_cooldown_seconds:
                continue
            payload = {
                "lag_seconds": round(lag_seconds, 3),
                "heartbeat_silence_seconds": round(heartbeat_silence, 3),
                "threshold_seconds": lag_threshold_seconds,
                "heartbeat_interval_seconds": heartbeat_interval_seconds,
            }
            try:
                _write_stack_dump(path, reason="event_loop_lag", payload=payload)
            except Exception as exc:
                payload["dump_error"] = str(exc)
            _queue_trace(handle, runtime, "runtime_event_loop_lag", payload)
            last_dump_at = now

    handle.watchdog_thread = threading.Thread(
        target=_watchdog,
        name="opencas-runtime-watchdog",
        daemon=True,
    )
    handle.watchdog_thread.start()
    runtime._runtime_diagnostics_handle = handle
    _trace(
        runtime,
        "runtime_diagnostics_started",
        {
            "stack_dump_path": str(path),
            "heartbeat_interval_seconds": heartbeat_interval_seconds,
            "lag_threshold_seconds": lag_threshold_seconds,
            "dump_cooldown_seconds": dump_cooldown_seconds,
            "signal_installed": handle.signal_file is not None,
        },
    )
    return handle


async def stop_runtime_diagnostics(
    runtime: Any,
    handle: RuntimeDiagnosticsHandle | None = None,
    *,
    join_timeout_seconds: float = 2.0,
) -> None:
    """Stop runtime diagnostics and release signal/file handles."""

    handle = handle or getattr(runtime, "_runtime_diagnostics_handle", None)
    if not isinstance(handle, RuntimeDiagnosticsHandle):
        return
    handle.watchdog_stop.set()
    handle.heartbeat_task.cancel()
    try:
        await asyncio.gather(handle.heartbeat_task, return_exceptions=True)
    except Exception:
        pass
    try:
        await asyncio.to_thread(handle.watchdog_thread.join, join_timeout_seconds)
    except Exception:
        pass
    if handle.signal_file is not None:
        signum = getattr(signal, "SIGUSR1", None)
        if signum is not None:
            try:
                faulthandler.unregister(signum)
            except Exception:
                pass
        try:
            handle.signal_file.close()
        except Exception:
            pass
    if getattr(runtime, "_runtime_diagnostics_handle", None) is handle:
        runtime._runtime_diagnostics_handle = None
    _trace(runtime, "runtime_diagnostics_stopped", {"stack_dump_path": str(handle.stack_dump_path)})
