"""Lifecycle orchestration helpers for AgentRuntime."""

from __future__ import annotations

import asyncio
import time
from typing import TYPE_CHECKING, Callable

import uvicorn

from opencas.api.server import create_app

from .continuity_breadcrumbs import current_runtime_focus, record_burst_continuity
from .conversation_recovery import recover_interrupted_conversation_turns
from .diagnostics import start_runtime_diagnostics, stop_runtime_diagnostics
from .provenance_hooks import emit_runtime_session_lifecycle
from .scheduler import AgentScheduler

if TYPE_CHECKING:
    from .agent_loop import AgentRuntime


_SERVER_SHUTDOWN_TIMEOUT_SECONDS = 4.0
_SERVER_CANCEL_TIMEOUT_SECONDS = 2.0
_DIAGNOSTICS_SHUTDOWN_TIMEOUT_SECONDS = 3.0
_SCHEDULER_SHUTDOWN_TIMEOUT_SECONDS = 10.0
_RESOURCE_SHUTDOWN_TIMEOUT_SECONDS = 20.0


async def _await_shutdown_step(
    runtime: "AgentRuntime",
    label: str,
    awaitable: object,
    *,
    timeout_seconds: float,
) -> None:
    """Run one shutdown awaitable with a bounded wait and trace the outcome."""

    started = time.perf_counter()
    try:
        await asyncio.wait_for(awaitable, timeout=timeout_seconds)  # type: ignore[arg-type]
        runtime._trace(
            "shutdown_step_complete",
            {
                "step": label,
                "elapsed_ms": int((time.perf_counter() - started) * 1000),
            },
        )
    except TimeoutError:
        runtime._trace(
            "shutdown_step_timeout",
            {
                "step": label,
                "timeout_seconds": timeout_seconds,
                "elapsed_ms": int((time.perf_counter() - started) * 1000),
            },
        )
    except Exception as exc:
        runtime._trace(
            "shutdown_step_error",
            {
                "step": label,
                "error": str(exc),
                "elapsed_ms": int((time.perf_counter() - started) * 1000),
            },
        )


async def _await_shutdown_task(
    runtime: "AgentRuntime",
    label: str,
    task: asyncio.Task[object],
    *,
    timeout_seconds: float,
    cancel_timeout_seconds: float = _SERVER_CANCEL_TIMEOUT_SECONDS,
    on_timeout: Callable[[], None] | None = None,
) -> None:
    """Wait for a task during shutdown, cancelling it if it misses the timeout."""

    started = time.perf_counter()
    try:
        await asyncio.wait_for(task, timeout=timeout_seconds)
        runtime._trace(
            "shutdown_step_complete",
            {
                "step": label,
                "elapsed_ms": int((time.perf_counter() - started) * 1000),
            },
        )
    except TimeoutError:
        if on_timeout is not None:
            try:
                on_timeout()
            except Exception:
                pass
        task.cancel()
        cancel_completed = True
        try:
            await asyncio.wait_for(
                asyncio.gather(task, return_exceptions=True),
                timeout=cancel_timeout_seconds,
            )
        except TimeoutError:
            cancel_completed = False
        runtime._trace(
            "shutdown_step_timeout",
            {
                "step": label,
                "timeout_seconds": timeout_seconds,
                "elapsed_ms": int((time.perf_counter() - started) * 1000),
                "cancelled": True,
                "cancel_completed": cancel_completed,
            },
        )
    except Exception as exc:
        runtime._trace(
            "shutdown_step_error",
            {
                "step": label,
                "error": str(exc),
                "elapsed_ms": int((time.perf_counter() - started) * 1000),
            },
        )


def install_runtime_signal_handlers(
    runtime: "AgentRuntime",
    shutdown_event: asyncio.Event,
) -> None:
    """Install best-effort SIGINT/SIGTERM handlers for autonomous runtime modes."""

    def _on_signal(sig: int) -> None:
        runtime._trace("signal_received", {"signal": sig})
        shutdown_event.set()

    try:
        import signal

        loop = asyncio.get_running_loop()
        loop.add_signal_handler(signal.SIGINT, _on_signal, signal.SIGINT)
        loop.add_signal_handler(signal.SIGTERM, _on_signal, signal.SIGTERM)
    except (NotImplementedError, ValueError):
        pass


async def shutdown_runtime_resources(runtime: "AgentRuntime") -> None:
    """Close runtime-owned services and stores in the correct shutdown order."""
    try:
        active_focus = current_runtime_focus(runtime, "shutdown")
        if active_focus and active_focus != "shutdown":
            await record_burst_continuity(
                runtime,
                trigger="work_burst_interrupted",
                phase="interrupt",
                intent="Interrupted work burst during runtime shutdown",
                focus=active_focus,
                next_step="recover the interrupted burst before starting new work",
                note="shutdown interruption",
            )
        emit_runtime_session_lifecycle(
            runtime,
            transition="shutdown",
            reason="runtime shutdown persisted",
            note="shutdown interruption" if active_focus and active_focus != "shutdown" else None,
            entrypoint="shutdown_runtime_resources",
        )
    except Exception:
        runtime._trace("continuity_breadcrumb_shutdown_error", {})
    if runtime.reliability:
        runtime.reliability.stop()
    if getattr(runtime, "process_supervisor", None):
        runtime.process_supervisor.shutdown()
    if getattr(runtime, "pty_supervisor", None):
        runtime.pty_supervisor.shutdown()
    if getattr(runtime, "browser_supervisor", None):
        await runtime.browser_supervisor.shutdown()
    if runtime._telegram is not None:
        try:
            await runtime._telegram.stop()
        except Exception:
            pass
    await runtime.ctx.close()
    runtime.ctx.identity.record_shutdown()


async def run_autonomous_runtime(
    runtime: "AgentRuntime",
    *,
    cycle_interval: int = 600,
    daydream_interval: int = 720,
    baa_heartbeat_interval: int = 120,
    consolidation_interval: int = 86400,
) -> None:
    """Run the scheduler-only autonomous mode until a shutdown signal arrives."""
    if not runtime._instance_lock.acquire():
        print(
            f"Error: Another instance of OpenCAS is already running in {runtime.ctx.config.state_dir}"
        )
        return

    await runtime._continuity_check()
    recovered_turns = await recover_interrupted_conversation_turns(runtime)
    if recovered_turns:
        runtime._trace("conversation_recovery_complete", {"recovered_turns": recovered_turns})
    scheduler = AgentScheduler(
        runtime=runtime,
        cycle_interval=cycle_interval,
        consolidation_interval=consolidation_interval,
        baa_heartbeat_interval=baa_heartbeat_interval,
        daydream_interval=daydream_interval,
        readiness=runtime.readiness,
        tracer=runtime.tracer,
    )
    shutdown_event = asyncio.Event()
    install_runtime_signal_handlers(runtime, shutdown_event)
    diagnostics = start_runtime_diagnostics(runtime)

    runtime.scheduler = scheduler
    await scheduler.start()
    await runtime.start_telegram()
    runtime.readiness.ready("autonomous_mode_active")
    emit_runtime_session_lifecycle(
        runtime,
        transition="boot",
        reason="autonomous runtime started",
        note="scheduler-only autonomous runtime",
        entrypoint="run_autonomous_runtime",
    )
    runtime._trace("autonomous_start", {})

    await shutdown_event.wait()

    runtime.readiness.shutdown("signal_received")
    await _await_shutdown_step(
        runtime,
        "scheduler_stop",
        scheduler.stop(),
        timeout_seconds=_SCHEDULER_SHUTDOWN_TIMEOUT_SECONDS,
    )
    await _await_shutdown_step(
        runtime,
        "runtime_diagnostics",
        stop_runtime_diagnostics(runtime, diagnostics),
        timeout_seconds=_DIAGNOSTICS_SHUTDOWN_TIMEOUT_SECONDS,
    )
    runtime.scheduler = None
    await _await_shutdown_step(
        runtime,
        "runtime_resources",
        shutdown_runtime_resources(runtime),
        timeout_seconds=_RESOURCE_SHUTDOWN_TIMEOUT_SECONDS,
    )
    runtime._trace("autonomous_shutdown", {})


async def run_autonomous_with_server_runtime(
    runtime: "AgentRuntime",
    *,
    host: str = "127.0.0.1",
    port: int = 8080,
    cycle_interval: int = 600,
    daydream_interval: int = 720,
    baa_heartbeat_interval: int = 120,
    consolidation_interval: int = 86400,
) -> None:
    """Run the scheduler and FastAPI server together until a shutdown signal arrives."""
    if not runtime._instance_lock.acquire():
        print(
            f"Error: Another instance of OpenCAS is already running in {runtime.ctx.config.state_dir}"
        )
        return

    scheduler = AgentScheduler(
        runtime=runtime,
        cycle_interval=cycle_interval,
        consolidation_interval=consolidation_interval,
        baa_heartbeat_interval=baa_heartbeat_interval,
        daydream_interval=daydream_interval,
        readiness=runtime.readiness,
        tracer=runtime.tracer,
    )
    runtime.server_base_url = _local_server_base_url(host=host, port=port)
    app = create_app(runtime)
    config = uvicorn.Config(
        app,
        host=host,
        port=port,
        log_level="info",
        timeout_graceful_shutdown=int(_SERVER_CANCEL_TIMEOUT_SECONDS),
    )
    server = uvicorn.Server(config)
    shutdown_event = asyncio.Event()
    install_runtime_signal_handlers(runtime, shutdown_event)
    diagnostics = start_runtime_diagnostics(runtime)

    await runtime._continuity_check()
    recovered_turns = await recover_interrupted_conversation_turns(runtime)
    if recovered_turns:
        runtime._trace("conversation_recovery_complete", {"recovered_turns": recovered_turns})
    runtime.scheduler = scheduler
    await scheduler.start()
    await runtime.start_telegram()
    runtime.readiness.ready("autonomous_mode_with_server")
    emit_runtime_session_lifecycle(
        runtime,
        transition="boot",
        reason="autonomous runtime with server started",
        note="scheduler and server runtime",
        entrypoint="run_autonomous_with_server_runtime",
    )
    runtime._trace("autonomous_with_server_start", {"host": host, "port": port})

    server_task = asyncio.create_task(server.serve())
    await shutdown_event.wait()

    runtime.readiness.shutdown("signal_received")
    server.should_exit = True
    await _await_shutdown_task(
        runtime,
        "uvicorn_server",
        server_task,
        timeout_seconds=_SERVER_SHUTDOWN_TIMEOUT_SECONDS,
        on_timeout=lambda: setattr(server, "force_exit", True),
    )
    await _await_shutdown_step(
        runtime,
        "scheduler_stop",
        scheduler.stop(),
        timeout_seconds=_SCHEDULER_SHUTDOWN_TIMEOUT_SECONDS,
    )
    await _await_shutdown_step(
        runtime,
        "runtime_diagnostics",
        stop_runtime_diagnostics(runtime, diagnostics),
        timeout_seconds=_DIAGNOSTICS_SHUTDOWN_TIMEOUT_SECONDS,
    )
    runtime.scheduler = None
    await _await_shutdown_step(
        runtime,
        "runtime_resources",
        shutdown_runtime_resources(runtime),
        timeout_seconds=_RESOURCE_SHUTDOWN_TIMEOUT_SECONDS,
    )
    runtime._trace("autonomous_with_server_shutdown", {})


def _local_server_base_url(*, host: str, port: int) -> str:
    usable_host = "127.0.0.1" if host in {"0.0.0.0", "::"} else host
    return f"http://{usable_host}:{port}"
