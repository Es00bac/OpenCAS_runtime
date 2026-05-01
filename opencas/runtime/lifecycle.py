"""Lifecycle orchestration helpers for AgentRuntime."""

from __future__ import annotations

import asyncio
from typing import TYPE_CHECKING

import uvicorn

from opencas.api.server import create_app

from .continuity_breadcrumbs import current_runtime_focus, record_burst_continuity
from .conversation_recovery import recover_interrupted_conversation_turns
from .provenance_hooks import emit_runtime_session_lifecycle
from .scheduler import AgentScheduler

if TYPE_CHECKING:
    from .agent_loop import AgentRuntime


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
    await scheduler.stop()
    runtime.scheduler = None
    await shutdown_runtime_resources(runtime)
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
    app = create_app(runtime)
    config = uvicorn.Config(app, host=host, port=port, log_level="info")
    server = uvicorn.Server(config)
    shutdown_event = asyncio.Event()
    install_runtime_signal_handlers(runtime, shutdown_event)

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
    await server_task
    await scheduler.stop()
    runtime.scheduler = None
    await shutdown_runtime_resources(runtime)
    runtime._trace("autonomous_with_server_shutdown", {})
