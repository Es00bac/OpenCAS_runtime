"""Tests for runtime lifecycle helpers."""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import pytest

from opencas.runtime.lifecycle import (
    run_autonomous_runtime,
    run_autonomous_with_server_runtime,
    shutdown_runtime_resources,
)


class _ImmediateEvent:
    def set(self) -> None:
        return None

    async def wait(self) -> None:
        return None


class _FakeScheduler:
    def __init__(self, **kwargs):
        self.kwargs = kwargs
        self.started = False
        self.stopped = False

    async def start(self) -> None:
        self.started = True

    async def stop(self) -> None:
        self.stopped = True


class _FakeReadiness:
    def __init__(self) -> None:
        self.events: list[tuple[str, str]] = []

    def ready(self, reason: str) -> None:
        self.events.append(("ready", reason))

    def shutdown(self, reason: str) -> None:
        self.events.append(("shutdown", reason))


class _FakeRuntime:
    def __init__(self, *, lock_acquired: bool = True) -> None:
        self._instance_lock = SimpleNamespace(acquire=lambda: lock_acquired)
        self.ctx = SimpleNamespace(config=SimpleNamespace(state_dir=Path("/tmp/opencas-test")))
        self.readiness = _FakeReadiness()
        self.tracer = object()
        self.scheduler = None
        self._continuity_check_called = False
        self._telegram_started = False
        self._shutdown_called = False
        self.traces: list[tuple[str, dict]] = []
        self._telegram = None

    async def _continuity_check(self) -> None:
        self._continuity_check_called = True

    async def start_telegram(self) -> None:
        self._telegram_started = True

    def _trace(self, event: str, payload: dict | None = None) -> None:
        self.traces.append((event, payload or {}))


@pytest.mark.asyncio
async def test_shutdown_runtime_resources_stops_services_in_order() -> None:
    events: list[str] = []

    class _Reliability:
        def stop(self) -> None:
            events.append("reliability")

    class _ProcessSupervisor:
        def shutdown(self) -> None:
            events.append("process")

    class _PtySupervisor:
        def shutdown(self) -> None:
            events.append("pty")

    class _BrowserSupervisor:
        async def shutdown(self) -> None:
            events.append("browser")

    class _Telegram:
        async def stop(self) -> None:
            events.append("telegram")

    class _Identity:
        def record_shutdown(self) -> None:
            events.append("identity")

    class _Ctx:
        identity = _Identity()

        async def close(self) -> None:
            events.append("ctx_close")

    runtime = SimpleNamespace(
        reliability=_Reliability(),
        process_supervisor=_ProcessSupervisor(),
        pty_supervisor=_PtySupervisor(),
        browser_supervisor=_BrowserSupervisor(),
        _telegram=_Telegram(),
        ctx=_Ctx(),
    )

    await shutdown_runtime_resources(runtime)

    assert events == [
        "reliability",
        "process",
        "pty",
        "browser",
        "telegram",
        "ctx_close",
        "identity",
    ]


@pytest.mark.asyncio
async def test_run_autonomous_runtime_sequences_start_and_shutdown(monkeypatch: pytest.MonkeyPatch) -> None:
    runtime = _FakeRuntime()

    async def _fake_shutdown_runtime_resources(fake_runtime: _FakeRuntime) -> None:
        fake_runtime._shutdown_called = True

    monkeypatch.setattr("opencas.runtime.lifecycle.AgentScheduler", _FakeScheduler)
    monkeypatch.setattr("opencas.runtime.lifecycle.install_runtime_signal_handlers", lambda runtime, event: None)
    monkeypatch.setattr("opencas.runtime.lifecycle.asyncio.Event", _ImmediateEvent)
    monkeypatch.setattr("opencas.runtime.lifecycle.shutdown_runtime_resources", _fake_shutdown_runtime_resources)

    await run_autonomous_runtime(runtime, cycle_interval=12, consolidation_interval=34)

    assert runtime._continuity_check_called is True
    assert runtime._telegram_started is True
    assert runtime._shutdown_called is True
    assert runtime.scheduler is None
    assert ("ready", "autonomous_mode_active") in runtime.readiness.events
    assert ("shutdown", "signal_received") in runtime.readiness.events
    assert any(event == "autonomous_start" for event, _ in runtime.traces)
    assert any(event == "autonomous_shutdown" for event, _ in runtime.traces)


@pytest.mark.asyncio
async def test_run_autonomous_with_server_runtime_sequences_server_shutdown(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    runtime = _FakeRuntime()
    created_server: dict[str, object] = {}

    class _FakeConfig:
        def __init__(self, app, host, port, log_level):
            self.app = app
            self.host = host
            self.port = port
            self.log_level = log_level

    class _FakeServer:
        def __init__(self, config):
            self.config = config
            self.should_exit = False
            self.serve_called = False
            created_server["server"] = self

        async def serve(self) -> None:
            self.serve_called = True

    async def _fake_shutdown_runtime_resources(fake_runtime: _FakeRuntime) -> None:
        fake_runtime._shutdown_called = True

    monkeypatch.setattr("opencas.runtime.lifecycle.AgentScheduler", _FakeScheduler)
    monkeypatch.setattr("opencas.runtime.lifecycle.create_app", lambda runtime: {"app": "fake"})
    monkeypatch.setattr("opencas.runtime.lifecycle.uvicorn.Config", _FakeConfig)
    monkeypatch.setattr("opencas.runtime.lifecycle.uvicorn.Server", _FakeServer)
    monkeypatch.setattr("opencas.runtime.lifecycle.install_runtime_signal_handlers", lambda runtime, event: None)
    monkeypatch.setattr("opencas.runtime.lifecycle.asyncio.Event", _ImmediateEvent)
    monkeypatch.setattr("opencas.runtime.lifecycle.shutdown_runtime_resources", _fake_shutdown_runtime_resources)

    await run_autonomous_with_server_runtime(
        runtime,
        host="127.0.0.1",
        port=20020,
        cycle_interval=15,
        consolidation_interval=45,
    )

    server = created_server["server"]
    assert runtime._continuity_check_called is True
    assert runtime._telegram_started is True
    assert runtime._shutdown_called is True
    assert runtime.scheduler is None
    assert ("ready", "autonomous_mode_with_server") in runtime.readiness.events
    assert ("shutdown", "signal_received") in runtime.readiness.events
    assert any(event == "autonomous_with_server_start" for event, _ in runtime.traces)
    assert any(event == "autonomous_with_server_shutdown" for event, _ in runtime.traces)
    assert getattr(server, "serve_called", False) is True
    assert getattr(server, "should_exit", False) is True
