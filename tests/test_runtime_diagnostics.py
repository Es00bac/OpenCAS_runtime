"""Tests for live runtime diagnostic helpers."""

from __future__ import annotations

import asyncio
import time
from pathlib import Path
from types import SimpleNamespace

import pytest

from opencas.runtime.diagnostics import start_runtime_diagnostics, stop_runtime_diagnostics


class _FakeRuntime:
    def __init__(self, state_dir: Path) -> None:
        self.ctx = SimpleNamespace(config=SimpleNamespace(state_dir=state_dir))
        self.traces: list[tuple[str, dict]] = []

    def _trace(self, event: str, payload: dict | None = None) -> None:
        self.traces.append((event, payload or {}))


@pytest.mark.asyncio
async def test_runtime_diagnostics_watchdog_dumps_stack_when_loop_stalls(tmp_path: Path) -> None:
    runtime = _FakeRuntime(tmp_path)
    handle = start_runtime_diagnostics(
        runtime,
        heartbeat_interval_seconds=0.01,
        lag_threshold_seconds=0.025,
        dump_cooldown_seconds=0.0,
        install_signal=False,
    )
    try:
        await asyncio.sleep(0.03)
        time.sleep(0.08)
        await asyncio.sleep(0.05)
    finally:
        await stop_runtime_diagnostics(runtime, handle)

    dump_text = handle.stack_dump_path.read_text(encoding="utf-8")
    assert "event_loop_lag" in dump_text
    assert "Thread" in dump_text
    assert any(event == "runtime_event_loop_lag" for event, _ in runtime.traces)
