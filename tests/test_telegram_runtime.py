"""Tests for runtime Telegram helper seams."""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import pytest

from opencas.runtime.telegram_runtime import (
    approve_runtime_telegram_pairing,
    build_runtime_telegram_service,
    configure_runtime_telegram,
    get_runtime_telegram_status,
    initialize_runtime_telegram,
    runtime_telegram_settings,
    start_runtime_telegram,
)
from opencas.telegram_config import TelegramRuntimeConfig


class _FakeTelegramService:
    def __init__(self, **kwargs):
        self.kwargs = kwargs
        self.started = False
        self.stopped = False
        self.approved_code = None
        self.status_payload = {"running": True, "bot": {"username": "opencas_bot"}}
        self.approve_result = object()

    async def start(self) -> None:
        self.started = True

    async def stop(self) -> None:
        self.stopped = True

    async def status(self):
        return dict(self.status_payload)

    async def approve_pairing(self, code: str):
        self.approved_code = code
        return self.approve_result


class _FailingTelegramService(_FakeTelegramService):
    async def start(self) -> None:
        raise RuntimeError("boom")


class _Runtime(SimpleNamespace):
    def __init__(self):
        super().__init__()
        self.ctx = SimpleNamespace(config=SimpleNamespace(state_dir=Path("/tmp/opencas-test")))
        self.tracer = object()
        self._telegram_config = TelegramRuntimeConfig()
        self._telegram = None
        self.events = []

    def _trace(self, event: str, payload: dict | None = None) -> None:
        self.events.append((event, payload or {}))


def test_initialize_runtime_telegram_loads_and_builds(monkeypatch: pytest.MonkeyPatch) -> None:
    runtime = _Runtime()
    loaded = TelegramRuntimeConfig(enabled=True, bot_token="abc123", allow_from=["42"])

    monkeypatch.setattr("opencas.runtime.telegram_runtime.load_telegram_runtime_config", lambda _state_dir: loaded)
    monkeypatch.setattr("opencas.runtime.telegram_runtime.TelegramBotService", _FakeTelegramService)

    initialize_runtime_telegram(runtime, runtime.ctx.config.state_dir)

    assert runtime_telegram_settings(runtime) == loaded
    assert isinstance(runtime._telegram, _FakeTelegramService)
    assert runtime._telegram.kwargs["allow_from"] == ["42"]


@pytest.mark.asyncio
async def test_get_runtime_telegram_status_returns_fallback_without_service() -> None:
    runtime = _Runtime()
    runtime._telegram_config = TelegramRuntimeConfig(enabled=False, bot_token=None, dm_policy="pairing")

    status = await get_runtime_telegram_status(runtime)

    assert status["running"] is False
    assert status["token_configured"] is False
    assert status["dm_policy"] == "pairing"


@pytest.mark.asyncio
async def test_configure_runtime_telegram_rebuilds_and_restarts(monkeypatch: pytest.MonkeyPatch) -> None:
    runtime = _Runtime()
    existing = _FakeTelegramService()
    runtime._telegram = existing
    saved = {}

    def _fake_save(state_dir, config):
        saved["state_dir"] = state_dir
        saved["config"] = config

    monkeypatch.setattr("opencas.runtime.telegram_runtime.save_telegram_runtime_config", _fake_save)
    monkeypatch.setattr("opencas.runtime.telegram_runtime.TelegramBotService", _FakeTelegramService)

    status = await configure_runtime_telegram(
        runtime,
        TelegramRuntimeConfig(enabled=True, bot_token="new-token", allow_from=["99"]),
    )

    assert existing.stopped is True
    assert saved["state_dir"] == runtime.ctx.config.state_dir
    assert saved["config"].bot_token == "new-token"
    assert isinstance(runtime._telegram, _FakeTelegramService)
    assert runtime._telegram.started is True
    assert status["saved"] is True
    assert status["provenance_events"][0]["event_type"] == "MUTATION"
    assert status["provenance_events"][0]["triggering_artifact"] == "setting|telegram|runtime"
    assert ("telegram_started", {}) in runtime.events


@pytest.mark.asyncio
async def test_start_runtime_telegram_traces_failure_without_raising() -> None:
    runtime = _Runtime()
    runtime._telegram = _FailingTelegramService()

    await start_runtime_telegram(runtime)

    assert runtime.events == [("telegram_start_failed", {"error": "boom"})]


@pytest.mark.asyncio
async def test_approve_runtime_telegram_pairing_returns_bool() -> None:
    runtime = _Runtime()
    service = _FakeTelegramService()
    runtime._telegram = service

    approved = await approve_runtime_telegram_pairing(runtime, "PAIR1234")

    assert approved is True
    assert service.approved_code == "PAIR1234"


def test_build_runtime_telegram_service_returns_none_when_disabled(monkeypatch: pytest.MonkeyPatch) -> None:
    runtime = _Runtime()
    monkeypatch.setattr("opencas.runtime.telegram_runtime.TelegramBotService", _FakeTelegramService)

    assert build_runtime_telegram_service(runtime) is None
