"""Tests for runtime phone helper seams."""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import pytest

from opencas.phone_config import PhoneRuntimeConfig
from opencas.runtime.phone_runtime import (
    autoconfigure_runtime_phone,
    build_runtime_phone_service,
    call_owner_via_runtime_phone,
    configure_runtime_phone,
    get_runtime_phone_status,
    initialize_runtime_phone,
    runtime_phone_settings,
)


class _FakePhoneService:
    def __init__(self, runtime, config):
        self.runtime = runtime
        self.config = config
        self.call_requests = []
        self.autoconfigure_requests = []

    def status(self):
        return {
            "enabled": self.config.enabled,
            "owner": {"phone_number": self.config.owner_phone_number},
            "twilio_credentials_configured": True,
        }

    async def place_owner_call(self, *, message: str, reason: str = ""):
        self.call_requests.append({"message": message, "reason": reason})
        return {
            "ok": True,
            "call_sid": "CA123",
            "status": "queued",
            "to": self.config.owner_phone_number,
        }

    async def autoconfigure_twilio(self, **kwargs):
        self.autoconfigure_requests.append(dict(kwargs))
        settings = PhoneRuntimeConfig(
            enabled=True if kwargs.get("enabled") is None else bool(kwargs["enabled"]),
            public_base_url=kwargs.get("public_base_url") or self.config.public_base_url,
            webhook_signature_required=(
                self.config.webhook_signature_required
                if kwargs.get("webhook_signature_required") is None
                else bool(kwargs["webhook_signature_required"])
            ),
            twilio_from_number="+15557654321",
            owner_phone_number=kwargs.get("owner_phone_number") or self.config.owner_phone_number,
            owner_display_name=kwargs.get("owner_display_name") or self.config.owner_display_name,
            owner_workspace_subdir=kwargs.get("owner_workspace_subdir") or self.config.owner_workspace_subdir,
            contacts=self.config.contacts,
        )
        return {
            "settings": settings,
            "selected_number": {"sid": "PN123", "phone_number": "+15557654321"},
            "twilio_number_candidates": [{"sid": "PN123", "phone_number": "+15557654321"}],
            "webhook_update": {
                "voice_url": f"{settings.public_base_url}/api/phone/twilio/voice",
                "voice_method": "POST",
            },
        }


class _Runtime(SimpleNamespace):
    def __init__(self):
        super().__init__()
        self.ctx = SimpleNamespace(config=SimpleNamespace(state_dir=Path("/tmp/opencas-phone-test")))
        self._phone_config = PhoneRuntimeConfig()
        self._phone = None
        self.events = []

    def _trace(self, event: str, payload: dict | None = None) -> None:
        self.events.append((event, payload or {}))


def test_initialize_runtime_phone_loads_and_builds(monkeypatch: pytest.MonkeyPatch) -> None:
    runtime = _Runtime()
    loaded = PhoneRuntimeConfig(enabled=True, owner_phone_number="+15551234567")

    monkeypatch.setattr("opencas.runtime.phone_runtime.load_phone_runtime_config", lambda _state_dir: loaded)
    monkeypatch.setattr("opencas.runtime.phone_runtime.PhoneBridgeService", _FakePhoneService)

    initialize_runtime_phone(runtime, runtime.ctx.config.state_dir)

    assert runtime_phone_settings(runtime) == loaded
    assert isinstance(runtime._phone, _FakePhoneService)
    assert runtime._phone.config.owner_phone_number == "+15551234567"


def test_build_runtime_phone_service_uses_current_config(monkeypatch: pytest.MonkeyPatch) -> None:
    runtime = _Runtime()
    runtime._phone_config = PhoneRuntimeConfig(enabled=True, owner_phone_number="+15551234567")

    monkeypatch.setattr("opencas.runtime.phone_runtime.PhoneBridgeService", _FakePhoneService)

    service = build_runtime_phone_service(runtime)

    assert isinstance(service, _FakePhoneService)
    assert service.config.owner_phone_number == "+15551234567"


@pytest.mark.asyncio
async def test_get_runtime_phone_status_returns_fallback_without_service() -> None:
    runtime = _Runtime()
    runtime._phone = None
    runtime._phone_config = PhoneRuntimeConfig(enabled=False, owner_phone_number=None)

    status = await get_runtime_phone_status(runtime)

    assert status["enabled"] is False
    assert status["twilio_credentials_configured"] is False
    assert status["contact_count"] == 0


@pytest.mark.asyncio
async def test_configure_runtime_phone_persists_and_rebuilds(monkeypatch: pytest.MonkeyPatch) -> None:
    runtime = _Runtime()
    saved = {}

    def _fake_save(state_dir, config):
        saved["state_dir"] = state_dir
        saved["config"] = config

    monkeypatch.setattr("opencas.runtime.phone_runtime.save_phone_runtime_config", _fake_save)
    monkeypatch.setattr("opencas.runtime.phone_runtime.PhoneBridgeService", _FakePhoneService)

    status = await configure_runtime_phone(
        runtime,
        PhoneRuntimeConfig(
            enabled=True,
            owner_phone_number="+15551234567",
            twilio_from_number="+15557654321",
            public_base_url="https://opencas.example.com",
        ),
    )

    assert saved["state_dir"] == runtime.ctx.config.state_dir
    assert saved["config"].owner_phone_number == "+15551234567"
    assert isinstance(runtime._phone, _FakePhoneService)
    assert status["saved"] is True
    assert status["owner"]["phone_number"] == "+15551234567"


@pytest.mark.asyncio
async def test_autoconfigure_runtime_phone_persists_selected_twilio_number(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    runtime = _Runtime()
    runtime._phone_config = PhoneRuntimeConfig(enabled=False, owner_phone_number="+15551234567")
    runtime._phone = _FakePhoneService(runtime, runtime._phone_config)
    saved = {}

    def _fake_save(state_dir, config):
        saved["state_dir"] = state_dir
        saved["config"] = config

    monkeypatch.setattr("opencas.runtime.phone_runtime.save_phone_runtime_config", _fake_save)
    monkeypatch.setattr("opencas.runtime.phone_runtime.PhoneBridgeService", _FakePhoneService)

    status = await autoconfigure_runtime_phone(
        runtime,
        enabled=True,
        public_base_url="https://opencas.example.com",
        owner_phone_number="+15551234567",
        owner_display_name="Operator",
    )

    assert saved["state_dir"] == runtime.ctx.config.state_dir
    assert saved["config"].twilio_from_number == "+15557654321"
    assert status["saved"] is True
    assert status["autoconfigured"] is True
    assert status["selected_number"]["sid"] == "PN123"
    assert status["twilio_number_candidates"][0]["phone_number"] == "+15557654321"


@pytest.mark.asyncio
async def test_call_owner_via_runtime_phone_traces_request() -> None:
    runtime = _Runtime()
    runtime._phone_config = PhoneRuntimeConfig(enabled=True, owner_phone_number="+15551234567")
    runtime._phone = _FakePhoneService(runtime, runtime._phone_config)

    result = await call_owner_via_runtime_phone(runtime, message="Call me back", reason="urgent")

    assert result["call_sid"] == "CA123"
    assert runtime._phone.call_requests == [{"message": "Call me back", "reason": "urgent"}]
    assert runtime.events == [("phone_owner_call_requested", {"to": "+15551234567", "call_sid": "CA123", "status": "queued"})]
