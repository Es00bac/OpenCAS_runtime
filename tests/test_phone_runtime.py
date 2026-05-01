"""Tests for runtime phone helper seams."""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import pytest

from opencas.phone_config import PhoneRuntimeConfig
from opencas.telemetry import EventKind, TelemetryEvent
from opencas.runtime.phone_runtime import (
    autoconfigure_runtime_phone,
    build_runtime_phone_service,
    call_owner_via_runtime_phone,
    configure_runtime_phone,
    configure_runtime_phone_session_profiles,
    get_runtime_phone_call_detail,
    get_runtime_recent_phone_calls,
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
            twilio_from_number="+14846736227",
            owner_phone_number=kwargs.get("owner_phone_number") or self.config.owner_phone_number,
            owner_display_name=kwargs.get("owner_display_name") or self.config.owner_display_name,
            owner_workspace_subdir=kwargs.get("owner_workspace_subdir") or self.config.owner_workspace_subdir,
            contacts=self.config.contacts,
        )
        return {
            "settings": settings,
            "selected_number": {"sid": "PN123", "phone_number": "+14846736227"},
            "twilio_number_candidates": [{"sid": "PN123", "phone_number": "+14846736227"}],
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
        self.tracer = SimpleNamespace(store=SimpleNamespace(query=lambda **_kwargs: []))

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
    assert status["recent_calls"] == []
    assert status["recent_events"] == []


@pytest.mark.asyncio
async def test_get_runtime_phone_status_summarizes_recent_phone_traces() -> None:
    runtime = _Runtime()
    runtime._phone = None
    runtime._phone_config = PhoneRuntimeConfig(enabled=True, owner_phone_number="+15551234567")
    events = [
        TelemetryEvent(
            kind=EventKind.TOM_EVAL,
            message="AgentRuntime: phone_stream_started",
            payload={"call_sid": "CA123", "caller_number": "+15550001111", "mode": "owner"},
        ),
        TelemetryEvent(
            kind=EventKind.TOM_EVAL,
            message="AgentRuntime: phone_stream_transcribed",
            payload={
                "call_sid": "CA123",
                "caller_number": "+15550001111",
                "mode": "owner",
                "transcript_preview": "Can you hear me now?",
            },
        ),
        TelemetryEvent(
            kind=EventKind.TOM_EVAL,
            message="AgentRuntime: phone_stream_closed",
            payload={
                "call_sid": "CA123",
                "caller_number": "+15550001111",
                "mode": "owner",
                "reason": "twilio_stop",
            },
        ),
    ]
    runtime.tracer = SimpleNamespace(store=SimpleNamespace(query=lambda **_kwargs: events))

    status = await get_runtime_phone_status(runtime)

    assert status["recent_calls"][0]["call_sid"] == "CA123"
    assert status["recent_calls"][0]["hangup_reason"] == "twilio_stop"
    assert status["recent_calls"][0]["hangup_class"] == "remote_disconnect"
    assert status["recent_calls"][0]["current_state"] == "closed"
    assert status["recent_calls"][0]["event_count"] == 3
    assert status["recent_events"][0]["event"] == "phone_stream_closed"
    assert status["recent_events"][-1]["event"] == "phone_stream_started"


@pytest.mark.asyncio
async def test_get_runtime_recent_phone_calls_returns_limited_summary() -> None:
    runtime = _Runtime()
    runtime._phone = None
    runtime._phone_config = PhoneRuntimeConfig(enabled=True, owner_phone_number="+15551234567")
    events = [
        TelemetryEvent(
            kind=EventKind.TOM_EVAL,
            message="AgentRuntime: phone_stream_started",
            payload={"call_sid": "CA123", "caller_number": "+15550001111", "mode": "owner"},
        ),
        TelemetryEvent(
            kind=EventKind.TOM_EVAL,
            message="AgentRuntime: phone_stream_started",
            payload={"call_sid": "CA999", "caller_number": "+15550002222", "mode": "workspace_assistant"},
        ),
        TelemetryEvent(
            kind=EventKind.TOM_EVAL,
            message="AgentRuntime: phone_stream_closed",
            payload={"call_sid": "CA123", "caller_number": "+15550001111", "mode": "owner", "reason": "twilio_stop"},
        ),
    ]
    runtime.tracer = SimpleNamespace(store=SimpleNamespace(query=lambda **_kwargs: events))

    result = await get_runtime_recent_phone_calls(runtime, limit=1)

    assert len(result["calls"]) == 1
    assert result["calls"][0]["call_sid"] in {"CA123", "CA999"}
    assert result["events"]


@pytest.mark.asyncio
async def test_get_runtime_phone_call_detail_returns_timeline_and_phase_durations() -> None:
    runtime = _Runtime()
    runtime._phone = None
    runtime._phone_config = PhoneRuntimeConfig(enabled=True, owner_phone_number="+15551234567")
    events = [
        TelemetryEvent(
            kind=EventKind.TOM_EVAL,
            message="AgentRuntime: phone_stream_started",
            payload={"call_sid": "CA123", "caller_number": "+15550001111", "mode": "owner"},
        ),
        TelemetryEvent(
            kind=EventKind.TOM_EVAL,
            message="AgentRuntime: phone_stream_transcribed",
            payload={"call_sid": "CA123", "caller_number": "+15550001111", "mode": "owner", "transcript_preview": "Hello there"},
        ),
        TelemetryEvent(
            kind=EventKind.TOM_EVAL,
            message="AgentRuntime: phone_owner_reply_started",
            payload={"call_sid": "CA123", "caller_number": "+15550001111"},
        ),
        TelemetryEvent(
            kind=EventKind.TOM_EVAL,
            message="AgentRuntime: phone_owner_reply_completed",
            payload={"call_sid": "CA123", "caller_number": "+15550001111", "response_preview": "Hi."},
        ),
        TelemetryEvent(
            kind=EventKind.TOM_EVAL,
            message="AgentRuntime: phone_stream_tts_sent",
            payload={"call_sid": "CA123", "caller_number": "+15550001111", "provider": "hosted"},
        ),
        TelemetryEvent(
            kind=EventKind.TOM_EVAL,
            message="AgentRuntime: phone_stream_closed",
            payload={"call_sid": "CA123", "caller_number": "+15550001111", "reason": "twilio_stop"},
        ),
    ]
    runtime.tracer = SimpleNamespace(store=SimpleNamespace(query=lambda **_kwargs: events))

    detail = await get_runtime_phone_call_detail(runtime, call_sid="CA123")

    assert detail["found"] is True
    assert detail["call"]["call_sid"] == "CA123"
    assert detail["call"]["hangup_reason"] == "twilio_stop"
    assert detail["call"]["hangup_class"] == "remote_disconnect"
    assert detail["call"]["current_state"] == "closed"
    assert "owner_live" in detail["call"]["visited_states"]
    assert "speaking" in detail["call"]["visited_states"]
    assert detail["events"][0]["event"] == "phone_stream_started"
    assert detail["events"][-1]["event"] == "phone_stream_closed"
    assert "time_to_transcription_seconds" in detail["phase_durations"]
    assert "time_to_first_reply_start_seconds" in detail["phase_durations"]
    assert "total_call_seconds" in detail["phase_durations"]
    assert detail["state_timeline"][-1]["state"] == "closed"


@pytest.mark.asyncio
async def test_get_runtime_phone_call_detail_projects_menu_state_transitions() -> None:
    runtime = _Runtime()
    runtime._phone = None
    runtime._phone_config = PhoneRuntimeConfig(enabled=True, owner_phone_number="+15551234567")
    events = [
        TelemetryEvent(
            kind=EventKind.TOM_EVAL,
            message="AgentRuntime: phone_stream_started",
            payload={"call_sid": "CA777", "caller_number": "+15550002222", "mode": "screening"},
        ),
        TelemetryEvent(
            kind=EventKind.TOM_EVAL,
            message="AgentRuntime: phone_stream_dtmf",
            payload={"call_sid": "CA777", "caller_number": "+15550002222", "mode": "screening", "digit": "1"},
        ),
        TelemetryEvent(
            kind=EventKind.TOM_EVAL,
            message="AgentRuntime: phone_stream_menu_choice",
            payload={
                "call_sid": "CA777",
                "caller_number": "+15550002222",
                "mode": "screening",
                "choice": "employer",
                "action": "workspace_assistant",
            },
        ),
        TelemetryEvent(
            kind=EventKind.TOM_EVAL,
            message="AgentRuntime: phone_workspace_reply_started",
            payload={"call_sid": "CA777", "caller_number": "+15550002222"},
        ),
        TelemetryEvent(
            kind=EventKind.TOM_EVAL,
            message="AgentRuntime: phone_stream_tts_sent",
            payload={"call_sid": "CA777", "caller_number": "+15550002222", "provider": "hosted"},
        ),
        TelemetryEvent(
            kind=EventKind.TOM_EVAL,
            message="AgentRuntime: phone_stream_closed",
            payload={"call_sid": "CA777", "caller_number": "+15550002222", "reason": "twilio_stop"},
        ),
    ]
    runtime.tracer = SimpleNamespace(store=SimpleNamespace(query=lambda **_kwargs: events))

    detail = await get_runtime_phone_call_detail(runtime, call_sid="CA777")

    assert detail["call"]["terminal_action"] is None
    assert detail["call"]["current_state"] == "closed"
    assert detail["call"]["visited_states"] == [
        "screening",
        "menu_input",
        "workspace_live",
        "generating_reply",
        "speaking",
        "closed",
    ]
    assert detail["state_timeline"][2]["state"] == "workspace_live"


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
    assert status["provenance_events"][0]["event_type"] == "MUTATION"
    assert status["provenance_events"][0]["triggering_artifact"] == "setting|phone|runtime"
    assert status["provenance_events"][0]["parent_link_id"].endswith("phone/config.json")
    assert status["provenance_events"][0]["linked_link_ids"] == [status["provenance_events"][0]["parent_link_id"]]


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
        owner_display_name="Cabew",
    )

    assert saved["state_dir"] == runtime.ctx.config.state_dir
    assert saved["config"].twilio_from_number == "+14846736227"
    assert status["saved"] is True
    assert status["autoconfigured"] is True
    assert status["selected_number"]["sid"] == "PN123"
    assert status["twilio_number_candidates"][0]["phone_number"] == "+14846736227"
    assert status["provenance_events"][0]["parent_link_id"].endswith("phone/config.json")


@pytest.mark.asyncio
async def test_call_owner_via_runtime_phone_traces_request() -> None:
    runtime = _Runtime()
    runtime._phone_config = PhoneRuntimeConfig(enabled=True, owner_phone_number="+15551234567")
    runtime._phone = _FakePhoneService(runtime, runtime._phone_config)

    result = await call_owner_via_runtime_phone(runtime, message="Call me back", reason="urgent")

    assert result["call_sid"] == "CA123"
    assert runtime._phone.call_requests == [{"message": "Call me back", "reason": "urgent"}]
    assert runtime.events == [("phone_owner_call_requested", {"to": "+15551234567", "call_sid": "CA123", "status": "queued"})]


@pytest.mark.asyncio
async def test_configure_runtime_phone_session_profiles_persists_editable_menu(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    runtime = _Runtime()
    runtime.ctx.config.state_dir = tmp_path
    runtime._phone_config = PhoneRuntimeConfig(enabled=True, owner_phone_number="+15551234567")

    monkeypatch.setattr("opencas.runtime.phone_runtime.PhoneBridgeService", _FakePhoneService)

    status = await configure_runtime_phone_session_profiles(
        runtime,
        {
            "owner_entry_prompt": "Press 1 for the owner.",
            "owner_entry_reprompt": "Press 1 for the owner now.",
            "owner_pin_prompt": "Enter owner pin.",
            "owner_pin_retry_prompt": "Retry pin.",
            "owner_pin_success_message": "Verified.",
            "owner_pin_failure_message": "Denied.",
            "public_prompt": "Potential employers press 1.",
            "public_reprompt": "Press 1 for employer mode.",
            "employer_enabled": True,
            "employer_digit": "1",
            "employer_label": "Employer lane",
            "employer_phrases": ["employer", "recruiter"],
            "employer_greeting": "Employer greeting.",
            "employer_prompt_profile": "worksafe_owner",
            "employer_allowed_actions": ["leave_message", "knowledge_qa"],
            "employer_shared_workspace_subdir": "phone/employer_shared",
            "employer_caller_workspace_subdir": "phone/employers/{phone_digits}",
            "reject_enabled": True,
            "reject_digit": "2",
            "reject_label": "Reject",
            "reject_phrases": ["other"],
            "reject_message": "Not for this line.",
        },
    )

    assert status["saved"] is True
    assert status["session_profiles_saved"] is True
    assert status["session_profiles"]["employer"]["label"] == "Employer lane"
    assert runtime._phone_config.menu_config_path is not None
    assert Path(runtime._phone_config.menu_config_path).exists()
    assert status["provenance_events"][0]["triggering_artifact"] == "setting|phone|session-profiles"
