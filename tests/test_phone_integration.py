"""Tests for Twilio-backed phone bridge behavior."""

from __future__ import annotations

import asyncio
import json
import shutil
import time
from pathlib import Path
from types import SimpleNamespace

import pytest
from fastapi import WebSocketDisconnect

import opencas.phone_integration as phone_integration
from opencas.context.models import MessageRole
from opencas.phone_config import PhoneRuntimeConfig
from opencas.phone_integration import (
    PendingPhoneReply,
    PhoneBridgeService,
    ResolvedPhoneReply,
    TwilioCredentials,
)
from opencas.phone_streaming import PhoneMediaStreamSession


class _ContextEntry(SimpleNamespace):
    pass


class _FakeContextStore:
    def __init__(self) -> None:
        self.entries: dict[str, list[_ContextEntry]] = {}
        self.session_names: dict[str, str] = {}

    async def ensure_session(self, session_id: str) -> None:
        self.entries.setdefault(session_id, [])

    async def update_session_name(self, session_id: str, name: str) -> None:
        self.session_names[session_id] = name

    async def list_recent(self, session_id: str, limit: int = 10, include_hidden: bool = False):
        return list(self.entries.get(session_id, []))[-limit:]

    async def append(self, session_id: str, role, content: str, meta=None) -> None:
        self.entries.setdefault(session_id, []).append(
            _ContextEntry(role=role, content=content, meta=meta or {})
        )


class _FakeLLM:
    def __init__(self, content: str = "Approved answer.", responses=None) -> None:
        self.content = content
        self.responses = list(responses or [])
        self.calls = []
        self.model_routing = SimpleNamespace(auto_escalation=False)

    async def chat_completion(self, **kwargs):
        self.calls.append(kwargs)
        if self.responses:
            return self.responses.pop(0)
        return {"choices": [{"message": {"content": self.content}}]}


class _FakeRuntime:
    def __init__(self, workspace_root: Path, llm_content: str = "Approved answer.") -> None:
        self.workspace_root = workspace_root
        self.ctx = SimpleNamespace(
            config=SimpleNamespace(
                state_dir=workspace_root / "state",
                agent_workspace_root=lambda: workspace_root,
            ),
            context_store=_FakeContextStore(),
            llm=SimpleNamespace(default_model="test-model"),
        )
        self.llm = _FakeLLM(content=llm_content)
        self.tracer = None
        self.approval = object()
        self.converse_calls = []
        self.recorded_episodes = []

    async def converse(self, text: str, session_id: str | None = None, user_meta=None):
        self.converse_calls.append(
            {"text": text, "session_id": session_id, "user_meta": user_meta}
        )
        return "Owner response."

    async def _record_episode(self, content, kind, *, session_id: str, role: str):
        self.recorded_episodes.append(
            {"content": content, "kind": kind, "session_id": session_id, "role": role}
        )


class _MultiValueForm:
    def __init__(self, pairs):
        self._pairs = list(pairs)

    def items(self):
        latest = {}
        for key, value in self._pairs:
            latest[key] = value
        return latest.items()

    def keys(self):
        return [key for key, _ in self._pairs]

    def get(self, key, default=None):
        values = self.getlist(key)
        return values[-1] if values else default

    def getlist(self, key):
        return [value for pair_key, value in self._pairs if pair_key == key]


class _FakeWebSocket:
    def __init__(self, messages) -> None:
        self._messages = list(messages)
        self.sent = []
        self.accepted = False
        self.closed = False

    async def accept(self) -> None:
        self.accepted = True

    async def receive_text(self) -> str:
        if self.closed or not self._messages:
            raise WebSocketDisconnect()
        return self._messages.pop(0)

    async def send_text(self, payload: str) -> None:
        self.sent.append(payload)

    async def close(self, code: int | None = None) -> None:
        self.closed = True


def _phone_service(
    tmp_path: Path,
    *,
    owner_phone_number: str = "+15551234567",
    contacts=None,
    llm_content: str = "Approved answer.",
    llm_responses=None,
) -> tuple[_FakeRuntime, PhoneBridgeService]:
    workspace_root = tmp_path / "workspace"
    workspace_root.mkdir(parents=True, exist_ok=True)
    runtime = _FakeRuntime(workspace_root, llm_content=llm_content)
    if llm_responses is not None:
        runtime.llm = _FakeLLM(content=llm_content, responses=llm_responses)
    config = PhoneRuntimeConfig(
        enabled=True,
        webhook_signature_required=False,
        public_base_url="https://opencas.example.com",
        twilio_from_number="+15557654321",
        owner_phone_number=owner_phone_number,
        owner_display_name="Cabew",
        contacts=contacts or [],
    )
    service = PhoneBridgeService(runtime=runtime, config=config)
    return runtime, service


def test_validate_webhook_signature_uses_twilio_request_validator(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _runtime, service = _phone_service(tmp_path)
    service.config = service.config.model_copy(update={"webhook_signature_required": True})
    captured = {}

    class _FakeValidator:
        def __init__(self, token: str) -> None:
            captured["token"] = token

        def validate(self, url, params, signature):
            captured["url"] = url
            captured["params"] = dict(params)
            captured["signature"] = signature
            return True

    monkeypatch.setattr(
        service,
        "_twilio_credentials",
        lambda: TwilioCredentials(
            account_sid="AC123",
            api_username="SK123",
            api_password="rest-secret",
            webhook_auth_token="secret",
        ),
    )
    monkeypatch.setattr("opencas.phone_integration.RequestValidator", _FakeValidator)

    valid = service.validate_webhook_signature(
        request_url="https://opencasagent.com/api/phone/twilio/voice",
        form_data={
            "CallSid": "CA123",
            "From": "+17203340532",
            "To": "+14846736227",
        },
        provided_signature="abc123",
    )

    assert valid is True
    assert captured == {
        "token": "secret",
        "url": "https://opencasagent.com/api/phone/twilio/voice",
        "params": {
            "CallSid": "CA123",
            "From": "+17203340532",
            "To": "+14846736227",
        },
        "signature": "abc123",
    }


@pytest.mark.asyncio
async def test_handle_voice_webhook_validates_against_original_multivalue_form(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _runtime, service = _phone_service(tmp_path)
    captured = {}
    form = _MultiValueForm(
        [
            ("From", "+15551234567"),
            ("To", "+15557654321"),
            ("CallSid", "CA123"),
            ("SipHeader_X-Test", "one"),
            ("SipHeader_X-Test", "two"),
        ]
    )

    def _fake_validate(
        *,
        request_url: str,
        form_data,
        provided_signature: str | None,
        bridge_token: str | None,
    ) -> bool:
        captured["request_url"] = request_url
        captured["form_data"] = form_data
        captured["provided_signature"] = provided_signature
        captured["bridge_token"] = bridge_token
        return False

    monkeypatch.setattr(service, "validate_webhook_request", _fake_validate)

    xml = await service.handle_voice_webhook(
        request_url="https://opencasagent.com/api/phone/twilio/voice",
        webhook_base_url="https://opencasagent.com",
        form_data=form,
        provided_signature="sig123",
    )

    assert captured["request_url"] == "https://opencasagent.com/api/phone/twilio/voice"
    assert captured["form_data"] is form
    assert captured["provided_signature"] == "sig123"
    assert captured["bridge_token"] is None
    assert "Unauthorized phone bridge request." in xml


def test_validate_webhook_request_allows_bridge_token_fallback_without_auth_token(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _runtime, service = _phone_service(tmp_path)
    service.config = service.config.model_copy(
        update={
            "webhook_signature_required": True,
            "webhook_secret": "bridge-secret",
        }
    )
    monkeypatch.setattr(
        service,
        "_twilio_credentials",
        lambda: TwilioCredentials(
            account_sid="AC123",
            api_username="SK123",
            api_password="rest-secret",
            webhook_auth_token=None,
        ),
    )

    assert service.validate_webhook_request(
        request_url="https://opencasagent.com/api/phone/twilio/voice?bridge_token=bridge-secret",
        form_data={"CallSid": "CA123", "From": "+17203340532", "To": "+14846736227"},
        provided_signature=None,
        bridge_token="bridge-secret",
    ) is True


@pytest.mark.asyncio
async def test_owner_gather_queues_background_reply_for_owner_request(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _runtime, service = _phone_service(tmp_path, llm_content="Owner response.")
    monkeypatch.setattr(service, "_start_owner_reply_task", lambda **kwargs: "reply-123")

    xml = await service.handle_gather_webhook(
        request_url="https://opencas.example.com/api/phone/twilio/gather",
        webhook_base_url="https://opencas.example.com",
        form_data={"From": "+15551234567", "SpeechResult": "Need you now", "CallSid": "CA-owner"},
        provided_signature=None,
    )

    assert "https://opencas.example.com/api/phone/twilio/poll?reply_token=reply-123" in xml
    assert "<Pause length=\"2\"/>" in xml


@pytest.mark.asyncio
async def test_owner_voice_webhook_returns_owner_menu_gather_twiml(tmp_path: Path) -> None:
    _runtime, service = _phone_service(tmp_path)

    xml = await service.handle_voice_webhook(
        request_url="https://opencasagent.com/api/phone/twilio/voice",
        webhook_base_url="https://opencasagent.com",
        form_data={"From": "+15551234567", "To": "+15557654321", "CallSid": "CA-owner"},
        provided_signature=None,
    )

    assert "<Gather" in xml
    assert 'input="dtmf"' in xml
    assert 'numDigits="1"' in xml
    assert "stream_mode=owner_menu" in xml
    assert "menu_key=owner_entry" in xml
    assert "Press 1 to continue as the owner" in xml
    assert "<Connect>" not in xml


@pytest.mark.asyncio
async def test_non_owner_voice_webhook_returns_screening_gather_twiml(tmp_path: Path) -> None:
    _runtime, service = _phone_service(tmp_path)

    xml = await service.handle_voice_webhook(
        request_url="https://opencasagent.com/api/phone/twilio/voice",
        webhook_base_url="https://opencasagent.com",
        form_data={"From": "+15550009999", "To": "+15557654321", "CallSid": "CA-employer"},
        provided_signature=None,
    )

    assert "<Gather" in xml
    assert 'input="dtmf speech"' in xml
    assert 'numDigits="1"' in xml
    assert "stream_mode=screening" in xml
    assert "menu_key=public_main" in xml
    assert "Potential employers, press 1" in xml
    assert "<Connect>" not in xml


@pytest.mark.asyncio
async def test_owner_menu_gather_digit_one_connects_owner_stream(tmp_path: Path) -> None:
    _runtime, service = _phone_service(tmp_path)

    xml = await service.handle_gather_webhook(
        request_url=(
            "https://opencas.example.com/api/phone/twilio/gather"
            "?stream_mode=owner_menu&menu_key=owner_entry"
        ),
        webhook_base_url="https://opencas.example.com",
        form_data={"From": "+15551234567", "Digits": "1", "CallSid": "CA-owner-menu"},
        provided_signature=None,
    )

    assert "<Connect>" in xml
    assert "<Stream " in xml
    assert "streamMode" in xml
    assert "owner" in xml
    assert "Go ahead." in xml


@pytest.mark.asyncio
async def test_screening_gather_digit_one_connects_workspace_stream(tmp_path: Path) -> None:
    _runtime, service = _phone_service(tmp_path)

    xml = await service.handle_gather_webhook(
        request_url=(
            "https://opencas.example.com/api/phone/twilio/gather"
            "?stream_mode=screening&menu_key=public_main"
        ),
        webhook_base_url="https://opencas.example.com",
        form_data={"From": "+15550009999", "Digits": "1", "CallSid": "CA-screening-menu"},
        provided_signature=None,
    )

    assert "<Connect>" in xml
    assert "<Stream " in xml
    assert "workspace_assistant" in xml
    assert "connected to the opencas phone bridge in work mode" in xml.lower()


@pytest.mark.asyncio
async def test_generate_owner_live_reply_uses_runtime_converse(tmp_path: Path) -> None:
    runtime, service = _phone_service(tmp_path)
    caller = service._resolve_caller(from_number="+15551234567")
    assert caller is not None

    response = await service.generate_owner_live_reply(
        caller=caller,
        transcript="Who am I?",
        call_sid="CA-owner",
    )

    assert response == "Owner response."
    assert runtime.converse_calls == [
        {
            "text": "Who am I?",
            "session_id": "phone:+15551234567",
            "user_meta": {
                "phone": {
                    "channel": "twilio_voice",
                    "caller_number": "+15551234567",
                    "display_name": "Cabew",
                    "trust_level": "owner",
                    "allowed_actions": ["leave_message", "knowledge_qa"],
                    "call_sid": "CA-owner",
                    "mode": "owner_live",
                },
                "voice_input": {
                    "provider": "elevenlabs",
                    "mode": "phone_stream",
                    "model": "scribe_v2",
                },
            },
        }
    ]


@pytest.mark.asyncio
async def test_generate_owner_live_stream_reply_times_out_with_persisted_fallback(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    runtime, service = _phone_service(tmp_path)
    caller = service._resolve_caller(from_number="+15551234567")
    assert caller is not None

    async def _slow_owner_reply(**kwargs):
        await asyncio.sleep(0.05)
        return "late"

    monkeypatch.setattr(service, "generate_owner_live_reply", _slow_owner_reply)
    monkeypatch.setattr(phone_integration, "_PHONE_OWNER_STREAM_REPLY_TIMEOUT_SECONDS", 0.01)

    reply = await service.generate_owner_live_stream_reply(
        caller=caller,
        transcript="Hello?",
        call_sid="CA-stream-timeout",
    )

    assert "phone line stalled" in reply
    session_entries = runtime.ctx.context_store.entries["phone:+15551234567"]
    assert session_entries[-1].role == MessageRole.ASSISTANT
    assert session_entries[-1].content == reply
    assert session_entries[-1].meta["phone"]["call_sid"] == "CA-stream-timeout"
    assert runtime.recorded_episodes[-1]["content"] == reply


@pytest.mark.asyncio
async def test_phone_media_stream_owner_uses_bounded_stream_reply(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _runtime, service = _phone_service(tmp_path)
    caller = service._resolve_caller(from_number="+15551234567")
    assert caller is not None
    spoken = []

    async def _fake_transcribe(self, payload: bytes) -> str:
        return "How are you?"

    async def _fake_speak(self, text: str) -> None:
        spoken.append(text)

    async def _fake_stream_reply(**kwargs) -> str:
        return "Stream fallback reply."

    monkeypatch.setattr(PhoneMediaStreamSession, "_transcribe", _fake_transcribe)
    monkeypatch.setattr(PhoneMediaStreamSession, "_speak_text", _fake_speak)
    monkeypatch.setattr(service, "generate_owner_live_stream_reply", _fake_stream_reply)

    session = PhoneMediaStreamSession(websocket=_FakeWebSocket([]), service=service)
    session.caller = caller
    session.mode = "owner"
    session.call_sid = "CA-stream-owner"
    await session._process_utterance(b"audio")

    assert spoken == ["Stream fallback reply."]


@pytest.mark.asyncio
async def test_activate_employer_caller_seeds_workspace_from_repo_seed(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    runtime, service = _phone_service(tmp_path)
    seed_dir = tmp_path / "seed"
    seed_dir.mkdir(parents=True, exist_ok=True)
    (seed_dir / "resume.md").write_text("Resume seed content", encoding="utf-8")
    monkeypatch.setattr(service, "_employment_shared_seed_dir", lambda: seed_dir)
    shutil.rmtree(runtime.workspace_root / "phone" / "employer_shared", ignore_errors=True)

    caller = await service.activate_employer_caller(
        caller_number="+15550001111",
        display_name="Recruiter",
    )

    assert caller.allowed_actions == ("leave_message", "knowledge_qa")
    shared_resume = runtime.workspace_root / "phone" / "employer_shared" / "resume.md"
    caller_notes = runtime.workspace_root / "phone" / "employers" / "15550001111" / "messages.md"
    assert shared_resume.read_text(encoding="utf-8") == "Resume seed content"
    assert caller_notes.exists()


@pytest.mark.asyncio
async def test_phone_media_stream_screening_digit_one_enters_employer_mode(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _runtime, service = _phone_service(tmp_path)
    spoken = []

    async def _fake_speak(self, text: str) -> None:
        spoken.append(text)

    async def _fake_wait(self, **kwargs) -> None:
        return None

    monkeypatch.setattr(PhoneMediaStreamSession, "_speak_text", _fake_speak)
    monkeypatch.setattr(PhoneMediaStreamSession, "_wait_for_playback_completion", _fake_wait)

    websocket = _FakeWebSocket(
        [
            json.dumps(
                {
                    "event": "start",
                    "start": {
                        "streamSid": "MZ123",
                        "callSid": "CA123",
                        "customParameters": {
                            "callerNumber": "+15550009999",
                            "displayName": "Caller",
                            "streamMode": "screening",
                            "introMessage": "Hi, this is the OpenCAS agent.",
                        },
                    },
                }
            ),
            json.dumps(
                {
                    "event": "dtmf",
                    "dtmf": {"digit": "1", "track": "inbound_track"},
                }
            ),
            json.dumps({"event": "stop"}),
        ]
    )

    session = PhoneMediaStreamSession(websocket=websocket, service=service)
    await session.run()

    assert websocket.accepted is True
    assert session.mode == "workspace_assistant"
    assert session.employer_mode_active is True
    assert any("work mode" in text.lower() for text in spoken)


@pytest.mark.asyncio
async def test_phone_media_stream_emits_structured_session_state_traces(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _runtime, service = _phone_service(tmp_path)
    spoken = []
    traced = []

    async def _fake_speak(self, text: str) -> None:
        spoken.append(text)

    async def _fake_wait(self, **kwargs) -> None:
        return None

    def _capture_trace(event, **payload):
        traced.append((event, payload))

    monkeypatch.setattr(PhoneMediaStreamSession, "_speak_text", _fake_speak)
    monkeypatch.setattr(PhoneMediaStreamSession, "_wait_for_playback_completion", _fake_wait)
    monkeypatch.setattr(service, "trace_phone_event", _capture_trace)

    websocket = _FakeWebSocket(
        [
            json.dumps(
                {
                    "event": "start",
                    "start": {
                        "streamSid": "MZ777",
                        "callSid": "CA777",
                        "customParameters": {
                            "callerNumber": "+15550009999",
                            "displayName": "Caller",
                            "streamMode": "screening",
                            "introMessage": "Hi, this is the OpenCAS agent.",
                        },
                    },
                }
            ),
            json.dumps({"event": "dtmf", "dtmf": {"digit": "1", "track": "inbound_track"}}),
            json.dumps({"event": "stop"}),
        ]
    )

    session = PhoneMediaStreamSession(websocket=websocket, service=service)
    await session.run()

    state_events = [payload for event, payload in traced if event == "phone_session_state_changed"]
    close_event = next(payload for event, payload in traced if event == "phone_stream_closed")

    assert state_events[0]["to_state"] == "screening"
    assert any(item["to_state"] == "menu_input" for item in state_events)
    assert any(item["to_state"] == "workspace_live" for item in state_events)
    assert close_event["current_state"] == "closed"
    assert close_event["hangup_class"] == "remote_disconnect"
    assert isinstance(close_event["phase_durations"], dict)
    assert "total_call_seconds" in close_event["phase_durations"]


@pytest.mark.asyncio
async def test_phone_media_stream_owner_pin_dtmf_verifies_before_owner_mode(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _runtime, service = _phone_service(tmp_path)
    service.config = service.config.model_copy(update={"owner_pin": "123456"})
    spoken = []

    async def _fake_speak(self, text: str) -> None:
        spoken.append(text)

    monkeypatch.setattr(PhoneMediaStreamSession, "_speak_text", _fake_speak)

    websocket = _FakeWebSocket(
        [
            json.dumps(
                {
                    "event": "start",
                    "start": {
                        "streamSid": "MZ123",
                        "callSid": "CA123",
                        "customParameters": {
                            "callerNumber": "+15551234567",
                            "displayName": "Cabew",
                            "streamMode": "owner_pin",
                            "introMessage": "Please enter your six digit owner PIN now.",
                        },
                    },
                }
            ),
            json.dumps({"event": "dtmf", "dtmf": {"digit": "1", "track": "inbound_track"}}),
            json.dumps({"event": "dtmf", "dtmf": {"digit": "2", "track": "inbound_track"}}),
            json.dumps({"event": "dtmf", "dtmf": {"digit": "3", "track": "inbound_track"}}),
            json.dumps({"event": "dtmf", "dtmf": {"digit": "4", "track": "inbound_track"}}),
            json.dumps({"event": "dtmf", "dtmf": {"digit": "5", "track": "inbound_track"}}),
            json.dumps({"event": "dtmf", "dtmf": {"digit": "6", "track": "inbound_track"}}),
            json.dumps({"event": "stop"}),
        ]
    )

    session = PhoneMediaStreamSession(websocket=websocket, service=service)
    await session.run()

    assert session.mode == "owner_menu"
    assert any("verified" in text.lower() for text in spoken)
    assert any("press 1 to continue as the owner" in text.lower() for text in spoken)


@pytest.mark.asyncio
async def test_phone_media_stream_owner_menu_digit_one_enters_owner_mode(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _runtime, service = _phone_service(tmp_path)
    spoken = []

    async def _fake_speak(self, text: str) -> None:
        spoken.append(text)

    monkeypatch.setattr(PhoneMediaStreamSession, "_speak_text", _fake_speak)

    websocket = _FakeWebSocket(
        [
            json.dumps(
                {
                    "event": "start",
                    "start": {
                        "streamSid": "MZ123",
                        "callSid": "CA123",
                        "customParameters": {
                            "callerNumber": "+15551234567",
                            "displayName": "Cabew",
                            "streamMode": "owner_menu",
                            "introMessage": "Press 1 to continue as the owner, or press 2 for the main menu.",
                        },
                    },
                }
            ),
            json.dumps({"event": "dtmf", "dtmf": {"digit": "1", "track": "inbound_track"}}),
            json.dumps({"event": "stop"}),
        ]
    )

    session = PhoneMediaStreamSession(websocket=websocket, service=service)
    await session.run()

    assert session.mode == "owner"
    assert any("what do you need" in text.lower() for text in spoken)


@pytest.mark.asyncio
async def test_phone_media_stream_owner_menu_digit_one_falls_back_to_local_speech(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _runtime, service = _phone_service(tmp_path)
    attempts = []
    sent_marks = []

    async def _fake_speak_once(self, text: str, *, prefer_local: bool) -> None:
        attempts.append(prefer_local)
        if not prefer_local:
            raise RuntimeError("hosted tts failed")
        sent_marks.append(text)

    monkeypatch.setattr(PhoneMediaStreamSession, "_speak_text_once", _fake_speak_once)

    websocket = _FakeWebSocket(
        [
            json.dumps(
                {
                    "event": "start",
                    "start": {
                        "streamSid": "MZ123",
                        "callSid": "CA123",
                        "customParameters": {
                            "callerNumber": "+15551234567",
                            "displayName": "Cabew",
                            "streamMode": "owner_menu",
                            "introMessage": "Press 1 to continue as the owner, or press 2 for the main menu.",
                        },
                    },
                }
            ),
            json.dumps({"event": "dtmf", "dtmf": {"digit": "1", "track": "inbound_track"}}),
            json.dumps({"event": "stop"}),
        ]
    )

    session = PhoneMediaStreamSession(websocket=websocket, service=service)
    await session.run()

    assert session.mode == "owner"
    assert attempts[:2] == [False, True]
    assert any("what do you need" in text.lower() for text in sent_marks)


@pytest.mark.asyncio
async def test_phone_media_stream_owner_menu_digit_one_does_not_drop_call_on_speech_failure(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _runtime, service = _phone_service(tmp_path)

    async def _failing_speak_once(self, text: str, *, prefer_local: bool) -> None:
        raise RuntimeError("tts failure")

    monkeypatch.setattr(PhoneMediaStreamSession, "_speak_text_once", _failing_speak_once)

    websocket = _FakeWebSocket(
        [
            json.dumps(
                {
                    "event": "start",
                    "start": {
                        "streamSid": "MZ123",
                        "callSid": "CA123",
                        "customParameters": {
                            "callerNumber": "+15551234567",
                            "displayName": "Cabew",
                            "streamMode": "owner_menu",
                            "introMessage": "Press 1 to continue as the owner, or press 2 for the main menu.",
                        },
                    },
                }
            ),
            json.dumps({"event": "dtmf", "dtmf": {"digit": "1", "track": "inbound_track"}}),
            json.dumps({"event": "stop"}),
        ]
    )

    session = PhoneMediaStreamSession(websocket=websocket, service=service)
    await session.run()

    assert session.mode == "owner"
    assert websocket.closed is True


@pytest.mark.asyncio
async def test_phone_media_stream_owner_menu_digit_two_routes_to_public_menu(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _runtime, service = _phone_service(tmp_path)
    spoken = []

    async def _fake_speak(self, text: str) -> None:
        spoken.append(text)

    monkeypatch.setattr(PhoneMediaStreamSession, "_speak_text", _fake_speak)

    websocket = _FakeWebSocket(
        [
            json.dumps(
                {
                    "event": "start",
                    "start": {
                        "streamSid": "MZ123",
                        "callSid": "CA123",
                        "customParameters": {
                            "callerNumber": "+15551234567",
                            "displayName": "Cabew",
                            "streamMode": "owner_menu",
                            "introMessage": "Press 1 to continue as the owner, or press 2 for the main menu.",
                        },
                    },
                }
            ),
            json.dumps({"event": "dtmf", "dtmf": {"digit": "2", "track": "inbound_track"}}),
            json.dumps({"event": "stop"}),
        ]
    )

    session = PhoneMediaStreamSession(websocket=websocket, service=service)
    await session.run()

    assert session.mode == "screening"
    assert session.active_menu_key == "public_main"
    assert any("potential employers, press 1" in text.lower() for text in spoken)


@pytest.mark.asyncio
async def test_finalize_employer_call_appends_owner_summary(tmp_path: Path) -> None:
    runtime, service = _phone_service(tmp_path, llm_content="Employer is interested in an AI workflow engagement.")
    notifications = []

    async def _fake_notify_owner(text: str, **kwargs):
        notifications.append({"text": text, **kwargs})
        return [42]

    runtime._telegram = SimpleNamespace(notify_owner=_fake_notify_owner)
    caller = await service.activate_employer_caller(
        caller_number="+15550007777",
        display_name="Hiring manager",
    )
    session_id = service._session_id(caller)
    await service._ensure_phone_session(session_id, caller)
    await runtime.ctx.context_store.append(
        session_id,
        MessageRole.USER,
        "We need help improving our operations with AI.",
        meta=service._user_meta(caller, call_sid="CA-emp"),
    )
    await runtime.ctx.context_store.append(
        session_id,
        MessageRole.ASSISTANT,
        "the owner builds autonomous AI systems and local-first workflow tooling.",
        meta={"phone": service._assistant_meta(caller, call_sid="CA-emp")},
    )

    monkeypatch = pytest.MonkeyPatch()
    audio_path = runtime.workspace_root / "phone" / "employers" / "15550007777" / "calls" / "CA-emp" / "caller-message.mp3"
    def _fake_store_audio(*, call_dir, caller_audio_mulaw):
        call_dir.mkdir(parents=True, exist_ok=True)
        audio_path.write_bytes(b"mp3")
        return audio_path

    monkeypatch.setattr(service, "_store_employer_call_audio_artifact", _fake_store_audio)
    try:
        await service.finalize_employer_call(caller=caller, call_sid="CA-emp", caller_audio_mulaw=b"audio")
    finally:
        monkeypatch.undo()

    owner_summary = runtime.workspace_root / "phone" / "owner" / "employer-call-summaries.md"
    employer_summary = runtime.workspace_root / "phone" / "employers" / "15550007777" / "call-summary.md"
    transcript_path = runtime.workspace_root / "phone" / "employers" / "15550007777" / "calls" / "CA-emp" / "transcript.md"
    summary_path = runtime.workspace_root / "phone" / "employers" / "15550007777" / "calls" / "CA-emp" / "summary.md"
    messages_path = runtime.workspace_root / "phone" / "employers" / "15550007777" / "messages.md"
    assert "Employer is interested in an AI workflow engagement." in owner_summary.read_text(encoding="utf-8")
    assert "Call SID: CA-emp" in employer_summary.read_text(encoding="utf-8")
    assert "Caller: We need help improving our operations with AI." in transcript_path.read_text(encoding="utf-8")
    assert "Employer is interested in an AI workflow engagement." in summary_path.read_text(encoding="utf-8")
    assert "Employer is interested in an AI workflow engagement." in messages_path.read_text(encoding="utf-8")
    assert notifications
    assert "the OpenCAS agent received an employment inquiry call." in notifications[0]["text"]
    assert notifications[0]["document_path"] == audio_path


@pytest.mark.asyncio
async def test_owner_poll_returns_background_reply_when_ready(tmp_path: Path) -> None:
    _runtime, service = _phone_service(tmp_path, llm_content="Owner response.")
    loop = asyncio.get_running_loop()
    task = loop.create_future()
    task.set_result(
        ResolvedPhoneReply(
            text="Owner response.",
            prompt_verb="<Say>Owner response.</Say>",
        )
    )
    service._pending_phone_replies["reply-123"] = PendingPhoneReply(
        token="reply-123",
        created_at=time.time(),
        call_sid="CA-owner",
        caller_number="+15551234567",
        task=task,
    )

    xml = await service.handle_poll_webhook(
        request_url="https://opencas.example.com/api/phone/twilio/poll?reply_token=reply-123",
        webhook_base_url="https://opencas.example.com",
        form_data={"From": "+15551234567", "CallSid": "CA-owner"},
        provided_signature=None,
        reply_token="reply-123",
    )

    assert "Owner response." in xml
    assert "<Gather" in xml
    assert "reply-123" not in service._pending_phone_replies


def test_validate_media_stream_request_allows_secret_fallback_without_auth_token(
    tmp_path: Path,
) -> None:
    _runtime, service = _phone_service(tmp_path)
    service.config = service.config.model_copy(update={"webhook_secret": "bridge-secret"})

    assert service.validate_media_stream_request(
        request_url="wss://opencasagent.com/api/phone/twilio/media/token",
        provided_signature=None,
        stream_secret=service._stream_bridge_token(),
    ) is True
    assert service.validate_media_stream_request(
        request_url="wss://opencasagent.com/api/phone/twilio/media/token",
        provided_signature=None,
        stream_secret="wrong-token",
    ) is False


@pytest.mark.asyncio
async def test_low_trust_qa_uses_workspace_knowledge_only(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    runtime, service = _phone_service(
        tmp_path,
        contacts=[
            {
                "phone_number": "+15550001111",
                "display_name": "Alex",
                "allowed_actions": ["knowledge_qa"],
                "workspace_subdir": "phone/contacts/alex",
                "notes": "Alex can ask about project Neptune.",
            }
        ],
        llm_content="Neptune is the approved internal codename.",
    )
    knowledge_dir = runtime.workspace_root / "phone" / "contacts" / "alex"
    knowledge_dir.mkdir(parents=True, exist_ok=True)
    (knowledge_dir / "notes.md").write_text(
        "Project Neptune is the approved internal codename for the migration.",
        encoding="utf-8",
    )

    async def _fake_voice_prompt(*, text: str, webhook_base_url: str, expressive: bool) -> str:
        assert text == "Neptune is the approved internal codename."
        assert expressive is False
        return "<Say>Neptune is the approved internal codename.</Say>"

    monkeypatch.setattr(service, "_build_voice_prompt_verb", _fake_voice_prompt)

    xml = await service.handle_gather_webhook(
        request_url="https://opencas.example.com/api/phone/twilio/gather",
        webhook_base_url="https://opencas.example.com",
        form_data={"From": "+15550001111", "SpeechResult": "What is Neptune?", "CallSid": "CA-qa"},
        provided_signature=None,
    )

    llm_call = runtime.llm.calls[0]
    assert llm_call["source"] == "tool_use_loop"
    assert llm_call["session_id"] == "phone:+15550001111"
    assert llm_call["tools"]
    tool_names = {tool["function"]["name"] for tool in llm_call["tools"]}
    assert {"fs_read_file", "fs_list_dir", "fs_write_file"} <= tool_names
    assert "bounded caller workspace toolset" in llm_call["messages"][0]["content"]
    assert "Project Neptune is the approved internal codename" in llm_call["messages"][0]["content"]
    assert "Alex can ask about project Neptune." in llm_call["messages"][0]["content"]
    session_entries = runtime.ctx.context_store.entries["phone:+15550001111"]
    assert [entry.role for entry in session_entries] == [MessageRole.USER, MessageRole.ASSISTANT]
    assert len(runtime.recorded_episodes) == 2
    assert "Neptune is the approved internal codename." in xml


@pytest.mark.asyncio
async def test_low_trust_workspace_tools_can_append_notes_without_escape(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    note_path = tmp_path / "workspace" / "phone" / "contacts" / "alex" / "call-log.md"
    runtime, service = _phone_service(
        tmp_path,
        contacts=[
            {
                "phone_number": "+15550001111",
                "display_name": "Alex",
                "allowed_actions": ["knowledge_qa"],
                "workspace_subdir": "phone/contacts/alex",
            }
        ],
        llm_responses=[
            {
                "choices": [
                    {
                        "message": {
                            "content": "",
                            "tool_calls": [
                                {
                                    "id": "call-1",
                                    "type": "function",
                                    "function": {
                                        "name": "fs_write_file",
                                        "arguments": (
                                            "{"
                                            f"\"file_path\": \"{note_path}\", "
                                            "\"content\": \"Client requested a Friday callback.\""
                                            "}"
                                        ),
                                    },
                                }
                            ],
                        }
                    }
                ]
            },
            {"choices": [{"message": {"content": "I recorded your callback request in your file."}}]},
        ],
    )

    async def _fake_voice_prompt(*, text: str, webhook_base_url: str, expressive: bool) -> str:
        return f"<Say>{text}</Say>"

    monkeypatch.setattr(service, "_build_voice_prompt_verb", _fake_voice_prompt)

    xml = await service.handle_gather_webhook(
        request_url="https://opencas.example.com/api/phone/twilio/gather",
        webhook_base_url="https://opencas.example.com",
        form_data={"From": "+15550001111", "SpeechResult": "Please note I need a Friday callback", "CallSid": "CA-note"},
        provided_signature=None,
    )

    assert note_path.read_text(encoding="utf-8") == "Client requested a Friday callback.\n"
    assert "I recorded your callback request in your file." in xml
    assert any(item["kind"].value == "action" for item in runtime.recorded_episodes)


@pytest.mark.asyncio
async def test_voicemail_only_contact_is_persisted_without_llm(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    runtime, service = _phone_service(
        tmp_path,
        contacts=[
            {
                "phone_number": "+15550002222",
                "display_name": "Jordan",
                "allowed_actions": ["leave_message"],
            }
        ],
    )

    async def _fake_voice_prompt(*, text: str, webhook_base_url: str, expressive: bool) -> str:
        return f"<Say>{text}</Say>"

    monkeypatch.setattr(service, "_build_voice_prompt_verb", _fake_voice_prompt)

    xml = await service.handle_gather_webhook(
        request_url="https://opencas.example.com/api/phone/twilio/gather",
        webhook_base_url="https://opencas.example.com",
        form_data={"From": "+15550002222", "SpeechResult": "Tell the OpenCAS agent I called", "CallSid": "CA-voicemail"},
        provided_signature=None,
    )

    assert runtime.llm.calls == []
    assert runtime.converse_calls == []
    session_entries = runtime.ctx.context_store.entries["phone:+15550002222"]
    assert len(session_entries) == 1
    assert session_entries[0].meta["phone"]["mode"] == "voicemail"
    assert session_entries[0].content == "Tell the OpenCAS agent I called"
    assert "Thanks. I saved your message for the OpenCAS agent." in xml


@pytest.mark.asyncio
async def test_place_owner_call_uses_twilio_rest_api(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    runtime, service = _phone_service(tmp_path)
    captured = {}

    class _FakeResponse:
        def raise_for_status(self) -> None:
            return None

        def json(self):
            return {"sid": "CA123", "status": "queued"}

    class _FakeAsyncClient:
        def __init__(self, *, auth, timeout):
            captured["auth"] = auth
            captured["timeout"] = timeout

        async def __aenter__(self):
            return self

        async def __aexit__(self, exc_type, exc, tb):
            return None

        async def post(self, url: str, data):
            captured["url"] = url
            captured["data"] = dict(data)
            return _FakeResponse()

    monkeypatch.setattr(
        service,
        "_twilio_credentials",
        lambda: TwilioCredentials("AC123", "SK123", "secret"),
    )
    monkeypatch.setattr("opencas.phone_integration.httpx.AsyncClient", _FakeAsyncClient)
    monkeypatch.setattr("opencas.phone_integration.secrets.token_urlsafe", lambda _length: "token123")

    result = await service.place_owner_call(message="Need you on the console.", reason="manual override")

    assert result["call_sid"] == "CA123"
    assert captured["auth"] == ("SK123", "secret")
    assert captured["url"] == "https://api.twilio.com/2010-04-01/Accounts/AC123/Calls.json"
    assert captured["data"]["From"] == "+15557654321"
    assert captured["data"]["To"] == "+15551234567"
    assert (
        captured["data"]["Url"]
        == "https://opencas.example.com/api/phone/twilio/voice?call_token=token123"
    )


@pytest.mark.asyncio
async def test_autoconfigure_twilio_selects_account_number_and_updates_webhook(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _runtime, service = _phone_service(tmp_path)
    captured = {"get_urls": [], "post_urls": []}

    class _FakeResponse:
        def __init__(self, payload):
            self._payload = payload

        def raise_for_status(self) -> None:
            return None

        def json(self):
            return self._payload

    class _FakeAsyncClient:
        def __init__(self, *, auth, timeout):
            captured["auth"] = auth
            captured["timeout"] = timeout

        async def __aenter__(self):
            return self

        async def __aexit__(self, exc_type, exc, tb):
            return None

        async def get(self, url: str):
            captured["get_urls"].append(url)
            return _FakeResponse(
                {
                    "incoming_phone_numbers": [
                        {
                            "sid": "PN123",
                            "friendly_name": "the OpenCAS agent",
                            "phone_number": "(484) 673-6227",
                            "voice_url": None,
                            "voice_method": None,
                        }
                    ]
                }
            )

        async def post(self, url: str, data):
            captured["post_urls"].append(url)
            captured["post_data"] = dict(data)
            return _FakeResponse(
                {
                    "sid": "PN123",
                    "friendly_name": "the OpenCAS agent",
                    "phone_number": "(484) 673-6227",
                    "voice_url": data["VoiceUrl"],
                    "voice_method": data["VoiceMethod"],
                }
            )

    monkeypatch.setattr(
        service,
        "_twilio_credentials",
        lambda: TwilioCredentials("AC123", "SK123", "secret"),
    )
    monkeypatch.setattr("opencas.phone_integration.httpx.AsyncClient", _FakeAsyncClient)
    monkeypatch.setattr("opencas.phone_integration.secrets.token_urlsafe", lambda _length: "bridge-secret")

    result = await service.autoconfigure_twilio(
        enabled=True,
        public_base_url="https://opencas.example.com",
        owner_phone_number="+17203340532",
        owner_display_name="Cabew",
    )

    settings = result["settings"]
    assert settings.enabled is True
    assert settings.owner_phone_number == "+17203340532"
    assert settings.twilio_from_number == "+14846736227"
    assert captured["auth"] == ("SK123", "secret")
    assert captured["get_urls"] == ["https://api.twilio.com/2010-04-01/Accounts/AC123/IncomingPhoneNumbers.json"]
    assert captured["post_urls"] == ["https://api.twilio.com/2010-04-01/Accounts/AC123/IncomingPhoneNumbers/PN123.json"]
    assert captured["post_data"] == {
        "VoiceUrl": "https://opencas.example.com/api/phone/twilio/voice?bridge_token=bridge-secret",
        "VoiceMethod": "POST",
    }
    assert result["selected_number"]["phone_number"] == "+14846736227"
    assert settings.webhook_secret == "bridge-secret"


@pytest.mark.asyncio
async def test_place_owner_call_falls_back_to_inline_twiml_without_public_base_url(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _runtime, service = _phone_service(tmp_path)
    service.config = service.config.model_copy(update={"public_base_url": None})
    captured = {}

    class _FakeResponse:
        def raise_for_status(self) -> None:
            return None

        def json(self):
            return {"sid": "CA124", "status": "queued"}

    class _FakeAsyncClient:
        def __init__(self, *, auth, timeout):
            captured["auth"] = auth
            captured["timeout"] = timeout

        async def __aenter__(self):
            return self

        async def __aexit__(self, exc_type, exc, tb):
            return None

        async def post(self, url: str, data):
            captured["url"] = url
            captured["data"] = dict(data)
            return _FakeResponse()

    monkeypatch.setattr(
        service,
        "_twilio_credentials",
        lambda: TwilioCredentials("AC123", "SK123", "secret"),
    )
    monkeypatch.setattr("opencas.phone_integration.httpx.AsyncClient", _FakeAsyncClient)

    result = await service.place_owner_call(message="Need you on the console.", reason="manual override")

    assert result["call_sid"] == "CA124"
    assert result["callback_url"] is None
    assert result["voice_mode"] == "inline_twiml"
    assert captured["auth"] == ("SK123", "secret")
    assert captured["data"]["Twiml"] == (
        '<?xml version="1.0" encoding="UTF-8"?><Response>'
        "<Say>Need you on the console.</Say><Hangup/></Response>"
    )
    assert "Url" not in captured["data"]
