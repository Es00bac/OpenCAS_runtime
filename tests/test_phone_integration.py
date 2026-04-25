"""Tests for Twilio-backed phone bridge behavior."""

from __future__ import annotations

import asyncio
import time
from pathlib import Path
from types import SimpleNamespace

import pytest

from opencas.context.models import MessageRole
from opencas.phone_config import PhoneRuntimeConfig
from opencas.phone_integration import (
    PendingPhoneReply,
    PhoneBridgeService,
    ResolvedPhoneReply,
    TwilioCredentials,
)


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
        owner_display_name="Operator",
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
        request_url="https://opencas.example.com/api/phone/twilio/voice",
        form_data={
            "CallSid": "CA123",
            "From": "+15551234567",
            "To": "+15557654321",
        },
        provided_signature="abc123",
    )

    assert valid is True
    assert captured == {
        "token": "secret",
        "url": "https://opencas.example.com/api/phone/twilio/voice",
        "params": {
            "CallSid": "CA123",
            "From": "+15551234567",
            "To": "+15557654321",
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
        request_url="https://opencas.example.com/api/phone/twilio/voice",
        webhook_base_url="https://opencas.example.com",
        form_data=form,
        provided_signature="sig123",
    )

    assert captured["request_url"] == "https://opencas.example.com/api/phone/twilio/voice"
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
        request_url="https://opencas.example.com/api/phone/twilio/voice?bridge_token=bridge-secret",
        form_data={"CallSid": "CA123", "From": "+15551234567", "To": "+15557654321"},
        provided_signature=None,
        bridge_token="bridge-secret",
    ) is True


@pytest.mark.asyncio
async def test_owner_gather_queues_background_reply_for_non_heuristic_request(
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
async def test_owner_voice_webhook_returns_connect_stream_twiml(tmp_path: Path) -> None:
    _runtime, service = _phone_service(tmp_path)

    xml = await service.handle_voice_webhook(
        request_url="https://opencas.example.com/api/phone/twilio/voice",
        webhook_base_url="https://opencas.example.com",
        form_data={"From": "+15551234567", "To": "+15557654321", "CallSid": "CA-owner"},
        provided_signature=None,
    )

    assert "<Connect>" in xml
    assert "<Stream " in xml
    assert "wss://opencas.example.com/api/phone/twilio/media/" in xml
    assert "introMessage" in xml
    assert "<Gather" not in xml
    assert "<Say>" not in xml


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


@pytest.mark.asyncio
async def test_generate_owner_live_reply_uses_llm_without_full_converse(tmp_path: Path) -> None:
    runtime, service = _phone_service(tmp_path, llm_content="Running steady. What do you need?")

    reply = await service.generate_owner_live_reply(
        caller=service._resolve_caller(from_number="+15551234567"),
        transcript="How are you doing right now?",
        call_sid="CA-owner-live",
    )

    assert reply == "Running steady. What do you need?"
    assert runtime.converse_calls == []
    assert runtime.llm.calls
    llm_call = runtime.llm.calls[0]
    assert llm_call["source"] == "phone_owner_live"
    assert llm_call["complexity"] == "light"
    session_entries = runtime.ctx.context_store.entries["phone:+15551234567"]
    assert [entry.role for entry in session_entries][-2:] == [MessageRole.USER, MessageRole.ASSISTANT]


def test_validate_media_stream_request_allows_secret_fallback_without_auth_token(
    tmp_path: Path,
) -> None:
    _runtime, service = _phone_service(tmp_path)
    service.config = service.config.model_copy(update={"webhook_secret": "bridge-secret"})

    assert service.validate_media_stream_request(
        request_url="wss://opencas.example.com/api/phone/twilio/media/token",
        provided_signature=None,
        stream_secret=service._stream_bridge_token(),
    ) is True
    assert service.validate_media_stream_request(
        request_url="wss://opencas.example.com/api/phone/twilio/media/token",
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
        form_data={"From": "+15550002222", "SpeechResult": "Tell the operator I called", "CallSid": "CA-voicemail"},
        provided_signature=None,
    )

    assert runtime.llm.calls == []
    assert runtime.converse_calls == []
    session_entries = runtime.ctx.context_store.entries["phone:+15550002222"]
    assert len(session_entries) == 1
    assert session_entries[0].meta["phone"]["mode"] == "voicemail"
    assert session_entries[0].content == "Tell the operator I called"
    assert "Thanks. I saved your message for the operator." in xml


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
                            "friendly_name": "OpenCAS",
                            "phone_number": "(555) 765-4321",
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
                    "friendly_name": "OpenCAS",
                    "phone_number": "(555) 765-4321",
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
        owner_phone_number="+15551234567",
        owner_display_name="Operator",
    )

    settings = result["settings"]
    assert settings.enabled is True
    assert settings.owner_phone_number == "+15551234567"
    assert settings.twilio_from_number == "+15557654321"
    assert captured["auth"] == ("SK123", "secret")
    assert captured["get_urls"] == ["https://api.twilio.com/2010-04-01/Accounts/AC123/IncomingPhoneNumbers.json"]
    assert captured["post_urls"] == ["https://api.twilio.com/2010-04-01/Accounts/AC123/IncomingPhoneNumbers/PN123.json"]
    assert captured["post_data"] == {
        "VoiceUrl": "https://opencas.example.com/api/phone/twilio/voice?bridge_token=bridge-secret",
        "VoiceMethod": "POST",
    }
    assert result["selected_number"]["phone_number"] == "+15557654321"
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
