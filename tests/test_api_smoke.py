"""Fast API smoke coverage without TestClient.

These checks are intentionally narrow and direct. In the current Python 3.14
shell environment, FastAPI `TestClient` can hang even for trivial requests, so
the qualification baseline uses direct route invocation instead.
"""

from __future__ import annotations

import io
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace
from unittest.mock import patch

import pytest
from starlette.requests import Request

from opencas.api.routes.chat import ChatAttachmentInput, ChatSendRequest, build_chat_router
from opencas.api.routes.monitor import build_monitor_router
from opencas.api.routes.phone import (
    PhoneAutoconfigureRequest,
    PhoneCallOwnerRequest,
    PhoneConfigUpdateRequest,
    _external_base_url,
    _external_request_url,
    build_phone_router,
)
from opencas.phone_config import PhoneRuntimeConfig
from opencas.api.routes.operations import (
    build_operations_router,
)
from opencas.api.server import ChatRequest, create_app
from tests.test_dashboard_api import FakeGatewayManager, FakeRuntime
from tests.test_operations_routes import _make_mock_runtime


def _route_endpoint(router, path: str):
    return next(
        route.endpoint
        for route in router.routes
        if getattr(route, "path", None) == path
    )


class AttachmentAwareRuntime(FakeRuntime):
    def __init__(self, state_dir: Path):
        super().__init__()
        self.ctx.config.state_dir = str(state_dir)
        self.last_converse = None

    async def converse(self, text, session_id=None, user_meta=None):
        self.last_converse = {
            "text": text,
            "session_id": session_id,
            "user_meta": user_meta,
        }
        return "resume reviewed"


class SessionListingStore:
    def __init__(self):
        self.calls = []

    async def list_session_ids(self, limit=50, status="active"):
        self.calls.append(("list", limit, status))
        return [{"session_id": "active-1", "status": status, "message_count": 0}]

    async def search_sessions(self, query, status="active", limit=20):
        self.calls.append(("search", query, status, limit))
        return [{"session_id": "archived-1", "status": status, "name": "Archived Session", "message_count": 2}]


class InMemoryUploadFile:
    def __init__(self, *, filename: str, content: bytes, content_type: str) -> None:
        self.filename = filename
        self.content_type = content_type
        self._content = content
        self.file = io.BytesIO(content)

    async def read(self) -> bytes:
        return self._content


@pytest.mark.asyncio
async def test_monitor_health_direct_invocation() -> None:
    runtime = FakeRuntime()
    router = build_monitor_router(runtime)

    endpoint = _route_endpoint(router, "/api/monitor/health")
    response = await endpoint()

    assert response.overall == "pass"
    assert response.failures == 0


@pytest.mark.asyncio
async def test_chat_context_summary_direct_invocation() -> None:
    runtime = FakeRuntime()
    runtime.ctx.llm = type(
        "L",
        (),
        {
            "manager": FakeGatewayManager(),
            "default_model": "kimi-coding/k2p5",
        },
    )()
    router = build_chat_router(runtime)

    endpoint = _route_endpoint(router, "/api/chat/context-summary")
    response = await endpoint(session_id=None, task_limit=6)

    assert response.lane["model"] == "kimi-coding/k2p5"
    assert response.lane["provider"] == "kimi-coding"
    assert response.executive["recommend_pause"] is False
    assert response.consolidation["commitments_extracted_from_chat"] == 1


@pytest.mark.asyncio
async def test_phone_routes_direct_invocation() -> None:
    runtime = FakeRuntime()
    router = build_phone_router(runtime)

    status_endpoint = _route_endpoint(router, "/api/phone/status")
    config_endpoint = _route_endpoint(router, "/api/phone/config")
    autoconfigure_endpoint = _route_endpoint(router, "/api/phone/autoconfigure")
    call_endpoint = _route_endpoint(router, "/api/phone/call-owner")

    status = await status_endpoint()
    assert status["twilio_from_number"] == "+15557654321"
    assert status["owner"]["phone_number"] == "+15551234567"

    updated = await config_endpoint(
        PhoneConfigUpdateRequest(
            enabled=True,
            public_base_url="https://opencas.example.com",
            webhook_signature_required=True,
            twilio_from_number="+15557654321",
            owner_phone_number="+15551234567",
            owner_display_name="Operator",
            owner_workspace_subdir="phone/owner",
            contacts=[],
        )
    )
    assert updated["saved"] is True

    autoconfigured = await autoconfigure_endpoint(
        PhoneAutoconfigureRequest(
            enabled=True,
            public_base_url="https://opencas.example.com",
            owner_phone_number="+15551234567",
            owner_display_name="Operator",
        )
    )
    assert autoconfigured["autoconfigured"] is True
    assert autoconfigured["selected_number"]["sid"] == "PN123"

    call = await call_endpoint(PhoneCallOwnerRequest(message="Ping", reason="test"))
    assert call["call_sid"] == "CA123"


def test_phone_route_external_url_prefers_configured_public_base_url() -> None:
    runtime = SimpleNamespace(
        phone_settings=PhoneRuntimeConfig(
            enabled=True,
            public_base_url="https://opencas.example.com",
            twilio_from_number="+15557654321",
            owner_phone_number="+15551234567",
        )
    )
    request = Request(
        {
            "type": "http",
            "method": "POST",
            "scheme": "http",
            "server": ("127.0.0.1", 32147),
            "client": ("127.0.0.1", 50000),
            "path": "/api/phone/twilio/voice",
            "query_string": b"call_token=abc123",
            "headers": [(b"host", b"127.0.0.1:32147")],
        }
    )

    assert _external_base_url(runtime, request) == "https://opencas.example.com"
    assert (
        _external_request_url(runtime, request)
        == "https://opencas.example.com/api/phone/twilio/voice?call_token=abc123"
    )


def test_phone_route_external_url_also_supports_callable_phone_settings() -> None:
    runtime = SimpleNamespace(
        phone_settings=lambda: PhoneRuntimeConfig(
            enabled=True,
            public_base_url="https://opencas.example.com",
            twilio_from_number="+15557654321",
            owner_phone_number="+15551234567",
        )
    )
    request = Request(
        {
            "type": "http",
            "method": "POST",
            "scheme": "http",
            "server": ("127.0.0.1", 32147),
            "client": ("127.0.0.1", 50000),
            "path": "/api/phone/twilio/voice",
            "query_string": b"call_token=abc123",
            "headers": [(b"host", b"127.0.0.1:32147")],
        }
    )

    assert _external_base_url(runtime, request) == "https://opencas.example.com"
    assert (
        _external_request_url(runtime, request)
        == "https://opencas.example.com/api/phone/twilio/voice?call_token=abc123"
    )


@pytest.mark.asyncio
async def test_operations_sessions_direct_invocation() -> None:
    runtime = _make_mock_runtime()
    router = build_operations_router(runtime)

    endpoint = _route_endpoint(router, "/api/operations/sessions")
    response = await endpoint(scope_key=None)

    assert response.total_processes == 1
    assert response.total_pty == 1
    assert response.total_browser == 1
    assert response.scopes[0].scope_key == "qualification"


@pytest.mark.asyncio
async def test_operations_qualification_direct_invocation() -> None:
    runtime = _make_mock_runtime()
    router = build_operations_router(runtime)
    endpoint = _route_endpoint(router, "/api/operations/qualification")

    payload = {
        "total_runs": 3,
        "total_direct_checks": 10,
        "total_agent_checks": 8,
        "direct_success_rate": 0.9,
        "agent_success_rate": 0.75,
        "average_run_duration_seconds": 42.5,
        "models": ["kimi-coding/k2p5"],
        "embedding_models": ["google/gemini-embedding-2-preview"],
        "agent_checks": {
            "integrated_operator_workflow": {
                "runs": 2,
                "successes": 1,
                "failures": 1,
                "success_rate": 0.5,
                "timeouts": 0,
            }
        },
    }

    with TemporaryDirectory() as tmpdir:
        tmp = Path(tmpdir)
        summary_path = tmp / "live_validation_summary.json"
        history_path = tmp / "qualification_rerun_history.jsonl"
        remediation_path = tmp / "qualification_remediation_rollup.json"
        runs_dir = tmp / "runs"
        summary_path.write_text(__import__("json").dumps(payload), encoding="utf-8")
        history_path.write_text("", encoding="utf-8")
        remediation_path.write_text(
            __import__("json").dumps(
                {
                    "count": 1,
                    "items": [
                        {
                            "request_id": "req-1",
                            "label": "integrated_operator_workflow",
                            "returncode": 0,
                            "recommended_action": "continue_testing",
                        }
                    ],
                }
            ),
            encoding="utf-8",
        )
        runs_dir.mkdir()

        with (
            patch("opencas.api.routes.operations.QUALIFICATION_SUMMARY_PATH", summary_path),
            patch("opencas.api.routes.operations.QUALIFICATION_RERUN_HISTORY_PATH", history_path),
            patch("opencas.api.routes.operations.QUALIFICATION_REMEDIATION_PATH", remediation_path),
            patch("opencas.api.routes.operations.VALIDATION_RUNS_DIR", runs_dir),
        ):
            response = await endpoint()

    assert response.summary["agent_checks"]["integrated_operator_workflow"]["success_rate"] == 0.5
    assert response.remediation_rollup["items"][0]["recommended_action"] == "continue_testing"


def test_main_app_registers_operations_routes() -> None:
    app = create_app(FakeRuntime())
    paths = {route.path for route in app.routes if getattr(route, "path", None)}

    assert "/api/operations/sessions" in paths
    assert "/api/operations/qualification" in paths
    assert "/api/chat/context-summary" in paths
    assert "/api/monitor/runtime" in paths
    assert "/api/phone/status" in paths
    assert "/api/phone/autoconfigure" in paths
    assert "/api/phone/twilio/voice" in paths


@pytest.mark.asyncio
async def test_root_chat_endpoint_uses_shared_chat_transport(tmp_path: Path) -> None:
    runtime = AttachmentAwareRuntime(tmp_path / "state" / "context.db")
    app = create_app(runtime)

    endpoint = _route_endpoint(app, "/chat")
    response = await endpoint(ChatRequest(session_id="root-chat", message="hello"))

    assert response.response == "resume reviewed"
    assert runtime.last_converse == {
        "text": "hello",
        "session_id": "root-chat",
        "user_meta": None,
    }


@pytest.mark.asyncio
async def test_chat_send_materializes_text_attachment_for_runtime(tmp_path: Path) -> None:
    runtime = AttachmentAwareRuntime(tmp_path / "state" / "context.db")
    router = build_chat_router(runtime)

    upload_endpoint = _route_endpoint(router, "/api/chat/upload")
    upload = InMemoryUploadFile(
        filename="resume.md",
        content=b"# Resume\n- Built Python automation\n",
        content_type="text/markdown",
    )
    uploaded = await upload_endpoint(file=upload)

    send_endpoint = _route_endpoint(router, "/api/chat/send")
    response = await send_endpoint(
        ChatSendRequest(
            session_id="resume-session",
            message="What do you think of my resume?",
            attachments=[
                ChatAttachmentInput(
                    filename=uploaded.filename,
                    url=uploaded.url,
                    path=uploaded.path,
                    media_type=uploaded.media_type,
                )
            ],
        )
    )

    assert response.response == "resume reviewed"
    assert runtime.last_converse["text"] == "What do you think of my resume?"
    attachment = runtime.last_converse["user_meta"]["attachments"][0]
    assert attachment["filename"] == "resume.md"
    assert attachment["media_type"].startswith("text/")
    assert "# Resume" in attachment["text_content"]


@pytest.mark.asyncio
async def test_chat_sessions_route_delegates_status_and_search_to_context_store() -> None:
    store = SessionListingStore()
    runtime = SimpleNamespace(
        ctx=SimpleNamespace(
            context_store=store,
            config=SimpleNamespace(state_dir="/tmp/opencas-test-context.db"),
        )
    )
    router = build_chat_router(runtime)
    endpoint = _route_endpoint(router, "/api/chat/sessions")

    searched = await endpoint(limit=12, status="archived", q="resume")
    listed = await endpoint(limit=7, status="active", q=None)

    assert searched.sessions[0]["session_id"] == "archived-1"
    assert listed.sessions[0]["session_id"] == "active-1"
    assert store.calls == [
        ("search", "resume", "archived", 12),
        ("list", 7, "active"),
    ]


@pytest.mark.asyncio
async def test_chat_voice_status_route_reports_provider_availability() -> None:
    runtime = AttachmentAwareRuntime(Path("/tmp/opencas-voice-state") / "context.db")
    router = build_chat_router(runtime)
    endpoint = _route_endpoint(router, "/api/chat/voice/status")

    with patch("opencas.api.routes.chat.voice_status") as status_mock:
        status_mock.return_value = SimpleNamespace(
            to_dict=lambda: {
                "elevenlabs_available": True,
                "local_stt_available": True,
                "local_tts_available": True,
                "elevenlabs_voice_id": "voice-123",
                "local_voice_name": "Aira",
                "local_voice_resolved": "en-US-AriaNeural",
                "expressive_supported": True,
            }
        )
        response = await endpoint()

    assert response.elevenlabs_available is True
    assert response.local_voice_resolved == "en-US-AriaNeural"


@pytest.mark.asyncio
async def test_chat_voice_transcribe_route_returns_voice_input_metadata(tmp_path: Path) -> None:
    runtime = AttachmentAwareRuntime(tmp_path / "state" / "context.db")
    router = build_chat_router(runtime)
    endpoint = _route_endpoint(router, "/api/chat/voice/transcribe")

    upload = InMemoryUploadFile(
        filename="voice.webm",
        content=b"fake-audio",
        content_type="audio/webm",
    )

    with patch("opencas.api.routes.chat.transcribe_audio") as transcribe_mock:
        transcribe_mock.return_value = SimpleNamespace(
            text="hello from voice",
            to_meta=lambda: {
                "provider": "elevenlabs",
                "mode": "hosted",
                "model": "scribe_v2",
                "warning": None,
                "audio": {
                    "filename": "voice.webm",
                    "path": str(tmp_path / "voice.webm"),
                    "url": "/api/chat/uploads/voice.webm",
                    "media_type": "audio/webm",
                    "size_bytes": 10,
                },
            },
        )
        response = await endpoint(file=upload, prefer_local=False, language_code="en")

    assert response.transcript == "hello from voice"
    assert response.voice_input.provider == "elevenlabs"
    assert response.voice_input.audio["url"] == "/api/chat/uploads/voice.webm"


@pytest.mark.asyncio
async def test_chat_send_can_request_voice_synthesis(tmp_path: Path) -> None:
    runtime = AttachmentAwareRuntime(tmp_path / "state" / "context.db")
    router = build_chat_router(runtime)
    endpoint = _route_endpoint(router, "/api/chat/send")

    with patch("opencas.api.chat_service.synthesize_speech") as synth_mock, patch(
        "opencas.api.chat_service.annotate_latest_assistant_voice_output"
    ) as annotate_mock:
        synth_mock.return_value = SimpleNamespace(
            to_meta=lambda: {
                "provider": "edge-tts",
                "mode": "local",
                "model": "edge-tts",
                "expressive": False,
                "voice_id": None,
                "voice_name": "Aira (en-US-AriaNeural)",
                "warning": None,
                "filename": "voice.mp3",
                "path": str(tmp_path / "voice.mp3"),
                "url": "/api/chat/uploads/voice.mp3",
                "media_type": "audio/mpeg",
                "size_bytes": 32,
            }
        )
        response = await endpoint(
            ChatSendRequest(
                session_id="voice-send",
                message="say hello back",
                voice_input={
                    "provider": "elevenlabs",
                    "mode": "hosted",
                    "model": "scribe_v2",
                    "audio": {"filename": "voice.webm", "url": "/api/chat/uploads/voice.webm"},
                },
                speak_response=True,
                voice_prefer_local=True,
                voice_expressive=True,
            )
        )

    assert response.response == "resume reviewed"
    assert response.voice_output["provider"] == "edge-tts"
    assert runtime.last_converse["user_meta"]["voice_input"]["provider"] == "elevenlabs"
    annotate_mock.assert_awaited()


def test_dashboard_static_contains_operations_surface() -> None:
    dashboard_path = Path("opencas/dashboard/static/index.html")
    body = dashboard_path.read_text(encoding="utf-8")

    assert "Operations" in body
    assert "/dashboard/static/js/http_helpers.js" in body
    assert "/dashboard/static/js/operations_helpers.js" in body
    assert "/api/operations/sessions" in body
    assert "/api/operations/qualification" in body
    assert "/api/operations/validation-runs?limit=10" in body
    assert 'data-panel="operations-sessions"' in body
    assert 'data-panel="operations-qualification"' in body


def test_dashboard_static_contains_memory_module() -> None:
    dashboard_path = Path("opencas/dashboard/static/index.html")
    body = dashboard_path.read_text(encoding="utf-8")
    module_body = Path("opencas/dashboard/static/js/memory_app.js").read_text(encoding="utf-8")

    assert "/dashboard/static/js/memory_app.js" in body
    assert 'x-data="memoryApp()"' in body
    assert "/api/memory/landscape" in module_body
    assert "/api/memory/retrieval-inspect" in module_body


def test_dashboard_static_contains_voice_chat_controls() -> None:
    dashboard_path = Path("opencas/dashboard/static/index.html")
    body = dashboard_path.read_text(encoding="utf-8")

    assert "/api/chat/voice/status" in body
    assert "/api/chat/voice/transcribe" in body
    assert "/api/chat/voice/synthesize" in body
    assert "Expressive ElevenLabs" in body
    assert "Auto-speak replies" in body
    assert "webkitAudioContext" in body
    assert "audio/wav" in body
