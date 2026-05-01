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
from fastapi import HTTPException
from starlette.requests import Request
from starlette.datastructures import Headers, UploadFile

from opencas.api.routes.chat import ChatAttachmentInput, ChatSendRequest, build_chat_router
from opencas.api.chat_service import chat_upload_dir
from opencas.api.routes.monitor import build_monitor_router
from opencas.api.routes.phone import (
    PhoneAutoconfigureRequest,
    PhoneCallOwnerRequest,
    PhoneConfigUpdateRequest,
    PhoneMenuConfigUpdateRequest,
    PhoneSessionProfilesUpdateRequest,
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
        self.ctx.config.agent_workspace_root = lambda: str(state_dir.parent / "workspace")
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
    recent_calls_endpoint = _route_endpoint(router, "/api/phone/recent-calls")
    call_detail_endpoint = _route_endpoint(router, "/api/phone/recent-calls/{call_sid}")
    config_endpoint = _route_endpoint(router, "/api/phone/config")
    session_profiles_endpoint = _route_endpoint(router, "/api/phone/session-profiles")
    menu_config_endpoint = _route_endpoint(router, "/api/phone/menu-config")
    autoconfigure_endpoint = _route_endpoint(router, "/api/phone/autoconfigure")
    call_endpoint = _route_endpoint(router, "/api/phone/call-owner")

    status = await status_endpoint()
    assert status["twilio_from_number"] == "+14846736227"
    assert status["owner"]["phone_number"] == "+17203340532"
    assert status["recent_calls"][0]["call_sid"] == "CA123"

    recent_calls = await recent_calls_endpoint()
    assert recent_calls["calls"][0]["call_sid"] == "CA123"

    call_detail = await call_detail_endpoint("CA123")
    assert call_detail["found"] is True
    assert call_detail["call"]["call_sid"] == "CA123"

    updated = await config_endpoint(
        PhoneConfigUpdateRequest(
            enabled=True,
            public_base_url="https://opencas.example.com",
            webhook_signature_required=True,
            twilio_from_number="+14846736227",
            owner_phone_number="+17203340532",
            owner_display_name="Cabew",
            owner_workspace_subdir="phone/owner",
            contacts=[],
        )
    )
    assert updated["saved"] is True

    session_profiles = await session_profiles_endpoint(
        PhoneSessionProfilesUpdateRequest(
            owner_entry_prompt="Press 1 for the owner.",
            owner_entry_reprompt="Press 1 for the owner again.",
            owner_pin_prompt="Enter owner pin.",
            owner_pin_retry_prompt="Try the pin again.",
            owner_pin_success_message="Verified.",
            owner_pin_failure_message="Denied.",
            public_prompt="Potential employers press 1.",
            public_reprompt="Press 1 for employer mode.",
            employer_enabled=True,
            employer_greeting="Employer greeting.",
            reject_enabled=True,
            reject_message="Not for this line.",
        )
    )
    assert session_profiles["saved"] is True

    menu_config = await menu_config_endpoint(
        PhoneMenuConfigUpdateRequest(
            config={
                "default_menu_key": "public_main",
                "owner_menu_key": "owner_entry",
                "menus": [
                    {"key": "owner_entry", "prompt": "Press 1 for the owner.", "options": []},
                    {"key": "public_main", "prompt": "Potential employers press 1.", "options": []},
                ],
            }
        )
    )
    assert menu_config["saved"] is True
    assert menu_config["menu_config_saved"] is True

    autoconfigured = await autoconfigure_endpoint(
        PhoneAutoconfigureRequest(
            enabled=True,
            public_base_url="https://opencas.example.com",
            owner_phone_number="+17203340532",
            owner_display_name="Cabew",
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
            twilio_from_number="+14846736227",
            owner_phone_number="+17203340532",
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
            twilio_from_number="+14846736227",
            owner_phone_number="+17203340532",
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
    assert "/api/phone/recent-calls" in paths
    assert "/api/phone/recent-calls/{call_sid}" in paths
    assert "/api/phone/menu-config" in paths
    assert "/api/phone/autoconfigure" in paths
    assert "/api/phone/session-profiles" in paths
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
    upload = UploadFile(
        filename="resume.md",
        file=io.BytesIO(b"# Resume\n- Built Python automation\n"),
        headers=Headers({"content-type": "text/markdown"}),
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


def test_chat_upload_dir_uses_managed_workspace(tmp_path: Path) -> None:
    runtime = AttachmentAwareRuntime(tmp_path / "state" / "context.db")

    upload_dir = chat_upload_dir(runtime)

    assert upload_dir == tmp_path / "state" / "workspace" / "chat_uploads"


@pytest.mark.asyncio
async def test_chat_voice_transcribe_route_returns_voice_input_metadata(tmp_path: Path) -> None:
    runtime = AttachmentAwareRuntime(tmp_path / "state" / "context.db")
    router = build_chat_router(runtime)
    endpoint = _route_endpoint(router, "/api/chat/voice/transcribe")

    upload = UploadFile(
        filename="voice.webm",
        file=io.BytesIO(b"fake-audio"),
        headers=Headers({"content-type": "audio/webm"}),
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
async def test_chat_voice_transcribe_route_reports_no_speech_detected(tmp_path: Path) -> None:
    runtime = AttachmentAwareRuntime(tmp_path / "state" / "context.db")
    router = build_chat_router(runtime)
    endpoint = _route_endpoint(router, "/api/chat/voice/transcribe")

    upload = UploadFile(
        filename="voice.webm",
        file=io.BytesIO(b"fake-audio"),
        headers=Headers({"content-type": "audio/webm"}),
    )

    with patch(
        "opencas.api.routes.chat.transcribe_audio",
        side_effect=HTTPException(
            status_code=422,
            detail="No speech was detected in the recording. Try again with a clearer utterance or higher microphone volume.",
        ),
    ):
        with pytest.raises(HTTPException) as exc_info:
            await endpoint(file=upload, prefer_local=True, language_code="en")

    assert exc_info.value.status_code == 422
    assert "No speech was detected" in str(exc_info.value.detail)


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
    assert "dashboard/static/js/http_helpers.js" in body
    assert "dashboard/static/js/operations_helpers.js" in body
    assert "/api/operations/sessions" in body
    assert "/api/operations/qualification" in body
    assert "/api/operations/validation-runs?limit=10" in body
    assert 'data-panel="operations-sessions"' in body
    assert 'data-panel="operations-qualification"' in body


def test_dashboard_static_uses_mounted_assets() -> None:
    dashboard_path = Path("opencas/dashboard/static/index.html")
    favicon_path = Path("opencas/dashboard/static/favicon.svg")
    body = dashboard_path.read_text(encoding="utf-8")

    assert 'href="dashboard/static/favicon.svg"' in body
    assert favicon_path.exists()
    assert 'src="images/logo-mark.png"' not in body


def test_dashboard_polling_components_clear_refresh_timers_on_destroy() -> None:
    dashboard_path = Path("opencas/dashboard/static/index.html")
    body = dashboard_path.read_text(encoding="utf-8")
    usage_section = body.split("window.usageApp = function()", 1)[1].split("window.overviewCharts = function()", 1)[0]
    overview_section = body.split("window.overviewCharts = function()", 1)[1]

    assert "destroy()" in usage_section
    assert "clearInterval(this._refreshTimer)" in usage_section
    assert "destroy()" in overview_section
    assert "clearInterval(this._refreshTimer)" in overview_section


def test_dashboard_charts_disable_animation_for_tab_teardown() -> None:
    dashboard_path = Path("opencas/dashboard/static/index.html")
    body = dashboard_path.read_text(encoding="utf-8")

    assert "Chart.defaults.animation = false" in body


def test_dashboard_log_viewer_closes_stream_on_destroy() -> None:
    dashboard_path = Path("opencas/dashboard/static/index.html")
    body = dashboard_path.read_text(encoding="utf-8")
    log_section = body.split("function logViewer()", 1)[1].split("kindColor(kind)", 1)[0]

    assert "destroy() {" in log_section
    assert "this.stopStreaming()" in log_section


def test_dashboard_static_contains_memory_module() -> None:
    dashboard_path = Path("opencas/dashboard/static/index.html")
    body = dashboard_path.read_text(encoding="utf-8")
    module_body = Path("opencas/dashboard/static/js/memory_app.js").read_text(encoding="utf-8")

    assert "/dashboard/static/js/memory_app.js" in body
    assert 'x-data="memoryApp()"' in body
    assert "/api/memory/landscape" in module_body
    assert "/api/memory/retrieval-inspect" in module_body


def test_dashboard_memory_atlas_css_is_not_globally_capped() -> None:
    css_body = Path("opencas/dashboard/static/css/app.css").read_text(encoding="utf-8")

    assert ".memory-atlas-wrap canvas" in css_body
    assert "height: clamp(420px, calc(100vh - 360px), 760px) !important;" in css_body
    assert "canvas {\n  max-height: 280px !important;\n}" not in css_body


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


def test_dashboard_static_keeps_chat_compose_outside_message_template() -> None:
    dashboard_path = Path("opencas/dashboard/static/index.html")
    body = dashboard_path.read_text(encoding="utf-8")

    template_start = body.index('<template x-for="m in filteredMessages()"')
    template_end = body.index("</template>", template_start)
    compose_index = body.index('class="chat-compose"')

    assert compose_index > template_end
    assert body.count('class="chat-compose"') == 1


def test_dashboard_static_contains_shadow_registry_triage_controls() -> None:
    dashboard_path = Path("opencas/dashboard/static/index.html")
    body = dashboard_path.read_text(encoding="utf-8")

    assert "/api/monitor/shadow-registry/cluster/triage" in body
    assert "Dismiss Cluster" in body
    assert "Restore Cluster" in body
    assert "Save Note" in body
