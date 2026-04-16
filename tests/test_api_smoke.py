"""Fast API smoke coverage without TestClient.

These checks are intentionally narrow and direct. In the current Python 3.14
shell environment, FastAPI `TestClient` can hang even for trivial requests, so
the qualification baseline uses direct route invocation instead.
"""

from __future__ import annotations

import io
from types import SimpleNamespace
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import patch

import pytest
from starlette.datastructures import Headers, UploadFile

from opencas.api.routes.chat import ChatAttachmentInput, ChatSendRequest, build_chat_router
from opencas.api.routes.monitor import build_monitor_router
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
