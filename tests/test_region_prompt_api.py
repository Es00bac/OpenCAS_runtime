from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

from fastapi import FastAPI
from fastapi.testclient import TestClient

from opencas.api.routes.chat import build_chat_router


class FakeDesktopContext:
    def __init__(self) -> None:
        self.calls: list[dict] = []

    async def analyze_region_for_conversation(self, **kwargs):
        self.calls.append(kwargs)
        return {
            "status": "observed",
            "conversation_prompt_note": "Selected desktop region context: fake visual evidence.",
            "capture": {"path": str(kwargs["image_path"])},
            "analysis": {"region_prompt": {"visual_summary": "fake visual evidence"}},
        }


class FakeRegionRuntime:
    def __init__(self, tmp_path: Path) -> None:
        self.desktop_context = FakeDesktopContext()
        self.ctx = SimpleNamespace(
            config=SimpleNamespace(
                state_dir=tmp_path / "state",
                session_id="default",
                agent_workspace_root=lambda: tmp_path / "workspace",
            ),
            somatic=None,
        )
        self.last_user_meta = None

    async def converse(self, message, *, session_id=None, user_meta=None):
        self.last_user_meta = user_meta
        return f"answered {message}"


def test_region_prompt_endpoint_analyzes_image_before_chat_turn(tmp_path: Path) -> None:
    runtime = FakeRegionRuntime(tmp_path)
    app = FastAPI()
    app.include_router(build_chat_router(runtime))
    client = TestClient(app)

    response = client.post(
        "/api/chat/region-prompt",
        data={
            "prompt": "What is this?",
            "session_id": "region-session",
            "selection": '{"x": 1, "y": 2, "width": 3, "height": 4}',
        },
        files={"file": ("region.png", b"fake-image", "image/png")},
    )

    assert response.status_code == 200
    assert response.json()["response"] == "answered What is this?"
    assert runtime.desktop_context.calls
    call = runtime.desktop_context.calls[0]
    assert call["prompt"] == "What is this?"
    assert call["session_id"] == "region-session"
    assert call["selection"] == {"x": 1, "y": 2, "width": 3, "height": 4}
    assert Path(call["image_path"]).exists()
    assert runtime.last_user_meta["desktop_context_turn"]["conversation_prompt_note"].startswith(
        "Selected desktop region context"
    )
    assert runtime.last_user_meta["attachments"][0]["media_type"] == "image/png"
    assert runtime.last_user_meta["conversation_actor"]["is_operator"] is True
