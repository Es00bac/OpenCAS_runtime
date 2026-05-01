"""Tests for the desktop-context body-double skill."""

from __future__ import annotations

import json
import os
from pathlib import Path
from types import SimpleNamespace

import pytest

from opencas.context import MessageRole
from opencas.desktop_context import (
    DesktopCapture,
    DesktopContextService,
    choose_screenshot_backend,
)
from opencas.execution.lanes import CommandLane
from opencas.plugins import PluginRegistry, SkillRegistry
from opencas.plugins.loader import load_plugin_from_manifest
from opencas.runtime.scheduler import AgentScheduler
from opencas.tools import ToolRegistry


class FakeContextStore:
    def __init__(self) -> None:
        self.entries: list[tuple[str, MessageRole, str, dict]] = []

    async def append(
        self,
        session_id: str,
        role: MessageRole,
        content: str,
        meta: dict | None = None,
    ) -> None:
        self.entries.append((session_id, role, content, meta or {}))


class FakeLLM:
    def __init__(self, payload: dict) -> None:
        self.payload = payload
        self.calls: list[dict] = []

    async def chat_completion(self, messages, **kwargs):
        self.calls.append({"messages": messages, "kwargs": kwargs})
        return {
            "choices": [
                {
                    "message": {
                        "content": json.dumps(self.payload),
                    }
                }
            ]
        }


class FakeRuntime:
    def __init__(self, tmp_path: Path, llm_payload: dict) -> None:
        self.ctx = SimpleNamespace(
            config=SimpleNamespace(
                state_dir=tmp_path / "state",
                session_id="default",
                agent_workspace_root=lambda: tmp_path / "workspace",
            ),
            context_store=FakeContextStore(),
        )
        self.llm = FakeLLM(llm_payload)
        self.traces: list[tuple[str, dict]] = []
        self.episodes: list[tuple[str, object, dict]] = []

    def _trace(self, event: str, payload: dict) -> None:
        self.traces.append((event, payload))

    async def _record_episode(self, content: str, kind, **kwargs):
        self.episodes.append((content, kind, kwargs))


def _capture_provider(path: Path) -> DesktopCapture:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(b"fake-png")
    return DesktopCapture(
        success=True,
        path=path,
        backend="fake",
        media_type="image/png",
        width=None,
        height=None,
    )


def test_choose_screenshot_backend_prefers_kde_spectacle() -> None:
    def fake_which(name: str) -> str | None:
        return f"/usr/bin/{name}" if name in {"spectacle", "import"} else None

    backend = choose_screenshot_backend("auto", which=fake_which)

    assert backend is not None
    assert backend.name == "spectacle"
    assert backend.command(Path("/tmp/shot.png")) == [
        "/usr/bin/spectacle",
        "-b",
        "-n",
        "-o",
        "/tmp/shot.png",
    ]


def test_desktop_context_config_is_disabled_by_default_and_persists(tmp_path: Path) -> None:
    runtime = FakeRuntime(tmp_path, {"should_speak": False, "activity_summary": "idle"})
    service = DesktopContextService(runtime=runtime, state_dir=runtime.ctx.config.state_dir)

    assert service.status()["config"]["enabled"] is False

    updated = service.configure(enabled=True, tts_enabled=False, capture_interval_seconds=42)
    assert updated["config"]["enabled"] is True
    assert updated["config"]["tts_enabled"] is False
    assert updated["config"]["capture_interval_seconds"] == 42

    reloaded = DesktopContextService(runtime=runtime, state_dir=runtime.ctx.config.state_dir)
    assert reloaded.status()["config"]["enabled"] is True
    assert reloaded.status()["config"]["tts_enabled"] is False
    assert reloaded.status()["config"]["capture_interval_seconds"] == 42


def test_large_screenshot_is_compressed_for_vision_payload(tmp_path: Path) -> None:
    from PIL import Image

    runtime = FakeRuntime(tmp_path, {"should_speak": False, "activity_summary": "idle"})
    service = DesktopContextService(runtime=runtime, state_dir=runtime.ctx.config.state_dir)
    service.configure(max_image_bytes=100_000, vision_max_dimension=320, vision_jpeg_quality=70)
    image_path = tmp_path / "large.png"
    payload = os.urandom(320 * 240 * 3)
    Image.frombytes("RGB", (320, 240), payload).save(image_path)

    data_uri = service._image_data_uri(image_path)

    assert data_uri is not None
    assert data_uri.startswith("data:image/jpeg;base64,")
    assert list((runtime.ctx.config.state_dir / "desktop_context" / "vision").glob("*_vision.jpg"))


@pytest.mark.asyncio
async def test_observe_creates_context_and_speaks_short_natural_text(tmp_path: Path) -> None:
    runtime = FakeRuntime(
        tmp_path,
        {
            "should_speak": True,
            "activity_summary": "The operator is reviewing tests.",
            "reason": "A small nudge is useful.",
            "spoken_text": "You are in the test files. Stay with the next small assertion.",
        },
    )
    spoken: list[str] = []
    service = DesktopContextService(
        runtime=runtime,
        state_dir=runtime.ctx.config.state_dir,
        capture_provider=_capture_provider,
        ocr_provider=lambda path: "pytest tests/test_desktop_context.py",
        speech_synthesizer=lambda text: spoken.append(text)
        or {"path": str(tmp_path / "voice.mp3"), "provider": "fake"},
        audio_player=lambda path: {"played": True, "path": str(path)},
    )
    service.configure(enabled=True, tts_enabled=True, play_audio=True)

    result = await service.observe_once(force=True, reason="test")

    assert result["status"] == "observed"
    assert result["analysis"]["should_speak"] is True
    assert spoken == ["You are in the test files. Stay with the next small assertion."]
    assert runtime.ctx.context_store.entries
    session_id, role, content, meta = runtime.ctx.context_store.entries[-1]
    assert session_id == "default"
    assert role == MessageRole.SYSTEM
    assert "Recent desktop context" in content
    assert meta["source"] == "desktop_context"
    assert runtime.episodes


@pytest.mark.asyncio
async def test_code_heavy_speech_is_redirected_to_note_file(tmp_path: Path) -> None:
    runtime = FakeRuntime(
        tmp_path,
        {
            "should_speak": True,
            "activity_summary": "A traceback is visible.",
            "reason": "The user may need a nudge.",
            "spoken_text": "```python\nraise RuntimeError('boom')\n```",
        },
    )
    spoken: list[str] = []
    service = DesktopContextService(
        runtime=runtime,
        state_dir=runtime.ctx.config.state_dir,
        capture_provider=_capture_provider,
        ocr_provider=lambda path: "Traceback (most recent call last): RuntimeError",
        speech_synthesizer=lambda text: spoken.append(text)
        or {"path": str(tmp_path / "voice.mp3"), "provider": "fake"},
        audio_player=lambda path: {"played": True},
    )
    service.configure(enabled=True, tts_enabled=True, play_audio=True)

    result = await service.observe_once(force=True, reason="test")

    assert result["speech"]["redirected_to_note"] is True
    note_path = Path(result["speech"]["note_path"])
    assert note_path.exists()
    assert "RuntimeError('boom')" in note_path.read_text(encoding="utf-8")
    assert len(spoken) == 1
    assert "desktop observation details" in spoken[0]
    assert "RuntimeError('boom')" not in spoken[0]


@pytest.mark.asyncio
async def test_desktop_context_plugin_registers_runtime_backed_tools(tmp_path: Path) -> None:
    runtime = FakeRuntime(tmp_path, {"should_speak": False, "activity_summary": "idle"})
    runtime.desktop_context = DesktopContextService(
        runtime=runtime,
        state_dir=runtime.ctx.config.state_dir,
        capture_provider=_capture_provider,
    )
    tools = ToolRegistry()
    tools.runtime = runtime
    plugin = load_plugin_from_manifest(
        Path("plugins/desktop_context/plugin.json"),
        PluginRegistry(),
        SkillRegistry(),
        tools,
    )

    assert plugin is not None
    assert tools.get("desktop_context_status") is not None
    assert tools.get("desktop_context_observe") is not None

    result = await tools.execute_async("desktop_context_status", {})
    assert result.success is True
    assert result.metadata["status"]["config"]["enabled"] is False


@pytest.mark.asyncio
async def test_scheduler_runs_desktop_context_loop_when_enabled() -> None:
    calls: list[str] = []
    scheduler_ref: dict[str, AgentScheduler] = {}

    async def run_once() -> dict:
        calls.append("ran")
        scheduler_ref["scheduler"]._running = False
        return {"status": "skipped"}

    scheduler = AgentScheduler(
        SimpleNamespace(
            maybe_run_desktop_context=run_once,
            baa=SimpleNamespace(start=lambda: None, stop=lambda: None),
        ),
        schedule_interval=0,
        initiative_contact_jitter_seconds=0,
    )
    scheduler_ref["scheduler"] = scheduler
    scheduler._running = True
    scheduler._should_run_cycle = lambda: True

    await scheduler._desktop_context_loop()

    assert calls == ["ran"]
    assert CommandLane.CRON in scheduler._lane_manager._lanes
