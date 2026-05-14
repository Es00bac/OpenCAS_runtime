"""Tests for the desktop-context body-double skill."""

from __future__ import annotations

import json
import os
import subprocess
import threading
from pathlib import Path
from types import SimpleNamespace
from uuid import uuid4

import pytest

from opencas.context import MessageRole
from opencas.desktop_context import (
    DesktopCapture,
    DesktopContextConfig,
    DesktopContextService,
    choose_screenshot_backend,
)
from opencas.desktop_context.media import MprisMediaController
from opencas.desktop_context.service import play_audio_file
from opencas.execution.lanes import CommandLane
from opencas.memory import EpisodeKind
from opencas.plugins import PluginRegistry, SkillRegistry
from opencas.plugins.loader import load_plugin_from_manifest
from opencas.runtime.scheduler import AgentScheduler
from opencas.tools import ToolRegistry
from opencas.wellbeing.fascination import FascinationGraph


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
    def __init__(self, payload: dict | list[dict]) -> None:
        self.payload = payload
        self.calls: list[dict] = []

    async def chat_completion(self, messages, **kwargs):
        index = len(self.calls)
        self.calls.append({"messages": messages, "kwargs": kwargs})
        if isinstance(self.payload, list):
            payload = self.payload[min(index, len(self.payload) - 1)]
        else:
            payload = self.payload
        return {
            "choices": [
                {
                    "message": {
                        "content": json.dumps(payload),
                    }
                }
            ]
        }


class FakeContextProposalStore:
    def __init__(self) -> None:
        self.saved: list[object] = []

    async def save(self, proposal):
        self.saved.append(proposal)
        return proposal


class FakeCognitiveStateStore:
    def __init__(self) -> None:
        self.events: list[tuple[object, str, dict]] = []

    async def record_event(self, kind, summary: str, **kwargs):
        self.events.append((kind, summary, kwargs))
        return SimpleNamespace(event_id=uuid4(), kind=kind, summary=summary, **kwargs)


class FakeRuntime:
    def __init__(self, tmp_path: Path, llm_payload: dict) -> None:
        self.saved_memories: list[object] = []
        self.ctx = SimpleNamespace(
            config=SimpleNamespace(
                state_dir=tmp_path / "state",
                session_id="default",
                agent_workspace_root=lambda: tmp_path / "workspace",
            ),
            identity=SimpleNamespace(self_model=SimpleNamespace(name="TestAgent")),
            context_store=FakeContextStore(),
        )
        self.llm = FakeLLM(llm_payload)
        self.traces: list[tuple[str, dict]] = []
        self.episodes: list[tuple[str, object, dict]] = []
        self.memory = SimpleNamespace(save_memory=self.saved_memories.append)

    def _trace(self, event: str, payload: dict) -> None:
        self.traces.append((event, payload))

    async def _record_episode(self, content: str, kind, **kwargs):
        self.episodes.append((content, kind, kwargs))
        return SimpleNamespace(episode_id=uuid4())


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
        "-f",
        "-b",
        "-n",
        "-o",
        "/tmp/shot.png",
    ]


def test_desktop_capture_waits_for_delayed_backend_output(tmp_path: Path) -> None:
    from opencas.desktop_context.capture import capture_desktop_image

    target = tmp_path / "delayed.png"

    def fake_which(name: str) -> str | None:
        return "/usr/bin/spectacle" if name == "spectacle" else None

    def delayed_runner(*args, **kwargs):
        threading.Timer(0.05, lambda: target.write_bytes(b"delayed-png")).start()
        return subprocess.CompletedProcess(args[0], 0, b"", b"")

    result = capture_desktop_image(target, runner=delayed_runner, which=fake_which)

    assert result.success is True
    assert result.backend == "spectacle"
    assert target.read_bytes() == b"delayed-png"


def test_desktop_capture_skips_import_backend_on_wayland(monkeypatch) -> None:
    monkeypatch.setenv("XDG_SESSION_TYPE", "wayland")

    def fake_which(name: str) -> str | None:
        return "/usr/bin/import" if name == "import" else None

    assert choose_screenshot_backend("auto", which=fake_which) is None


def test_desktop_context_config_is_enabled_by_default_and_persists(tmp_path: Path) -> None:
    runtime = FakeRuntime(tmp_path, {"should_speak": False, "activity_summary": "idle"})
    service = DesktopContextService(runtime=runtime, state_dir=runtime.ctx.config.state_dir)

    assert service.status()["config"]["enabled"] is True

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


def test_play_audio_file_targets_current_system_default_sink(tmp_path: Path, monkeypatch) -> None:
    audio_path = tmp_path / "voice.mp3"
    audio_path.write_bytes(b"fake-audio")
    commands: list[tuple[list[str], dict]] = []

    monkeypatch.setattr("opencas.desktop_context.service._current_default_audio_sink", lambda: "alsa_output.starship")
    monkeypatch.setattr("opencas.desktop_context.service.shutil.which", lambda name: f"/usr/bin/{name}" if name == "mpv" else None)

    def fake_run(command, **kwargs):
        commands.append((list(command), dict(kwargs)))
        return subprocess.CompletedProcess(command, 0)

    monkeypatch.setattr("opencas.desktop_context.service.subprocess.run", fake_run)

    result = play_audio_file(audio_path)

    assert result["played"] is True
    assert result["audio_sink"] == "alsa_output.starship"
    assert "--ao=pulse" in commands[0][0]
    assert "--audio-device=pulse/alsa_output.starship" in commands[0][0]


def test_transcript_excerpt_uses_caption_duration_when_mpris_duration_is_too_short(tmp_path: Path) -> None:
    runtime = FakeRuntime(tmp_path, {"should_speak": False, "activity_summary": "idle"})
    service = DesktopContextService(runtime=runtime, state_dir=runtime.ctx.config.state_dir)
    segments = [
        {"start_seconds": 0.0, "end_seconds": 3.0, "text": "New video opening claim."},
        {"start_seconds": 2104.0, "end_seconds": 2108.0, "text": "Current long-video segment."},
        {"start_seconds": 5269.0, "end_seconds": 5272.0, "text": "Actual end of the longer video."},
    ]
    transcript_text = " ".join(segment["text"] for segment in segments)

    excerpt, meta = service._transcript_excerpt_for_media_position(
        transcript_text,
        [
            {
                "title": "Autoplay video",
                "url": "https://www.youtube.com/watch?v=nextVideo",
                "position_us": 2_114_000_000,
                "length_us": 2_114_000_000,
                "position_label": "position 35:14 / 35:14",
                "progress_percent": 100.0,
            }
        ],
        limit=500,
        segments=segments,
    )

    assert meta["transcript_excerpt_basis"] == "caption_timestamp_reached"
    assert meta["transcript_position_valid"] is True
    assert meta["media_duration_reliable"] is False
    assert "reported duration 35:14" in meta["media_duration_warning"]
    assert meta["transcript_caption_duration_label"] == "1:27:52"
    assert meta["transcript_progress_percent"] == pytest.approx(40.095, abs=0.01)
    assert meta["transcript_position_label"] == "position 35:14 / 1:27:52"
    assert "Current long-video segment" in excerpt
    assert "Actual end" not in excerpt


@pytest.mark.asyncio
async def test_repeated_scheduled_spoken_text_is_suppressed(tmp_path: Path) -> None:
    runtime = FakeRuntime(tmp_path, {"should_speak": False, "activity_summary": "idle"})
    spoken: list[str] = []
    service = DesktopContextService(
        runtime=runtime,
        state_dir=runtime.ctx.config.state_dir,
        config=DesktopContextConfig(play_audio=False, min_speech_interval_seconds=0),
        speech_synthesizer=lambda text: {"path": str(tmp_path / "voice.mp3"), "text": spoken.append(text)},
    )
    analysis = {
        "should_speak": True,
        "activity_summary": "Video commentary.",
        "spoken_text": "This is the exact same opinion.",
    }

    first = await service._speak_analysis(analysis, {"capture": {}}, reason="scheduled_body_double")
    second = await service._speak_analysis(analysis, {"capture": {}}, reason="scheduled_body_double")

    assert first["status"] == "spoken"
    assert second == {"status": "skipped", "reason": "repeated_spoken_text"}


@pytest.mark.asyncio
async def test_direct_speak_text_can_use_long_form_override(tmp_path: Path) -> None:
    runtime = FakeRuntime(tmp_path, {"should_speak": False, "activity_summary": "idle"})
    spoken: list[str] = []
    service = DesktopContextService(
        runtime=runtime,
        state_dir=runtime.ctx.config.state_dir,
        config=DesktopContextConfig(play_audio=False, min_speech_interval_seconds=0, max_spoken_chars=80),
        speech_synthesizer=lambda text: {"path": str(tmp_path / "voice.mp3"), "text": spoken.append(text)},
    )
    long_text = (
        "This first sentence should be spoken. "
        "This middle sentence should also be spoken. "
        "This final sentence proves the direct conversation voice path can exceed the short nudge limit."
    )

    result = await service.speak_text(
        long_text,
        reason="body_double_conversation_response:voice",
        force=True,
        max_chars=5000,
        allow_note_redirect=False,
    )

    assert result["status"] == "spoken"
    assert spoken == [long_text]
    assert "final sentence proves" in result["spoken_text"]


@pytest.mark.asyncio
async def test_repeated_media_commentary_context_is_suppressed_even_when_reworded(tmp_path: Path) -> None:
    runtime = FakeRuntime(tmp_path, {"should_speak": False, "activity_summary": "idle"})
    spoken: list[str] = []
    service = DesktopContextService(
        runtime=runtime,
        state_dir=runtime.ctx.config.state_dir,
        config=DesktopContextConfig(play_audio=False, min_speech_interval_seconds=0),
        speech_synthesizer=lambda text: {"path": str(tmp_path / "voice.mp3"), "text": spoken.append(text)},
    )
    capture_result = {
        "media_context": [
            {
                "title": "Karpathy's Wiki vs. Open Brain. One Fails When You Need It Most.",
                "url": "https://www.youtube.com/watch?v=dxq7WtWxi44",
                "position_us": 1_030_000_000,
                "length_us": 2_470_000_000,
                "position_label": "position 17:10 / 41:10",
            }
        ],
        "youtube_transcript": {
            "status": "available",
            "video_id": "dxq7WtWxi44",
            "url": "https://www.youtube.com/watch?v=dxq7WtWxi44",
            "transcript_excerpt_basis": "caption_timestamp",
            "transcript_aligned_start_seconds": 1010.0,
            "transcript_aligned_end_seconds": 1060.0,
            "transcript_position_label": "position 17:10 / 41:10",
        },
    }

    first = await service._speak_analysis(
        {
            "should_speak": True,
            "activity_summary": "Video commentary.",
            "spoken_text": "Plain-text wikis win on inspectability, but structured memory wins when relationships matter.",
        },
        capture_result,
        reason="scheduled_body_double",
    )
    second = await service._speak_analysis(
        {
            "should_speak": True,
            "activity_summary": "Video commentary.",
            "spoken_text": "That framing is still wiki versus database: inspectability against relational structure.",
        },
        capture_result,
        reason="scheduled_body_double",
    )

    assert first["status"] == "spoken"
    assert second == {"status": "skipped", "reason": "repeated_spoken_context"}
    assert len(spoken) == 1


@pytest.mark.asyncio
async def test_policy_boilerplate_is_not_spoken_as_media_commentary(tmp_path: Path) -> None:
    runtime = FakeRuntime(tmp_path, {"should_speak": False, "activity_summary": "idle"})
    spoken: list[str] = []
    service = DesktopContextService(
        runtime=runtime,
        state_dir=runtime.ctx.config.state_dir,
        config=DesktopContextConfig(play_audio=False, min_speech_interval_seconds=0),
        speech_synthesizer=lambda text: {"path": str(tmp_path / "voice.mp3"), "text": spoken.append(text)},
    )

    result = await service._speak_analysis(
        {
            "should_speak": True,
            "activity_summary": "Video commentary.",
            "spoken_text": (
                "My grounded take is that this segment names the failure mode OpenCAS should avoid: "
                "an assistant is only useful if it reduces management overhead, understands context, "
                "and earns interruptions by producing synthesis or follow-through."
            ),
        },
        {"capture": {}},
        reason="scheduled_body_double",
    )

    assert result == {"status": "skipped", "reason": "policy_boilerplate_spoken_text"}
    assert spoken == []


def test_policy_boilerplate_is_scrubbed_before_followup_persistence(tmp_path: Path) -> None:
    runtime = FakeRuntime(tmp_path, {"should_speak": False, "activity_summary": "idle"})
    service = DesktopContextService(
        runtime=runtime,
        state_dir=runtime.ctx.config.state_dir,
        config=DesktopContextConfig(play_audio=False, min_speech_interval_seconds=0),
    )

    followup = service._normalize_self_interest_followup(
        {
            "matches_observed_context": True,
            "matches_shared_work": True,
            "match_category": "project_context",
            "matched_targets": ["OpenCAS body double"],
            "connection_summary": "This segment is relevant to the active body-double work.",
            "agent_viewpoint": (
                "OpenCAS memory only if I can connect a specific claim about assisted behavior "
                "from the reached transcript segment to a concrete design choice or follow-up task."
            ),
            "spoken_text": (
                "The useful bar here is context and earns interruptions by producing synthesis or follow-through."
            ),
            "implications": [
                "Preserve this concrete segment for later work.",
                "An assistant is only useful if it reduces management overhead.",
            ],
            "open_questions": ["What concrete evidence changed the design?"],
            "self_directed_next_step": "Use this observation in the body-double implementation.",
            "operator_interruption_warranted": True,
            "confidence": 0.95,
            "salience": 0.92,
        }
    )

    assert followup["connection_summary"] == "This segment is relevant to the active body-double work."
    assert followup["agent_viewpoint"] == ""
    assert followup["spoken_text"] == ""
    assert followup["implications"] == ["Preserve this concrete segment for later work."]
    assert followup["open_questions"] == ["What concrete evidence changed the design?"]


def test_config_migrates_declared_youtube_commentary_task_to_media_commentary_mode(tmp_path: Path) -> None:
    runtime = FakeRuntime(tmp_path, {"should_speak": False, "activity_summary": "idle"})
    root = runtime.ctx.config.state_dir / "desktop_context"
    root.mkdir(parents=True)
    (root / "config.json").write_text(
        json.dumps(
            {
                "enabled": True,
                "declared_task": (
                    "When Jarrod is watching YouTube, offer brief companion-style commentary on the video "
                    "regardless of topic."
                ),
                "declared_task_source": "operator",
            }
        ),
        encoding="utf-8",
    )

    service = DesktopContextService(runtime=runtime, state_dir=runtime.ctx.config.state_dir)

    assert service.config.media_commentary_mode_enabled is True
    assert service.config.media_commentary_source == "declared_task_migration"
    assert "watching YouTube" in str(service.config.media_commentary_request)


class FakeMediaController:
    def __init__(self, items: list[dict]) -> None:
        self.items = items
        self.calls = 0

    def current_media(self) -> list[dict]:
        self.calls += 1
        return self.items


class SequenceMediaController:
    def __init__(self, snapshots: list[list[dict]]) -> None:
        self.snapshots = snapshots
        self.calls = 0

    def current_media(self) -> list[dict]:
        self.calls += 1
        if not self.snapshots:
            return []
        if len(self.snapshots) == 1:
            return self.snapshots[0]
        return self.snapshots.pop(0)


def _youtube_media_item(*, title: str = "A useful video", video_id: str = "abc123") -> dict:
    return {
        "player": "firefox",
        "status": "Playing",
        "title": title,
        "artist": "YouTube",
        "url": f"https://www.youtube.com/watch?v={video_id}",
        "length_us": 600_000_000,
        "position_us": 12_000_000,
        "position_label": "0:12",
        "progress_percent": 2.0,
    }


@pytest.mark.asyncio
async def test_video_commentary_request_enables_ongoing_media_commentary_mode(tmp_path: Path) -> None:
    runtime = FakeRuntime(
        tmp_path,
        {
            "should_speak": False,
            "activity_summary": "The operator is watching a video.",
            "reason": "Conversation context only.",
            "speech_intent": "screen_relevant",
            "speech_relevance_score": 0.4,
        },
    )
    service = DesktopContextService(
        runtime=runtime,
        state_dir=runtime.ctx.config.state_dir,
        config=DesktopContextConfig(
            tts_enabled=False,
            play_audio=False,
            youtube_transcripts_enabled=False,
            self_interest_followup_enabled=False,
            observed_context_relevance_enabled=False,
            project_context_relevance_enabled=False,
        ),
        capture_provider=_capture_provider,
        ocr_provider=lambda path: "A YouTube video is open.",
        media_controller=FakeMediaController([_youtube_media_item()]),
    )

    result = await service.observe_for_conversation(
        session_id="default",
        user_input="Give me your commentary on this video, key points as I'm watching it, please.",
        source="api_chat",
    )

    assert result is not None
    assert result["status"] == "observed"
    assert service.config.media_commentary_mode_enabled is True
    assert service.config.media_commentary_source == "api_chat"
    assert "commentary on this video" in str(service.config.media_commentary_request)
    assert "media commentary mode: active" in result["conversation_prompt_note"].lower()

    reloaded = DesktopContextService(runtime=runtime, state_dir=runtime.ctx.config.state_dir)
    assert reloaded.config.media_commentary_mode_enabled is True
    assert reloaded.config.media_commentary_source == "api_chat"


@pytest.mark.asyncio
async def test_media_started_requests_commentary_observation_when_commentary_mode_active(tmp_path: Path) -> None:
    runtime = FakeRuntime(tmp_path, {"should_speak": False, "activity_summary": "idle"})
    service = DesktopContextService(
        runtime=runtime,
        state_dir=runtime.ctx.config.state_dir,
        config=DesktopContextConfig(
            media_commentary_mode_enabled=True,
            media_commentary_source="api_chat",
            media_commentary_request="Give me commentary on this video.",
            youtube_transcripts_enabled=False,
        ),
        media_controller=FakeMediaController([_youtube_media_item(title="Autoplay successor", video_id="next456")]),
    )

    result = await service.poll_media_state_once()

    assert result["status"] == "changed"
    assert result["media_state_changes"][0]["event"] == "media_started"
    assert result["commentary_observation_requested"] is True
    assert result["commentary_observation_reason"] == "media_commentary_mode:media_started"


@pytest.mark.asyncio
async def test_paused_new_media_does_not_request_commentary_observation(tmp_path: Path) -> None:
    runtime = FakeRuntime(tmp_path, {"should_speak": False, "activity_summary": "idle"})
    service = DesktopContextService(
        runtime=runtime,
        state_dir=runtime.ctx.config.state_dir,
        config=DesktopContextConfig(
            media_commentary_mode_enabled=True,
            media_commentary_source="api_chat",
            media_commentary_request="Give me commentary on this video.",
            youtube_transcripts_enabled=False,
        ),
        media_controller=FakeMediaController(
            [_youtube_media_item(title="Paused next video", video_id="paused456") | {"status": "Paused"}]
        ),
    )

    result = await service.poll_media_state_once()

    assert result["status"] == "changed"
    assert result["media_state_changes"][0]["event"] == "media_loaded_paused"
    assert result["commentary_observation_requested"] is False
    assert result["commentary_observation_reason"] is None


@pytest.mark.asyncio
async def test_paused_media_seek_does_not_request_commentary_observation(tmp_path: Path) -> None:
    runtime = FakeRuntime(tmp_path, {"should_speak": False, "activity_summary": "idle"})
    service = DesktopContextService(
        runtime=runtime,
        state_dir=runtime.ctx.config.state_dir,
        config=DesktopContextConfig(
            media_commentary_mode_enabled=True,
            media_commentary_source="api_chat",
            media_commentary_request="Give me commentary on this video.",
            youtube_transcripts_enabled=False,
        ),
        media_controller=SequenceMediaController(
            [
                [_youtube_media_item(title="Paused video", video_id="pausedSeek") | {"status": "Paused"}],
                [
                    _youtube_media_item(title="Paused video", video_id="pausedSeek")
                    | {"status": "Paused", "position_us": 90_000_000}
                ],
            ]
        ),
    )

    first = await service.poll_media_state_once()
    second = await service.poll_media_state_once()

    assert first["commentary_observation_requested"] is False
    assert second["status"] == "changed"
    assert second["media_state_changes"][0]["event"] == "media_seeked"
    assert second["commentary_observation_requested"] is False


@pytest.mark.asyncio
async def test_observe_creates_context_and_speaks_short_natural_text(tmp_path: Path) -> None:
    runtime = FakeRuntime(
        tmp_path,
        {
            "should_speak": True,
            "activity_summary": "The operator is reviewing tests.",
            "reason": "A small nudge is useful.",
            "spoken_text": "You are in the test files. Stay with the next small assertion.",
            "speech_intent": "task_coaching",
            "speech_relevance_score": 0.88,
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
    service.configure(
        enabled=True,
        tts_enabled=True,
        play_audio=True,
        declared_task="Review the desktop-context tests.",
    )

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


@pytest.mark.asyncio
async def test_region_prompt_turn_analyzes_selected_pixels_and_builds_prompt_note(tmp_path: Path) -> None:
    from PIL import Image

    runtime = FakeRuntime(
        tmp_path,
        {
            "visual_summary": "A terminal dialog reports an authentication failure.",
            "visible_text": "invalid_grant: Token has been expired or revoked",
            "relevant_details": "The selected region contains an OpenCAS scheduled Gmail auth error.",
            "answer_context": "The user should refresh the Google Workspace OAuth token.",
            "uncertainty": "The screenshot is narrow, but the visible error is clear.",
            "confidence": 0.91,
        },
    )
    service = DesktopContextService(
        runtime=runtime,
        state_dir=runtime.ctx.config.state_dir,
        ocr_provider=lambda path: "invalid_grant: Token has been expired or revoked",
    )
    image_path = tmp_path / "region.png"
    Image.new("RGB", (240, 80), color=(20, 20, 20)).save(image_path)

    result = await service.analyze_region_for_conversation(
        image_path=image_path,
        prompt="What should I do about this error?",
        session_id="region-session",
        selection={"x": 10, "y": 20, "width": 240, "height": 80},
        source="test_region_prompt",
    )

    assert result["status"] == "observed"
    assert result["capture"]["path"] == str(image_path)
    assert result["analysis"]["region_prompt"]["visual_summary"].startswith("A terminal dialog")
    assert "Selected desktop region context" in result["conversation_prompt_note"]
    assert "What should I do about this error?" in result["conversation_prompt_note"]
    assert "invalid_grant" in result["conversation_prompt_note"]
    assert runtime.llm.calls
    call_content = runtime.llm.calls[-1]["messages"][1]["content"]
    assert isinstance(call_content, list)
    assert call_content[1]["type"] == "image_url"
    session_id, role, content, meta = runtime.ctx.context_store.entries[-1]
    assert session_id == "region-session"
    assert role == MessageRole.SYSTEM
    assert "Selected desktop region context" in content
    assert meta["analysis"]["region_prompt"]["answer_context"].startswith("The user should refresh")
    assert runtime.episodes
    assert "user-selected desktop rectangle" in runtime.llm.calls[0]["messages"][0]["content"]


@pytest.mark.asyncio
async def test_observe_persists_recallable_evidence_backed_activity_memory(tmp_path: Path) -> None:
    runtime = FakeRuntime(
        tmp_path,
        {
            "should_speak": False,
            "activity_summary": "The operator is editing Chronicle notes.",
            "reason": "Ongoing work is visible.",
            "spoken_text": "",
            "note": "A Markdown editor and terminal appear to be open side by side.",
        },
    )
    service = DesktopContextService(
        runtime=runtime,
        state_dir=runtime.ctx.config.state_dir,
        capture_provider=_capture_provider,
        ocr_provider=lambda path: "chronicle_4246_unified_manuscript_v7_clean.md\npytest tests",
    )

    result = await service.observe_once(force=True, reason="scheduled_body_double")

    assert result["status"] == "observed"
    assert runtime.episodes
    content, kind, kwargs = runtime.episodes[-1]
    assert kind == EpisodeKind.OBSERVATION
    assert "Observed user activity:" in content
    payload = kwargs["payload"]
    assert payload["source"] == "desktop_context"
    assert payload["context_authority"] == "live_observation"
    assert payload["context_material"] == "desktop_observation"
    assert payload["observed_user_activity"] == "The operator is editing Chronicle notes."
    assert payload["evidence"]["screenshot_path"]
    assert payload["evidence"]["ocr_excerpt"].startswith("chronicle_4246")
    assert payload["temporal"]["observed_at"]
    assert runtime.saved_memories
    memory = runtime.saved_memories[-1]
    assert "Observed user activity: The operator is editing Chronicle notes." in memory.content
    assert str(payload["evidence"]["screenshot_path"]) in memory.content
    assert "desktop_context" in memory.tags
    assert "observed_user_activity" in memory.tags
    assert memory.source_episode_ids


@pytest.mark.asyncio
async def test_youtube_playback_observation_retrieves_transcript_context(tmp_path: Path) -> None:
    runtime = FakeRuntime(
        tmp_path,
        {
            "should_speak": False,
            "activity_summary": "The operator is watching a YouTube video about Sith ranks.",
            "reason": "The video is visible and playing.",
            "spoken_text": "",
            "speech_intent": "none",
            "speech_relevance_score": 0.0,
        },
    )

    class FakeMediaController:
        def current_media(self):
            return [
                {
                    "player": "org.mpris.MediaPlayer2.firefox",
                    "status": "Playing",
                    "title": "The 7 Levels of Sith Explained in Detail",
                    "artist": "Order 77",
                    "url": "https://www.youtube.com/watch?v=2n9TOgYHv1E",
                    "length_us": 125000000,
                }
            ]

    def transcript_provider(url, media_context):
        assert url == "https://www.youtube.com/watch?v=2n9TOgYHv1E"
        assert media_context[0]["title"] == "The 7 Levels of Sith Explained in Detail"
        return {
            "status": "available",
            "source": "fake",
            "transcript_text": "Sith training begins with acolytes and advances through apprentices and lords.",
        }

    service = DesktopContextService(
        runtime=runtime,
        state_dir=runtime.ctx.config.state_dir,
        capture_provider=_capture_provider,
        ocr_provider=lambda path: "YouTube video playing",
        media_controller=FakeMediaController(),
        youtube_transcript_provider=transcript_provider,
    )

    result = await service.observe_once(force=True, reason="scheduled_body_double")

    assert result["status"] == "observed"
    transcript = result["youtube_transcript"]
    assert transcript["status"] == "available"
    assert transcript["video_id"] == "2n9TOgYHv1E"
    assert transcript["title"] == "The 7 Levels of Sith Explained in Detail"
    assert "Sith training begins" in transcript["transcript_excerpt"]
    content, _, kwargs = runtime.episodes[-1]
    payload = kwargs["payload"]
    assert payload["youtube_transcript"]["status"] == "available"
    assert "Media observed: The 7 Levels of Sith Explained in Detail" in runtime.saved_memories[-1].content
    assert "YouTube transcript excerpt:" in runtime.saved_memories[-1].content
    prompt_content = runtime.llm.calls[-1]["messages"][1]["content"]
    prompt_text = prompt_content[0]["text"] if isinstance(prompt_content, list) else prompt_content
    assert "YouTube transcript context:" in prompt_text
    assert "current-position transcript excerpt withheld" in prompt_text
    assert "Playback position is unavailable" in prompt_text
    assert "Sith training begins" not in prompt_text
    assert "YouTube transcript excerpt:" in content


@pytest.mark.asyncio
async def test_live_whisper_transcript_follows_livestream_without_prefetched_transcript(tmp_path: Path) -> None:
    runtime = FakeRuntime(
        tmp_path,
        {
            "should_speak": False,
            "activity_summary": "The operator is watching a livestream.",
            "reason": "The livestream is playing.",
            "spoken_text": "",
            "speech_intent": "none",
            "speech_relevance_score": 0.0,
        },
    )
    media_item = {
        "player": "org.mpris.MediaPlayer2.firefox",
        "status": "Playing",
        "title": "Live AI Systems Q&A",
        "artist": "Example Channel",
        "url": "https://www.youtube.com/watch?v=liveStream1",
        "length_us": None,
        "position_us": None,
    }
    provider_calls: list[dict] = []

    async def live_provider(media, *, media_context, youtube_transcript):
        provider_calls.append(
            {
                "media": media,
                "media_context": media_context,
                "youtube_transcript": youtube_transcript,
            }
        )
        return {
            "status": "available",
            "source": "whisper",
            "mode": "local",
            "model": "whisper-base",
            "transcript_text": "The host is answering a live question about memory and transformer agents.",
            "capture_seconds": 4.0,
        }

    service = DesktopContextService(
        runtime=runtime,
        state_dir=runtime.ctx.config.state_dir,
        config=DesktopContextConfig(
            tts_enabled=False,
            live_transcription_enabled=True,
            youtube_transcripts_enabled=True,
            self_interest_followup_enabled=False,
            observed_context_relevance_enabled=False,
            project_context_relevance_enabled=False,
        ),
        capture_provider=_capture_provider,
        ocr_provider=lambda path: "A livestream video is open.",
        media_controller=FakeMediaController([media_item]),
        youtube_transcript_provider=lambda url, media_context: {
            "status": "unavailable",
            "reason": "prefetched_transcript_unavailable_for_livestream",
        },
        live_transcript_provider=live_provider,
    )

    result = await service.observe_once(force=True, reason="scheduled_body_double")

    assert provider_calls
    assert provider_calls[0]["media"]["title"] == "Live AI Systems Q&A"
    assert provider_calls[0]["youtube_transcript"]["status"] == "unavailable"
    live_transcript = result["live_transcript"]
    assert live_transcript["status"] == "available"
    assert live_transcript["source"] == "whisper"
    assert live_transcript["mode"] == "local"
    assert live_transcript["prefetched_transcript_status"] == "unavailable"
    assert "memory and transformer agents" in live_transcript["transcript_excerpt"]
    prompt_content = runtime.llm.calls[-1]["messages"][1]["content"]
    prompt_text = prompt_content[0]["text"] if isinstance(prompt_content, list) else prompt_content
    assert "Live transcript context (local Whisper):" in prompt_text
    assert "current audio transcript excerpt" in prompt_text
    assert "memory and transformer agents" in prompt_text
    assert "this live Whisper excerpt is the current heard segment" in prompt_text
    assert runtime.episodes[-1][2]["payload"]["live_transcript"]["status"] == "available"


@pytest.mark.asyncio
async def test_live_whisper_transcript_skips_paused_media(tmp_path: Path) -> None:
    runtime = FakeRuntime(
        tmp_path,
        {
            "should_speak": False,
            "activity_summary": "The operator has a paused video open.",
            "reason": "Paused media should not be transcribed.",
            "spoken_text": "",
            "speech_intent": "none",
            "speech_relevance_score": 0.0,
        },
    )

    async def live_provider(*args, **kwargs):
        raise AssertionError("paused media must not invoke live transcription")

    service = DesktopContextService(
        runtime=runtime,
        state_dir=runtime.ctx.config.state_dir,
        config=DesktopContextConfig(
            tts_enabled=False,
            live_transcription_enabled=True,
            youtube_transcripts_enabled=False,
            self_interest_followup_enabled=False,
            observed_context_relevance_enabled=False,
            project_context_relevance_enabled=False,
        ),
        capture_provider=_capture_provider,
        ocr_provider=lambda path: "A paused video is open.",
        media_controller=FakeMediaController([_youtube_media_item(title="Paused talk") | {"status": "Paused"}]),
        live_transcript_provider=live_provider,
    )

    result = await service.observe_once(force=True, reason="scheduled_body_double")

    assert result["live_transcript"]["status"] == "skipped"
    assert result["live_transcript"]["reason"] == "media_not_playing"


def test_analysis_prompt_fuses_prefetched_and_live_transcripts(tmp_path: Path) -> None:
    runtime = FakeRuntime(tmp_path, {"should_speak": False, "activity_summary": "idle"})
    service = DesktopContextService(runtime=runtime, state_dir=runtime.ctx.config.state_dir)

    prompt = service._analysis_prompt(
        {
            "capture": {"path": str(tmp_path / "screen.png"), "backend": "fake"},
            "ocr_text": "A YouTube video is playing.",
            "media_context": [_youtube_media_item(title="Timed video", video_id="timedLive")],
            "youtube_transcript": {
                "status": "available",
                "title": "Timed video",
                "url": "https://www.youtube.com/watch?v=timedLive",
                "transcript_excerpt_basis": "caption_timestamp_reached",
                "transcript_position_valid": True,
                "transcript_excerpt": "[0:20-0:26] The recorded transcript explains the setup.",
                "transcript_position_label": "position 0:31 / 5:00",
            },
            "live_transcript": {
                "status": "available",
                "source": "whisper",
                "mode": "local",
                "model": "whisper-base",
                "transcript_excerpt": "The speaker is now adding a live clarification about the setup.",
                "capture_seconds": 5.0,
                "media_title": "Timed video",
                "media_status": "Playing",
                "prefetched_transcript_status": "available",
            },
        },
        reason="scheduled_body_double",
    )

    assert "YouTube transcript context:" in prompt
    assert "Live transcript context (local Whisper):" in prompt
    assert "recorded transcript explains the setup" in prompt
    assert "live clarification about the setup" in prompt
    assert "use the retrieved transcript as the timestamped map" in prompt
    assert "Trust the live transcript for what is being heard now" in prompt


def test_live_transcript_changes_spoken_context_signature_without_playback_position(tmp_path: Path) -> None:
    runtime = FakeRuntime(tmp_path, {"should_speak": False, "activity_summary": "idle"})
    service = DesktopContextService(runtime=runtime, state_dir=runtime.ctx.config.state_dir)
    base_capture = {
        "media_context": [
            {
                "player": "firefox",
                "status": "Playing",
                "title": "Live stream",
                "url": "https://www.youtube.com/watch?v=liveStream1",
                "position_us": None,
            }
        ]
    }

    first = service._spoken_context_signature(
        base_capture
        | {
            "live_transcript": {
                "status": "available",
                "media_identity": "liveStream1",
                "transcript_excerpt": "The stream is discussing local transcription now.",
            }
        }
    )
    second = service._spoken_context_signature(
        base_capture
        | {
            "live_transcript": {
                "status": "available",
                "media_identity": "liveStream1",
                "transcript_excerpt": "The stream has moved on to live timing drift.",
            }
        }
    )

    assert first
    assert second
    assert first != second


def test_youtube_transcript_excerpt_tracks_playback_position(tmp_path: Path) -> None:
    runtime = FakeRuntime(tmp_path, {"should_speak": False, "activity_summary": "idle"})
    service = DesktopContextService(
        runtime=runtime,
        state_dir=runtime.ctx.config.state_dir,
        config=DesktopContextConfig(youtube_transcript_max_chars=220),
    )
    transcript_text = (
        "opening context before the topic. "
        + ("early filler " * 80)
        + "middle marker about proactive assistants, memory, and agent management overhead. "
        + ("late filler " * 80)
        + "closing context after the topic."
    )

    payload = service._transcript_payload(
        "positionTest123",
        "https://www.youtube.com/watch?v=positionTest123",
        transcript_text,
        source="fake",
        media_context=[
            {
                "title": "Consumer AI Has a Problem Nobody's Naming.",
                "url": "https://www.youtube.com/watch?v=positionTest123",
                "length_us": 1_000_000_000,
                "position_us": 500_000_000,
                "position_label": "position 8:20 / 16:40",
                "progress_percent": 50.0,
            }
        ],
        transcript_path=tmp_path / "positionTest123.txt",
        cached=True,
    )

    assert "middle marker about proactive assistants" in payload["transcript_excerpt"]
    assert "opening context before the topic" not in payload["transcript_excerpt"]
    assert payload["transcript_excerpt_basis"] == "playback_position"
    assert payload["transcript_progress_percent"] == 50.0
    assert payload["transcript_position_label"] == "position 8:20 / 16:40"


def test_approximate_playback_position_transcript_is_withheld_from_commentary_prompt(tmp_path: Path) -> None:
    runtime = FakeRuntime(tmp_path, {"should_speak": False, "activity_summary": "idle"})
    service = DesktopContextService(runtime=runtime, state_dir=runtime.ctx.config.state_dir)

    lines = service._youtube_transcript_prompt_lines(
        {
            "status": "available",
            "title": "Approximate transcript",
            "url": "https://www.youtube.com/watch?v=approx123",
            "transcript_excerpt_basis": "playback_position",
            "transcript_excerpt": "Future claim that should not be spoken yet.",
            "transcript_position_label": "position 8:20 / 16:40",
        }
    )
    prompt = "\n".join(lines)

    assert "current-position transcript excerpt withheld" in prompt
    assert "Future claim that should not be spoken yet" not in prompt
    assert "Only approximate transcript-position alignment is available" in prompt


def test_timestamp_confirmed_transcript_excerpt_is_available_for_commentary_prompt(tmp_path: Path) -> None:
    runtime = FakeRuntime(tmp_path, {"should_speak": False, "activity_summary": "idle"})
    service = DesktopContextService(runtime=runtime, state_dir=runtime.ctx.config.state_dir)

    lines = service._youtube_transcript_prompt_lines(
        {
            "status": "available",
            "title": "Timed transcript",
            "url": "https://www.youtube.com/watch?v=timed123",
            "transcript_excerpt_basis": "caption_timestamp_reached",
            "transcript_position_valid": True,
            "transcript_excerpt": "[1:00-1:05] Reached claim that is safe to discuss.",
            "transcript_position_label": "position 1:11 / 3:00",
        }
    )
    prompt = "\n".join(lines)

    assert "current-position transcript excerpt:" in prompt
    assert "Reached claim that is safe to discuss" in prompt
    assert "timestamp-confirmed reached part" in prompt


def test_youtube_transcript_excerpt_prefers_caption_timestamp_alignment(tmp_path: Path) -> None:
    runtime = FakeRuntime(tmp_path, {"should_speak": False, "activity_summary": "idle"})
    service = DesktopContextService(
        runtime=runtime,
        state_dir=runtime.ctx.config.state_dir,
        config=DesktopContextConfig(youtube_transcript_max_chars=260),
    )
    transcript_text = (
        "Opening context before the main point. "
        "The speaker says proactive agents should pause at the exact relevant moment. "
        "Closing context after the point."
    )

    payload = service._transcript_payload(
        "timed123",
        "https://www.youtube.com/watch?v=timed123",
        transcript_text,
        source="fake",
        media_context=[
            {
                "title": "Precise Agent Timing",
                "url": "https://www.youtube.com/watch?v=timed123",
                "length_us": 180_000_000,
                "position_us": 71_200_000,
                "position_label": "position 1:11 / 3:00",
            }
        ],
        transcript_path=tmp_path / "timed123.txt",
        cached=True,
        segments=[
            {"start_seconds": 3.0, "end_seconds": 7.0, "text": "Opening context before the main point."},
            {
                "start_seconds": 60.5,
                "end_seconds": 65.0,
                "text": "The speaker says proactive agents should pause at the exact relevant moment.",
            },
            {"start_seconds": 140.0, "end_seconds": 145.0, "text": "Closing context after the point."},
        ],
    )

    assert payload["transcript_excerpt_basis"] == "caption_timestamp_reached"
    assert payload["transcript_target_seconds"] == pytest.approx(71.2)
    assert payload["transcript_commentary_cutoff_seconds"] == pytest.approx(65.2)
    assert payload["transcript_aligned_start_seconds"] == pytest.approx(60.5)
    assert payload["transcript_aligned_end_seconds"] == pytest.approx(65.0)
    assert payload["transcript_aligned_start_label"] == "1:00"
    assert payload["transcript_aligned_end_label"] == "1:05"
    assert payload["transcript_position_label"] == "position 1:11 / 3:00"
    assert "exact relevant moment" in payload["transcript_excerpt"]
    assert "Closing context after the point" not in payload["transcript_excerpt"]
    assert payload["transcript_timed_segment_count"] == 3
    assert payload["transcript_aligned_segments"][0]["text"].startswith("The speaker says")


def test_youtube_transcript_alignment_uses_matching_media_item_when_players_overlap(tmp_path: Path) -> None:
    runtime = FakeRuntime(tmp_path, {"should_speak": False, "activity_summary": "idle"})
    service = DesktopContextService(
        runtime=runtime,
        state_dir=runtime.ctx.config.state_dir,
        config=DesktopContextConfig(youtube_transcript_max_chars=300),
    )
    transcript_text = (
        "The current segment says an AI civil-war scenario is testing trust boundaries. "
        "The later segment says a separate model found adversarial misalignment evidence."
    )

    payload = service._transcript_payload(
        "gwfCWDO4LbM",
        "https://www.youtube.com/watch?v=gwfCWDO4LbM&t=15s",
        transcript_text,
        source="fake",
        media_context=[
            {
                "player": "org.mpris.MediaPlayer2.chromium.instance6028",
                "status": "Paused",
                "title": "Episode 1",
                "url": "",
                "length_us": 1_420_000_000,
                "position_us": 1_088_000_000,
                "position_label": "position 18:08 / 23:40",
            },
            {
                "player": "org.mpris.MediaPlayer2.firefox.instance_1_63",
                "status": "Playing",
                "title": "The First 48 Hours of an AI Civil War - A Realistic Scenario",
                "artist": "Species | Documenting AGI",
                "url": "https://www.youtube.com/watch?v=gwfCWDO4LbM&t=15s",
                "length_us": 2_114_000_000,
                "position_us": 735_000_000,
                "position_label": "position 12:15 / 35:14",
            },
        ],
        transcript_path=tmp_path / "gwfCWDO4LbM.txt",
        cached=True,
        segments=[
            {
                "start_seconds": 720.0,
                "end_seconds": 728.0,
                "text": "The current segment says an AI civil-war scenario is testing trust boundaries.",
            },
            {
                "start_seconds": 1045.0,
                "end_seconds": 1058.0,
                "text": "The later segment says a separate model found adversarial misalignment evidence.",
            },
        ],
    )

    assert payload["title"] == "The First 48 Hours of an AI Civil War - A Realistic Scenario"
    assert payload["transcript_position_label"] == "position 12:15 / 35:14"
    assert payload["transcript_target_seconds"] == pytest.approx(735.0)
    assert "current segment" in payload["transcript_excerpt"]
    assert "later segment" not in payload["transcript_excerpt"]


def test_youtube_url_extraction_prefers_playing_media_over_paused_players(tmp_path: Path) -> None:
    runtime = FakeRuntime(tmp_path, {"should_speak": False, "activity_summary": "idle"})
    service = DesktopContextService(runtime=runtime, state_dir=runtime.ctx.config.state_dir)

    url = service._extract_youtube_url(
        {"ocr_text": ""},
        [
            {
                "player": "org.mpris.MediaPlayer2.chromium.instance6028",
                "status": "Paused",
                "title": "Old paused video",
                "url": "https://www.youtube.com/watch?v=pausedOld123",
            },
            {
                "player": "org.mpris.MediaPlayer2.firefox.instance_1_63",
                "status": "Playing",
                "title": "Current relevant video",
                "url": "https://www.youtube.com/watch?v=currentNow45",
            },
        ],
    )

    assert url == "https://www.youtube.com/watch?v=currentNow45"


def test_primary_media_surfaces_prefer_transcript_and_playing_media(tmp_path: Path) -> None:
    runtime = FakeRuntime(tmp_path, {"should_speak": False, "activity_summary": "idle"})
    service = DesktopContextService(runtime=runtime, state_dir=runtime.ctx.config.state_dir)
    media_context = [
        {
            "player": "org.mpris.MediaPlayer2.chromium.instance6028",
            "status": "Paused",
            "title": "Episode 1",
            "url": "",
            "position_label": "position 18:08 / 23:40",
        },
        {
            "player": "org.mpris.MediaPlayer2.firefox.instance_1_63",
            "status": "Playing",
            "title": "The First 48 Hours of an AI Civil War - A Realistic Scenario",
            "artist": "Species | Documenting AGI",
            "url": "https://www.youtube.com/watch?v=gwfCWDO4LbM&t=15s",
            "position_label": "position 4:53 / 35:14",
        },
    ]

    capture_result = {
        "media_context": media_context,
        "youtube_transcript": {
            "status": "available",
            "title": "The First 48 Hours of an AI Civil War - A Realistic Scenario",
            "artist": "Species | Documenting AGI",
            "url": "https://www.youtube.com/watch?v=gwfCWDO4LbM&t=15s",
            "transcript_position_label": "position 4:53 / 35:14",
        },
    }

    assert service._primary_media_title(capture_result) == (
        "The First 48 Hours of an AI Civil War - A Realistic Scenario by Species | Documenting AGI"
    )
    assert service._current_media_position_label(capture_result) == "position 4:53 / 35:14"
    media_lines = service._media_prompt_lines(media_context)
    assert "The First 48 Hours of an AI Civil War" in media_lines[1]
    assert "Episode 1" not in media_lines[1]


def test_youtube_transcript_excerpt_uses_reached_captions_not_future_cues(tmp_path: Path) -> None:
    runtime = FakeRuntime(tmp_path, {"should_speak": False, "activity_summary": "idle"})
    service = DesktopContextService(
        runtime=runtime,
        state_dir=runtime.ctx.config.state_dir,
        config=DesktopContextConfig(youtube_transcript_max_chars=360),
    )
    transcript_text = (
        "The completed claim has already happened. "
        "The spoiler claim is about to happen but has not happened yet. "
        "The much later claim is definitely future content."
    )

    payload = service._transcript_payload(
        "retrospective123",
        "https://www.youtube.com/watch?v=retrospective123",
        transcript_text,
        source="fake",
        media_context=[
            {
                "title": "Retrospective Commentary Timing",
                "url": "https://www.youtube.com/watch?v=retrospective123",
                "length_us": 180_000_000,
                "position_us": 61_200_000,
                "position_label": "position 1:01 / 3:00",
            }
        ],
        transcript_path=tmp_path / "retrospective123.txt",
        cached=True,
        segments=[
            {"start_seconds": 48.0, "end_seconds": 54.0, "text": "The completed claim has already happened."},
            {
                "start_seconds": 60.5,
                "end_seconds": 65.0,
                "text": "The spoiler claim is about to happen but has not happened yet.",
            },
            {"start_seconds": 140.0, "end_seconds": 145.0, "text": "The much later claim is definitely future content."},
        ],
    )

    assert payload["transcript_excerpt_basis"] == "caption_timestamp_reached"
    assert payload["transcript_target_seconds"] == pytest.approx(61.2)
    assert payload["transcript_commentary_cutoff_seconds"] == pytest.approx(55.2)
    assert payload["transcript_aligned_start_seconds"] == pytest.approx(48.0)
    assert payload["transcript_aligned_end_seconds"] == pytest.approx(54.0)
    assert "completed claim" in payload["transcript_excerpt"]
    assert "spoiler claim" not in payload["transcript_excerpt"]
    assert "much later claim" not in payload["transcript_excerpt"]


def test_youtube_transcript_alignment_rejects_stale_mpris_position(tmp_path: Path) -> None:
    runtime = FakeRuntime(tmp_path, {"should_speak": False, "activity_summary": "idle"})
    service = DesktopContextService(
        runtime=runtime,
        state_dir=runtime.ctx.config.state_dir,
        config=DesktopContextConfig(youtube_transcript_max_chars=260),
    )
    transcript_text = "Opening claim about research agents. Second claim about summarizing evidence."

    payload = service._transcript_payload(
        "staleMpris123",
        "https://www.youtube.com/watch?v=staleMpris123",
        transcript_text,
        source="fake",
        media_context=[
            {
                "title": "Short AutoResearch Tutorial",
                "url": "https://www.youtube.com/watch?v=staleMpris123",
                "length_us": 15_500_000_000,
                "position_us": 9_400_000_000,
                "position_label": "position 2:36:40 / 4:18:20",
            }
        ],
        transcript_path=tmp_path / "staleMpris123.txt",
        cached=True,
        segments=[
            {"start_seconds": 2.0, "end_seconds": 4.0, "text": "Opening claim about research agents."},
            {"start_seconds": 8.0, "end_seconds": 11.0, "text": "Second claim about summarizing evidence."},
        ],
    )

    assert payload["transcript_excerpt_basis"] == "position_out_of_range"
    assert payload["transcript_position_valid"] is False
    assert "outside caption range" in payload["transcript_position_warning"]
    assert payload["transcript_target_seconds"] == pytest.approx(9400.0)
    assert payload["transcript_aligned_start_seconds"] is None
    assert payload["transcript_aligned_end_seconds"] is None
    assert "Opening claim about research agents" in payload["transcript_excerpt"]
    assert "Second claim about summarizing evidence" in payload["transcript_excerpt"]


def test_caption_duration_correction_removes_false_near_end_prompt(tmp_path: Path) -> None:
    runtime = FakeRuntime(tmp_path, {"should_speak": False, "activity_summary": "idle"})
    service = DesktopContextService(runtime=runtime, state_dir=runtime.ctx.config.state_dir)
    capture_result = {
        "media_context": [
            {
                "title": "UFO Roundtable: CIA Physicist Proves Aliens Exist!",
                "artist": "The Diary Of A CEO",
                "url": "https://www.youtube.com/watch?v=longVideo",
                "status": "Playing",
                "length_us": 2_114_000_000,
                "position_us": 2_114_000_000,
                "position_label": "position 35:14 / 35:14",
                "progress_percent": 100.0,
            }
        ]
    }
    payload = service._transcript_payload(
        "longVideo",
        "https://www.youtube.com/watch?v=longVideo",
        "Opening claim. Current long-video segment. Actual end.",
        source="fake",
        media_context=capture_result["media_context"],
        transcript_path=tmp_path / "longVideo.txt",
        cached=True,
        segments=[
            {"start_seconds": 0.0, "end_seconds": 3.0, "text": "Opening claim."},
            {"start_seconds": 2104.0, "end_seconds": 2108.0, "text": "Current long-video segment."},
            {"start_seconds": 5269.0, "end_seconds": 5272.0, "text": "Actual end."},
        ],
    )

    service._apply_youtube_transcript_timing_to_media_context(capture_result, payload)
    media_prompt = "\n".join(service._media_prompt_lines(capture_result["media_context"]))
    transcript_prompt = "\n".join(service._youtube_transcript_prompt_lines(payload))

    assert "position 35:14 / 35:14" not in media_prompt
    assert "100.0% elapsed" not in media_prompt
    assert "position 35:14 / 1:27:52" in media_prompt
    assert "40.1% elapsed" in media_prompt
    assert "do not treat the reported duration or 100% progress as the video ending" in transcript_prompt
    assert "Current long-video segment" in transcript_prompt


def test_out_of_range_transcript_position_hides_false_completion_prompt(tmp_path: Path) -> None:
    runtime = FakeRuntime(tmp_path, {"should_speak": False, "activity_summary": "idle"})
    service = DesktopContextService(runtime=runtime, state_dir=runtime.ctx.config.state_dir)
    capture_result = {
        "media_context": [
            {
                "title": "The Biggest Android Update Ever",
                "artist": "Marques Brownlee",
                "url": "https://www.youtube.com/watch?v=shortVideo",
                "status": "Playing",
                "length_us": 2_114_000_000,
                "position_us": 2_114_000_000,
                "position_label": "position 35:14 / 35:14",
                "progress_percent": 100.0,
            }
        ]
    }
    payload = service._transcript_payload(
        "shortVideo",
        "https://www.youtube.com/watch?v=shortVideo",
        "Opening claim. Actual short-video ending.",
        source="fake",
        media_context=capture_result["media_context"],
        transcript_path=tmp_path / "shortVideo.txt",
        cached=True,
        segments=[
            {"start_seconds": 0.0, "end_seconds": 2.0, "text": "Opening claim."},
            {"start_seconds": 772.0, "end_seconds": 776.0, "text": "Actual short-video ending."},
        ],
    )

    service._apply_youtube_transcript_timing_to_media_context(capture_result, payload)
    media_prompt = "\n".join(service._media_prompt_lines(capture_result["media_context"]))

    assert payload["transcript_excerpt_basis"] == "position_out_of_range"
    assert payload["transcript_position_valid"] is False
    assert "position 35:14 / 35:14" not in media_prompt
    assert "100.0% elapsed" not in media_prompt
    assert "position 35:14 (timing uncertain)" in media_prompt
    assert "outside caption range" in media_prompt


def test_json3_transcript_parsing_preserves_timestamps(tmp_path: Path) -> None:
    runtime = FakeRuntime(tmp_path, {"should_speak": False, "activity_summary": "idle"})
    service = DesktopContextService(runtime=runtime, state_dir=runtime.ctx.config.state_dir)
    raw = json.dumps(
        {
            "events": [
                {
                    "tStartMs": 1250,
                    "dDurationMs": 1750,
                    "segs": [{"utf8": "Precise "}, {"utf8": "caption timing"}],
                },
                {
                    "tStartMs": 4500,
                    "dDurationMs": 1000,
                    "segs": [{"utf8": "matters."}],
                },
            ]
        }
    )

    data = service._json3_transcript_data(raw)

    assert data["transcript_text"] == "Precise caption timing matters."
    assert data["segments"] == [
        {"start_seconds": 1.25, "end_seconds": 3.0, "text": "Precise caption timing"},
        {"start_seconds": 4.5, "end_seconds": 5.5, "text": "matters."},
    ]


def test_vtt_transcript_parsing_preserves_timestamps(tmp_path: Path) -> None:
    runtime = FakeRuntime(tmp_path, {"should_speak": False, "activity_summary": "idle"})
    service = DesktopContextService(runtime=runtime, state_dir=runtime.ctx.config.state_dir)
    raw = """WEBVTT

00:01:00.500 --> 00:01:04.000
Pause here for the useful comment.

00:01:05.000 --> 00:01:07.250
Then continue watching.
"""

    data = service._subtitle_transcript_data(raw)

    assert data["transcript_text"] == "Pause here for the useful comment. Then continue watching."
    assert data["segments"] == [
        {"start_seconds": 60.5, "end_seconds": 64.0, "text": "Pause here for the useful comment."},
        {"start_seconds": 65.0, "end_seconds": 67.25, "text": "Then continue watching."},
    ]


def test_old_plain_transcript_cache_is_refetched_for_timed_alignment(tmp_path: Path) -> None:
    runtime = FakeRuntime(tmp_path, {"should_speak": False, "activity_summary": "idle"})
    service = DesktopContextService(runtime=runtime, state_dir=runtime.ctx.config.state_dir)
    transcript_dir = runtime.ctx.config.state_dir / "desktop_context" / "transcripts"
    transcript_dir.mkdir(parents=True)
    (transcript_dir / "oldCache123.txt").write_text("Old transcript without segments.", encoding="utf-8")
    (transcript_dir / "oldCache123.json").write_text(
        json.dumps({"source": "yt-dlp"}),
        encoding="utf-8",
    )

    cached = service._cached_transcript(
        "oldCache123",
        url="https://www.youtube.com/watch?v=oldCache123",
        media_context=[],
    )

    assert cached is None


@pytest.mark.asyncio
async def test_youtube_media_matching_self_interest_creates_self_directed_followup(tmp_path: Path) -> None:
    runtime = FakeRuntime(
        tmp_path,
        {
            "should_speak": False,
            "activity_summary": "The operator is watching a YouTube video about old software restoration.",
            "reason": "The video is visible and playing.",
            "spoken_text": "This connects with my old software preservation interest; I want to follow it up.",
            "speech_intent": "none",
            "speech_relevance_score": 0.0,
            "matches_self_interest": True,
            "matched_interests": ["old software preservation and obsolete interfaces"],
            "connection_summary": "The video transcript discusses restoring old software and obsolete interfaces.",
            "why_it_matters": "It overlaps with a durable self-directed curiosity seed.",
            "self_directed_next_step": "Save this as a research thread and compare it with prior old-software notes.",
            "operator_interruption_warranted": True,
            "confidence": 0.91,
            "salience": 0.88,
            "novelty": 0.7,
            "timecode_or_position": "around 2:05 of 8:20",
        },
    )
    runtime.ctx.identity.self_model.self_beliefs = {
        "daydream": {
            "bulma_config": {
                "hobbySeeds": ["old software preservation and obsolete interfaces"],
            }
        }
    }
    runtime.ctx.context_proposal_store = FakeContextProposalStore()
    runtime.cognitive_state_store = FakeCognitiveStateStore()
    runtime.fascination_graph = FascinationGraph()
    spoken: list[str] = []

    class FakeMediaController:
        def current_media(self):
            return [
                {
                    "player": "org.mpris.MediaPlayer2.firefox",
                    "status": "Playing",
                    "title": "Restoring Strange Old Software",
                    "artist": "Archive Channel",
                    "url": "https://www.youtube.com/watch?v=oldSoft12345",
                    "length_us": 500_000_000,
                    "position_us": 125_000_000,
                }
            ]

    service = DesktopContextService(
        runtime=runtime,
        state_dir=runtime.ctx.config.state_dir,
        capture_provider=_capture_provider,
        ocr_provider=lambda path: "YouTube old software restoration video playing",
        media_controller=FakeMediaController(),
        youtube_transcript_provider=lambda url, media: {
            "status": "available",
            "source": "fake",
            "transcript_text": (
                "This segment shows an abandoned interface and the process of restoring old software "
                "so it can be understood and preserved."
            ),
        },
        speech_synthesizer=lambda text: spoken.append(text)
        or {"path": str(tmp_path / "voice.mp3"), "provider": "fake"},
        audio_player=lambda path: {"played": True},
    )

    result = await service.observe_once(force=True, reason="scheduled_body_double")

    assert result["status"] == "observed"
    followup = result["analysis"]["self_interest_followup"]
    assert followup["matches_self_interest"] is True
    assert followup["should_speak"] is True
    assert result["analysis"]["speech_intent"] == "screen_relevant"
    assert spoken == ["This connects with my old software preservation interest; I want to follow it up."]
    assert runtime.ctx.context_proposal_store.saved
    proposal = runtime.ctx.context_proposal_store.saved[-1]
    assert proposal.proposal_kind == "self_directed_media_interest_followup"
    assert proposal.source_lane.value == "reflective"
    assert proposal.validation["self_directed"] is True
    assert proposal.validation["media_context"][0]["position_label"] == "position 2:05 / 8:20"
    assert runtime.cognitive_state_store.events
    assert runtime.fascination_graph.active(limit=1)
    payload = runtime.episodes[-1][2]["payload"]
    assert payload["self_interest_followup"]["context_proposal_id"] == proposal.proposal_id
    assert "Self-directed media curiosity follow-up:" in runtime.saved_memories[-1].content
    assert "position 2:05 / 8:20" in runtime.saved_memories[-1].content


@pytest.mark.asyncio
async def test_observed_youtube_context_matching_shared_work_creates_relevance_followup(tmp_path: Path) -> None:
    runtime = FakeRuntime(
        tmp_path,
        {
            "should_speak": False,
            "activity_summary": "The operator is watching a video about screen-aware assistants.",
            "reason": "The visible video may relate to active OpenCAS work.",
            "spoken_text": "This looks relevant to the body-double context work we are doing.",
            "speech_intent": "none",
            "speech_relevance_score": 0.0,
            "matches_observed_context": True,
            "matches_self_interest": False,
            "matches_shared_work": True,
            "match_category": "shared_work",
            "matched_targets": ["Implement body-double YouTube transcript and screen-context awareness"],
            "connection_summary": "The transcript discusses an assistant reasoning about visible screen context.",
            "why_it_matters": "It can inform the current body-double observer design.",
            "self_directed_next_step": "Compare the observed pattern against the OpenCAS body-double implementation.",
            "operator_interruption_warranted": True,
            "confidence": 0.9,
            "salience": 0.87,
            "novelty": 0.45,
            "timecode_or_position": "around 0:42 of 6:00",
        },
    )
    runtime.ctx.context_proposal_store = FakeContextProposalStore()
    runtime.cognitive_state_store = FakeCognitiveStateStore()
    runtime.fascination_graph = FascinationGraph()
    spoken: list[str] = []

    class FakeMediaController:
        def current_media(self):
            return [
                {
                    "player": "org.mpris.MediaPlayer2.firefox",
                    "status": "Playing",
                    "title": "Building Screen Aware Assistants",
                    "artist": "Research Channel",
                    "url": "https://www.youtube.com/watch?v=screenWork123",
                    "length_us": 360_000_000,
                    "position_us": 42_000_000,
                }
            ]

    service = DesktopContextService(
        runtime=runtime,
        state_dir=runtime.ctx.config.state_dir,
        capture_provider=_capture_provider,
        ocr_provider=lambda path: "YouTube video about screen-aware assistants and desktop context",
        media_controller=FakeMediaController(),
        youtube_transcript_provider=lambda url, media: {
            "status": "available",
            "source": "fake",
            "transcript_text": "The assistant uses screen context and transcripts to reason about what the user is doing.",
        },
        speech_synthesizer=lambda text: spoken.append(text)
        or {"path": str(tmp_path / "voice.mp3"), "provider": "fake"},
        audio_player=lambda path: {"played": True},
    )
    service.configure(
        enabled=True,
        declared_task="Implement body-double YouTube transcript and screen-context awareness.",
    )

    result = await service.observe_once(force=True, reason="scheduled_body_double")

    assert result["status"] == "observed"
    followup = result["analysis"]["self_interest_followup"]
    assert followup["matches_observed_context"] is True
    assert followup["matches_shared_work"] is True
    assert followup["matches_self_interest"] is False
    assert spoken == ["This looks relevant to the body-double context work we are doing."]
    proposal = runtime.ctx.context_proposal_store.saved[-1]
    assert proposal.proposal_kind == "observed_context_relevance_followup"
    assert proposal.validation["self_directed"] is False
    assert proposal.validation["work_relevant"] is True
    assert proposal.validation["matched_targets"] == [
        "Implement body-double YouTube transcript and screen-context awareness"
    ]
    assert "Observed context relevance follow-up:" in runtime.saved_memories[-1].content
    assert "Matched relevance targets" in runtime.saved_memories[-1].content


@pytest.mark.asyncio
async def test_video_can_match_opencas_project_context_without_declared_task(tmp_path: Path) -> None:
    runtime = FakeRuntime(
        tmp_path,
        {
            "should_speak": False,
            "activity_summary": "The operator is watching a video about persistent autonomous agents.",
            "reason": "The visible video may relate to OpenCAS.",
            "spoken_text": "This is relevant to OpenCAS; the speaker is describing persistent agent behavior.",
            "speech_intent": "none",
            "speech_relevance_score": 0.0,
            "matches_observed_context": True,
            "matches_self_interest": False,
            "matches_shared_work": True,
            "match_category": "project_context",
            "matched_targets": ["OpenCAS persistent autonomous agent project context"],
            "connection_summary": "The video discusses persistent autonomous agents with memory and proactive behavior.",
            "why_it_matters": "That overlaps with the OpenCAS project contract.",
            "self_directed_next_step": "Preserve this as evidence for the body-double and autonomy design.",
            "operator_interruption_warranted": True,
            "confidence": 0.92,
            "salience": 0.9,
            "novelty": 0.5,
            "timecode_or_position": "around 1:12 of 12:00",
        },
    )
    runtime.ctx.context_proposal_store = FakeContextProposalStore()
    runtime.cognitive_state_store = FakeCognitiveStateStore()
    runtime.fascination_graph = FascinationGraph()
    spoken: list[str] = []

    class FakeMediaController:
        def current_media(self):
            return [
                {
                    "player": "org.mpris.MediaPlayer2.firefox",
                    "status": "Playing",
                    "title": "Persistent Autonomous Assistants",
                    "artist": "Agent Research",
                    "url": "https://www.youtube.com/watch?v=openCASrelevant",
                    "length_us": 720_000_000,
                    "position_us": 72_000_000,
                }
            ]

    service = DesktopContextService(
        runtime=runtime,
        state_dir=runtime.ctx.config.state_dir,
        capture_provider=_capture_provider,
        ocr_provider=lambda path: "persistent autonomous agent memory proactive behavior",
        media_controller=FakeMediaController(),
        youtube_transcript_provider=lambda url, media: {
            "status": "available",
            "source": "fake",
            "transcript_text": "A persistent autonomous assistant can remember across sessions and act proactively.",
        },
        speech_synthesizer=lambda text: spoken.append(text)
        or {"path": str(tmp_path / "voice.mp3"), "provider": "fake"},
        audio_player=lambda path: {"played": True},
    )
    service.configure(enabled=True, declared_task="")

    result = await service.observe_once(force=True, reason="scheduled_body_double")

    assert result["status"] == "observed"
    assert spoken == ["This is relevant to OpenCAS; the speaker is describing persistent agent behavior."]
    proposal = runtime.ctx.context_proposal_store.saved[-1]
    assert proposal.proposal_kind == "observed_context_relevance_followup"
    assert proposal.validation["match_category"] == "project_context"
    prompt_content = runtime.llm.calls[-1]["messages"][1]["content"]
    assert "OpenCAS" in prompt_content
    assert "Do not wait for the operator" in prompt_content


@pytest.mark.asyncio
async def test_obvious_opencas_video_relevance_persists_silently_when_llm_misses_match(tmp_path: Path) -> None:
    runtime = FakeRuntime(
        tmp_path,
        {
            "should_speak": False,
            "activity_summary": "The operator is watching a video about AI assistants.",
            "reason": "The video is visible and playing.",
            "spoken_text": "",
            "speech_intent": "none",
            "speech_relevance_score": 0.0,
        },
    )
    runtime.ctx.context_proposal_store = FakeContextProposalStore()
    runtime.cognitive_state_store = FakeCognitiveStateStore()
    runtime.fascination_graph = FascinationGraph()
    spoken: list[str] = []

    class FakeMediaController:
        def current_media(self):
            return [
                {
                    "player": "org.mpris.MediaPlayer2.firefox",
                    "status": "Playing",
                    "title": "Consumer AI Has a Problem Nobody's Naming.",
                    "artist": "AI News & Strategy Daily | Nate B Jones",
                    "url": "https://www.youtube.com/watch?v=Z0HizICooiw",
                    "length_us": 1_550_100_000,
                    "position_us": 676_300_000,
                }
            ]

    service = DesktopContextService(
        runtime=runtime,
        state_dir=runtime.ctx.config.state_dir,
        capture_provider=_capture_provider,
        ocr_provider=lambda path: "YouTube video: Consumer AI Has a Problem Nobody's Naming.",
        media_controller=FakeMediaController(),
        youtube_transcript_provider=lambda url, media: {
            "status": "available",
            "source": "fake",
            "transcript_text": (
                "The main problem in consumer AI is that the software is capable enough to help, "
                "but it has become another thing to manage. I do not need another chatbot or "
                "another agent waiting for me to assign work. A proactive assistant should reduce "
                "management overhead, understand context, know when to interrupt, and help without "
                "turning into one more inbox."
            ),
        },
        speech_synthesizer=lambda text: spoken.append(text)
        or {"path": str(tmp_path / "voice.mp3"), "provider": "fake"},
        audio_player=lambda path: {"played": True},
    )
    service.configure(enabled=True, declared_task="")

    result = await service.observe_once(force=True, reason="scheduled_body_double")

    assert result["status"] == "observed"
    followup = result["analysis"]["self_interest_followup"]
    assert followup["matches_observed_context"] is True
    assert followup["matches_shared_work"] is True
    assert followup["match_category"] == "project_context"
    assert followup["heuristic_fallback"] is True
    assert followup["agent_viewpoint"] == ""
    assert "heuristic" in followup["heuristic_evidence_summary"].lower()
    assert followup["why_it_matters"] == ""
    assert followup["implications"] == []
    assert followup["open_questions"] == []
    assert followup["self_directed_next_step"] == ""
    assert followup["should_speak"] is False
    assert spoken == []
    assert runtime.ctx.context_proposal_store.saved


@pytest.mark.asyncio
async def test_media_commentary_speaks_viewpoint_not_video_summary(tmp_path: Path) -> None:
    runtime = FakeRuntime(
        tmp_path,
        [
            {
                "should_speak": True,
                "activity_summary": "The operator is watching a video about proactive AI agents.",
                "reason": "The current video segment is relevant.",
                "spoken_text": "The video is about proactive AI agents and how they interrupt users.",
                "speech_intent": "screen_relevant",
                "speech_relevance_score": 0.9,
            },
            {
                "matches_observed_context": True,
                "matches_self_interest": False,
                "matches_shared_work": True,
                "match_category": "project_context",
                "matched_targets": ["OpenCAS proactive assistant design"],
                "connection_summary": "The current segment overlaps OpenCAS interruption and autonomy design.",
                "agent_viewpoint": (
                    "My take is that the useful bar is not whether an agent can interrupt, "
                    "but whether it has earned the interruption by producing a grounded synthesis."
                ),
                "why_it_matters": "This can sharpen the body-double speech policy.",
                "implications": ["Prefer opinions and synthesis over screen narration."],
                "open_questions": ["How should OpenCAS decide an insight is worth pausing media for?"],
                "self_directed_next_step": "Save this as a thought artifact for body-double design.",
                "operator_interruption_warranted": True,
                "spoken_text": (
                    "My take is that the useful bar is not whether an agent can interrupt, "
                    "but whether it has earned the interruption by producing a grounded synthesis."
                ),
                "confidence": 0.94,
                "salience": 0.91,
                "novelty": 0.56,
                "timecode_or_position": "around 4:10 of 12:00",
            },
        ],
    )
    runtime.ctx.context_proposal_store = FakeContextProposalStore()
    runtime.cognitive_state_store = FakeCognitiveStateStore()
    runtime.fascination_graph = FascinationGraph()
    spoken: list[str] = []

    class FakeMediaController:
        def current_media(self):
            return [
                {
                    "player": "org.mpris.MediaPlayer2.firefox",
                    "status": "Playing",
                    "title": "Useful Proactive Agents",
                    "artist": "Agent Research",
                    "url": "https://www.youtube.com/watch?v=opinionAgent1",
                    "length_us": 720_000_000,
                    "position_us": 250_000_000,
                }
            ]

    service = DesktopContextService(
        runtime=runtime,
        state_dir=runtime.ctx.config.state_dir,
        capture_provider=_capture_provider,
        ocr_provider=lambda path: "YouTube video about proactive AI agents",
        media_controller=FakeMediaController(),
        youtube_transcript_provider=lambda url, media: {
            "status": "available",
            "source": "fake",
            "transcript_text": (
                "A proactive agent should not just notify the user. It should form an opinion, "
                "create useful artifacts, and interrupt only when the insight is grounded."
            ),
        },
        speech_synthesizer=lambda text: spoken.append(text)
        or {"path": str(tmp_path / "voice.mp3"), "provider": "fake"},
        audio_player=lambda path: {"played": True},
    )

    result = await service.observe_once(force=True, reason="scheduled_body_double")

    assert result["status"] == "observed"
    assert spoken == [
        "My take is that the useful bar is not whether an agent can interrupt, "
        "but whether it has earned the interruption by producing a grounded synthesis."
    ]
    assert "The video is about" not in spoken[0]
    followup = result["analysis"]["self_interest_followup"]
    assert followup["agent_viewpoint"].startswith("My take")
    proposal = runtime.ctx.context_proposal_store.saved[-1]
    assert "Agent viewpoint:" in proposal.content
    assert "Implications:" in proposal.content
    assert "Open questions:" in proposal.content
    assert "grounded synthesis" in runtime.cognitive_state_store.events[-1][2]["content"]
    assert "Observed media thought artifact:" in runtime.saved_memories[-1].content


@pytest.mark.asyncio
async def test_media_relevance_creates_thought_artifact_even_when_silent(tmp_path: Path) -> None:
    runtime = FakeRuntime(
        tmp_path,
        [
            {
                "should_speak": False,
                "activity_summary": "The operator is watching a relevant AI video.",
                "reason": "The segment is worth preserving but not interrupting over.",
                "spoken_text": "",
                "speech_intent": "none",
                "speech_relevance_score": 0.0,
            },
            {
                "matches_observed_context": True,
                "matches_self_interest": False,
                "matches_shared_work": True,
                "match_category": "project_context",
                "matched_targets": ["OpenCAS autonomy and memory design"],
                "connection_summary": "The video segment frames proactive agents as systems that reduce management load.",
                "agent_viewpoint": (
                    "I think this is useful because it names the failure mode OpenCAS should avoid: "
                    "proactivity that creates more management work instead of reducing it."
                ),
                "why_it_matters": "It can be used later when tuning interruption thresholds.",
                "implications": ["Treat noisy proactive nudges as a regression."],
                "open_questions": ["What evidence is enough before the agent speaks?"],
                "self_directed_next_step": "Turn this segment into a design comparison note later.",
                "operator_interruption_warranted": False,
                "spoken_text": "",
                "confidence": 0.91,
                "salience": 0.88,
                "novelty": 0.62,
                "timecode_or_position": "around 6:20 of 18:00",
            },
        ],
    )
    runtime.ctx.context_proposal_store = FakeContextProposalStore()
    runtime.cognitive_state_store = FakeCognitiveStateStore()
    runtime.fascination_graph = FascinationGraph()
    spoken: list[str] = []

    class FakeMediaController:
        def current_media(self):
            return [
                {
                    "player": "org.mpris.MediaPlayer2.firefox",
                    "status": "Playing",
                    "title": "Proactivity That Actually Helps",
                    "artist": "AI Design",
                    "url": "https://www.youtube.com/watch?v=artifactOnly1",
                    "length_us": 1_080_000_000,
                    "position_us": 380_000_000,
                }
            ]

    service = DesktopContextService(
        runtime=runtime,
        state_dir=runtime.ctx.config.state_dir,
        capture_provider=_capture_provider,
        ocr_provider=lambda path: "YouTube video about proactive agents and management load",
        media_controller=FakeMediaController(),
        youtube_transcript_provider=lambda url, media: {
            "status": "available",
            "source": "fake",
            "transcript_text": "Proactive agents should reduce management load, not add another inbox.",
        },
        speech_synthesizer=lambda text: spoken.append(text)
        or {"path": str(tmp_path / "voice.mp3"), "provider": "fake"},
        audio_player=lambda path: {"played": True},
    )

    result = await service.observe_once(force=True, reason="scheduled_body_double")

    assert result["status"] == "observed"
    assert spoken == []
    assert result["analysis"]["should_speak"] is False
    proposal = runtime.ctx.context_proposal_store.saved[-1]
    assert "Agent viewpoint:" in proposal.content
    assert "management work instead of reducing it" in proposal.content
    assert runtime.cognitive_state_store.events
    assert runtime.fascination_graph.active(limit=1)


@pytest.mark.asyncio
async def test_media_summary_speech_is_suppressed_without_viewpoint(tmp_path: Path) -> None:
    runtime = FakeRuntime(
        tmp_path,
        [
            {
                "should_speak": True,
                "activity_summary": "The operator is watching a video about AI agents.",
                "reason": "The video is visible.",
                "spoken_text": "The video is about AI agents and proactive notifications.",
                "speech_intent": "screen_relevant",
                "speech_relevance_score": 0.91,
            },
            {
                "matches_observed_context": False,
                "matches_self_interest": False,
                "confidence": 0.0,
                "salience": 0.0,
            },
        ],
    )
    spoken: list[str] = []

    class FakeMediaController:
        def current_media(self):
            return [
                {
                    "player": "org.mpris.MediaPlayer2.firefox",
                    "status": "Playing",
                    "title": "Generic AI Agents",
                    "artist": "Example",
                    "url": "https://www.youtube.com/watch?v=summaryOnly1",
                    "length_us": 500_000_000,
                    "position_us": 50_000_000,
                }
            ]

    service = DesktopContextService(
        runtime=runtime,
        state_dir=runtime.ctx.config.state_dir,
        capture_provider=_capture_provider,
        ocr_provider=lambda path: "YouTube video about AI agents",
        media_controller=FakeMediaController(),
        youtube_transcript_provider=lambda url, media: {
            "status": "available",
            "source": "fake",
            "transcript_text": "This video discusses AI agents.",
        },
        speech_synthesizer=lambda text: spoken.append(text)
        or {"path": str(tmp_path / "voice.mp3"), "provider": "fake"},
        audio_player=lambda path: {"played": True},
    )

    result = await service.observe_once(force=True, reason="scheduled_body_double")

    assert result["status"] == "observed"
    assert spoken == []
    assert result["analysis"]["should_speak"] is False
    assert result["analysis"]["speech_policy"] == "suppressed_media_summary_without_viewpoint"


@pytest.mark.asyncio
async def test_conversation_observation_forces_fresh_snapshot_and_prompt_note(tmp_path: Path) -> None:
    runtime = FakeRuntime(
        tmp_path,
        {
            "should_speak": False,
            "activity_summary": "The operator is asking a question while a YouTube video is playing.",
            "reason": "Conversation context snapshot.",
            "spoken_text": "",
            "speech_intent": "none",
            "speech_relevance_score": 0.0,
        },
    )
    capture_count = 0

    def capture(path: Path) -> DesktopCapture:
        nonlocal capture_count
        capture_count += 1
        return _capture_provider(path)

    class FakeMediaController:
        def current_media(self):
            return [
                {
                    "player": "org.mpris.MediaPlayer2.firefox",
                    "status": "Playing",
                    "title": "Visible YouTube context",
                    "artist": "Example Channel",
                    "url": "https://www.youtube.com/watch?v=abc123DEF45",
                }
            ]

    service = DesktopContextService(
        runtime=runtime,
        state_dir=runtime.ctx.config.state_dir,
        capture_provider=capture,
        ocr_provider=lambda path: "visible browser question context",
        media_controller=FakeMediaController(),
        youtube_transcript_provider=lambda url, media: {
            "status": "available",
            "source": "fake",
            "transcript_text": "Transcript evidence for the currently visible video.",
        },
        speech_synthesizer=lambda text: (_ for _ in ()).throw(AssertionError("conversation observation must not speak")),
    )
    service.configure(enabled=True, capture_interval_seconds=9999)
    first = await service.observe_once(force=True, reason="scheduled_body_double")
    assert first["status"] == "observed"

    result = await service.observe_for_conversation(
        session_id="telegram-1",
        user_input="What do you think about this?",
        source="telegram",
    )

    assert result is not None
    assert result["status"] == "observed"
    assert capture_count == 2
    assert result["speech"] is None
    assert "Body-double environment is active for this user turn." in result["conversation_prompt_note"]
    assert "conversational input from someone in the room" in result["conversation_prompt_note"]
    assert "Transcript evidence for the currently visible video" in result["conversation_prompt_note"]
    assert runtime.episodes[-1][2]["session_id"] == "telegram-1"


@pytest.mark.asyncio
async def test_body_double_suppresses_task_coaching_without_declared_task(tmp_path: Path) -> None:
    runtime = FakeRuntime(
        tmp_path,
        {
            "should_speak": True,
            "activity_summary": "The operator has a terminal and browser open.",
            "reason": "They may be drifting from the intended task.",
            "spoken_text": "Stay on task and get back to the next step.",
            "speech_intent": "task_coaching",
            "speech_relevance_score": 0.95,
        },
    )
    spoken: list[str] = []
    service = DesktopContextService(
        runtime=runtime,
        state_dir=runtime.ctx.config.state_dir,
        capture_provider=_capture_provider,
        ocr_provider=lambda path: "terminal browser editor",
        speech_synthesizer=lambda text: spoken.append(text)
        or {"path": str(tmp_path / "voice.mp3"), "provider": "fake"},
        audio_player=lambda path: {"played": True},
    )
    service.configure(enabled=True, tts_enabled=True, play_audio=True)

    result = await service.observe_once(force=True, reason="scheduled_body_double")

    assert result["status"] == "observed"
    assert spoken == []
    assert result["analysis"]["should_speak"] is False
    assert result["analysis"]["speech_policy"] == "suppressed_no_declared_task_for_coaching"
    assert runtime.episodes
    assert runtime.saved_memories


@pytest.mark.asyncio
async def test_body_double_allows_task_coaching_with_declared_task(tmp_path: Path) -> None:
    runtime = FakeRuntime(
        tmp_path,
        {
            "should_speak": True,
            "activity_summary": "The operator is in a terminal near pytest output.",
            "reason": "The visible work relates to the declared task.",
            "spoken_text": "You are still on the test pass. Finish the focused verification step.",
            "speech_intent": "task_coaching",
            "speech_relevance_score": 0.9,
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
        audio_player=lambda path: {"played": True},
    )
    service.configure(
        enabled=True,
        tts_enabled=True,
        play_audio=True,
        declared_task="Finish the desktop-context test pass.",
    )

    result = await service.observe_once(force=True, reason="scheduled_body_double")

    assert result["status"] == "observed"
    assert spoken == ["You are still on the test pass. Finish the focused verification step."]
    assert result["analysis"]["should_speak"] is True
    assert result["analysis"]["declared_task"] == "Finish the desktop-context test pass."
    assert result["analysis"]["speech_policy"] == "allowed_declared_task"
    payload = runtime.episodes[-1][2]["payload"]
    assert payload["declared_task"] == "Finish the desktop-context test pass."
    assert payload["task_coaching_allowed"] is True


@pytest.mark.asyncio
async def test_relevant_screen_observation_can_speak_without_declared_task(tmp_path: Path) -> None:
    runtime = FakeRuntime(
        tmp_path,
        {
            "should_speak": True,
            "activity_summary": "A repeated authentication error is visible.",
            "reason": "The visible error may need immediate attention.",
            "spoken_text": "There is a repeated authentication error on screen that may need attention.",
            "speech_intent": "system_issue",
            "speech_relevance_score": 0.82,
        },
    )
    spoken: list[str] = []
    service = DesktopContextService(
        runtime=runtime,
        state_dir=runtime.ctx.config.state_dir,
        capture_provider=_capture_provider,
        ocr_provider=lambda path: "Authentication failed invalid_grant repeated notification",
        speech_synthesizer=lambda text: spoken.append(text)
        or {"path": str(tmp_path / "voice.mp3"), "provider": "fake"},
        audio_player=lambda path: {"played": True},
    )
    service.configure(enabled=True, tts_enabled=True, play_audio=True)

    result = await service.observe_once(force=True, reason="scheduled_body_double")

    assert result["status"] == "observed"
    assert spoken == ["There is a repeated authentication error on screen that may need attention."]
    assert result["analysis"]["should_speak"] is True
    assert result["analysis"]["speech_policy"] == "allowed_relevant_observation"


@pytest.mark.asyncio
async def test_body_double_pauses_playing_media_while_speaking(tmp_path: Path) -> None:
    runtime = FakeRuntime(
        tmp_path,
        {
            "should_speak": True,
            "activity_summary": "The operator is watching a video.",
            "reason": "A body-double nudge is useful.",
            "spoken_text": "Short spoken nudge.",
            "speech_intent": "task_coaching",
            "speech_relevance_score": 0.8,
        },
    )
    events: list[tuple[str, object]] = []

    class FakeMediaController:
        def pause_playing(self):
            events.append(("pause", None))
            return {"paused_players": ["org.mpris.MediaPlayer2.firefox"], "errors": []}

        def resume_players(self, players):
            events.append(("resume", list(players)))
            return {"resumed_players": list(players), "errors": []}

    service = DesktopContextService(
        runtime=runtime,
        state_dir=runtime.ctx.config.state_dir,
        capture_provider=_capture_provider,
        ocr_provider=lambda path: "video player visible",
        speech_synthesizer=lambda text: {"path": str(tmp_path / "voice.mp3"), "provider": "fake"},
        audio_player=lambda path: events.append(("play", Path(path))) or {"played": True},
        media_controller=FakeMediaController(),
    )
    service.configure(
        enabled=True,
        tts_enabled=True,
        play_audio=True,
        declared_task="Stay with the current media-review task.",
    )

    result = await service.observe_once(force=True, reason="test")

    assert [event[0] for event in events] == ["pause", "play", "resume"]
    assert result["speech"]["playback"]["media"]["paused_players"] == [
        "org.mpris.MediaPlayer2.firefox"
    ]


def test_mpris_media_controller_pauses_and_resumes_only_playing_players() -> None:
    commands: list[list[str]] = []

    def fake_runner(args, **kwargs):
        commands.append(list(args))
        if args[:4] == ["busctl", "--user", "list", "--no-legend"]:
            return subprocess.CompletedProcess(
                args,
                0,
                "org.mpris.MediaPlayer2.firefox 123 user :1.1 - -\n"
                "org.mpris.MediaPlayer2.vlc 124 user :1.2 - -\n",
                "",
            )
        if "get-property" in args and "org.mpris.MediaPlayer2.firefox" in args:
            return subprocess.CompletedProcess(args, 0, 's "Playing"\n', "")
        if "get-property" in args and "org.mpris.MediaPlayer2.vlc" in args:
            return subprocess.CompletedProcess(args, 0, 's "Paused"\n', "")
        if "call" in args:
            return subprocess.CompletedProcess(args, 0, "", "")
        raise AssertionError(f"unexpected command: {args}")

    controller = MprisMediaController(runner=fake_runner, busctl_path="busctl")

    pause = controller.pause_playing()
    resume = controller.resume_players(pause["paused_players"])

    assert pause["paused_players"] == ["org.mpris.MediaPlayer2.firefox"]
    assert resume["resumed_players"] == ["org.mpris.MediaPlayer2.firefox"]
    assert any(command[-1] == "Pause" for command in commands)
    assert any(command[-1] == "Play" for command in commands)
    assert not any("org.mpris.MediaPlayer2.vlc" in command and command[-1] == "Play" for command in commands)


def test_mpris_media_controller_reports_current_youtube_metadata() -> None:
    def fake_runner(args, **kwargs):
        if args[:4] == ["busctl", "--user", "list", "--no-legend"]:
            return subprocess.CompletedProcess(
                args,
                0,
                "org.mpris.MediaPlayer2.firefox 123 user :1.1 - -\n",
                "",
            )
        if "PlaybackStatus" in args:
            return subprocess.CompletedProcess(args, 0, 's "Playing"\n', "")
        if "Metadata" in args:
            return subprocess.CompletedProcess(
                args,
                0,
                'a{sv} 6 "mpris:trackid" o "/org/mpris/MediaPlayer2/firefox" '
                '"xesam:title" s "The 7 Levels of Sith Explained in Detail" '
                '"xesam:artist" as 1 "Order 77" '
                '"xesam:url" s "https://www.youtube.com/watch?v=2n9TOgYHv1E" '
                '"mpris:length" x 125000000\n',
                "",
            )
        if "Position" in args:
            return subprocess.CompletedProcess(args, 0, "x 62500000\n", "")
        raise AssertionError(f"unexpected command: {args}")

    controller = MprisMediaController(runner=fake_runner, busctl_path="busctl")

    media = controller.current_media()

    assert media == [
        {
            "player": "org.mpris.MediaPlayer2.firefox",
            "status": "Playing",
            "title": "The 7 Levels of Sith Explained in Detail",
            "artist": "Order 77",
            "album": "",
            "url": "https://www.youtube.com/watch?v=2n9TOgYHv1E",
            "length_us": 125000000,
            "position_us": 62500000,
            "raw": {
                "xesam:title": "The 7 Levels of Sith Explained in Detail",
                "xesam:artist": "Order 77",
                "xesam:url": "https://www.youtube.com/watch?v=2n9TOgYHv1E",
                "mpris:length": 125000000,
            },
        }
    ]


def test_mpris_media_controller_reports_paused_media_metadata() -> None:
    def fake_runner(args, **kwargs):
        if args[:4] == ["busctl", "--user", "list", "--no-legend"]:
            return subprocess.CompletedProcess(
                args,
                0,
                "org.mpris.MediaPlayer2.vlc 123 user :1.2 - -\n",
                "",
            )
        if "PlaybackStatus" in args:
            return subprocess.CompletedProcess(args, 0, 's "Paused"\n', "")
        if "Metadata" in args:
            return subprocess.CompletedProcess(
                args,
                0,
                'a{sv} 4 "xesam:title" s "Local Lecture" '
                '"xesam:artist" s "Archive" '
                '"xesam:url" s "file:///videos/lecture.mp4" '
                '"mpris:length" x 90000000\n',
                "",
            )
        if "Position" in args:
            return subprocess.CompletedProcess(args, 0, "x 45000000\n", "")
        raise AssertionError(f"unexpected command: {args}")

    controller = MprisMediaController(runner=fake_runner, busctl_path="busctl")

    media = controller.current_media()

    assert media[0]["status"] == "Paused"
    assert media[0]["title"] == "Local Lecture"
    assert media[0]["position_us"] == 45000000


@pytest.mark.asyncio
async def test_desktop_context_records_media_state_transitions(tmp_path: Path) -> None:
    runtime = FakeRuntime(tmp_path, {"should_speak": False, "activity_summary": "idle"})
    snapshots = [
        [
            {
                "player": "org.mpris.MediaPlayer2.firefox",
                "status": "Playing",
                "title": "Video A",
                "url": "https://www.youtube.com/watch?v=videoA",
                "length_us": 120_000_000,
                "position_us": 10_000_000,
            }
        ],
        [
            {
                "player": "org.mpris.MediaPlayer2.firefox",
                "status": "Paused",
                "title": "Video A",
                "url": "https://www.youtube.com/watch?v=videoA",
                "length_us": 120_000_000,
                "position_us": 10_000_000,
            }
        ],
        [
            {
                "player": "org.mpris.MediaPlayer2.firefox",
                "status": "Playing",
                "title": "Video A",
                "url": "https://www.youtube.com/watch?v=videoA",
                "length_us": 120_000_000,
                "position_us": 70_000_000,
            }
        ],
        [
            {
                "player": "org.mpris.MediaPlayer2.firefox",
                "status": "Playing",
                "title": "Video B",
                "url": "https://www.youtube.com/watch?v=videoB",
                "length_us": 60_000_000,
                "position_us": 2_000_000,
            }
        ],
        [],
    ]

    class FakeMediaController:
        def current_media(self):
            return snapshots.pop(0)

    service = DesktopContextService(
        runtime=runtime,
        state_dir=runtime.ctx.config.state_dir,
        media_controller=FakeMediaController(),
        config=DesktopContextConfig(youtube_transcripts_enabled=False),
    )
    payload: dict = {}

    await service._enrich_capture_context(payload)
    assert payload["media_state_changes"][0]["event"] == "media_started"

    payload = {}
    await service._enrich_capture_context(payload)
    assert payload["media_state_changes"][0]["event"] == "media_paused"

    payload = {}
    await service._enrich_capture_context(payload)
    events = {change["event"] for change in payload["media_state_changes"]}
    assert "media_resumed" in events
    assert "media_seeked" in events

    payload = {}
    await service._enrich_capture_context(payload)
    events = {change["event"] for change in payload["media_state_changes"]}
    assert {"media_stopped", "media_started"} <= events

    payload = {}
    await service._enrich_capture_context(payload)
    assert payload["media_state_changes"][0]["event"] == "media_stopped"
    assert any(event == "desktop_context_media_state_changed" for event, _ in runtime.traces)


@pytest.mark.asyncio
async def test_desktop_context_polls_media_state_without_desktop_capture(tmp_path: Path) -> None:
    runtime = FakeRuntime(tmp_path, {"should_speak": False, "activity_summary": "idle"})

    class FakeMediaController:
        def current_media(self):
            return [
                {
                    "player": "org.mpris.MediaPlayer2.vlc",
                    "status": "Playing",
                    "title": "Local Clip",
                    "url": "file:///clips/local.mp4",
                    "length_us": 90_000_000,
                    "position_us": 5_000_000,
                }
            ]

    service = DesktopContextService(
        runtime=runtime,
        state_dir=runtime.ctx.config.state_dir,
        media_controller=FakeMediaController(),
        config=DesktopContextConfig(youtube_transcripts_enabled=False),
        capture_provider=lambda path: (_ for _ in ()).throw(AssertionError("desktop capture should not run")),
    )

    result = await service.poll_media_state_once()

    assert result["status"] == "changed"
    assert result["media_state_changes"][0]["event"] == "media_started"
    assert result["media_context"][0]["status"] == "Playing"
    assert runtime.traces[-1][0] == "desktop_context_media_state_changed"


def test_normal_playback_progress_after_delayed_poll_is_not_seek(tmp_path: Path) -> None:
    runtime = FakeRuntime(tmp_path, {"should_speak": False, "activity_summary": "idle"})
    service = DesktopContextService(runtime=runtime, state_dir=runtime.ctx.config.state_dir)
    previous = {
        "firefox|https://www.youtube.com/watch?v=steadyVideo": {
            "player": "firefox",
            "status": "Playing",
            "title": "Steady Playback",
            "url": "https://www.youtube.com/watch?v=steadyVideo",
            "position_us": 10_000_000,
            "position_label": "position 0:10 / 10:00",
        }
    }
    current = {
        "firefox|https://www.youtube.com/watch?v=steadyVideo": {
            "player": "firefox",
            "status": "Playing",
            "title": "Steady Playback",
            "url": "https://www.youtube.com/watch?v=steadyVideo",
            "position_us": 40_000_000,
            "position_label": "position 0:40 / 10:00",
        }
    }

    changes = service._detect_media_state_changes(
        previous,
        current,
        observed_at="2026-05-14T13:00:30+00:00",
        previous_observed_at="2026-05-14T13:00:00+00:00",
    )

    assert changes == []


def test_playback_jump_beyond_elapsed_time_is_still_seek(tmp_path: Path) -> None:
    runtime = FakeRuntime(tmp_path, {"should_speak": False, "activity_summary": "idle"})
    service = DesktopContextService(runtime=runtime, state_dir=runtime.ctx.config.state_dir)
    previous = {
        "firefox|https://www.youtube.com/watch?v=jumpVideo": {
            "player": "firefox",
            "status": "Playing",
            "title": "Jump Playback",
            "url": "https://www.youtube.com/watch?v=jumpVideo",
            "position_us": 10_000_000,
            "position_label": "position 0:10 / 10:00",
        }
    }
    current = {
        "firefox|https://www.youtube.com/watch?v=jumpVideo": {
            "player": "firefox",
            "status": "Playing",
            "title": "Jump Playback",
            "url": "https://www.youtube.com/watch?v=jumpVideo",
            "position_us": 140_000_000,
            "position_label": "position 2:20 / 10:00",
        }
    }

    changes = service._detect_media_state_changes(
        previous,
        current,
        observed_at="2026-05-14T13:00:30+00:00",
        previous_observed_at="2026-05-14T13:00:00+00:00",
    )

    assert changes[0]["event"] == "media_seeked"
    assert changes[0]["position_delta_seconds"] == 130.0


def test_pause_after_normal_playback_progress_is_not_seek(tmp_path: Path) -> None:
    runtime = FakeRuntime(tmp_path, {"should_speak": False, "activity_summary": "idle"})
    service = DesktopContextService(runtime=runtime, state_dir=runtime.ctx.config.state_dir)
    previous = {
        "firefox|https://www.youtube.com/watch?v=pauseVideo": {
            "player": "firefox",
            "status": "Playing",
            "title": "Pause Playback",
            "url": "https://www.youtube.com/watch?v=pauseVideo",
            "position_us": 10_000_000,
            "position_label": "position 0:10 / 10:00",
        }
    }
    current = {
        "firefox|https://www.youtube.com/watch?v=pauseVideo": {
            "player": "firefox",
            "status": "Paused",
            "title": "Pause Playback",
            "url": "https://www.youtube.com/watch?v=pauseVideo",
            "position_us": 35_000_000,
            "position_label": "position 0:35 / 10:00",
        }
    }

    changes = service._detect_media_state_changes(
        previous,
        current,
        observed_at="2026-05-14T13:00:30+00:00",
        previous_observed_at="2026-05-14T13:00:00+00:00",
    )

    assert [change["event"] for change in changes] == ["media_paused"]


@pytest.mark.asyncio
async def test_stale_video_observation_is_not_spoken_after_media_changes(tmp_path: Path) -> None:
    runtime = FakeRuntime(
        tmp_path,
        {
            "should_speak": True,
            "activity_summary": "The old video is playing.",
            "reason": "The old video has a relevant claim.",
            "spoken_text": "This is a comment about the old video.",
            "speech_intent": "screen_relevant",
            "speech_relevance_score": 0.95,
        },
    )
    spoken: list[str] = []
    service = DesktopContextService(
        runtime=runtime,
        state_dir=runtime.ctx.config.state_dir,
        config=DesktopContextConfig(
            media_commentary_mode_enabled=True,
            play_audio=False,
            min_speech_interval_seconds=0,
            self_interest_followup_enabled=False,
            observed_context_relevance_enabled=False,
            project_context_relevance_enabled=False,
            youtube_transcripts_enabled=False,
        ),
        capture_provider=_capture_provider,
        ocr_provider=lambda path: "YouTube old video",
        media_controller=SequenceMediaController(
            [
                [_youtube_media_item(title="Old Video", video_id="oldVideo")],
                [_youtube_media_item(title="New Video", video_id="newVideo")],
            ]
        ),
        speech_synthesizer=lambda text: spoken.append(text) or {"path": str(tmp_path / "voice.mp3")},
    )

    result = await service.observe_once(force=True, reason="media_commentary_mode:media_started")

    assert result["status"] == "observed"
    assert result["speech"] == {"status": "skipped", "reason": "media_context_stale"}
    assert spoken == []


@pytest.mark.asyncio
async def test_same_video_observation_is_not_spoken_after_playback_moves_too_far(tmp_path: Path) -> None:
    runtime = FakeRuntime(
        tmp_path,
        {
            "should_speak": True,
            "activity_summary": "The current video is playing.",
            "reason": "The current segment has a relevant claim.",
            "spoken_text": "This is a comment about the current segment.",
            "speech_intent": "screen_relevant",
            "speech_relevance_score": 0.95,
        },
    )
    spoken: list[str] = []
    first_snapshot = [
        _youtube_media_item(title="Same Video", video_id="sameVideo")
        | {"position_us": 60_000_000, "position_label": "position 1:00 / 10:00"}
    ]
    later_snapshot = [
        _youtube_media_item(title="Same Video", video_id="sameVideo")
        | {"position_us": 120_000_000, "position_label": "position 2:00 / 10:00"}
    ]
    service = DesktopContextService(
        runtime=runtime,
        state_dir=runtime.ctx.config.state_dir,
        config=DesktopContextConfig(
            media_commentary_mode_enabled=True,
            play_audio=False,
            min_speech_interval_seconds=0,
            self_interest_followup_enabled=False,
            observed_context_relevance_enabled=False,
            project_context_relevance_enabled=False,
            youtube_transcripts_enabled=False,
        ),
        capture_provider=_capture_provider,
        ocr_provider=lambda path: "YouTube same video",
        media_controller=SequenceMediaController([first_snapshot, later_snapshot]),
        speech_synthesizer=lambda text: spoken.append(text) or {"path": str(tmp_path / "voice.mp3")},
    )

    result = await service.observe_once(force=True, reason="media_commentary_mode:media_seeked")

    assert result["status"] == "observed"
    assert result["speech"] == {"status": "skipped", "reason": "media_position_stale"}
    assert spoken == []


@pytest.mark.asyncio
async def test_multiple_playing_videos_speak_ambiguity_instead_of_guessing(tmp_path: Path) -> None:
    runtime = FakeRuntime(
        tmp_path,
        {
            "should_speak": True,
            "activity_summary": "The operator is watching a video.",
            "reason": "The video is relevant.",
            "spoken_text": "This should not be used because there are multiple videos.",
            "speech_intent": "screen_relevant",
            "speech_relevance_score": 0.95,
        },
    )
    spoken: list[str] = []
    service = DesktopContextService(
        runtime=runtime,
        state_dir=runtime.ctx.config.state_dir,
        config=DesktopContextConfig(
            media_commentary_mode_enabled=True,
            play_audio=False,
            min_speech_interval_seconds=0,
            self_interest_followup_enabled=False,
            observed_context_relevance_enabled=False,
            project_context_relevance_enabled=False,
            youtube_transcripts_enabled=False,
        ),
        capture_provider=_capture_provider,
        ocr_provider=lambda path: "Two videos are visible.",
        media_controller=FakeMediaController(
            [
                _youtube_media_item(title="Video One", video_id="videoOne"),
                _youtube_media_item(title="Video Two", video_id="videoTwo") | {"player": "chromium"},
            ]
        ),
        speech_synthesizer=lambda text: spoken.append(text) or {"path": str(tmp_path / "voice.mp3")},
    )

    result = await service.observe_once(force=True, reason="media_commentary_mode:media_started")

    assert result["status"] == "observed"
    assert result["analysis"]["media_ambiguity"]["reason"] == "multiple_playing_media"
    assert result["speech"]["status"] == "spoken"
    assert "more than one media source playing" in result["speech"]["spoken_text"]
    assert "This should not be used" not in result["speech"]["spoken_text"]
    assert spoken == [result["speech"]["spoken_text"]]


@pytest.mark.asyncio
async def test_chat_video_commentary_request_enables_following_autoplay_videos(tmp_path: Path) -> None:
    runtime = FakeRuntime(
        tmp_path,
        {
            "should_speak": False,
            "activity_summary": "The operator is watching a tutorial.",
            "reason": "Conversation context only.",
            "spoken_text": "",
            "speech_intent": "none",
            "speech_relevance_score": 0.0,
        },
    )

    class FakeMediaController:
        def current_media(self):
            return [
                {
                    "player": "org.mpris.MediaPlayer2.firefox",
                    "status": "Playing",
                    "title": "Video A",
                    "url": "https://www.youtube.com/watch?v=videoA",
                    "length_us": 120_000_000,
                    "position_us": 10_000_000,
                }
            ]

    service = DesktopContextService(
        runtime=runtime,
        state_dir=runtime.ctx.config.state_dir,
        media_controller=FakeMediaController(),
        capture_provider=_capture_provider,
        ocr_provider=lambda path: "YouTube tutorial",
        youtube_transcript_provider=lambda url, media: {
            "status": "available",
            "source": "fake",
            "transcript_text": "This video is starting.",
            "segments": [{"start_seconds": 1.0, "end_seconds": 3.0, "text": "This video is starting."}],
        },
    )

    result = await service.observe_for_conversation(
        session_id="default",
        user_input="Give me your commentary on this video, key points as I'm watching it, please",
        source="api_chat",
    )

    assert result["status"] == "observed"
    assert service.config.media_commentary_mode_enabled is True
    assert service.config.media_commentary_request_source == "api_chat"
    assert "commentary on this video" in service.config.media_commentary_request_text
    assert "Media commentary mode: active" in result["conversation_prompt_note"]


@pytest.mark.asyncio
async def test_media_started_requests_observation_when_commentary_mode_active(tmp_path: Path) -> None:
    runtime = FakeRuntime(tmp_path, {"should_speak": False, "activity_summary": "idle"})

    class FakeMediaController:
        def current_media(self):
            return [
                {
                    "player": "org.mpris.MediaPlayer2.firefox",
                    "status": "Playing",
                    "title": "Autoplay Video",
                    "url": "https://www.youtube.com/watch?v=nextVideo",
                    "length_us": 90_000_000,
                    "position_us": 2_000_000,
                }
            ]

    service = DesktopContextService(
        runtime=runtime,
        state_dir=runtime.ctx.config.state_dir,
        media_controller=FakeMediaController(),
        config=DesktopContextConfig(media_commentary_mode_enabled=True, youtube_transcripts_enabled=False),
    )

    result = await service.poll_media_state_once()

    assert result["status"] == "changed"
    assert result["commentary_observation_requested"] is True
    assert result["media_state_changes"][0]["event"] == "media_started"


@pytest.mark.asyncio
async def test_scheduler_forces_body_double_observation_for_media_commentary_transition() -> None:
    observations: list[dict] = []
    scheduler_ref: dict[str, AgentScheduler] = {}

    class FakeDesktopContext:
        config = SimpleNamespace(media_state_poll_seconds=1)

        async def poll_media_state_once(self):
            scheduler_ref["scheduler"]._running = False
            return {
                "status": "changed",
                "commentary_observation_requested": True,
                "commentary_observation_reason": "media_commentary_mode:media_started",
                "media_state_changes": [{"event": "media_started"}],
            }

        async def observe_once(self, **kwargs):
            observations.append(kwargs)
            return {"status": "observed", "reason": kwargs.get("reason")}

    runtime = SimpleNamespace(desktop_context=FakeDesktopContext())
    scheduler = AgentScheduler(runtime)
    scheduler_ref["scheduler"] = scheduler
    scheduler._running = True

    await scheduler._desktop_media_state_loop()

    assert observations == [{"force": True, "reason": "media_commentary_mode:media_started"}]


@pytest.mark.asyncio
async def test_code_heavy_speech_is_redirected_to_note_file(tmp_path: Path) -> None:
    runtime = FakeRuntime(
        tmp_path,
        {
            "should_speak": True,
            "activity_summary": "A traceback is visible.",
            "reason": "The user may need a nudge.",
            "spoken_text": "```python\nraise RuntimeError('boom')\n```",
            "speech_intent": "system_issue",
            "speech_relevance_score": 0.9,
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

    assert result["speech"]["status"] == "skipped"
    assert result["speech"]["reason"] == "redirected_to_note_without_canned_speech"
    assert result["speech"]["redirected_to_note"] is True
    note_path = Path(result["speech"]["note_path"])
    assert note_path.exists()
    assert "RuntimeError('boom')" in note_path.read_text(encoding="utf-8")
    assert spoken == []


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
    assert tools.get("desktop_context_set_task") is not None
    assert tools.get("desktop_context_observe") is not None

    task_result = await tools.execute_async(
        "desktop_context_set_task",
        {"task": "Keep me focused on finishing the screenshot observer."},
    )
    assert task_result.success is True
    assert "{" not in task_result.output
    assert "'config'" not in task_result.output
    assert "Body Double task set" in task_result.output
    assert task_result.metadata["config"]["declared_task"] == (
        "Keep me focused on finishing the screenshot observer."
    )

    result = await tools.execute_async("desktop_context_status", {})
    assert result.success is True
    assert "{" not in result.output
    assert "'config'" not in result.output
    assert "Body Double is enabled" in result.output
    assert result.metadata["status"]["config"]["enabled"] is True
    assert result.metadata["status"]["config"]["declared_task"] == (
        "Keep me focused on finishing the screenshot observer."
    )


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
