"""Desktop-context body-double plugin."""

from __future__ import annotations

from opencas.autonomy.models import ActionRiskTier
from opencas.plugins.models import SkillEntry
from opencas.tools.adapters.desktop_context import DesktopContextToolAdapter


def register_skills(skill_registry, tools) -> None:
    adapter = DesktopContextToolAdapter(tools=tools)
    skill_registry.register(
        SkillEntry(
            skill_id="desktop_context",
            name="Desktop Context",
            description=(
                "Toggle desktop screenshot context, create body-double observations, "
                "and speak short local TTS nudges when useful."
            ),
            plugin_id="desktop_context",
        )
    )
    tools.register(
        "desktop_context_status",
        "Inspect whether desktop context observation is enabled, available, and when it last ran.",
        adapter,
        ActionRiskTier.READONLY,
        {"type": "object", "properties": {}, "required": []},
    )
    tools.register(
        "desktop_context_configure",
        (
            "Enable, disable, or tune the desktop context body-double skill. Body Double mode means "
            "desktop_context enabled=true. If the operator asks to turn on Body Double, call this "
            "tool with enabled=true before setting or relying on a task."
        ),
        adapter,
        ActionRiskTier.WORKSPACE_WRITE,
        {
            "type": "object",
            "properties": {
                "enabled": {"type": "boolean"},
                "capture_interval_seconds": {"type": "integer"},
                "min_speech_interval_seconds": {"type": "integer"},
                "tts_enabled": {"type": "boolean"},
                "play_audio": {"type": "boolean"},
                "vision_enabled": {"type": "boolean"},
                "ocr_enabled": {"type": "boolean"},
                "capture_backend": {"type": "string"},
                "vision_model": {"type": "string"},
                "session_id": {"type": "string"},
                "declared_task": {
                    "type": "string",
                    "description": "Explicit operator task for body-double coaching. Empty clears the task.",
                },
                "declared_task_source": {"type": "string"},
                "speech_relevance_threshold": {
                    "type": "number",
                    "description": "Minimum relevance score for non-task spoken observations when no task is declared.",
                },
                "youtube_transcripts_enabled": {
                    "type": "boolean",
                    "description": "When playing YouTube media is detected, retrieve transcript evidence with yt-dlp when possible.",
                },
                "youtube_transcript_max_chars": {"type": "integer"},
                "yt_dlp_path": {"type": "string"},
                "live_transcription_enabled": {
                    "type": "boolean",
                    "description": "Capture short chunks of currently playing system audio and transcribe them with local Whisper for live media following.",
                },
                "live_transcription_capture_seconds": {"type": "number"},
                "live_transcription_min_interval_seconds": {"type": "number"},
                "live_transcription_max_chars": {"type": "integer"},
                "live_transcription_whisper_model": {"type": "string"},
                "live_transcription_audio_input": {
                    "type": "string",
                    "description": "Optional Pulse/PipeWire monitor source for ffmpeg, such as alsa_output.device.monitor.",
                },
                "live_transcription_ffmpeg_path": {"type": "string"},
                "live_transcription_whisper_path": {"type": "string"},
                "live_transcription_timeout_seconds": {"type": "integer"},
                "media_commentary_mode_enabled": {
                    "type": "boolean",
                    "description": "Enable ongoing media/YouTube co-watching commentary while Body Double is active.",
                },
                "media_commentary_source": {"type": "string"},
                "media_commentary_request": {"type": "string"},
                "livestream_resume_catchup_enabled": {
                    "type": "boolean",
                    "description": "After Body Double pauses a browser livestream to speak, resume it with a temporary catch-up playback rate.",
                },
                "livestream_resume_catchup_rate": {"type": "number"},
                "livestream_resume_catchup_max_seconds": {"type": "number"},
                "max_image_bytes": {"type": "integer"},
                "vision_max_dimension": {"type": "integer"},
                "vision_jpeg_quality": {"type": "integer"},
            },
            "required": [],
        },
    )
    tools.register(
        "desktop_context_set_task",
        (
            "Set or clear the explicit operator task used for body-double coaching. This does not "
            "enable desktop observation by itself; call desktop_context_configure with enabled=true "
            "when the operator asks to turn on Body Double. Without a declared task, the body double "
            "observes silently unless the screen shows a concrete relevant issue."
        ),
        adapter,
        ActionRiskTier.WORKSPACE_WRITE,
        {
            "type": "object",
            "properties": {
                "task": {
                    "type": "string",
                    "description": "Task the operator wants body-double support for. Empty string clears the task.",
                },
                "source": {
                    "type": "string",
                    "description": "Where the task declaration came from. Defaults to operator.",
                },
            },
            "required": [],
        },
    )
    tools.register(
        "desktop_context_capture",
        "Capture one active-desktop screenshot and OCR excerpt without asking Bulma to comment.",
        adapter,
        ActionRiskTier.WORKSPACE_WRITE,
        {
            "type": "object",
            "properties": {
                "force": {
                    "type": "boolean",
                    "description": "Capture even if the skill is currently disabled.",
                }
            },
            "required": [],
        },
    )
    tools.register(
        "desktop_context_observe",
        "Capture the active desktop, create context, and let Bulma decide whether a short spoken comment is useful.",
        adapter,
        ActionRiskTier.EXTERNAL_WRITE,
        {
            "type": "object",
            "properties": {
                "force": {"type": "boolean"},
                "reason": {"type": "string"},
                "speak": {
                    "type": "boolean",
                    "description": "Override configured TTS for this observation.",
                },
            },
            "required": [],
        },
    )
    tools.register(
        "desktop_context_speak",
        "Speak one short natural-language local TTS message through the desktop-context voice path.",
        adapter,
        ActionRiskTier.EXTERNAL_WRITE,
        {
            "type": "object",
            "properties": {
                "text": {"type": "string"},
                "reason": {"type": "string"},
            },
            "required": ["text"],
        },
    )
