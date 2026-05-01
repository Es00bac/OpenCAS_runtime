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
        "Enable, disable, or tune the desktop context body-double skill. Disabled by default for privacy.",
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
                "max_image_bytes": {"type": "integer"},
                "vision_max_dimension": {"type": "integer"},
                "vision_jpeg_quality": {"type": "integer"},
            },
            "required": [],
        },
    )
    tools.register(
        "desktop_context_capture",
        "Capture one active-desktop screenshot and OCR excerpt without asking the OpenCAS agent to comment.",
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
        "Capture the active desktop, create context, and let the OpenCAS agent decide whether a short spoken comment is useful.",
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
