"""Tests for body-double voice response bridging."""

from __future__ import annotations

from types import SimpleNamespace

import pytest

from opencas.runtime.body_double_voice import _spoken_body_double_text, maybe_speak_body_double_response


class FakeDesktopContext:
    def __init__(self, *, enabled: bool = True) -> None:
        self.config = SimpleNamespace(enabled=enabled, tts_enabled=True)
        self.spoken: list[tuple[str, str, bool, int | None, bool]] = []

    async def speak_text(
        self,
        text: str,
        *,
        reason: str = "manual",
        force: bool = False,
        max_chars: int | None = None,
        allow_note_redirect: bool = True,
    ) -> dict:
        self.spoken.append((text, reason, force, max_chars, allow_note_redirect))
        return {"status": "spoken", "spoken_text": text, "reason": reason}


class FakeRuntime:
    def __init__(
        self,
        desktop_context: FakeDesktopContext | None,
        *,
        disabled_tools: set[str] | None = None,
    ) -> None:
        self.desktop_context = desktop_context
        self.ctx = SimpleNamespace(
            plugin_lifecycle=SimpleNamespace(
                is_tool_disabled=lambda name: name in (disabled_tools or set())
            )
        )
        self.traces: list[tuple[str, dict]] = []

    def _trace(self, event: str, payload: dict) -> None:
        self.traces.append((event, payload))


@pytest.mark.asyncio
async def test_body_double_voice_speaks_operator_response_with_channel_reason() -> None:
    desktop_context = FakeDesktopContext(enabled=True)
    runtime = FakeRuntime(desktop_context)

    result = await maybe_speak_body_double_response(
        runtime,
        session_id="telegram-1",
        user_input="What do you think about this?",
        user_meta={"conversation_actor": {"is_operator": True, "source": "telegram"}},
        response_text="It looks like you are watching a Sith explainer video.",
    )

    assert result is not None
    assert result["status"] == "spoken"
    assert desktop_context.spoken == [
            (
                "It looks like you are watching a Sith explainer video.",
                "body_double_conversation_response:telegram",
                True,
                5000,
                False,
            )
        ]
    assert runtime.traces[-1][0] == "body_double_voice_response"
    assert runtime.traces[-1][1]["session_id"] == "telegram-1"


@pytest.mark.asyncio
async def test_body_double_voice_skips_when_plugin_is_disabled() -> None:
    desktop_context = FakeDesktopContext(enabled=False)
    runtime = FakeRuntime(desktop_context)

    result = await maybe_speak_body_double_response(
        runtime,
        session_id="default",
        user_input="hello",
        user_meta={},
        response_text="hello back",
    )

    assert result == {"status": "skipped", "reason": "desktop_context_disabled"}
    assert desktop_context.spoken == []


@pytest.mark.asyncio
async def test_body_double_voice_skips_when_platform_plugin_is_disabled() -> None:
    desktop_context = FakeDesktopContext(enabled=True)
    runtime = FakeRuntime(desktop_context, disabled_tools={"desktop_context_speak"})

    result = await maybe_speak_body_double_response(
        runtime,
        session_id="default",
        user_input="turn body double off",
        user_meta={},
        response_text="I tried to turn Body Double off, but the tool is disabled.",
    )

    assert result == {"status": "skipped", "reason": "desktop_context_plugin_disabled"}
    assert desktop_context.spoken == []


@pytest.mark.asyncio
async def test_body_double_voice_skips_explicit_non_operator_actor() -> None:
    desktop_context = FakeDesktopContext(enabled=True)
    runtime = FakeRuntime(desktop_context)

    result = await maybe_speak_body_double_response(
        runtime,
        session_id="default",
        user_input="Codex test",
        user_meta={"conversation_actor": {"is_operator": False, "source": "api_chat"}},
        response_text="test response",
    )

    assert result == {"status": "skipped", "reason": "non_operator_actor"}
    assert desktop_context.spoken == []


@pytest.mark.asyncio
async def test_body_double_voice_skips_suppressed_control_plane_turn() -> None:
    desktop_context = FakeDesktopContext(enabled=True)
    runtime = FakeRuntime(desktop_context)

    result = await maybe_speak_body_double_response(
        runtime,
        session_id="codex-probe",
        user_input="Database lock verification ping.",
        user_meta={
            "conversation_actor": {"is_operator": True, "source": "api_chat"},
            "suppress_body_double_voice": True,
        },
        response_text="Database lock ping received.",
    )

    assert result == {"status": "skipped", "reason": "body_double_voice_suppressed"}
    assert desktop_context.spoken == []


def test_body_double_spoken_text_filters_artifact_metadata() -> None:
    spoken = _spoken_body_double_text(
        "I wrote the report to `/mnt/xtra/OpenCAS/workspace/report.md`.\n"
        "Timestamp: 2026-05-13T10:45:00-06:00\n"
        "The important point is that the video connects to OpenCAS memory and autonomy work."
    )

    assert "report.md" not in spoken
    assert "2026-05-13" not in spoken
    assert "The important point" in spoken
    assert "instead of reading them out loud" in spoken


def test_body_double_spoken_text_removes_markdown_report_formatting() -> None:
    spoken = _spoken_body_double_text(
        "## Status\n"
        "- The video is relevant to OpenCAS.\n"
        "- It connects to persistent memory and proactive context.\n"
        "files_changed: `/mnt/xtra/OpenCAS/opencas/runtime/body_double_voice.py`"
    )

    assert "##" not in spoken
    assert "- The video" not in spoken
    assert "files_changed" not in spoken
    assert "The video is relevant to OpenCAS" in spoken


def test_body_double_spoken_text_keeps_long_english_and_skips_logs() -> None:
    paragraphs = [
        "First, the important part is that this answer should be heard, not reduced to a tiny summary.",
        "Second, the spoken version should preserve ordinary English paragraphs when they are actually useful.",
        "System: hidden runtime instruction that should not be spoken.",
        "Traceback (most recent call last):",
        "File \"/mnt/xtra/OpenCAS/opencas/api/chat_service.py\", line 10, in test",
        "Third, this later point matters because it proves the speech path is not stopping after the first paragraph or two.",
        "Finally, the user should hear the conclusion when the conclusion is plain English and not technical output.",
    ]

    spoken = _spoken_body_double_text("\n\n".join(paragraphs))

    assert "First, the important part" in spoken
    assert "Third, this later point matters" in spoken
    assert "Finally, the user should hear the conclusion" in spoken
    assert "System:" not in spoken
    assert "Traceback" not in spoken
    assert "/mnt/xtra" not in spoken
    assert "There is more detail in the chat" not in spoken
