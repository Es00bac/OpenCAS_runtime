from __future__ import annotations

from datetime import datetime, timezone
from types import SimpleNamespace

import pytest

from opencas.telegram_commands import (
    BOT_COMMAND_MENU,
    TelegramCommandRouter,
)
from opencas.telegram_integration import TelegramBotService


class FakeTelegramClient:
    def __init__(self):
        self.sent_messages = []
        self.edited_messages = []
        self.actions = []
        self.commands_set = None

    async def close(self):
        return None

    async def get_me(self):
        return {"id": 1, "username": "opencas_bot", "first_name": "OpenCAS"}

    async def send_message(self, chat_id, text, reply_to_message_id=None):
        message_id = len(self.sent_messages) + 100
        self.sent_messages.append(
            {
                "chat_id": chat_id,
                "text": text,
                "reply_to_message_id": reply_to_message_id,
                "message_id": message_id,
            }
        )
        return {"message_id": message_id}

    async def edit_message_text(self, chat_id, message_id, text):
        self.edited_messages.append({"chat_id": chat_id, "message_id": message_id, "text": text})
        return {"message_id": message_id}

    async def send_chat_action(self, chat_id, action="typing"):
        self.actions.append({"chat_id": chat_id, "action": action})
        return {"ok": True}

    async def set_my_commands(self, commands):
        self.commands_set = list(commands)
        return {"ok": True}


def _build_runtime() -> SimpleNamespace:
    now = datetime(2026, 4, 19, 12, 0, tzinfo=timezone.utc)

    somatic_state = SimpleNamespace(
        state_id="ss-1",
        updated_at=now,
        arousal=0.4,
        fatigue=0.2,
        tension=0.3,
        valence=0.1,
        focus=0.7,
        energy=0.8,
        certainty=0.6,
        somatic_tag="engaged",
        to_memory_salience_modifier=lambda: 1.1,
    )
    somatic_mgr = SimpleNamespace(state=somatic_state)

    musubi_state = SimpleNamespace(
        state_id="ms-1",
        updated_at=now,
        musubi=0.45,
        dimensions={"trust": 0.6, "resonance": 0.3, "presence": 0.5, "attunement": 0.4},
        source_tag="warm",
        continuity_breadcrumb="last turn felt grounded",
    )
    relational = SimpleNamespace(state=musubi_state)

    self_model = SimpleNamespace(
        name="OpenCAS",
        version="0.9",
        narrative="Quiet, durable, helpful.",
        values=["curiosity", "honesty"],
        traits=["steady"],
        current_goals=["ship the Telegram commands"],
        current_intention="help the operator",
        self_beliefs={},
        relational_state_id=None,
        recent_activity=[],
        model_id="id-1",
        updated_at=now,
        source_system="test",
        imported_identity_profile={},
        memory_anchors=[],
        recent_themes=[],
        identity_rebuild_audit={},
    )
    user_model = SimpleNamespace(
        model_id="um-1",
        updated_at=now,
        explicit_preferences={"tone": "direct"},
        inferred_goals=["ship reliable features"],
        known_boundaries=["no surprise pushes"],
        trust_level=0.8,
        uncertainty_areas=["unfamiliar codebases"],
    )
    continuity = SimpleNamespace(
        state_id="cs-1",
        updated_at=now,
        last_session_id="telegram:private:42",
        last_shutdown_time=None,
        boot_count=17,
        version="1.0",
    )
    identity = SimpleNamespace(self_model=self_model, user_model=user_model, continuity=continuity)

    ctx = SimpleNamespace(
        somatic=somatic_mgr,
        relational=relational,
        identity=identity,
        config=SimpleNamespace(default_llm_model="anthropic/claude-sonnet-4-6"),
    )
    runtime = SimpleNamespace(ctx=ctx)

    async def converse(text, session_id=None):
        runtime.calls.append({"text": text, "session_id": session_id})
        return "ok"

    runtime.calls = []
    runtime.converse = converse
    return runtime


@pytest.mark.asyncio
async def test_new_command_bumps_session(tmp_path):
    runtime = _build_runtime()
    router = TelegramCommandRouter(runtime, state_path=tmp_path / "sessions.json")
    before = router.session_id_for("private", 42)
    assert before == "telegram:private:42"

    reply = await router.dispatch(chat_id=42, user_id=42, text="/new")
    assert "new conversation" in reply.lower()

    after = router.session_id_for("private", 42)
    assert after == "telegram:private:42:1"


@pytest.mark.asyncio
async def test_help_lists_every_menu_command(tmp_path):
    runtime = _build_runtime()
    router = TelegramCommandRouter(runtime, state_path=tmp_path / "sessions.json")
    reply = await router.dispatch(chat_id=1, user_id=1, text="/help")
    for name, _ in BOT_COMMAND_MENU:
        assert f"/{name}" in reply


@pytest.mark.asyncio
async def test_somatic_command_renders_state(tmp_path):
    runtime = _build_runtime()
    router = TelegramCommandRouter(runtime, state_path=tmp_path / "sessions.json")
    reply = await router.dispatch(chat_id=1, user_id=1, text="/somatic")
    assert "arousal" in reply
    assert "somatic tag:" in reply
    assert "0.40" in reply


@pytest.mark.asyncio
async def test_musubi_command_includes_all_dimensions(tmp_path):
    runtime = _build_runtime()
    router = TelegramCommandRouter(runtime, state_path=tmp_path / "sessions.json")
    reply = await router.dispatch(chat_id=1, user_id=1, text="/musubi")
    for dim in ("trust", "resonance", "presence", "attunement"):
        assert dim in reply
    assert "composite musubi" in reply


@pytest.mark.asyncio
async def test_identity_and_user_and_continuity(tmp_path):
    runtime = _build_runtime()
    router = TelegramCommandRouter(runtime, state_path=tmp_path / "sessions.json")

    identity = await router.dispatch(chat_id=1, user_id=1, text="/identity")
    assert "OpenCAS" in identity
    assert "curiosity" in identity

    user = await router.dispatch(chat_id=1, user_id=1, text="/user")
    assert "trust level" in user
    assert "tone" in user

    continuity = await router.dispatch(chat_id=1, user_id=1, text="/continuity")
    assert "boot count:      17" in continuity


@pytest.mark.asyncio
async def test_status_combines_subsystems(tmp_path):
    runtime = _build_runtime()
    router = TelegramCommandRouter(runtime, state_path=tmp_path / "sessions.json")
    reply = await router.dispatch(chat_id=42, user_id=42, text="/status")
    assert "OpenCAS" in reply
    assert "somatic:" in reply
    assert "musubi:" in reply
    assert "session:" in reply


@pytest.mark.asyncio
async def test_unknown_command_hints_help(tmp_path):
    runtime = _build_runtime()
    router = TelegramCommandRouter(runtime, state_path=tmp_path / "sessions.json")
    reply = await router.dispatch(chat_id=1, user_id=1, text="/nope")
    assert "/help" in reply


@pytest.mark.asyncio
async def test_mention_suffix_is_stripped(tmp_path):
    runtime = _build_runtime()
    router = TelegramCommandRouter(runtime, state_path=tmp_path / "sessions.json")
    reply = await router.dispatch(
        chat_id=1, user_id=1, text="/help@opencas_bot", bot_username="opencas_bot"
    )
    assert "/help" in reply


@pytest.mark.asyncio
async def test_command_dispatch_skips_runtime_converse(tmp_path):
    runtime = _build_runtime()
    client = FakeTelegramClient()
    service = TelegramBotService(
        runtime=runtime,
        enabled=True,
        token="123:abc",
        state_dir=tmp_path,
        dm_policy="pairing",
        client=client,
    )
    request = await service.pairing_store.create_or_get_request("42")
    await service.pairing_store.approve(request.code)

    await service.handle_update(
        {
            "update_id": 1,
            "message": {
                "message_id": 9,
                "text": "/somatic",
                "chat": {"id": 42, "type": "private"},
                "from": {"id": 42, "username": "operator"},
            },
        }
    )

    assert runtime.calls == []
    assert any("arousal" in msg["text"] for msg in client.sent_messages)


@pytest.mark.asyncio
async def test_new_command_changes_next_converse_session(tmp_path):
    runtime = _build_runtime()
    client = FakeTelegramClient()
    service = TelegramBotService(
        runtime=runtime,
        enabled=True,
        token="123:abc",
        state_dir=tmp_path,
        dm_policy="pairing",
        client=client,
    )
    request = await service.pairing_store.create_or_get_request("42")
    await service.pairing_store.approve(request.code)

    async def immediate_placeholder(chat_id, reply_to_message_id, stop_event):
        return None

    service._delayed_placeholder = immediate_placeholder  # type: ignore[assignment]

    await service.handle_update(
        {
            "update_id": 1,
            "message": {
                "message_id": 1,
                "text": "hello",
                "chat": {"id": 42, "type": "private"},
                "from": {"id": 42},
            },
        }
    )
    assert runtime.calls[-1]["session_id"] == "telegram:private:42"

    await service.handle_update(
        {
            "update_id": 2,
            "message": {
                "message_id": 2,
                "text": "/new",
                "chat": {"id": 42, "type": "private"},
                "from": {"id": 42},
            },
        }
    )

    await service.handle_update(
        {
            "update_id": 3,
            "message": {
                "message_id": 3,
                "text": "how are you?",
                "chat": {"id": 42, "type": "private"},
                "from": {"id": 42},
            },
        }
    )

    assert runtime.calls[-1]["session_id"] == "telegram:private:42:1"
