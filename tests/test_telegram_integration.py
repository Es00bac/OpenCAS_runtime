from __future__ import annotations

import pytest

from opencas.telegram_integration import TelegramBotService, TelegramPairingStore


class FakeTelegramClient:
    def __init__(self):
        self.sent_messages = []
        self.edited_messages = []
        self.actions = []

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
        self.edited_messages.append(
            {"chat_id": chat_id, "message_id": message_id, "text": text}
        )
        return {"message_id": message_id}

    async def send_chat_action(self, chat_id, action="typing"):
        self.actions.append({"chat_id": chat_id, "action": action})
        return {"ok": True}


class FakeRuntime:
    def __init__(self, response: str = "done"):
        self.response = response
        self.calls = []

    async def converse(self, text, session_id=None):
        self.calls.append({"text": text, "session_id": session_id})
        return self.response


@pytest.mark.asyncio
async def test_pairing_store_create_and_approve(tmp_path):
    store = TelegramPairingStore(tmp_path / "pairings.json", ttl_seconds=600)
    request = await store.create_or_get_request(
        "42",
        username="operator",
        first_name="Op",
    )
    snapshot = await store.snapshot()
    assert snapshot["pending_requests"][0]["code"] == request.code

    approved = await store.approve(request.code)
    assert approved is not None
    updated = await store.snapshot()
    assert "42" in updated["approved_user_ids"]
    assert updated["pending_requests"] == []


@pytest.mark.asyncio
async def test_unauthorized_private_message_creates_pairing_request(tmp_path):
    runtime = FakeRuntime()
    client = FakeTelegramClient()
    service = TelegramBotService(
        runtime=runtime,
        enabled=True,
        token="123:abc",
        state_dir=tmp_path,
        dm_policy="pairing",
        client=client,
    )

    await service.handle_update(
        {
            "update_id": 1,
            "message": {
                "message_id": 9,
                "text": "hello",
                "chat": {"id": 42, "type": "private"},
                "from": {"id": 42, "username": "operator", "first_name": "Op"},
            },
        }
    )

    assert runtime.calls == []
    assert len(client.sent_messages) == 1
    assert "Pairing code:" in client.sent_messages[0]["text"]


@pytest.mark.asyncio
async def test_authorized_message_edits_placeholder_into_final_reply(tmp_path, monkeypatch):
    runtime = FakeRuntime(response="final answer")
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
        sent = await client.send_message(
            chat_id,
            "Thinking…",
            reply_to_message_id=reply_to_message_id,
        )
        return sent["message_id"]

    monkeypatch.setattr(service, "_delayed_placeholder", immediate_placeholder)

    await service.handle_update(
        {
            "update_id": 2,
            "message": {
                "message_id": 10,
                "text": "status?",
                "chat": {"id": 42, "type": "private"},
                "from": {"id": 42, "username": "operator", "first_name": "Op"},
            },
        }
    )

    assert runtime.calls[0]["session_id"] == "telegram:private:42"
    assert client.actions
    assert client.edited_messages[0]["text"] == "final answer"
