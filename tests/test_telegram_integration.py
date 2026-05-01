from __future__ import annotations

from pathlib import Path

import pytest

from opencas.telegram_integration import TelegramBotService, TelegramPairingStore


class FakeTelegramClient:
    def __init__(self):
        self.sent_messages = []
        self.edited_messages = []
        self.actions = []
        self.file_payloads = {
            "photo-small": {
                "file_path": "photos/small.jpg",
                "content": b"small-photo",
            },
            "photo-large": {
                "file_path": "photos/large.jpg",
                "content": b"large-photo-content",
            },
        }

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

    async def get_file(self, file_id):
        payload = self.file_payloads[file_id]
        return {
            "file_id": file_id,
            "file_path": payload["file_path"],
            "file_size": len(payload["content"]),
        }

    async def download_file(self, file_path):
        for payload in self.file_payloads.values():
            if payload["file_path"] == file_path:
                return payload["content"]
        raise KeyError(file_path)


class FakeRuntime:
    def __init__(self, response: str = "done"):
        self.response = response
        self.calls = []
        self.ctx = type(
            "Ctx",
            (),
            {
                "config": type(
                    "Config",
                    (),
                    {
                        "state_dir": "",
                        "session_id": "default",
                    },
                )()
            },
        )()

    async def converse(self, text, session_id=None, user_meta=None):
        self.calls.append({"text": text, "session_id": session_id, "user_meta": user_meta})
        return self.response


class FakeVisionLLM:
    def __init__(self):
        self.calls = []

    async def chat_completion(self, **kwargs):
        self.calls.append(kwargs)
        return {"choices": [{"message": {"content": "A screenshot of a Telegram photo card."}}]}


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
    assert runtime.calls[0]["user_meta"] is None
    assert client.actions
    assert client.edited_messages[0]["text"] == "final answer"


@pytest.mark.asyncio
async def test_authorized_photo_without_caption_is_materialized_as_chat_attachment(tmp_path, monkeypatch):
    runtime = FakeRuntime(response="saw the attachment")
    runtime.ctx.config.state_dir = str(tmp_path / "state")
    runtime.ctx.config.agent_workspace_root = lambda: str(tmp_path / "workspace")
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

    async def no_placeholder(chat_id, reply_to_message_id, stop_event):
        return None

    monkeypatch.setattr(service, "_delayed_placeholder", no_placeholder)

    await service.handle_update(
        {
            "update_id": 3,
            "message": {
                "message_id": 11,
                "photo": [
                    {
                        "file_id": "photo-small",
                        "file_unique_id": "small",
                        "width": 90,
                        "height": 90,
                        "file_size": 12,
                    },
                    {
                        "file_id": "photo-large",
                        "file_unique_id": "large",
                        "width": 1280,
                        "height": 720,
                        "file_size": 19,
                    },
                ],
                "chat": {"id": 42, "type": "private"},
                "from": {"id": 42, "username": "operator", "first_name": "Op"},
            },
        }
    )

    assert runtime.calls[0]["session_id"] == "telegram:private:42"
    assert runtime.calls[0]["text"] == "Please review the attached Telegram media."
    attachment = runtime.calls[0]["user_meta"]["attachments"][0]
    assert attachment["filename"].endswith(".jpg")
    assert attachment["media_type"] == "image/jpeg"
    assert attachment["telegram"]["kind"] == "photo"
    assert attachment["telegram"]["file_id"] == "photo-large"
    assert Path(attachment["path"]).read_bytes() == b"large-photo-content"
    assert "not supported yet" not in " ".join(item["text"] for item in client.sent_messages)


@pytest.mark.asyncio
async def test_authorized_captioned_photo_keeps_caption_and_attachment(tmp_path, monkeypatch):
    runtime = FakeRuntime(response="caption response")
    runtime.ctx.config.state_dir = str(tmp_path / "state")
    runtime.ctx.config.agent_workspace_root = lambda: str(tmp_path / "workspace")
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

    async def no_placeholder(chat_id, reply_to_message_id, stop_event):
        return None

    monkeypatch.setattr(service, "_delayed_placeholder", no_placeholder)

    await service.handle_update(
        {
            "update_id": 4,
            "message": {
                "message_id": 12,
                "caption": "What is in this photo?",
                "photo": [{"file_id": "photo-large", "file_unique_id": "large", "width": 1280, "height": 720}],
                "chat": {"id": 42, "type": "private"},
                "from": {"id": 42, "username": "operator", "first_name": "Op"},
            },
        }
    )

    assert runtime.calls[0]["text"] == "What is in this photo?"
    attachment = runtime.calls[0]["user_meta"]["attachments"][0]
    assert attachment["media_type"] == "image/jpeg"
    assert attachment["telegram"]["message_id"] == 12


@pytest.mark.asyncio
async def test_authorized_photo_adds_vision_description_when_available(tmp_path, monkeypatch):
    runtime = FakeRuntime(response="vision response")
    runtime.ctx.config.state_dir = str(tmp_path / "state")
    runtime.ctx.config.agent_workspace_root = lambda: str(tmp_path / "workspace")
    runtime.llm = FakeVisionLLM()
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

    async def no_placeholder(chat_id, reply_to_message_id, stop_event):
        return None

    monkeypatch.setattr(service, "_delayed_placeholder", no_placeholder)

    await service.handle_update(
        {
            "update_id": 5,
            "message": {
                "message_id": 13,
                "photo": [{"file_id": "photo-large", "file_unique_id": "large", "width": 1280, "height": 720}],
                "chat": {"id": 42, "type": "private"},
                "from": {"id": 42, "username": "operator", "first_name": "Op"},
            },
        }
    )

    attachment = runtime.calls[0]["user_meta"]["attachments"][0]
    assert "Image analysis from Telegram media" in attachment["text_content"]
    assert "Telegram photo card" in attachment["text_content"]
    assert attachment["telegram"]["image_analysis"] == "vision"
    vision_content = runtime.llm.calls[0]["messages"][1]["content"]
    assert vision_content[0]["type"] == "text"
    assert vision_content[1]["type"] == "image_url"


@pytest.mark.asyncio
async def test_notify_owner_sends_to_allowlisted_owner(tmp_path):
    runtime = FakeRuntime()
    client = FakeTelegramClient()
    service = TelegramBotService(
        runtime=runtime,
        enabled=True,
        token="123:abc",
        state_dir=tmp_path,
        dm_policy="pairing",
        allow_from=["42"],
        client=client,
    )

    result = await service.notify_owner(
        "I should tell you this.",
        reason="initiative contact",
        urgency="normal",
        source="unit-test",
    )

    assert result["sent"] == 1
    assert result["chat_ids"] == ["42"]
    assert client.sent_messages[0]["chat_id"] == 42
    assert client.sent_messages[0]["text"] == "I should tell you this."
