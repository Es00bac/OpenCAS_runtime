"""Telegram long-poll integration for OpenCAS."""

from __future__ import annotations

import asyncio
import base64
import json
import mimetypes
import re
import secrets
import time
from io import BytesIO
from pathlib import Path
from typing import Any, Dict, List, Optional

import httpx
from pydantic import BaseModel, Field

from opencas.api.chat_service import (
    chat_upload_dir,
    resolve_uploaded_attachment,
    store_uploaded_file,
)
from opencas.telegram_commands import BOT_COMMAND_MENU, TelegramCommandRouter
from opencas.telemetry import EventKind, Tracer

TELEGRAM_TEXT_LIMIT = 4096
TELEGRAM_ATTACHMENT_MAX_BYTES = 20_000_000
TELEGRAM_VISION_MAX_BYTES = 5_000_000


class TelegramPairingRequest(BaseModel):
    """Pending or approved Telegram pairing request."""

    code: str
    user_id: str
    username: Optional[str] = None
    first_name: Optional[str] = None
    last_name: Optional[str] = None
    requested_at: float = Field(default_factory=time.time)
    approved_at: Optional[float] = None

    @property
    def approved(self) -> bool:
        return self.approved_at is not None


class TelegramPairingStore:
    """Small JSON-backed store for Telegram pairing approvals."""

    def __init__(self, path: Path | str, ttl_seconds: int = 3600) -> None:
        self.path = Path(path)
        self.ttl_seconds = max(60, int(ttl_seconds))
        self._lock = asyncio.Lock()

    async def snapshot(self) -> Dict[str, Any]:
        async with self._lock:
            state = self._load_state()
            self._prune_expired(state)
            self._save_state(state)
            return self._serialize_state(state)

    async def is_authorized(self, user_id: str, allow_from: List[str]) -> bool:
        async with self._lock:
            state = self._load_state()
            self._prune_expired(state)
            if user_id in set(state.get("approved_user_ids", [])):
                self._save_state(state)
                return True
            return user_id in {str(item).strip() for item in allow_from if str(item).strip()}

    async def create_or_get_request(
        self,
        user_id: str,
        *,
        username: Optional[str] = None,
        first_name: Optional[str] = None,
        last_name: Optional[str] = None,
    ) -> TelegramPairingRequest:
        async with self._lock:
            state = self._load_state()
            self._prune_expired(state)
            if user_id in set(state.get("approved_user_ids", [])):
                request = TelegramPairingRequest(
                    code=self._generate_code(),
                    user_id=user_id,
                    username=username,
                    first_name=first_name,
                    last_name=last_name,
                    approved_at=time.time(),
                )
                return request
            for item in state.get("requests", []):
                if item.get("user_id") == user_id and item.get("approved_at") is None:
                    request = TelegramPairingRequest.model_validate(item)
                    self._save_state(state)
                    return request
            request = TelegramPairingRequest(
                code=self._generate_code(),
                user_id=user_id,
                username=username,
                first_name=first_name,
                last_name=last_name,
            )
            state.setdefault("requests", []).append(request.model_dump(mode="json"))
            self._save_state(state)
            return request

    async def approve(self, code: str) -> Optional[TelegramPairingRequest]:
        normalized = code.strip().upper()
        if not normalized:
            return None
        async with self._lock:
            state = self._load_state()
            self._prune_expired(state)
            for item in state.get("requests", []):
                if str(item.get("code", "")).upper() != normalized:
                    continue
                item["approved_at"] = time.time()
                approved_ids = set(state.get("approved_user_ids", []))
                approved_ids.add(str(item.get("user_id", "")))
                state["approved_user_ids"] = sorted(approved_ids)
                self._save_state(state)
                return TelegramPairingRequest.model_validate(item)
            self._save_state(state)
            return None

    def _load_state(self) -> Dict[str, Any]:
        if not self.path.exists():
            return {"approved_user_ids": [], "requests": []}
        try:
            raw = json.loads(self.path.read_text(encoding="utf-8"))
            if isinstance(raw, dict):
                raw.setdefault("approved_user_ids", [])
                raw.setdefault("requests", [])
                return raw
        except Exception:
            pass
        return {"approved_user_ids": [], "requests": []}

    def _save_state(self, state: Dict[str, Any]) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.path.write_text(json.dumps(state, ensure_ascii=True, indent=2), encoding="utf-8")

    def _prune_expired(self, state: Dict[str, Any]) -> None:
        cutoff = time.time() - self.ttl_seconds
        kept: List[Dict[str, Any]] = []
        for item in state.get("requests", []):
            approved_at = item.get("approved_at")
            if approved_at is not None:
                kept.append(item)
                continue
            requested_at = float(item.get("requested_at", 0.0) or 0.0)
            if requested_at >= cutoff:
                kept.append(item)
        state["requests"] = kept

    def _serialize_state(self, state: Dict[str, Any]) -> Dict[str, Any]:
        requests = [TelegramPairingRequest.model_validate(item) for item in state.get("requests", [])]
        pending = [item.model_dump(mode="json") for item in requests if not item.approved]
        approved_requests = [item.model_dump(mode="json") for item in requests if item.approved]
        return {
            "approved_user_ids": sorted({str(item) for item in state.get("approved_user_ids", []) if str(item)}),
            "pending_requests": pending,
            "approved_requests": approved_requests,
        }

    @staticmethod
    def _generate_code() -> str:
        return secrets.token_hex(4).upper()


class TelegramApiClient:
    """Very small Telegram Bot API client backed by httpx."""

    def __init__(
        self,
        token: str,
        *,
        base_url: str = "https://api.telegram.org",
        timeout: float = 40.0,
        transport: Any = None,
    ) -> None:
        self.token = token
        self.base_url = f"{base_url.rstrip('/')}/bot{token}"
        self.file_base_url = f"{base_url.rstrip('/')}/file/bot{token}"
        self._client = httpx.AsyncClient(timeout=timeout, transport=transport)

    async def close(self) -> None:
        await self._client.aclose()

    async def get_me(self) -> Dict[str, Any]:
        return await self._request("getMe")

    async def get_updates(self, *, offset: int, timeout: int = 30) -> List[Dict[str, Any]]:
        result = await self._request("getUpdates", json_payload={"offset": offset, "timeout": timeout})
        return result if isinstance(result, list) else []

    async def send_message(
        self,
        chat_id: int,
        text: str,
        *,
        reply_to_message_id: Optional[int] = None,
    ) -> Dict[str, Any]:
        payload: Dict[str, Any] = {"chat_id": chat_id, "text": text}
        if reply_to_message_id is not None:
            payload["reply_to_message_id"] = reply_to_message_id
        return await self._request("sendMessage", json_payload=payload)

    async def edit_message_text(self, chat_id: int, message_id: int, text: str) -> Dict[str, Any]:
        return await self._request(
            "editMessageText",
            json_payload={"chat_id": chat_id, "message_id": message_id, "text": text},
        )

    async def send_chat_action(self, chat_id: int, action: str = "typing") -> Dict[str, Any]:
        return await self._request(
            "sendChatAction",
            json_payload={"chat_id": chat_id, "action": action},
        )

    async def set_my_commands(self, commands: List[Dict[str, str]]) -> Dict[str, Any]:
        return await self._request("setMyCommands", json_payload={"commands": commands})

    async def get_file(self, file_id: str) -> Dict[str, Any]:
        result = await self._request("getFile", json_payload={"file_id": file_id})
        return result if isinstance(result, dict) else {}

    async def download_file(self, file_path: str) -> bytes:
        response = await self._client.get(f"{self.file_base_url}/{file_path.lstrip('/')}")
        response.raise_for_status()
        return response.content

    async def _request(self, method: str, *, json_payload: Optional[Dict[str, Any]] = None) -> Any:
        response = await self._client.post(f"{self.base_url}/{method}", json=json_payload or {})
        response.raise_for_status()
        payload = response.json()
        if not payload.get("ok", False):
            raise RuntimeError(str(payload.get("description", f"Telegram API error in {method}")))
        return payload.get("result")


class TelegramBotService:
    """Long-poll Telegram bot service that routes DMs into OpenCAS conversations."""

    def __init__(
        self,
        runtime: Any,
        *,
        enabled: bool,
        token: str,
        state_dir: Path | str,
        dm_policy: str = "pairing",
        allow_from: Optional[List[str]] = None,
        poll_interval_seconds: float = 1.0,
        pairing_ttl_seconds: int = 3600,
        api_base_url: str = "https://api.telegram.org",
        tracer: Optional[Tracer] = None,
        client: Optional[TelegramApiClient] = None,
    ) -> None:
        self.runtime = runtime
        self.enabled = bool(enabled)
        self.token = token.strip()
        self.dm_policy = dm_policy
        self.allow_from = [str(item).strip() for item in (allow_from or []) if str(item).strip()]
        self.poll_interval_seconds = max(0.2, float(poll_interval_seconds))
        self.tracer = tracer
        telegram_state_dir = Path(state_dir).expanduser() / "telegram"
        self._offset_path = telegram_state_dir / "update_offset.json"
        self.pairing_store = TelegramPairingStore(
            telegram_state_dir / "pairings.json",
            ttl_seconds=pairing_ttl_seconds,
        )
        self.client = client or TelegramApiClient(self.token, base_url=api_base_url)
        self._task: Optional[asyncio.Task[None]] = None
        self._stop_event = asyncio.Event()
        self._last_update_id = self._load_offset()
        self._last_error: Optional[str] = None
        self._bot_info: Optional[Dict[str, Any]] = None
        self._chat_locks: Dict[int, asyncio.Lock] = {}
        self.commands = TelegramCommandRouter(
            runtime,
            state_path=telegram_state_dir / "command_sessions.json",
        )

    @property
    def configured(self) -> bool:
        return bool(self.token)

    async def start(self) -> None:
        if not self.enabled or not self.configured or self._task is not None:
            return
        self._stop_event.clear()
        try:
            self._bot_info = await self.client.get_me()
        except Exception as exc:
            self._last_error = str(exc)
            self._trace("telegram_start_failed", {"error": str(exc)}, kind=EventKind.ERROR)
            raise
        try:
            await self.client.set_my_commands(
                [{"command": name, "description": desc} for name, desc in BOT_COMMAND_MENU]
            )
        except Exception as exc:
            self._last_error = str(exc)
            self._trace(
                "telegram_set_commands_failed",
                {"error": str(exc)},
                kind=EventKind.ERROR,
            )
        self._task = asyncio.create_task(self._poll_loop())
        self._trace("telegram_started", {"bot_username": self._bot_info.get("username")})

    async def stop(self) -> None:
        self._stop_event.set()
        if self._task is not None:
            self._task.cancel()
            try:
                await self._task
            except asyncio.CancelledError:
                pass
            self._task = None
        await self.client.close()
        self._trace("telegram_stopped", {})

    async def status(self) -> Dict[str, Any]:
        pairings = await self.pairing_store.snapshot()
        return {
            "enabled": self.enabled,
            "configured": self.configured,
            "token_configured": self.configured,
            "running": self._task is not None and not self._task.done(),
            "dm_policy": self.dm_policy,
            "allow_from": sorted(set(self.allow_from)),
            "bot": {
                "id": self._bot_info.get("id") if self._bot_info else None,
                "username": self._bot_info.get("username") if self._bot_info else None,
                "first_name": self._bot_info.get("first_name") if self._bot_info else None,
                "link": (
                    f"https://t.me/{self._bot_info.get('username')}"
                    if self._bot_info and self._bot_info.get("username")
                    else None
                ),
            },
            "last_update_id": self._last_update_id,
            "last_error": self._last_error,
            "pairings": pairings,
        }

    async def approve_pairing(self, code: str) -> Optional[TelegramPairingRequest]:
        request = await self.pairing_store.approve(code)
        if request is None:
            return None
        try:
            await self.client.send_message(
                int(request.user_id),
                "Pairing approved. You can chat with OpenCAS now.",
            )
        except Exception as exc:
            self._last_error = str(exc)
            self._trace(
                "telegram_pairing_notify_failed",
                {"error": str(exc), "user_id": request.user_id},
                kind=EventKind.ERROR,
            )
        self._trace("telegram_pairing_approved", {"user_id": request.user_id, "code": request.code})
        return request

    async def notify_owner(
        self,
        text: str,
        *,
        reason: str = "",
        urgency: str = "normal",
        source: str = "runtime",
        document_path: Optional[Path] = None,
        document_filename: Optional[str] = None,
        document_caption: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Send an owner-directed proactive notification to approved private chats."""
        del document_path, document_filename, document_caption
        chat_ids = await self._owner_chat_ids()
        if not chat_ids:
            self._trace(
                "telegram_owner_notify_skipped",
                {"reason": "no_owner_chat", "source": source, "urgency": urgency},
            )
            return {"sent": 0, "chat_ids": [], "failed": [], "reason": "no_owner_chat"}

        sent: list[str] = []
        failed: list[Dict[str, str]] = []
        for raw_chat_id in chat_ids:
            try:
                chat_id = int(raw_chat_id)
                chunks = _chunk_telegram_text(text)
                for chunk in chunks:
                    await self.client.send_message(chat_id, chunk)
                sent.append(str(raw_chat_id))
            except Exception as exc:
                self._last_error = str(exc)
                failed.append({"chat_id": str(raw_chat_id), "error": str(exc)})
        self._trace(
            "telegram_owner_notified",
            {
                "sent": len(sent),
                "failed": len(failed),
                "reason": reason,
                "urgency": urgency,
                "source": source,
            },
            kind=EventKind.TOOL_CALL if sent else EventKind.ERROR,
        )
        return {"sent": len(sent), "chat_ids": sent, "failed": failed}

    async def _owner_chat_ids(self) -> list[str]:
        snapshot = await self.pairing_store.snapshot()
        approved = {
            str(item).strip()
            for item in snapshot.get("approved_user_ids", [])
            if str(item).strip()
        }
        allowed = {str(item).strip() for item in self.allow_from if str(item).strip()}
        return sorted(allowed | approved)

    async def handle_update(self, update: Dict[str, Any]) -> None:
        message = update.get("message") or update.get("edited_message")
        if not isinstance(message, dict):
            return
        chat = message.get("chat") or {}
        sender = message.get("from") or {}
        chat_id = chat.get("id")
        user_id = sender.get("id")
        if not isinstance(chat_id, int) or not isinstance(user_id, int):
            return
        if sender.get("is_bot"):
            return

        text = message.get("text") or message.get("caption") or ""
        if not isinstance(text, str):
            text = ""

        authorized = await self._is_authorized(chat.get("type"), str(user_id))
        if not authorized:
            if chat.get("type") == "private" and self.dm_policy == "pairing":
                request = await self.pairing_store.create_or_get_request(
                    str(user_id),
                    username=sender.get("username"),
                    first_name=sender.get("first_name"),
                    last_name=sender.get("last_name"),
                )
                reply = (
                    "This Telegram account is not paired with OpenCAS yet.\n"
                    f"Your Telegram user id: {user_id}\n"
                    f"Pairing code: {request.code}\n"
                    "Approve it in the OpenCAS dashboard Telegram panel."
                )
                await self.client.send_message(chat_id, reply, reply_to_message_id=message.get("message_id"))
                self._trace("telegram_pairing_requested", {"user_id": user_id, "code": request.code})
            return

        lock = self._chat_locks.setdefault(chat_id, asyncio.Lock())
        async with lock:
            reply_to_message_id = message.get("message_id")
            try:
                attachments = await self._materialize_telegram_attachments(message)
            except Exception as exc:
                self._last_error = str(exc)
                self._trace(
                    "telegram_media_download_failed",
                    {
                        "chat_id": chat_id,
                        "user_id": user_id,
                        "message_id": message.get("message_id"),
                        "error": str(exc),
                    },
                    kind=EventKind.ERROR,
                )
                if chat.get("type") == "private":
                    await self.client.send_message(
                        chat_id,
                        f"I received Telegram media, but could not download it: {exc}",
                        reply_to_message_id=reply_to_message_id,
                    )
                return
            if not text.strip() and not attachments:
                if chat.get("type") == "private":
                    await self.client.send_message(
                        chat_id,
                        "I received that Telegram message, but it did not include "
                        "supported text or downloadable media.",
                        reply_to_message_id=reply_to_message_id,
                    )
                return

            if text.strip() and self.commands.is_command(text):
                bot_username = self._bot_info.get("username") if self._bot_info else None
                command_reply = await self.commands.dispatch(
                    chat_id=chat_id,
                    user_id=user_id,
                    text=text,
                    bot_username=bot_username,
                )
                if command_reply is not None:
                    self._trace(
                        "telegram_command_dispatched",
                        {"chat_id": chat_id, "user_id": user_id, "command": text.split()[0]},
                    )
                    await self._deliver_response(
                        chat_id,
                        command_reply,
                        reply_to_message_id=reply_to_message_id,
                        placeholder_message_id=None,
                    )
                    return

            session_id = self.commands.session_id_for(chat.get("type", "chat"), chat_id)
            conversation_text = text.strip() or "Please review the attached Telegram media."
            stop_typing = asyncio.Event()
            try:
                await self.client.send_chat_action(chat_id, "typing")
            except Exception as exc:
                self._last_error = str(exc)
                self._trace(
                    "telegram_typing_failed",
                    {"chat_id": chat_id, "error": str(exc)},
                    kind=EventKind.ERROR,
                )
            typing_task = asyncio.create_task(self._typing_keepalive(chat_id, stop_typing))
            placeholder_task = asyncio.create_task(
                self._delayed_placeholder(chat_id, reply_to_message_id, stop_typing)
            )
            try:
                self._trace(
                    "telegram_message_received",
                    {
                        "chat_id": chat_id,
                        "user_id": user_id,
                        "session_id": session_id,
                        "attachment_count": len(attachments),
                    },
                )
                if attachments:
                    response = await self.runtime.converse(
                        conversation_text,
                        session_id=session_id,
                        user_meta={"attachments": attachments},
                    )
                else:
                    response = await self.runtime.converse(
                        conversation_text,
                        session_id=session_id,
                    )
            except Exception as exc:
                response = f"[Error: {exc}]"
                self._last_error = str(exc)
                self._trace(
                    "telegram_message_failed",
                    {"chat_id": chat_id, "user_id": user_id, "error": str(exc)},
                    kind=EventKind.ERROR,
                )
            finally:
                stop_typing.set()
                await typing_task

            placeholder_message_id: Optional[int] = None
            if placeholder_task.done():
                try:
                    placeholder_message_id = placeholder_task.result()
                except Exception:
                    placeholder_message_id = None
            else:
                placeholder_task.cancel()
                try:
                    await placeholder_task
                except asyncio.CancelledError:
                    pass

            await self._deliver_response(
                chat_id,
                response,
                reply_to_message_id=reply_to_message_id,
                placeholder_message_id=placeholder_message_id,
            )

    async def _materialize_telegram_attachments(self, message: Dict[str, Any]) -> List[Dict[str, Any]]:
        specs = self._telegram_attachment_specs(message)
        if not specs:
            return []
        upload_dir = chat_upload_dir(self.runtime)
        attachments: List[Dict[str, Any]] = []
        for spec in specs:
            file_id = str(spec.get("file_id") or "").strip()
            if not file_id:
                continue
            file_info = await self.client.get_file(file_id)
            file_path = str(file_info.get("file_path") or "").strip()
            if not file_path:
                raise RuntimeError(f"Telegram did not return a file path for {spec.get('kind')}")
            declared_size = int(file_info.get("file_size") or spec.get("file_size") or 0)
            if declared_size > TELEGRAM_ATTACHMENT_MAX_BYTES:
                raise RuntimeError(
                    f"Telegram {spec.get('kind')} is too large "
                    f"({declared_size} bytes > {TELEGRAM_ATTACHMENT_MAX_BYTES})"
                )
            content = await self.client.download_file(file_path)
            if len(content) > TELEGRAM_ATTACHMENT_MAX_BYTES:
                raise RuntimeError(
                    f"Telegram {spec.get('kind')} is too large "
                    f"({len(content)} bytes > {TELEGRAM_ATTACHMENT_MAX_BYTES})"
                )
            filename = _safe_telegram_filename(
                str(spec.get("filename") or Path(file_path).name or "telegram_attachment")
            )
            media_type = str(spec.get("media_type") or "").strip()
            if not media_type:
                media_type = mimetypes.guess_type(filename)[0] or "application/octet-stream"
            stored = store_uploaded_file(
                upload_dir,
                filename=filename,
                content_type=media_type,
                fileobj=BytesIO(content),
            )
            resolved = resolve_uploaded_attachment(upload_dir, stored)
            resolved["telegram"] = {
                "kind": spec.get("kind"),
                "message_id": message.get("message_id"),
                "file_id": file_id,
                "file_unique_id": spec.get("file_unique_id"),
                "file_path": file_path,
                "declared_size_bytes": declared_size or None,
            }
            for key in ("width", "height", "duration", "performer", "title"):
                if spec.get(key) is not None:
                    resolved["telegram"][key] = spec.get(key)
            await self._attach_image_analysis(resolved)
            attachments.append(resolved)
        return attachments

    async def _attach_image_analysis(self, attachment: Dict[str, Any]) -> None:
        media_type = str(attachment.get("media_type") or "")
        if not media_type.startswith("image/"):
            return
        path = Path(str(attachment.get("path") or ""))
        try:
            image_bytes = path.read_bytes()
        except OSError:
            return
        if not image_bytes or len(image_bytes) > TELEGRAM_VISION_MAX_BYTES:
            telegram_meta = attachment.setdefault("telegram", {})
            if isinstance(telegram_meta, dict):
                telegram_meta["image_analysis"] = "skipped_size"
            return
        llm = getattr(self.runtime, "llm", None)
        if llm is None or not hasattr(llm, "chat_completion"):
            telegram_meta = attachment.setdefault("telegram", {})
            if isinstance(telegram_meta, dict):
                telegram_meta["image_analysis"] = "unavailable"
            return

        image_uri = "data:{media_type};base64,{payload}".format(
            media_type=media_type,
            payload=base64.b64encode(image_bytes).decode("ascii"),
        )
        prompt = (
            "Describe this Telegram image attachment for the OpenCAS agent's chat context. "
            "Be factual and concise. If it is a screenshot or document-like image, "
            "summarize visible text and layout without inventing missing details."
        )
        try:
            response = await llm.chat_completion(
                messages=[
                    {
                        "role": "system",
                        "content": (
                            "You analyze user-provided Telegram images so the agent can respond "
                            "from grounded visual evidence. Do not infer private facts not visible "
                            "in the image."
                        ),
                    },
                    {
                        "role": "user",
                        "content": [
                            {"type": "text", "text": prompt},
                            {"type": "image_url", "image_url": {"url": image_uri}},
                        ],
                    },
                ],
                complexity="light",
                payload={"temperature": 0.1, "max_tokens": 400},
                source="telegram_media_image_analysis",
            )
            description = self._extract_llm_response_text(response)
        except Exception as exc:
            telegram_meta = attachment.setdefault("telegram", {})
            if isinstance(telegram_meta, dict):
                telegram_meta["image_analysis"] = "failed"
                telegram_meta["image_analysis_error"] = str(exc)[:160]
            return

        if not description:
            return
        attachment["text_content"] = (
            "Image analysis from Telegram media:\n"
            + self._truncate_attachment_description(description)
        )
        attachment["text_truncated"] = len(description) > 4000
        attachment["language_hint"] = "image-description"
        telegram_meta = attachment.setdefault("telegram", {})
        if isinstance(telegram_meta, dict):
            telegram_meta["image_analysis"] = "vision"

    @staticmethod
    def _telegram_attachment_specs(message: Dict[str, Any]) -> List[Dict[str, Any]]:
        specs: List[Dict[str, Any]] = []
        photos = message.get("photo")
        if isinstance(photos, list) and photos:
            photo_items = [item for item in photos if isinstance(item, dict) and item.get("file_id")]
            if photo_items:
                best = max(
                    photo_items,
                    key=lambda item: (
                        int(item.get("file_size") or 0),
                        int(item.get("width") or 0) * int(item.get("height") or 0),
                    ),
                )
                specs.append(
                    {
                        "kind": "photo",
                        "file_id": best.get("file_id"),
                        "file_unique_id": best.get("file_unique_id"),
                        "file_size": best.get("file_size"),
                        "width": best.get("width"),
                        "height": best.get("height"),
                        "filename": f"telegram_{message.get('message_id') or 'message'}_photo.jpg",
                        "media_type": "image/jpeg",
                    }
                )

        media_fields = {
            "document": ("document", "telegram_document", "application/octet-stream"),
            "video": ("video", "telegram_video.mp4", "video/mp4"),
            "animation": ("animation", "telegram_animation.mp4", "video/mp4"),
            "audio": ("audio", "telegram_audio.mp3", "audio/mpeg"),
            "voice": ("voice", "telegram_voice.oga", "audio/ogg"),
            "video_note": ("video_note", "telegram_video_note.mp4", "video/mp4"),
            "sticker": ("sticker", "telegram_sticker.webp", "image/webp"),
        }
        for field, (kind, default_name, default_media_type) in media_fields.items():
            item = message.get(field)
            if not isinstance(item, dict) or not item.get("file_id"):
                continue
            filename = str(item.get("file_name") or default_name)
            specs.append(
                {
                    "kind": kind,
                    "file_id": item.get("file_id"),
                    "file_unique_id": item.get("file_unique_id"),
                    "file_size": item.get("file_size"),
                    "width": item.get("width"),
                    "height": item.get("height"),
                    "duration": item.get("duration"),
                    "performer": item.get("performer"),
                    "title": item.get("title"),
                    "filename": filename,
                    "media_type": item.get("mime_type") or default_media_type,
                }
            )
        return specs

    @staticmethod
    def _extract_llm_response_text(response: Dict[str, Any]) -> str:
        choices = response.get("choices", []) if isinstance(response, dict) else []
        if not choices:
            return ""
        message = choices[0].get("message", {}) if isinstance(choices[0], dict) else {}
        return str(message.get("content") or "").strip()

    @staticmethod
    def _truncate_attachment_description(text: str, limit: int = 4000) -> str:
        normalized = str(text or "").strip()
        if len(normalized) <= limit:
            return normalized
        return normalized[: max(0, limit - 3)].rstrip() + "..."

    async def _poll_loop(self) -> None:
        while not self._stop_event.is_set():
            try:
                updates = await self.client.get_updates(offset=self._last_update_id, timeout=30)
                for update in updates:
                    update_id = int(update.get("update_id", 0) or 0)
                    await self.handle_update(update)
                    self._last_update_id = max(self._last_update_id, update_id + 1)
                    self._save_offset(self._last_update_id)
                self._last_error = None
            except asyncio.CancelledError:
                raise
            except Exception as exc:
                self._last_error = str(exc)
                self._trace("telegram_poll_error", {"error": str(exc)}, kind=EventKind.ERROR)
                await asyncio.sleep(self.poll_interval_seconds)

    async def _is_authorized(self, chat_type: Optional[str], user_id: str) -> bool:
        if self.dm_policy == "disabled":
            return False
        if self.dm_policy == "open":
            return True
        if self.dm_policy == "allowlist":
            return user_id in set(self.allow_from)
        if chat_type != "private":
            return await self.pairing_store.is_authorized(user_id, self.allow_from)
        return await self.pairing_store.is_authorized(user_id, self.allow_from)

    async def _typing_keepalive(self, chat_id: int, stop_event: asyncio.Event) -> None:
        while not stop_event.is_set():
            try:
                await self.client.send_chat_action(chat_id, "typing")
            except Exception as exc:
                self._last_error = str(exc)
                self._trace(
                    "telegram_typing_failed",
                    {"chat_id": chat_id, "error": str(exc)},
                    kind=EventKind.ERROR,
                )
                return
            try:
                await asyncio.wait_for(stop_event.wait(), timeout=3.5)
            except asyncio.TimeoutError:
                continue

    async def _delayed_placeholder(
        self,
        chat_id: int,
        reply_to_message_id: Optional[int],
        stop_event: asyncio.Event,
    ) -> Optional[int]:
        try:
            await asyncio.wait_for(stop_event.wait(), timeout=1.0)
            return None
        except asyncio.TimeoutError:
            sent = await self.client.send_message(
                chat_id,
                "Thinking…",
                reply_to_message_id=reply_to_message_id,
            )
            message_id = sent.get("message_id")
            return int(message_id) if isinstance(message_id, int) else None

    async def _deliver_response(
        self,
        chat_id: int,
        text: str,
        *,
        reply_to_message_id: Optional[int],
        placeholder_message_id: Optional[int],
    ) -> None:
        chunks = _chunk_telegram_text(text)
        if not chunks:
            return
        if placeholder_message_id is not None:
            await self.client.edit_message_text(chat_id, placeholder_message_id, chunks[0])
            remaining = chunks[1:]
        else:
            await self.client.send_message(
                chat_id,
                chunks[0],
                reply_to_message_id=reply_to_message_id,
            )
            remaining = chunks[1:]
        for chunk in remaining:
            await self.client.send_message(chat_id, chunk)

    def _trace(self, message: str, payload: Dict[str, Any], *, kind: EventKind = EventKind.TOOL_CALL) -> None:
        if self.tracer is None:
            return
        self.tracer.log(kind, message, payload)

    def _load_offset(self) -> int:
        if not self._offset_path.exists():
            return 0
        try:
            payload = json.loads(self._offset_path.read_text(encoding="utf-8"))
            return int(payload.get("offset", 0) or 0)
        except Exception:
            return 0

    def _save_offset(self, offset: int) -> None:
        self._offset_path.parent.mkdir(parents=True, exist_ok=True)
        self._offset_path.write_text(json.dumps({"offset": offset}, ensure_ascii=True), encoding="utf-8")


def _chunk_telegram_text(text: str, limit: int = TELEGRAM_TEXT_LIMIT) -> List[str]:
    normalized = str(text or "").strip()
    if not normalized:
        return []
    if len(normalized) <= limit:
        return [normalized]

    chunks: List[str] = []
    remaining = normalized
    while remaining:
        if len(remaining) <= limit:
            chunks.append(remaining)
            break
        split_at = remaining.rfind("\n", 0, limit)
        if split_at <= 0:
            split_at = remaining.rfind(" ", 0, limit)
        if split_at <= 0:
            split_at = limit
        chunk = remaining[:split_at].rstrip()
        if chunk:
            chunks.append(chunk)
        remaining = remaining[split_at:].lstrip()
    return chunks


def _safe_telegram_filename(filename: str) -> str:
    raw = Path(str(filename or "telegram_attachment")).name.strip()
    if not raw:
        raw = "telegram_attachment"
    safe = re.sub(r"[^A-Za-z0-9._-]+", "_", raw)
    return safe.strip("._") or "telegram_attachment"
